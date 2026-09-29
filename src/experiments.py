"""Improvement experiments: what actually makes the pre-match model better?

    python -m src.experiments

Every experiment is scored the same way, on walk-forward seasons inside the
training period: train on every allowed season before N, predict N, for
N = 2019/20 to 2023/24. The validation (2024/25) and test (2025/26) seasons
are never touched here, so the choices made from this table (config.yaml
``features.groups`` and the first training season) cannot be tuned to them.

Hyperparameters are held at the values tuned in Phase 3 so the table
isolates the effect of data and features. The pipeline re-tunes the final
configuration.

Writes reports/experiments.json, reports/experiments.md and
reports/figures/experiments.png.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402

from src.data.clean import build_match_table  # noqa: E402
from src.data.collect import collect_seasons, raw_path  # noqa: E402
from src.data.odds import attach_odds, download_odds, implied_probabilities, odds_matrix  # noqa: E402
from src.features.build_features import FEATURE_GROUPS, build_features, feature_columns  # noqa: E402
from src.features.elo import EloSettings, elo_fit_error  # noqa: E402
from src.models.evaluate import to_markdown  # noqa: E402
from src.models.train_random_forest import build_random_forest  # noqa: E402
from src.models.train_xgboost import build_xgboost  # noqa: E402
from src.utils.config import load_config, resolve_path, season_label, setup_logging  # noqa: E402
from src.utils.plotting import ENTITY_COLORS, NEUTRAL, apply_style  # noqa: E402

logger = logging.getLogger("experiments")

# Phase 3's walk-forward-tuned hyperparameters, held fixed for every row.
PHASE3_PARAMS = {
    "Random Forest": {"n_estimators": 500, "min_samples_split": 10, "min_samples_leaf": 20,
                      "max_features": 0.3, "max_depth": 4, "class_weight": None},
    "XGBoost": {"subsample": 0.6, "reg_lambda": 1.0, "reg_alpha": 0, "n_estimators": 300,
                "min_child_weight": 10, "max_depth": 2, "learning_rate": 0.01, "gamma": 0,
                "colsample_bytree": 0.5},
}
MARKET_FEATURES = ["market_p_home", "market_p_draw", "market_p_away"]
ELO_GRID = {"k": [8, 10, 12, 15, 20, 25], "home_advantage": [40, 50, 60, 70, 80],
            "season_regression": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]}


@dataclass(frozen=True)
class Experiment:
    name: str
    groups: tuple[str, ...] = ()
    train_start: int = 2005
    base_features: bool = True
    pooled_leagues: bool = False
    market: bool = False
    note: str = field(default="", compare=False)

    def columns(self) -> list[str]:
        if self.base_features:
            cols = feature_columns(True, list(self.groups))
        else:
            cols = [c for g in FEATURE_GROUPS if g in self.groups for c in FEATURE_GROUPS[g]]
        return cols + (MARKET_FEATURES if self.market else [])


EXPERIMENTS = [
    Experiment("Phase 3 setup", (), 2015, note="42 features, trained from 2015/16"),
    Experiment("+ more history", (), 2005, note="same features, trained from 2005/06"),
    Experiment("+ Elo ratings", ("elo",), 2005),
    Experiment("+ weighted form", ("elo", "ewm"), 2005, note="exponentially weighted form incl. shots conceded"),
    Experiment("+ match context", ("elo", "ewm", "context"), 2005,
               note="rest days, unbeaten/winless runs, last season's finish, head-to-head"),
    Experiment("Elo + weighted form only", ("elo", "ewm"), 2005, base_features=False,
               note="the 42 base features dropped"),
    Experiment("+ other top leagues", ("elo", "ewm"), 2005, pooled_leagues=True,
               note="La Liga, Bundesliga, Serie A and Ligue 1 added to training; scored on the EPL only"),
    Experiment("+ bookmaker odds as inputs", ("elo", "ewm"), 2005, market=True,
               note="market-informed variant; needs live odds to be used for real predictions"),
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_epl(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg, feat_cfg = config["data"], config["features"]
    raw_dir = resolve_path(data_cfg["raw_dir"])
    first, last = data_cfg["history_start_season"], data_cfg["last_season"]
    collect_seasons(first, last, data_cfg["sources"], raw_dir)
    matches = build_match_table({y: raw_path(raw_dir, y) for y in range(first, last + 1)})
    features = build_features(matches, feat_cfg["form_window"], feat_cfg["venue_window"],
                              feat_cfg["venue_min_periods"], feat_cfg["ewm_halflife"], feat_cfg["elo"])
    odds_cfg = config["odds"]
    odds = download_odds(odds_cfg["url"], resolve_path(odds_cfg["cache"]), odds_cfg["bookmaker"], first)
    features = attach_odds(features, odds)
    has_odds = features["odds_home"].notna()
    implied = implied_probabilities(odds_matrix(features[has_odds]))
    for i, column in enumerate(["market_p_away", "market_p_draw", "market_p_home"]):
        features.loc[has_odds, column] = implied[:, i]
    return matches, features


def load_other_leagues(config: dict, last_season: int) -> pd.DataFrame:
    """Feature tables for the other big-five leagues, each built on its own."""
    exp_cfg = config["experiments"]["other_leagues"]
    feat_cfg = config["features"]
    first = config["data"]["history_start_season"]
    frames = []
    for league in exp_cfg["leagues"]:
        sources = [{"name": s["name"], "url_template": s["url_template"].replace("{league}", league["slug"])
                    .replace("{league_code}", league["code"])} for s in exp_cfg["sources"]]
        raw_dir = resolve_path(exp_cfg["raw_dir"])
        collect_seasons(first, last_season, sources, raw_dir, league_code=league["code"])
        matches = build_match_table({y: raw_path(raw_dir, y, league["code"]) for y in range(first, last_season + 1)},
                                    team_prefix=league["code"])
        features = build_features(matches, feat_cfg["form_window"], feat_cfg["venue_window"],
                                  feat_cfg["venue_min_periods"], feat_cfg["ewm_halflife"], feat_cfg["elo"])
        frames.append(features.assign(league=league["code"]))
        logger.info("%s: %d matches", league["name"], len(features))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _build(name: str, seed: int):
    params = PHASE3_PARAMS[name]
    return build_random_forest(params, seed) if name == "Random Forest" else build_xgboost(params, seed)


def score_experiment(exp: Experiment, epl: pd.DataFrame, other: pd.DataFrame | None, folds: list[int],
                     seed: int) -> dict:
    columns = exp.columns()
    result = {"experiment": exp.name, "note": exp.note, "n_features": len(columns),
              "train_start": season_label(exp.train_start)}
    for name in ("Random Forest", "XGBoost"):
        losses, accuracies, draw_shares = [], [], []
        for season in folds:
            train = epl[(epl["season"] >= exp.train_start) & (epl["season"] < season)]
            if exp.pooled_leagues and other is not None:
                train = pd.concat([train, other[(other["season"] >= exp.train_start) & (other["season"] < season)]])
            test = epl[epl["season"] == season]
            model = _build(name, seed).fit(train[columns], train["target"].astype(int))
            proba = model.predict_proba(test[columns])
            losses.append(log_loss(test["target"], proba, labels=[0, 1, 2]))
            accuracies.append(accuracy_score(test["target"], proba.argmax(axis=1)))
            draw_shares.append(float((proba.argmax(axis=1) == 1).mean()))
        result[f"{name} log loss"] = float(np.mean(losses))
        result[f"{name} accuracy"] = float(np.mean(accuracies))
        result[f"{name} draws predicted"] = float(np.mean(draw_shares))
        result[f"{name} per season"] = [round(v, 4) for v in losses]
    logger.info("%-28s RF %.4f / %.1f%%   XGB %.4f / %.1f%%", exp.name, result["Random Forest log loss"],
                100 * result["Random Forest accuracy"], result["XGBoost log loss"], 100 * result["XGBoost accuracy"])
    return result


def bookmaker_reference(epl: pd.DataFrame, folds: list[int]) -> dict:
    losses, accuracies, draws = [], [], []
    for season in folds:
        block = epl[epl["season"] == season]
        proba = block[["market_p_away", "market_p_draw", "market_p_home"]].to_numpy()
        losses.append(log_loss(block["target"], proba, labels=[0, 1, 2]))
        accuracies.append(accuracy_score(block["target"], proba.argmax(axis=1)))
        draws.append(float((proba.argmax(axis=1) == 1).mean()))
    return {"log_loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies)),
            "draws_predicted": float(np.mean(draws))}


def tune_elo(matches: pd.DataFrame, seasons: list[int]) -> dict:
    """Grid search for the Elo settings on training seasons only."""
    rows = []
    for k, home_advantage, regression in itertools.product(*ELO_GRID.values()):
        settings = EloSettings(k=k, home_advantage=home_advantage, season_regression=regression)
        rows.append({"k": k, "home_advantage": home_advantage, "season_regression": regression,
                     "squared_error": elo_fit_error(matches, settings, seasons)})
    table = pd.DataFrame(rows).sort_values("squared_error")
    return {"best": table.iloc[0].to_dict(), "top10": table.head(10).to_dict(orient="records"),
            "seasons": [season_label(s) for s in (seasons[0], seasons[-1])]}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _plot(table: pd.DataFrame, bookmaker: dict, path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(table))[::-1]
    height = 0.38
    ax.barh(y + height / 2, table["Random Forest log loss"], height, color=ENTITY_COLORS["Random Forest"],
            label="Random Forest")
    ax.barh(y - height / 2, table["XGBoost log loss"], height, color=ENTITY_COLORS["XGBoost"], label="XGBoost")
    ax.axvline(bookmaker["log_loss"], color=ENTITY_COLORS["Bookmaker"], linewidth=2, label="Bookmaker")
    ax.set_yticks(y, table["experiment"])
    low = min(bookmaker["log_loss"], table[["Random Forest log loss", "XGBoost log loss"]].min().min())
    high = table[["Random Forest log loss", "XGBoost log loss"]].max().max()
    ax.set_xlim(low - 0.01, high + 0.005)
    ax.set_xlabel("Mean walk-forward log loss, 2019/20 to 2023/24 (lower is better)")
    ax.set_title("What improved the model")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    for spine in ("left",):
        ax.spines[spine].set_color(NEUTRAL)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render(results: list[dict], bookmaker: dict, elo: dict, folds: list[int]) -> str:
    table = pd.DataFrame([{
        "Experiment": r["experiment"], "Features": r["n_features"], "Trained from": r["train_start"],
        "RF log loss": r["Random Forest log loss"], "RF accuracy": r["Random Forest accuracy"],
        "XGB log loss": r["XGBoost log loss"], "XGB accuracy": r["XGBoost accuracy"],
    } for r in results])
    table.loc[len(table)] = ["Bookmaker (Bet365, margin removed)", 0, "", bookmaker["log_loss"],
                             bookmaker["accuracy"], bookmaker["log_loss"], bookmaker["accuracy"]]
    table["Features"] = table["Features"].map(lambda v: "" if v == 0 else str(v))
    notes = [f"* **{r['experiment']}**: {r['note']}" for r in results if r["note"]]
    best = elo["best"]
    return "\n".join([
        "# Improvement experiments",
        "",
        "_Generated by `python -m src.experiments`._",
        "",
        f"Each row is scored on walk-forward seasons {season_label(folds[0])} to {season_label(folds[-1])}: train on "
        "every allowed season before N, predict N, average over the five seasons. The validation and test seasons "
        "are not used. Hyperparameters are held at Phase 3's values, so differences come from data and features "
        "only. Each row after the first adds to the row above unless its note says otherwise.",
        "",
        to_markdown(table, "{:.4f}"),
        "",
        *notes,
        "",
        "![Experiments](figures/experiments.png)",
        "",
        "## Elo settings",
        "",
        f"Chosen by the lowest squared error of Elo's expected home score on seasons {elo['seasons'][0]} to "
        f"{elo['seasons'][1]} (training seasons only): K = {best['k']:.0f}, home advantage = "
        f"{best['home_advantage']:.0f} rating points, {best['season_regression']:.0%} pulled back to the mean "
        "between seasons.",
        "",
    ])


def run(config_path: str | None = None, include_other_leagues: bool = True) -> dict:
    config = load_config(config_path)
    seed = config["random_state"]
    folds = config["split"]["walk_forward_validation_seasons"]
    matches, epl = load_epl(config)
    epl = epl[epl["target"].notna()]
    other = load_other_leagues(config, max(folds) - 1) if include_other_leagues else None
    experiments = [e for e in EXPERIMENTS if include_other_leagues or not e.pooled_leagues]

    results = [score_experiment(exp, epl, other, folds, seed) for exp in experiments]
    bookmaker = bookmaker_reference(epl, folds)
    elo = tune_elo(matches, list(range(config["data"]["history_start_season"] + 3, max(folds) + 1)))

    reports = resolve_path(config["output"]["reports_dir"])
    (reports / "experiments.json").write_text(json.dumps(
        {"folds": folds, "results": results, "bookmaker": bookmaker, "elo_tuning": elo,
         "hyperparameters": PHASE3_PARAMS}, indent=2, default=str))
    _plot(pd.DataFrame(results), bookmaker, reports / "figures" / "experiments.png")
    (reports / "experiments.md").write_text(render(results, bookmaker, elo, folds))
    logger.info("Wrote %s", reports / "experiments.md")
    return {"results": results, "bookmaker": bookmaker, "elo": elo}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score model-improvement experiments on walk-forward seasons.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-other-leagues", action="store_true", help="Skip downloading the other leagues")
    args = parser.parse_args()
    setup_logging()
    run(args.config, include_other_leagues=not args.skip_other_leagues)
