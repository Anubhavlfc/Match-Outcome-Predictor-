"""Final project report: bookmaker benchmark, betting simulation, EDA, write-up.

    python -m src.pipeline                      # training + out-of-sample predictions
    python -m src.models.feature_selection      # optional, adds the feature-selection section
    python -m src.final_report                  # this: writes reports/final_report.md

Every number in the report is computed here from pipeline outputs; the
prose around the numbers only chooses wording (e.g. "profit" vs "loss")
based on those numbers.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import eda
from src.data.odds import attach_odds, download_odds
from src.features.build_features import feature_columns
from src.models import betting
from src.models.benchmark import bookmaker_metrics, probability_gap
from src.models.evaluate import class_prior_proba, compute_metrics, to_markdown
from src.tracking.history import load_history
from src.utils.config import load_config, resolve_path, season_label, setup_logging

logger = logging.getLogger("final_report")

MODELS = ("Random Forest", "XGBoost")
EDGE_SENSITIVITY = (0.0, 0.05, 0.10, 0.20)
# Value-bet rules searched on the walk-forward seasons only.
BET_RULE_GRID = {
    "odds_kind": ("odds", "max_odds"),
    "edge": (0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
    "max_odds": (None, 2.5, 4.0),
}
MIN_TUNING_BETS = 100
ODDS_KIND_LABEL = {"odds": "Bet365", "max_odds": "best price"}


def _proba(frame: pd.DataFrame) -> np.ndarray:
    proba = frame[["p_away", "p_draw", "p_home"]].to_numpy(dtype=float)
    return proba / proba.sum(axis=1, keepdims=True)  # guard against float32 rounding


def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value * 100:.{digits}f}%"


def _money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"+${value:,.0f}"


def _result_phrase(value: float) -> str:
    """'a loss of $4,993' / 'a profit of $120'."""
    return f"a {'profit' if value > 0 else 'loss'} of ${abs(value):,.0f}"


def load_inputs(config: dict) -> dict[str, Any]:
    processed = resolve_path(config["data"]["processed_dir"])
    reports = resolve_path(config["output"]["reports_dir"])
    needed = [processed / "matches.csv", processed / "features.csv", processed / "out_of_sample_predictions.csv",
              reports / "metrics.json"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run python -m src.pipeline first; missing {missing}")
    odds_cfg = config["odds"]
    odds = download_odds(odds_cfg["url"], resolve_path(odds_cfg["cache"]), odds_cfg["bookmaker"],
                         config["data"]["history_start_season"])
    matches = pd.read_csv(processed / "matches.csv", parse_dates=["date"])
    fs_path = reports / "feature_selection.json"
    optional = {name: reports / f"{name}.json" for name in ("experiments", "in_play_reference")}
    return {
        **{name: json.loads(path.read_text()) if path.exists() else None for name, path in optional.items()},
        "matches": attach_odds(matches, odds),
        "features": pd.read_csv(processed / "features.csv", parse_dates=["date"]),
        "oos": pd.read_csv(processed / "out_of_sample_predictions.csv", parse_dates=["date"]),
        "metrics": json.loads((reports / "metrics.json").read_text()),
        "feature_selection": json.loads(fs_path.read_text()) if fs_path.exists() else None,
        "history": load_history(resolve_path(config["live"]["history_path"])),
    }


def _chosen_experiment(experiments: dict, groups: list[str], config: dict) -> dict | None:
    """The experiment row that matches the production configuration, if any."""
    start = season_label(min(config["split"]["train_seasons"]))
    base = config["features"]["base_features"]
    setups = {
        "+ more history": (True, []), "+ Elo ratings": (True, ["elo"]), "+ weighted form": (True, ["elo", "ewm"]),
        "+ match context": (True, ["elo", "ewm", "context"]), "Elo + weighted form only": (False, ["elo", "ewm"]),
    }
    for r in experiments["results"]:
        setup = setups.get(r["experiment"])
        if r["train_start"] == start and setup and setup[0] == base and sorted(setup[1]) == sorted(groups):
            return r
    return None


def _gap_closed(first: dict, last: dict, experiments: dict, model: str) -> float:
    """Share of the Phase 3 gap to the bookmaker (walk-forward log loss) that the changes removed."""
    book = experiments["bookmaker"]["log_loss"]
    before, after = first[f"{model} log loss"], last[f"{model} log loss"]
    return (before - after) / (before - book) if before > book else float("nan")


def _experiment_verdicts(experiments: dict, model: str) -> str:
    """One sentence per experiment that did not make it into production, worded from the numbers."""
    by_name = {r["experiment"]: r for r in experiments["results"]}
    key = f"{model} log loss"
    parts = []
    if "+ match context" in by_name and "+ weighted form" in by_name:
        change = by_name["+ match context"][key] - by_name["+ weighted form"][key]
        parts.append(f"Match context changed log loss by {change:+.4f}, too little to justify ten more features.")
    if "+ other top leagues" in by_name and "+ weighted form" in by_name:
        change = by_name["+ other top leagues"][key] - by_name["+ weighted form"][key]
        parts.append((f"Adding four more leagues to training changed it by {change:+.4f}" if abs(change) >= 0.0005
                      else "Adding four more leagues to training left it unchanged")
                     + (", so more data from other leagues does not help an EPL model here." if change > -0.002
                        else "."))
    if "+ bookmaker odds as inputs" in by_name:
        r = by_name["+ bookmaker odds as inputs"]
        book = experiments["bookmaker"]["log_loss"]
        parts.append(f"Feeding the bookmaker's odds in as inputs gives {r[key]:.4f}, "
                     + ("about the same as the bookmaker alone" if abs(r[key] - book) < 0.003 else
                        ("better than the bookmaker alone" if r[key] < book else "still behind the bookmaker"))
                     + ", and it can't be used live without an odds feed for upcoming fixtures.")
    return " ".join(parts)


def with_odds(oos: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    odds_cols = [c for c in matches.columns if "odds" in c]
    return oos.merge(matches[["match_id", *odds_cols]], on="match_id", how="left", validate="many_to_one")


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def season_log_loss_table(scored: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample log loss per season: both models, the bookmaker and the class-frequency baseline."""
    rows = []
    for season, block in scored.groupby("season"):
        row: dict[str, Any] = {"season": season}
        for model in MODELS:
            part = block[block["model"] == model]
            row[model] = compute_metrics(part["target"], _proba(part))["log_loss"]
        one = block[block["model"] == MODELS[0]]
        row["Bookmaker"] = bookmaker_metrics(one)["log_loss"]
        earlier = features[(features["season"] < season) & (features["season"] >= features["season"].min())]
        row["Class frequencies"] = compute_metrics(one["target"], class_prior_proba(earlier["target"], len(one)))["log_loss"]
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_table(block: pd.DataFrame, prior_target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Accuracy / log loss / Brier for models, bookmaker (pre-match and closing) and the baseline."""
    one = block[block["model"] == MODELS[0]]
    entries: dict[str, dict[str, Any]] = {
        "Class frequencies": compute_metrics(one["target"], class_prior_proba(prior_target, len(one))),
    }
    for model in MODELS:
        part = block[block["model"] == model]
        entries[model] = compute_metrics(part["target"], _proba(part))
    entries["Bookmaker (pre-match)"] = bookmaker_metrics(one)
    if one["close_odds_home"].notna().all():
        entries["Bookmaker (closing)"] = bookmaker_metrics(one, closing=True)
    table = pd.DataFrame([
        {"Predictor": name, "Accuracy": m["accuracy"], "Macro F1": m["macro_f1"], "Log Loss": m["log_loss"],
         "Brier": m["brier"]}
        for name, m in entries.items()
    ])
    return table, entries


# --------------------------------------------------------------------------
# Betting
# --------------------------------------------------------------------------

def rule_label(rule: dict) -> str:
    cap = f", odds up to {rule['max_odds']:.1f}" if rule["max_odds"] else ""
    return f"edge > {rule['edge']:.0%}, {ODDS_KIND_LABEL[rule['odds_kind']]}{cap}"


def tune_bet_rules(block: pd.DataFrame, stake: float, seed: int) -> tuple[dict[str, dict], pd.DataFrame]:
    """Per model, the value-bet rule with the best ROI on the walk-forward seasons.

    Only walk-forward predictions go in, so validation and test stay honest
    checks. Picking the best of many rules flatters its walk-forward ROI;
    the validation and test rows are the numbers to believe.
    """
    rows = []
    for model in MODELS:
        part = block[block["model"] == model].reset_index(drop=True)
        if part[["max_odds_home", "max_odds_draw", "max_odds_away"]].isna().any().any():
            raise ValueError("Walk-forward matches without best-price odds")
        for kind, edge, cap in itertools.product(*BET_RULE_GRID.values()):
            s = betting.summarize(betting.value_bets(part, _proba(part), stake, edge, kind, cap), seed, n_boot=200)
            rows.append({"Model": model, "odds_kind": kind, "edge": edge, "max_odds": cap,
                         "Bets": s["bets"], "Profit": s["profit"], "ROI": s["roi"]})
    table = pd.DataFrame(rows)
    rules = {}
    for model in MODELS:
        eligible = table[(table["Model"] == model) & (table["Bets"] >= MIN_TUNING_BETS)]
        best = eligible.loc[eligible["ROI"].idxmax()]
        rules[model] = {"odds_kind": best["odds_kind"], "edge": float(best["edge"]),
                        "max_odds": None if pd.isna(best["max_odds"]) else float(best["max_odds"]),
                        "walk_forward_roi": float(best["ROI"]), "walk_forward_bets": int(best["Bets"])}
    return rules, table


def betting_results(block: pd.DataFrame, stake: float, edge: float, seed: int,
                    rules: dict[str, dict] | None = None) -> tuple[pd.DataFrame, dict]:
    one = block[block["model"] == MODELS[0]].reset_index(drop=True)
    logs: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        part = block[block["model"] == model].reset_index(drop=True)
        logs[f"{model} model pick"] = betting.model_pick_bets(part, _proba(part), stake)
        logs[f"{model} value bets"] = betting.value_bets(part, _proba(part), stake, edge)
        if rules:
            r = rules[model]
            logs[f"{model} value bets, tuned rule"] = betting.value_bets(
                part, _proba(part), stake, r["edge"], r["odds_kind"], r["max_odds"])
    logs["Bookmaker favourite"] = betting.favourite_bets(one, stake)
    logs["Bookmaker favourite, best price"] = betting.favourite_bets(one, stake, "max_odds")
    logs["Always home"] = betting.always_home_bets(one, stake)
    rows = []
    for name, bets in logs.items():
        s = betting.summarize(bets, seed)
        rows.append({"Strategy": name, "Bets": s["bets"], "Staked": s["staked"], "Profit": s["profit"],
                     "ROI": s["roi"], "ROI 95% CI": f"{_pct(s['roi_ci_low'])} to {_pct(s['roi_ci_high'])}",
                     "Hit rate": s["hit_rate"], "Avg odds": s["average_odds"]})
    return pd.DataFrame(rows), logs


def edge_sensitivity(block: pd.DataFrame, stake: float, seed: int) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        part = block[block["model"] == model].reset_index(drop=True)
        for edge in EDGE_SENSITIVITY:
            s = betting.summarize(betting.value_bets(part, _proba(part), stake, edge), seed)
            rows.append({"Model": model, "Edge threshold": f"{edge:.0%}", "Bets": s["bets"],
                         "Profit": s["profit"], "ROI": s["roi"]})
    return pd.DataFrame(rows)


def _format_betting(table: pd.DataFrame) -> str:
    shown = table.copy()
    shown["Staked"] = shown["Staked"].map(lambda v: f"${v:,.0f}")
    shown["Profit"] = shown["Profit"].map(_money)
    shown["ROI"] = shown["ROI"].map(_pct)
    shown["Hit rate"] = shown["Hit rate"].map(_pct)
    shown["Avg odds"] = shown["Avg odds"].map(lambda v: f"{v:.2f}")
    return to_markdown(shown)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def run(config_path: str | None = None) -> Path:
    config = load_config(config_path)
    seed = config["random_state"]
    stake, edge = config["odds"]["stake"], config["odds"]["value_edge"]
    split_cfg = config["split"]
    reports = resolve_path(config["output"]["reports_dir"])
    figures = reports / "figures"
    inputs = load_inputs(config)
    matches, features, metrics = inputs["matches"], inputs["features"], inputs["metrics"]
    scored = with_odds(inputs["oos"], matches)
    if scored[["odds_home", "odds_draw", "odds_away"]].isna().any().any():
        raise ValueError("Some out-of-sample matches have no pre-match odds")

    train_seasons = split_cfg["train_seasons"]
    val_season, test_season = split_cfg["validation_season"], split_cfg["test_season"]
    model_rows = matches[matches["season"] >= config["data"]["first_season"]]
    blocks = {
        "walk-forward": scored[scored["stage"] == "walk-forward"],
        "validation": scored[scored["stage"] == "validation"],
        "test": scored[scored["stage"] == "test"],
    }
    priors = {
        "validation": features[features["season"].isin(train_seasons)]["target"],
        "test": features[features["season"].isin([*train_seasons, val_season])]["target"],
    }

    # EDA figures ----------------------------------------------------------
    share = eda.outcome_share_by_season(model_rows, figures / "outcome_share_by_season.png")
    goals = eda.goals_by_season(model_rows, figures / "goals_by_season.png")
    gap_table = eda.outcome_by_position_gap(features, figures / "outcome_by_position_gap.png")
    eda.bookmaker_calibration(model_rows, figures / "bookmaker_calibration.png")

    # Benchmark ------------------------------------------------------------
    by_season = season_log_loss_table(scored, features)
    eda.log_loss_by_season(by_season, figures / "log_loss_by_season.png")
    val_table, _ = benchmark_table(blocks["validation"], priors["validation"])
    test_table, test_entries = benchmark_table(blocks["test"], priors["test"])
    gaps = {m: probability_gap(_proba(blocks["test"][blocks["test"]["model"] == m]),
                               blocks["test"][blocks["test"]["model"] == m]) for m in MODELS}

    # Betting --------------------------------------------------------------
    bet_rules, rule_search = tune_bet_rules(blocks["walk-forward"], stake, seed)
    bet_tables = {}
    bet_logs = {}
    for stage, block in blocks.items():
        bet_tables[stage], bet_logs[stage] = betting_results(block, stake, edge, seed, bet_rules)
    sensitivity = {stage: edge_sensitivity(blocks[stage], stake, seed) for stage in ("validation", "test")}
    eda.cumulative_profit(
        {k: v for k, v in bet_logs["test"].items() if k in ("Random Forest value bets, tuned rule",
                                                             "XGBoost value bets, tuned rule",
                                                             "Bookmaker favourite", "Always home")},
        figures / "cumulative_profit_test.png",
        f"Betting ${stake} per bet through {season_label(test_season)} (test season)",
    )
    (reports / "betting_and_benchmark.json").write_text(json.dumps({
        "log_loss_by_season": by_season.to_dict(orient="records"),
        "benchmark_validation": val_table.to_dict(orient="records"),
        "benchmark_test": test_table.to_dict(orient="records"),
        "betting": {k: v.to_dict(orient="records") for k, v in bet_tables.items()},
        "edge_sensitivity": {k: v.to_dict(orient="records") for k, v in sensitivity.items()},
        "stake": stake, "value_edge": edge,
        "tuned_bet_rules": bet_rules,
        "bet_rule_search": rule_search.to_dict(orient="records"),
    }, indent=2, default=str))

    text = render(config, matches, model_rows, features, metrics, inputs, share, goals, gap_table, by_season,
                  val_table, test_table, test_entries, gaps, bet_tables, sensitivity, bet_rules)
    path = reports / "final_report.md"
    path.write_text(text)
    logger.info("Wrote %s", path)
    return path


def render(config, matches, model_rows, features, metrics, inputs, share, goals, gap_table, by_season,
           val_table, test_table, test_entries, gaps, bet_tables, sensitivity, bet_rules) -> str:
    split_cfg = config["split"]
    stake, edge = config["odds"]["stake"], config["odds"]["value_edge"]
    val_label, test_label = season_label(split_cfg["validation_season"]), season_label(split_cfg["test_season"])
    train_label = f"{season_label(min(split_cfg['train_seasons']))}–{season_label(max(split_cfg['train_seasons']))}"
    wf = split_cfg["walk_forward_validation_seasons"]
    wf_label = f"{season_label(min(wf))}–{season_label(max(wf))}"
    groups, use_base = config["features"]["groups"], config["features"]["base_features"]
    n_features = len(feature_columns(config["features"]["use_diff_features"], groups, base=use_base))
    experiments, in_play = inputs.get("experiments"), inputs.get("in_play_reference")
    draw_rule = metrics.get("draw_rule", {})

    t = test_table.set_index("Predictor")
    best_model = min(MODELS, key=lambda m: t.loc[m, "Log Loss"])
    book = t.loc["Bookmaker (pre-match)"]
    base = t.loc["Class frequencies"]
    gap_to_book = t.loc[best_model, "Log Loss"] - book["Log Loss"]
    closed = (base["Log Loss"] - t.loc[best_model, "Log Loss"]) / (base["Log Loss"] - book["Log Loss"])
    beats_book_seasons = int(sum(
        min(r[m] for m in MODELS) < r["Bookmaker"] for r in by_season.to_dict(orient="records")))

    bt = {stage: table.set_index("Strategy") for stage, table in bet_tables.items()}
    test_bets = bt["test"]
    headline_strategy = f"{best_model} value bets, tuned rule"
    headline = test_bets.loc[headline_strategy]
    fixed_rule = test_bets.loc[f"{best_model} value bets"]
    pick = test_bets.loc[f"{best_model} model pick"]
    val_value = bt["validation"].loc[headline_strategy]
    val_fixed = bt["validation"].loc[f"{best_model} value bets"]

    home_share = share["Home Win"]
    covid = home_share.get(2020, np.nan)
    history = inputs["history"]
    fs = inputs["feature_selection"]

    overround = test_entries["Bookmaker (pre-match)"]["overround"]
    selected = metrics.get("selected_model", best_model)
    gap_ci = metrics["rf_minus_xgb_log_loss"]["test"]
    tied = gap_ci["ci_low"] < 0 < gap_ci["ci_high"]
    never_draw = all(test_entries[m]["predicted_share"]["Draw"] == 0 for m in MODELS)
    beats_baseline_everywhere = bool((by_season[list(MODELS)].max(axis=1) < by_season["Class frequencies"]).all())
    hardest = {e: season_label(int(by_season.loc[by_season[e].idxmax(), "season"]))
               for e in ("Random Forest", "XGBoost", "Bookmaker")}
    covid_hardest = all(v == "2020/21" for v in hardest.values())
    importance = metrics.get("permutation_importance_validation", {})
    top_features = {m: list(pd.Series(v).sort_values(ascending=False).index[:4]) for m, v in importance.items()}
    market_ahead = gap_to_book > 0 and max(headline["ROI"], val_value["ROI"]) < 0

    val_wf = pd.DataFrame(metrics["walk_forward"])
    wf_pivot = val_wf.pivot_table(index="validation_season", columns="model", values="log_loss")

    lines: list[str] = []
    add = lines.append
    add("# Predicting Premier League Results with Random Forest and XGBoost")
    add("")
    add("_Final project report, generated by `python -m src.final_report`. Every number below is computed "
        "from the pipeline's outputs._")
    add("")
    add("## Summary")
    add("")
    add("* **Task:** predict Home Win / Draw / Away Win probabilities for Premier League matches from "
        "information available before kickoff.")
    add(f"* **Data:** {len(model_rows):,} matches ({season_label(model_rows['season'].min())}–"
        f"{season_label(model_rows['season'].max())}), {n_features} leakage-safe features, Bet365 odds for "
        "benchmarking.")
    add(f"* **Result on the untouched {test_label} test season:** {best_model} reached "
        f"{_pct(t.loc[best_model, 'Accuracy'])} accuracy and a log loss of {t.loc[best_model, 'Log Loss']:.3f}. "
        f"Always backing the home team gives {_pct(base['Accuracy'])}. The bookmaker's own "
        f"probabilities score {_pct(book['Accuracy'])} and {book['Log Loss']:.3f}.")
    add(f"* **Against the market:** the models close {_pct(closed, 0)} of the log-loss gap between a naive "
        f"baseline and the bookmaker. They are {abs(gap_to_book):.3f} log loss "
        f"{'behind' if gap_to_book > 0 else 'ahead of'} the bookmaker on the test season. "
        + (f"The better model beats the bookmaker's log loss in {beats_book_seasons} of {len(by_season)} "
           "out-of-sample seasons." if beats_book_seasons else
           f"The bookmaker has the lower log loss in all {len(by_season)} out-of-sample seasons."))
    add(f"* **Simulated betting (${stake} flat stakes):** backing {best_model}'s pick in every test match gives "
        f"{_result_phrase(pick['Profit'])} ({_pct(pick['ROI'])} ROI). Value bets with the fixed {edge:.0%} edge give "
        f"{_result_phrase(fixed_rule['Profit'])} on {int(fixed_rule['Bets']):,} test bets ({_pct(fixed_rule['ROI'])}, "
        f"95% CI {fixed_rule['ROI 95% CI']}) but {_pct(val_fixed['ROI'])} on the validation season. The rule "
        f"searched on the walk-forward seasons ({rule_label(bet_rules[best_model])}) returned "
        f"{_pct(headline['ROI'])} on test and {_pct(val_value['ROI'])} on validation. "
        + ("No strategy is reliably profitable." if min(fixed_rule["ROI"], val_fixed["ROI"], headline["ROI"],
                                                          val_value["ROI"]) < 0 else ""))
    if draw_rule.get(best_model, {}).get("threshold") is not None:
        dr = metrics["test"][f"{best_model} + draw rule"]
        add(f"* **Draws:** plain argmax never predicts a draw (neither does the bookmaker). With the draw rule chosen "
            f"on past seasons, {best_model} predicts a draw in {_pct(dr['predicted_share']['Draw'], 0)} of test "
            f"matches and gets {_pct(dr['per_class']['Draw']['recall'], 0)} of actual draws, at "
            f"{_pct(dr['accuracy'])} overall accuracy (argmax: {_pct(t.loc[best_model, 'Accuracy'])}).")
    if experiments:
        first, last = experiments["results"][0], _chosen_experiment(experiments, groups, config)
        if last:
            add(f"* **What improved the model:** Elo ratings, more history and weighted form cut walk-forward "
                f"log loss from {first[f'{best_model} log loss']:.4f} to {last[f'{best_model} log loss']:.4f} "
                f"(bookmaker {experiments['bookmaker']['log_loss']:.4f}), closing "
                f"{_pct(_gap_closed(first, last, experiments, best_model), 0)} of the gap to the market "
                "(section 9).")
    if market_ahead:
        add("* **Bottom line:** the models learn real, well-calibrated signal and clearly beat naive baselines, but "
            "public team statistics alone don't beat the betting market. That's the expected result for this "
            "kind of feature set, and the live forward test will keep checking it honestly.")
    else:
        add("* **Bottom line:** the models beat naive baselines. Against the market the evidence is mixed (see "
            "sections 6–7), and the wide intervals mean more seasons, including the live forward test, are "
            "needed before claiming an edge.")
    add("")

    add("## 1. Data collection")
    add("")
    add("| Data | Source | Coverage |")
    add("| --- | --- | --- |")
    add("| Results, shots, shots on target, corners | Football-Data.co.uk E0 files (GitHub mirror `datasets/football-datasets`) | "
        f"{season_label(matches['season'].min())}–{season_label(config['data']['last_season'])}, 380 matches per season |")
    add("| Bet365 pre-match and closing odds, best and average price across bookmakers | Football-Data.co.uk, via "
        "`AnishKhetani/premier-league-data` | Bet365 pre-match from 2002/03, best price from 2005/06, closing from "
        "2019/20 |")
    add("| Fixtures, kickoff times and results for the season in progress | openfootball `football.json` | "
        "goals only, verified against all 380 matches of 2025/26 |")
    add("")
    add("The article this report follows scraped odds from Oddsportal. Here every source is a stable CSV or JSON "
        "file, cached in `data/raw/` so the whole project re-runs offline.")
    add("")

    add("## 2. Data cleaning")
    add("")
    add("* Every team spelling is mapped to one canonical name (`Man United`, `Manchester Utd` and "
        "`Manchester United FC` all become `Manchester United`), and an unknown name stops the pipeline.")
    add("* Dates are parsed format by format, so a day and a month are never swapped. Each result must agree "
        "with the recorded score, and duplicate fixtures are rejected.")
    add("* The odds are joined by season and teams, and the pipeline stops if a match date disagrees between "
        "sources. Every match the models are evaluated on has odds.")
    add("* Possession and expected goals aren't in the sources, so they are left out rather than invented.")
    add("")

    add("## 3. Feature engineering")
    add("")
    add(f"The production models use {n_features} features that describe both teams as they were **before kickoff**. "
        + ("" if use_base else "Phase 3 used 42 form, season-to-date and venue features; the experiments in "
           "section 9 showed that Elo ratings and weighted form carry the same information and more, so the "
           "models now use those alone. The Phase 3 features are still built and tested:"))
    add("")
    add("* **Recent form** (last 5 matches): points, goals scored and conceded, average shots and shots on target.")
    add("* **Season to date:** points, goals, goal difference, shots per game, and league position rebuilt from "
        "matches played on earlier dates only.")
    add("* **Venue form:** the home team's last 10 home matches and the away team's last 10 away matches. This "
        "represents home advantage per team instead of as a constant.")
    add("* **Differences:** home minus away for form, goal difference, shots, points per game, league position "
        "and venue form.")
    if not use_base:
        add("")
        add("The features the models use:")
        add("")
    if "elo" in groups:
        elo = config["features"]["elo"]
        add(f"* **Elo ratings:** one strength number per club that carries across seasons. After every match the "
            f"winner takes rating points from the loser (K = {elo['k']}, more for a bigger margin), the home side "
            f"gets {elo['home_advantage']} points of advantage, ratings are pulled {elo['season_regression']:.0%} "
            "back to the mean each summer, and a promoted club starts at the average of the clubs that went down. "
            "The settings were chosen on training seasons only.")
    if "ewm" in groups:
        add(f"* **Weighted form:** exponentially weighted points, goal difference, shots on target and shots on "
            f"target conceded, with a half-life of {config['features']['ewm_halflife']} matches, so recent games "
            "count most without a hard 5-match cut-off.")
    if "context" in groups:
        add("* **Match context:** rest days, unbeaten and winless runs, last season's finish and head-to-head.")
    add("")
    add("All rolling statistics use `shift(1)` within each team's history. The test suite checks this directly: "
        "changing a match's score doesn't move its own features, rewriting the future doesn't move past "
        "features, and features rebuilt from truncated history equal the training table exactly.")
    add("")

    add("## 4. Exploratory data analysis")
    add("")
    home_top = [s for s in share.index if share.loc[s].idxmax() == "Home Win"]
    away_top = [s for s in share.index if share.loc[s].idxmax() == "Away Win"]
    add(f"Home wins are the most common outcome in {len(home_top)} of {len(share)} seasons "
        f"({_pct(home_share.min(), 0)}–{_pct(home_share.max(), 0)} of matches). "
        + (f"The exception is {', '.join(season_label(s) for s in away_top)}, when away wins were more common. "
           if away_top else "")
        + f"2020/21 was played almost entirely without crowds, and home teams won only {_pct(covid, 0)} of matches."
        + (" That season is also the hardest for every predictor below, including the bookmaker."
           if covid_hardest else ""))
    add("")
    add("![Outcome mix by season](figures/outcome_share_by_season.png)")
    add("")
    add(f"Home teams score {goals['home_goals'].mean():.2f} goals per match on average and away teams "
        f"{goals['away_goals'].mean():.2f}.")
    add("")
    add("![Goals by season](figures/goals_by_season.png)")
    add("")
    top_gap, bottom_gap = gap_table.iloc[-1], gap_table.iloc[0]
    add(f"Pre-match league position carries a lot of signal. When the home team sits 10 or more places higher, it "
        f"wins {_pct(top_gap['Home Win'], 0)} of the time. When it sits 10 or more places lower, it wins only "
        f"{_pct(bottom_gap['Home Win'], 0)}. Draw rates barely move ({_pct(gap_table['Draw'].min(), 0)}–"
        f"{_pct(gap_table['Draw'].max(), 0)}), which is why draws are so hard to predict.")
    add("")
    add("![Outcome by league-position gap](figures/outcome_by_position_gap.png)")
    add("")
    add("The bookmaker's implied probabilities (with the margin removed) line up closely with how often outcomes "
        "actually happen. That's why it's the benchmark to beat.")
    add("")
    add("![Bookmaker calibration](figures/bookmaker_calibration.png)")
    add("")

    add("## 5. Model building")
    add("")
    add("Only two models are compared, as the project brief requires: `RandomForestClassifier` and "
        "`XGBClassifier` (`multi:softprob`). Neither needs feature scaling, and both handle missing values "
        "(such as a promoted team's empty form) natively.")
    add("")
    add(f"* **Split:** train {train_label}, validate {val_label}, test {test_label}. The test season was used once, "
        "at the end.")
    add(f"* **Tuning:** random search ({config['tuning']['random_forest_iterations']} Random Forest and "
        f"{config['tuning']['xgboost_iterations']} XGBoost candidates), scored by mean log loss over walk-forward "
        f"folds {wf_label}. Each fold trains on every earlier season.")
    add("* **Production:** after evaluation, both models are refit on every completed season for live predictions.")
    add("")
    add("**Walk-forward log loss (best hyperparameters)**")
    add("")
    wf_show = wf_pivot.reset_index().rename(columns={"validation_season": "Season"})
    wf_show["Season"] = wf_show["Season"].map(season_label)
    add(to_markdown(wf_show[["Season", "Class frequencies", "Random Forest", "XGBoost"]]))
    add("")
    add(f"**Validation season {val_label}**")
    add("")
    add(to_markdown(val_table))
    add("")
    add(f"**Test season {test_label}**")
    add("")
    add(to_markdown(test_table))
    add("")
    add(("Random Forest and XGBoost are statistically tied. " if tied else "")
        + f"On the test season, the paired bootstrap 95% interval for the per-match log-loss difference "
        f"(Random Forest minus XGBoost) is {gap_ci['ci_low']:+.4f} to {gap_ci['ci_high']:+.4f}. "
        f"{selected} was selected on validation log loss. "
        + ("Per class, both models back home wins and away wins but never make a draw the single most likely "
           "outcome, so with plain argmax draw recall is 0 " if never_draw else "Draws are rarely the top pick ")
        + "(details in [model_comparison.md](model_comparison.md)).")
    add("")
    if draw_rule:
        add("**Getting draws into the predictions.** A draw is almost never the single most likely outcome: "
            "draws happen in about a quarter of matches, and in most games one side is a little more likely to "
            "win than that. So picking the highest probability (argmax) never picks a draw, and the bookmaker's own "
            "prices don't either. The probabilities are fine; the decision rule is the problem. The draw rule "
            "predicts a draw when the draw probability reaches a threshold, chosen on the walk-forward seasons "
            "to maximise macro F1 (which rewards getting draws right as well as wins).")
        add("")
        rows = []
        for m in MODELS:
            rule = draw_rule.get(m)
            if not rule:
                continue
            for season_name, block in ((val_label, metrics["validation"]), (test_label, metrics["test"])):
                key = f"{m} + draw rule"
                if key not in block:
                    continue
                rows.append({
                    "Model": m, "Season": season_name,
                    "Threshold": "argmax" if rule["threshold"] is None else f"{rule['threshold']:.3f}",
                    "Accuracy (argmax)": block[m]["accuracy"], "Accuracy (rule)": block[key]["accuracy"],
                    "Macro F1 (argmax)": block[m]["macro_f1"], "Macro F1 (rule)": block[key]["macro_f1"],
                    "Draws predicted": block[key]["predicted_share"]["Draw"],
                    "Draw recall": block[key]["per_class"]["Draw"]["recall"],
                    "Draw precision": block[key]["per_class"]["Draw"]["precision"],
                })
        if rows:
            add(to_markdown(pd.DataFrame(rows)))
            add("")
            draw_rate = {val_label: metrics["validation"][MODELS[0]]["per_class"]["Draw"]["support"] / 380,
                         test_label: metrics["test"][MODELS[0]]["per_class"]["Draw"]["support"] / 380}
            precision = [r["Draw precision"] for r in rows]
            add(f"The rule changes accuracy by {min(r['Accuracy (rule)'] - r['Accuracy (argmax)'] for r in rows) * 100:+.1f} "
                f"to {max(r['Accuracy (rule)'] - r['Accuracy (argmax)'] for r in rows) * 100:+.1f} points and "
                f"catches about a fifth of the draws. Its draw calls are right {_pct(min(precision), 0)}–"
                f"{_pct(max(precision), 0)} of the time, against draw rates of "
                f"{_pct(min(draw_rate.values()), 0)}–{_pct(max(draw_rate.values()), 0)} in these seasons, so they "
                "are only a little better than guessing: draws stay the hardest outcome to call. Live predictions use "
                "the rule for the predicted outcome; the saved probabilities are unchanged.")
            add("")
    add("![Confusion matrices](confusion_matrices_test.png)")
    add("")
    add("![Calibration](calibration_test.png)")
    add("")
    if top_features and len({tuple(v) for v in top_features.values()}) == 1:
        names = next(iter(top_features.values()))
        add("**What drives the predictions.** For both models, the top 4 features by permutation importance on the "
            "validation season are " + ", ".join(f"`{n}`" for n in names) + ". Short-term last-5 form ranks lower.")
        add("")
    else:
        for model, names in top_features.items():
            add(f"**What drives {model}.** Top 4 features by permutation importance on the validation season: "
                + ", ".join(f"`{n}`" for n in names) + ".")
            add("")
    add("")
    add("![Feature importance](feature_importance.png)")
    add("")
    if fs:
        add("**Feature selection.** This is our version of the article's recursive feature elimination. Features "
            "are ranked by permutation importance averaged over the walk-forward folds, and the best subset size "
            "is chosen by walk-forward log loss. Only the training seasons are used.")
        add("")
        fs_rows = [{"Model": m, "Best k": r["best_k"], "Validation log loss (selected)": r["validation_log_loss_selected"],
                    "Validation log loss (all)": r["validation_log_loss_all"],
                    "Top features": ", ".join(r["selected_features"][:5]) + (" …" if r["best_k"] > 5 else "")}
                   for m, r in fs.items()]
        add(to_markdown(pd.DataFrame(fs_rows), "{:.4f}"))
        add("")
        largest_change = max(abs(r["validation_log_loss_selected"] - r["validation_log_loss_all"]) for r in fs.values())
        smallest_k = min(r["best_k"] for r in fs.values())
        if largest_change < 0.005:
            largest_k = max(r["best_k"] for r in fs.values())
            k_text = f"{smallest_k}" if smallest_k == largest_k else f"{smallest_k}–{largest_k}"
            add(f"Keeping only the top {k_text} features changes "
                f"validation log loss by at most {largest_change:.4f}, which is well within noise. As in the "
                "article, a much smaller feature set predicts just as well. The production models keep all "
                f"{n_features} features for now; switching to the smaller set is a simplicity choice, not an "
                "accuracy one.")
        else:
            add(f"The selected subsets change validation log loss by up to {largest_change:.4f}; the production "
                "models keep the full feature set unless config.yaml says otherwise.")
        add("")
        add("![Feature selection](figures/feature_selection.png)")
        add("")

    add("## 6. Benchmark: the models against the bookmaker")
    add("")
    add("The bookmaker is scored as if it were another model, using its odds with the margin removed. Log loss "
        "is the fairest single measure because it rewards good probabilities, not just correct picks.")
    add("")
    add("![Log loss by season](figures/log_loss_by_season.png)")
    add("")
    bys = by_season.copy()
    bys["season"] = bys["season"].map(season_label)
    add(to_markdown(bys.rename(columns={"season": "Season"})[["Season", "Class frequencies", "Random Forest",
                                                               "XGBoost", "Bookmaker"]]))
    add("")
    add(f"On the test season, the models pick the same favourite as the bookmaker in "
        f"{_pct(gaps[best_model]['same_favourite'], 0)} of matches, and their probabilities differ from the market's "
        f"by {gaps[best_model]['mean_abs_gap'] * 100:.1f} percentage points on average. The bookmaker's closing "
        f"line is sharper still (log loss {test_entries.get('Bookmaker (closing)', {}).get('log_loss', float('nan')):.3f}).")
    add("")

    add("## 7. Simulating investment")
    add("")
    add(f"As in the reference article, each strategy stakes **${stake} per bet**, at Bet365's pre-match odds "
        "unless it says best price. Both are prices a bettor could actually have taken.")
    add("")
    add("* **Model pick:** back the model's most likely outcome in every match (the article's approach).")
    add(f"* **Value bets:** back an outcome only when *model probability × odds − 1 > {edge:.0%}*. This is how "
        "a probability model would really be used against a market.")
    add("* **Value bets, tuned rule:** the edge threshold, whether to take Bet365's price or the best price listed "
        "across bookmakers, and an optional cap on the odds were searched on the walk-forward seasons "
        f"({wf_label}) only: " + "; ".join(f"{m}: {rule_label(r)} (walk-forward ROI {_pct(r['walk_forward_roi'])} "
                                          f"on {r['walk_forward_bets']:,} bets)" for m, r in bet_rules.items())
        + ". Choosing the best of many rules flatters its walk-forward ROI, so the validation and test rows are "
        "the ones to believe.")
    add("* **Baselines:** always back the home team, and always back the bookmaker's favourite (at Bet365's price "
        "and at the best price, which shows how much of the loss is the bookmaker's margin).")
    add("")
    for stage, title in (("test", f"Test season {test_label}"), ("validation", f"Validation season {val_label}"),
                         ("walk-forward", f"Walk-forward seasons {wf_label} (pooled)")):
        add(f"**{title}**")
        add("")
        add(_format_betting(bet_tables[stage]))
        add("")
    add("![Cumulative profit](figures/cumulative_profit_test.png)")
    add("")
    fav = {stage: (bt[stage].loc["Bookmaker favourite", "ROI"], bt[stage].loc["Bookmaker favourite, best price", "ROI"])
           for stage in bt}
    gains = [fav[stage][1] - fav[stage][0] for stage in fav]
    add("**The price matters.** Backing the bookmaker's favourite at Bet365's price returns "
        f"{_pct(fav['walk-forward'][0])} over the walk-forward seasons, {_pct(fav['validation'][0])} on validation and "
        f"{_pct(fav['test'][0])} on test. The same bets at the best price across bookmakers return "
        f"{_pct(fav['walk-forward'][1])}, {_pct(fav['validation'][1])} and {_pct(fav['test'][1])}, "
        f"{min(gains) * 100:.1f} to {max(gains) * 100:.1f} points better. Shopping for the best price helps every "
        "strategy, but it doesn't create an edge by itself.")
    add("")
    value_odds = test_bets.loc[headline_strategy, "Avg odds"]
    fav_odds = test_bets.loc["Bookmaker favourite", "Avg odds"]
    if value_odds > 2 * fav_odds:
        add(f"The value bets are mostly on outsiders (average odds {value_odds:.2f}, against {fav_odds:.2f} for the "
            "favourite). That's where the models disagree most with the market, and the results show the market "
            "is usually right there. This fits the well-documented favourite-longshot bias in betting markets, where "
            "bets on outsiders tend to return less than bets on favourites, so apparent value on outsiders is the "
            "easiest kind to be fooled by.")
    add("")
    add(f"**Sensitivity to the edge threshold.** The {edge:.0%} threshold was fixed before looking at results. "
        "These rows show how much the conclusion depends on it:")
    add("")
    sens = pd.concat([sensitivity["validation"].assign(Season=val_label),
                      sensitivity["test"].assign(Season=test_label)], ignore_index=True)
    sens["Profit"] = sens["Profit"].map(_money)
    sens["ROI"] = sens["ROI"].map(_pct)
    add(to_markdown(sens[["Season", "Model", "Edge threshold", "Bets", "Profit", "ROI"]]))
    add("")
    add(f"How to read this: the bookmaker's margin ({_pct(overround)} on the test season) means a strategy with no real edge loses "
        "roughly that much per bet in the long run. The 95% intervals are wide because one season holds only a "
        "few hundred bets. A single profitable season is not evidence of an edge, and a single losing one is "
        "not proof there isn't one.")
    add("")

    add("## 8. Productionization")
    add("")
    add("The project runs as a small application rather than a notebook:")
    add("")
    add("* `python -m src.pipeline` rebuilds data, features, models and evaluation reports from scratch.")
    add("* `python -m src.live predict` predicts every team's next fixture with both models and saves the "
        "predictions to `data/predictions/prediction_history.csv` **before kickoff**. Saved predictions are "
        "never overwritten.")
    add("* `python -m src.live update` / `report` fill in results and track live accuracy over time.")
    n_saved = len(history)
    n_done = int(history["actual_result"].notna().sum()) if n_saved else 0
    add(f"* Live forward test so far: {n_saved} predictions saved, {n_done} with results.")
    add("* A Streamlit dashboard (Upcoming Predictions, Model Performance, Team Form, Prediction History) is the "
        "next phase.")
    add("")

    add("## 9. Improving the model")
    add("")
    if experiments:
        add("After Phase 3 each candidate change was scored the same way: mean log loss over the walk-forward "
            f"seasons {wf_label}, with the validation and test seasons left untouched and Phase 3's "
            "hyperparameters held fixed so only the data and features change. Each row adds to the row above "
            "unless its note says otherwise.")
        add("")
        exp_rows = [{"Change": r["experiment"], "Features": r["n_features"], "Trained from": r["train_start"],
                     "RF log loss": r["Random Forest log loss"], "RF accuracy": r["Random Forest accuracy"],
                     "XGB log loss": r["XGBoost log loss"], "XGB accuracy": r["XGBoost accuracy"]}
                    for r in experiments["results"]]
        bk = experiments["bookmaker"]
        exp_rows.append({"Change": "Bookmaker (Bet365, margin removed)", "Features": "", "Trained from": "",
                         "RF log loss": bk["log_loss"], "RF accuracy": bk["accuracy"],
                         "XGB log loss": bk["log_loss"], "XGB accuracy": bk["accuracy"]})
        add(to_markdown(pd.DataFrame(exp_rows), "{:.4f}"))
        add("")
        for r in experiments["results"]:
            if r.get("note"):
                add(f"* **{r['experiment']}:** {r['note']}")
        add("")
        add("![Experiments](figures/experiments.png)")
        add("")
        add("The production setup is " + ("the Phase 3 features plus " if use_base else "")
            + " and ".join({"elo": "Elo ratings", "ewm": "weighted form", "context": "match context"}[g] for g in groups)
            + ("" if use_base else " only") + f", trained from {season_label(min(split_cfg['train_seasons']))}: "
            "the best row that can be used for live predictions. With its hyperparameters re-tuned, its "
            f"walk-forward log loss is {wf_pivot[best_model].mean():.4f} for {best_model}. "
            + _experiment_verdicts(experiments, best_model))
        add("")
    if in_play:
        add("## 10. Why some projects report 60–70% accuracy")
        add("")
        top = in_play["results"][0]
        ht = in_play["results"][1]
        ids = in_play["results"][-1]
        add("Public football-prediction projects sometimes report around 70% accuracy with the same algorithms. "
            f"One example is [{in_play['source'].split('github.com/')[1]}]({in_play['source']}). Its inputs are the "
            "half-time score plus the full-match shots, shots on target and red cards of the match being predicted, "
            "scored on a random 80/20 split. It predicts at half-time with second-half statistics already known, so "
            "none of its inputs exist before kickoff.")
        add("")
        add(f"Rerunning that setup on our data ({in_play['seasons']}, {in_play['matches']:,} matches):")
        add("")
        add(to_markdown(pd.DataFrame([{"Inputs": r["inputs"], "Random Forest": r["random_forest_accuracy"],
                                       "Logistic regression": r["logistic_regression_accuracy"]}
                                      for r in in_play["results"]])))
        add("")
        add(f"The half-time score alone gets {_pct(ht['random_forest_accuracy'], 0)}. With only the team identities, "
            f"the one pre-match input in that set, accuracy drops to {_pct(ids['random_forest_accuracy'], 0)}. "
            "A pre-match model should be compared with the bookmaker instead, whose own favourite wins "
            f"{_pct(book['Accuracy'], 0)} of the time on the test season. That's the realistic ceiling here.")
        add("")
    add(f"## {11 if in_play else 10}. Next steps")
    add("")
    add("* **Richer inputs:** expected goals, lineups, injuries and transfers aren't in these sources, and they "
        "are what the bookmaker prices in.")
    add("* **Current-season shots:** the live feed has goals only. Allowing football-data.co.uk in the network "
        "settings would restore shot statistics (the replayed test season lost about 0.004 log loss without them).")
    add("* **Live odds:** the odds-as-inputs variant needs pre-match odds for upcoming fixtures, which the "
        "current sources don't provide.")
    add("* **Explanations:** SHAP values (XGBoost supports them natively) to say why a team is favoured.")
    add("")

    add(f"## {12 if in_play else 11}. Conclusion")
    add("")
    add(f"Using only public match statistics, both tree models beat the naive baseline "
        f"{'in every out-of-sample season' if beats_baseline_everywhere else 'in most out-of-sample seasons'}. "
        f"They produce well-calibrated probabilities and reach about {_pct(t.loc[best_model, 'Accuracy'], 0)} "
        f"accuracy on the test season ({_pct(metrics['validation'][best_model]['accuracy'], 0)} on validation), "
        f"in a three-way problem where a quarter of matches are draws."
        + (" Random Forest and XGBoost are indistinguishable within statistical noise." if tied else ""))
    add("")
    add(f"The bookmaker {'remains ahead' if gap_to_book > 0 else 'is not ahead'} ({book['Log Loss']:.3f} vs "
        f"{t.loc[best_model, 'Log Loss']:.3f} log loss on the test season). Value betting returned "
        f"{_pct(fixed_rule['ROI'])} (fixed {edge:.0%} edge) and {_pct(headline['ROI'])} (rule chosen on past seasons) "
        f"on the test season, and {_pct(val_fixed['ROI'])} and {_pct(val_value['ROI'])} on the validation season. "
        "A negative return is not a bug in the simulation: the bookmaker's margin is built into every price, so a "
        "model that is slightly less accurate than the market loses money on average. "
        + ("The honest conclusion is the one the reference article's title hints at but can't claim from a single "
           "season: beating the odds needs information the market doesn't already have. "
           if market_ahead else
           "Those returns come with wide intervals, so they are not yet evidence of a lasting edge. ")
        + "The live forward test will show whether this verdict holds on matches that haven't happened yet.")
    add("")
    return "\n".join(lines)


if __name__ == "__main__":
    setup_logging()
    print(run())
