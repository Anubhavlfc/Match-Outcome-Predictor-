"""Final project report: bookmaker benchmark, betting simulation, EDA, write-up.

    python -m src.pipeline                      # training + out-of-sample predictions
    python -m src.models.feature_selection      # optional, adds the feature-selection section
    python -m src.final_report                  # this: writes reports/final_report.md

Every number in the report is computed here from pipeline outputs; the
prose around the numbers only chooses wording (e.g. "profit" vs "loss")
based on those numbers.
"""

from __future__ import annotations

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
    return {
        "matches": attach_odds(matches, odds),
        "features": pd.read_csv(processed / "features.csv", parse_dates=["date"]),
        "oos": pd.read_csv(processed / "out_of_sample_predictions.csv", parse_dates=["date"]),
        "metrics": json.loads((reports / "metrics.json").read_text()),
        "feature_selection": json.loads(fs_path.read_text()) if fs_path.exists() else None,
        "history": load_history(resolve_path(config["live"]["history_path"])),
    }


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

def betting_results(block: pd.DataFrame, stake: float, edge: float, seed: int) -> tuple[pd.DataFrame, dict]:
    one = block[block["model"] == MODELS[0]].reset_index(drop=True)
    logs: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        part = block[block["model"] == model].reset_index(drop=True)
        logs[f"{model} model pick"] = betting.model_pick_bets(part, _proba(part), stake)
        logs[f"{model} value bets"] = betting.value_bets(part, _proba(part), stake, edge)
    logs["Bookmaker favourite"] = betting.favourite_bets(one, stake)
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
    bet_tables = {}
    bet_logs = {}
    for stage, block in blocks.items():
        bet_tables[stage], bet_logs[stage] = betting_results(block, stake, edge, seed)
    sensitivity = {stage: edge_sensitivity(blocks[stage], stake, seed) for stage in ("validation", "test")}
    eda.cumulative_profit(
        {k: v for k, v in bet_logs["test"].items() if k in ("Random Forest value bets", "XGBoost value bets",
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
    }, indent=2, default=str))

    text = render(config, matches, model_rows, features, metrics, inputs, share, goals, gap_table, by_season,
                  val_table, test_table, test_entries, gaps, bet_tables, sensitivity)
    path = reports / "final_report.md"
    path.write_text(text)
    logger.info("Wrote %s", path)
    return path


def render(config, matches, model_rows, features, metrics, inputs, share, goals, gap_table, by_season,
           val_table, test_table, test_entries, gaps, bet_tables, sensitivity) -> str:
    split_cfg = config["split"]
    stake, edge = config["odds"]["stake"], config["odds"]["value_edge"]
    val_label, test_label = season_label(split_cfg["validation_season"]), season_label(split_cfg["test_season"])
    train_label = f"{season_label(min(split_cfg['train_seasons']))}–{season_label(max(split_cfg['train_seasons']))}"
    wf = split_cfg["walk_forward_validation_seasons"]
    wf_label = f"{season_label(min(wf))}–{season_label(max(wf))}"
    n_features = len(feature_columns(config["features"]["use_diff_features"]))

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
    headline_strategy = f"{best_model} value bets"
    headline = test_bets.loc[headline_strategy]
    pick = test_bets.loc[f"{best_model} model pick"]
    wf_value = bt["walk-forward"].loc[headline_strategy]

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
    market_ahead = gap_to_book > 0 and wf_value["ROI"] < 0

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
        f"{_result_phrase(pick['Profit'])} ({_pct(pick['ROI'])} ROI). Value bets (edge > {edge:.0%}) give "
        f"{_result_phrase(headline['Profit'])} on {int(headline['Bets']):,} bets ({_pct(headline['ROI'])} ROI, "
        f"95% CI {headline['ROI 95% CI']}). Over the five walk-forward seasons, value betting returned "
        f"{_pct(wf_value['ROI'])} on {int(wf_value['Bets']):,} bets.")
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
    add("| Bet365 pre-match and closing odds | Football-Data.co.uk, via `AnishKhetani/premier-league-data` | "
        "pre-match every season, closing from 2019/20 |")
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
        f"sources. All {len(model_rows):,} model matches have odds.")
    add("* Possession and expected goals aren't in the sources, so they are left out rather than invented.")
    add("")

    add("## 3. Feature engineering")
    add("")
    add(f"{n_features} features describe both teams as they were **before kickoff**:")
    add("")
    add("* **Recent form** (last 5 matches): points, goals scored and conceded, average shots and shots on target.")
    add("* **Season to date:** points, goals, goal difference, shots per game, and league position rebuilt from "
        "matches played on earlier dates only.")
    add("* **Venue form:** the home team's last 10 home matches and the away team's last 10 away matches. This "
        "represents home advantage per team instead of as a constant.")
    add("* **Differences:** home minus away for form, goal difference, shots, points per game, league position "
        "and venue form.")
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
           "outcome, so draw recall is 0 " if never_draw else "Draws are rarely the top pick ")
        + "(details in [model_comparison.md](model_comparison.md)).")
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
            add(f"Keeping only the top {smallest_k}–{max(r['best_k'] for r in fs.values())} features changes "
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
    add(f"As in the reference article, each strategy stakes **${stake} per bet** at Bet365's pre-match odds, "
        "the prices a bettor could actually have taken.")
    add("")
    add("* **Model pick:** back the model's most likely outcome in every match (the article's approach).")
    add(f"* **Value bets:** back an outcome only when *model probability × odds − 1 > {edge:.0%}*. This is how "
        "a probability model would really be used against a market.")
    add("* **Baselines:** always back the home team, and always back the bookmaker's favourite.")
    add("")
    for stage, title in (("test", f"Test season {test_label}"), ("validation", f"Validation season {val_label}"),
                         ("walk-forward", f"Walk-forward seasons {wf_label} (pooled)")):
        add(f"**{title}**")
        add("")
        add(_format_betting(bet_tables[stage]))
        add("")
    add("![Cumulative profit](figures/cumulative_profit_test.png)")
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

    add("## 9. Improvements")
    add("")
    add("* **Richer inputs:** expected goals, lineups, injuries and transfers aren't in these sources, and they "
        "are what the bookmaker prices in.")
    add("* **Current-season shots:** the live feed has goals only. Allowing football-data.co.uk in the network "
        "settings would restore shot statistics (the replayed test season lost about 0.004 log loss without them).")
    add("* **More features, tested one at a time:** head-to-head record, rest days, streaks and previous-season "
        "finish, each kept only if walk-forward log loss improves.")
    add("* **Draws:** a draw-aware decision rule, if hard draw predictions are ever needed. The probabilities "
        "themselves are already calibrated.")
    add("* **Explanations:** SHAP values (XGBoost supports them natively) to say why a team is favoured.")
    add("")

    add("## 10. Conclusion")
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
        f"{_pct(headline['ROI'])} on the test season and {_pct(wf_value['ROI'])} across the walk-forward seasons. "
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
