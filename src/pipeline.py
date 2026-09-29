"""Training pipeline: data -> features -> tuning -> evaluation -> saved models.

    python -m src.pipeline            # full run
    python -m src.pipeline --quick    # fewer tuning candidates, for a smoke test

Steps
-----
1. Download (or reuse cached) season CSVs.
2. Clean and normalize them into data/processed/matches.csv.
3. Build leakage-safe features into data/processed/features.csv.
4. Split chronologically: train / validation / test seasons.
5. Tune each model with walk-forward folds inside the training seasons.
6. Fit on the training seasons and evaluate on the validation season
   (this is where models and feature sets are compared).
7. Refit on train + validation and evaluate once on the untouched test season.
   The draw rule's threshold is chosen on the walk-forward predictions of
   step 5's folds, never on validation or test.
8. Refit on every completed season (train + validation + test) for live
   use, save both models, and write reports/model_comparison.md. The
   reported test numbers come from step 7, before the test season was seen.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.data.clean import TARGET_LABELS, build_match_table
from src.data.collect import collect_seasons, raw_path
from src.features.build_features import build_features, feature_columns
from src.models import evaluate as ev
from src.models.decision import choose_draw_threshold, predict_outcome
from src.models.split import season_split, walk_forward_folds
from src.models.train_random_forest import (
    save_random_forest,
    train_random_forest,
    tune_random_forest,
)
from src.models.train_xgboost import save_xgboost, train_xgboost, tune_xgboost
from src.utils.config import load_config, resolve_path, season_label, setup_logging

logger = logging.getLogger("pipeline")

MODEL_NAMES = ("Random Forest", "XGBoost")


def _fit(name: str, X: pd.DataFrame, y: pd.Series, params: dict[str, Any], seed: int):
    trainer = train_random_forest if name == "Random Forest" else train_xgboost
    return trainer(X, y, params, seed)


def _prediction_frame(rows: pd.DataFrame, model: str, stage: str, proba: np.ndarray) -> pd.DataFrame:
    """Out-of-sample probabilities for one model on one block of matches."""
    frame = rows[["match_id", "season", "date", "home_team", "away_team", "target"]].copy()
    frame["model"], frame["stage"] = model, stage
    # XGBoost returns float32 probabilities; store them as float64 rows that sum to exactly 1.
    proba = np.asarray(proba, dtype=float)
    proba = proba / proba.sum(axis=1, keepdims=True)
    frame["p_away"], frame["p_draw"], frame["p_home"] = proba[:, 0], proba[:, 1], proba[:, 2]
    return frame


def _feature_args(config: dict[str, Any]) -> dict[str, Any]:
    feat_cfg = config["features"]
    return {"include_diffs": feat_cfg["use_diff_features"], "groups": feat_cfg["groups"],
            "base": feat_cfg["base_features"]}


def _walk_forward_summary(search_results: pd.DataFrame, best_params: dict[str, Any]) -> pd.DataFrame:
    best = search_results[search_results["params"].apply(lambda p: p == best_params)]
    return best[["validation_season", "accuracy", "log_loss"]].reset_index(drop=True)


def run(config_path: str | None = None, quick: bool = False, force_download: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    seed = config["random_state"]
    np.random.seed(seed)
    data_cfg, feat_cfg, split_cfg = config["data"], config["features"], config["split"]
    raw_dir = resolve_path(data_cfg["raw_dir"])
    processed_dir = resolve_path(data_cfg["processed_dir"])
    models_dir = resolve_path(config["output"]["models_dir"])
    reports_dir = resolve_path(config["output"]["reports_dir"])
    for directory in (processed_dir, models_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 1-2. Data ---------------------------------------------------------------
    first, last = data_cfg["history_start_season"], data_cfg["last_season"]
    collect_seasons(first, last, data_cfg["sources"], raw_dir, force=force_download)
    matches = build_match_table({year: raw_path(raw_dir, year) for year in range(first, last + 1)})
    matches.to_csv(processed_dir / "matches.csv", index=False)

    # 3. Features -------------------------------------------------------------
    feature_settings = {
        "form_window": feat_cfg["form_window"],
        "venue_window": feat_cfg["venue_window"],
        "venue_min_periods": feat_cfg["venue_min_periods"],
        "ewm_halflife": feat_cfg["ewm_halflife"],
        "elo": feat_cfg["elo"],
    }
    features = build_features(matches, **feature_settings)
    # Warm-up seasons only feed rolling history and Elo; they are never model rows.
    features = features[features["season"] >= data_cfg["first_season"]].reset_index(drop=True)
    features.to_csv(processed_dir / "features.csv", index=False)
    use_diffs, groups, use_base = feat_cfg["use_diff_features"], feat_cfg["groups"], feat_cfg["base_features"]
    columns = feature_columns(include_diffs=use_diffs, groups=groups, base=use_base)
    logger.info("Feature table: %d matches x %d model features", len(features), len(columns))

    # 4. Split ----------------------------------------------------------------
    split = season_split(features, split_cfg["train_seasons"], split_cfg["validation_season"],
                         split_cfg["test_season"])
    X_train, y_train = split.train[columns], split.train["target"]
    X_val, y_val = split.validation[columns], split.validation["target"]
    X_test, y_test = split.test[columns], split.test["target"]
    logger.info("Train %d | validation %d | test %d", len(X_train), len(X_val), len(X_test))

    # 5. Walk-forward tuning inside the training seasons -----------------------
    tuning_cfg = config["tuning"]
    wf_seasons = split_cfg["walk_forward_validation_seasons"]
    rf_iter = 3 if quick else tuning_cfg["random_forest_iterations"]
    xgb_iter = 3 if quick else tuning_cfg["xgboost_iterations"]
    rf_params, rf_search = tune_random_forest(X_train, y_train, split.train["season"], wf_seasons, rf_iter, seed)
    xgb_params, xgb_search = tune_xgboost(X_train, y_train, split.train["season"], wf_seasons, xgb_iter, seed)
    best_params = {"Random Forest": rf_params, "XGBoost": xgb_params}

    walk_forward = []
    for name, search, params in (("Random Forest", rf_search, rf_params), ("XGBoost", xgb_search, xgb_params)):
        summary = _walk_forward_summary(search, params).assign(model=name)
        walk_forward.append(summary)
    for season in wf_seasons:
        train_part = split.train[split.train["season"] < season]
        eval_part = split.train[split.train["season"] == season]
        base = ev.compute_metrics(eval_part["target"], ev.class_prior_proba(train_part["target"], len(eval_part)))
        walk_forward.append(pd.DataFrame([{
            "validation_season": season, "accuracy": base["accuracy"], "log_loss": base["log_loss"],
            "model": "Class frequencies",
        }]))
    walk_forward = pd.concat(walk_forward, ignore_index=True)

    # 6. Validation season: model comparison and feature-set ablation ---------
    val_models = {name: _fit(name, X_train, y_train, best_params[name], seed) for name in MODEL_NAMES}
    val_probas = {name: model.predict_proba(X_val) for name, model in val_models.items()}
    val_results = ev.baseline_metrics(y_train, y_val)
    val_results.update({name: ev.compute_metrics(y_val, proba) for name, proba in val_probas.items()})

    # Validation-season check of the chosen feature set against the
    # alternatives (the choice itself was made on walk-forward seasons with
    # python -m src.experiments).
    variants = {
        f"chosen set ({len(columns)})": columns,
        "Phase 3 set (42)": feature_columns(True),
        "Phase 3 set + " + "+".join(groups): feature_columns(True, groups),
        "+".join(groups) + " only": feature_columns(True, groups, base=False),
    }
    seen: list[list[str]] = []
    ablation_rows = []
    for label, cols in variants.items():
        if not cols or cols in seen:
            continue
        seen.append(cols)
        for name in MODEL_NAMES:
            model = _fit(name, split.train[cols], y_train, best_params[name], seed)
            m = ev.compute_metrics(y_val, model.predict_proba(split.validation[cols]))
            ablation_rows.append({
                "Model": name, "Features": label if "(" in label else f"{label} ({len(cols)})",
                "Accuracy": m["accuracy"], "Macro F1": m["macro_f1"], "Log Loss": m["log_loss"],
            })
    ablation = pd.DataFrame(ablation_rows)

    importances = {
        name: ev.permutation_importance_logloss(model, X_val, y_val, seed) for name, model in val_models.items()
    }
    builtin = {name: ev.builtin_importance(model, columns) for name, model in val_models.items()}

    # 7. Final models: refit on train + validation, evaluate once on test -----
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    final_models = {name: _fit(name, X_full, y_full, best_params[name], seed) for name in MODEL_NAMES}
    test_probas = {name: model.predict_proba(X_test) for name, model in final_models.items()}
    test_results = ev.baseline_metrics(y_full, y_test)
    test_results.update({name: ev.compute_metrics(y_test, proba) for name, proba in test_probas.items()})

    # Every prediction below was made by a model that had not seen that season.
    # The odds benchmark and betting simulation (src.final_report) use them.
    out_of_sample = []
    for season, train_idx, val_idx in walk_forward_folds(split.train["season"], wf_seasons):
        for name in MODEL_NAMES:
            model = _fit(name, X_train.iloc[train_idx], y_train.iloc[train_idx], best_params[name], seed)
            out_of_sample.append(_prediction_frame(split.train.iloc[val_idx], name, "walk-forward",
                                                   model.predict_proba(X_train.iloc[val_idx])))
    for name in MODEL_NAMES:
        out_of_sample.append(_prediction_frame(split.validation, name, "validation", val_probas[name]))
        out_of_sample.append(_prediction_frame(split.test, name, "test", test_probas[name]))
    out_of_sample = pd.concat(out_of_sample, ignore_index=True)
    out_of_sample.to_csv(processed_dir / "out_of_sample_predictions.csv", index=False)

    # Draw rule: threshold chosen on the walk-forward folds only, then
    # applied unchanged to validation and test.
    draw_rule = {}
    for name in MODEL_NAMES:
        wf = out_of_sample[(out_of_sample["model"] == name) & (out_of_sample["stage"] == "walk-forward")]
        draw_rule[name] = choose_draw_threshold(wf["target"].to_numpy(), wf[["p_away", "p_draw", "p_home"]].to_numpy())
        threshold = draw_rule[name]["threshold"]
        logger.info("%s draw threshold %.3f (walk-forward macro F1 %.3f vs %.3f with argmax)", name, threshold,
                    draw_rule[name]["selected"]["macro_f1"], draw_rule[name]["argmax"]["macro_f1"])
        val_results[f"{name} + draw rule"] = ev.compute_metrics(
            y_val, val_probas[name], y_pred=predict_outcome(val_probas[name], threshold))
        test_results[f"{name} + draw rule"] = ev.compute_metrics(
            y_test, test_probas[name], y_pred=predict_outcome(test_probas[name], threshold))

    selected = min(MODEL_NAMES, key=lambda name: val_results[name]["log_loss"])
    gaps = {
        "validation": ev.paired_bootstrap_logloss_diff(y_val.to_numpy(), val_probas["Random Forest"],
                                                        val_probas["XGBoost"], seed),
        "test": ev.paired_bootstrap_logloss_diff(y_test.to_numpy(), test_probas["Random Forest"],
                                                  test_probas["XGBoost"], seed),
    }

    # 8. Production models: same hyperparameters, every completed season ----
    # Evaluation is finished, so the test season can now be used as training
    # data. Live predictions for the next season should learn from the most
    # recent season too.
    all_seasons = [*split_cfg["train_seasons"], split_cfg["validation_season"], split_cfg["test_season"]]
    production_rows = features[features["season"].isin(all_seasons)]
    production_models = {
        name: _fit(name, production_rows[columns], production_rows["target"], best_params[name], seed)
        for name in MODEL_NAMES
    }
    save_random_forest(production_models["Random Forest"], models_dir / "random_forest.pkl")
    save_xgboost(production_models["XGBoost"], models_dir / "xgboost.json")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": columns,
        "feature_groups": groups,
        "base_features": use_base,
        "feature_settings": feature_settings,
        # None = plain argmax for the predicted outcome (config decision.draw_rule: false).
        "draw_threshold": {name: draw_rule[name]["threshold"] if config["decision"]["draw_rule"] else None
                           for name in MODEL_NAMES},
        "target_encoding": {str(k): v for k, v in TARGET_LABELS.items()},
        "trained_on_seasons": [season_label(s) for s in all_seasons],
        "evaluated_on_season": season_label(split_cfg["test_season"]),
        "hyperparameters": best_params,
        "random_state": seed,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__},
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    ev.plot_confusion_matrices({k: test_results[k] for k in MODEL_NAMES}, reports_dir / "confusion_matrices_test.png",
                               f"Test season {season_label(split_cfg['test_season'])}")
    ev.plot_calibration(y_test.to_numpy(), test_probas, reports_dir / "calibration_test.png",
                        f"Calibration, test season {season_label(split_cfg['test_season'])}")
    ev.plot_importance(importances, reports_dir / "feature_importance.png")

    metrics = {
        "validation": val_results, "test": test_results, "hyperparameters": best_params,
        "walk_forward": walk_forward.to_dict(orient="records"),
        "ablation": ablation.to_dict(orient="records"),
        "selected_model": selected, "rf_minus_xgb_log_loss": gaps,
        "draw_rule": {name: {k: v for k, v in rule.items()} for name, rule in draw_rule.items()},
        "permutation_importance_validation": {
            name: table["importance"].round(5).to_dict() for name, table in importances.items()
        },
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    report = _render_report(config, matches, features, split, walk_forward, val_results, test_results,
                            ablation, importances, builtin, best_params, quick, selected, gaps, draw_rule)
    (reports_dir / "model_comparison.md").write_text(report)
    logger.info("Report written to %s", reports_dir / "model_comparison.md")
    print("\nValidation season\n" + ev.comparison_table(val_results).round(4).to_string(index=False))
    print("\nTest season\n" + ev.comparison_table(test_results).round(4).to_string(index=False))
    return metrics


def _render_report(config, matches, features, split, walk_forward, val_results, test_results,
                   ablation, importances, builtin, best_params, quick, selected, gaps, draw_rule) -> str:
    split_cfg = config["split"]
    train_label = f"{season_label(min(split_cfg['train_seasons']))} to {season_label(max(split_cfg['train_seasons']))}"
    val_label = season_label(split_cfg["validation_season"])
    test_label = season_label(split_cfg["test_season"])

    wf_table = walk_forward.pivot_table(index="validation_season", columns="model",
                                        values=["accuracy", "log_loss"]).round(3)
    wf_table.columns = [f"{model} {metric.replace('_', ' ')}" for metric, model in wf_table.columns]
    wf_table = wf_table.reset_index().rename(columns={"validation_season": "Validated on"})
    wf_table["Validated on"] = wf_table["Validated on"].map(season_label)
    wf_table = wf_table[["Validated on"] + sorted(c for c in wf_table.columns if c != "Validated on")]

    def cm_block(results, name):
        cm = pd.DataFrame(results[name]["confusion_matrix"],
                          index=[f"Actual {c}" for c in ev.CLASS_NAMES],
                          columns=[f"Pred {c}" for c in ev.CLASS_NAMES]).reset_index().rename(columns={"index": ""})
        return f"**{name}**\n\n" + ev.to_markdown(cm, "{:.0f}")

    importance_tables = []
    for name, table in importances.items():
        top = table.head(10).reset_index().rename(columns={"index": "Feature", "importance": "Log-loss increase",
                                                           "std": "Std"})
        top.insert(2, "Built-in rank", top["Feature"].map(
            {f: i + 1 for i, f in enumerate(builtin[name].index)}).astype(int))
        importance_tables.append(f"**{name}**\n\n" + ev.to_markdown(top, "{:.4f}") + "\n")

    target_share = split.train["target"].value_counts(normalize=True)
    lines = [
        "# Model comparison",
        "",
        "_Generated by `python -m src.pipeline`" + (" in --quick mode (reduced tuning)" if quick else "") + "._",
        "",
        "## Data",
        "",
        f"* {len(matches)} Premier League matches loaded ({season_label(matches['season'].min())} to "
        f"{season_label(matches['season'].max())}); seasons before {season_label(config['data']['first_season'])} "
        "only warm up rolling features and Elo ratings.",
        f"* {len(features)} model rows. Train {len(split.train)} ({train_label}), validation "
        f"{len(split.validation)} ({val_label}), test {len(split.test)} ({test_label}).",
        f"* Training outcome mix: Home {target_share.get(2, 0):.1%}, Draw {target_share.get(1, 0):.1%}, "
        f"Away {target_share.get(0, 0):.1%}.",
        f"* {len(feature_columns(**_feature_args(config)))} features "
        f"({'base set plus ' if config['features']['base_features'] else ''}groups: "
        f"{', '.join(config['features']['groups']) or 'none'}). No possession or xG (not in the source data).",
        "",
        "## Walk-forward validation (inside training seasons, best hyperparameters)",
        "",
        "Each row trains on every earlier training season and predicts the named season.",
        "",
        ev.to_markdown(wf_table),
        "",
        f"## Validation season {val_label} (models fit on {train_label})",
        "",
        ev.to_markdown(ev.comparison_table(val_results)),
        "",
        "### Model selection",
        "",
        f"Selected by validation log loss: **{selected}**. Random Forest minus XGBoost log loss per match: "
        f"validation {gaps['validation']['mean']:+.4f} (95% CI {gaps['validation']['ci_low']:+.4f} to "
        f"{gaps['validation']['ci_high']:+.4f}), test {gaps['test']['mean']:+.4f} (95% CI "
        f"{gaps['test']['ci_low']:+.4f} to {gaps['test']['ci_high']:+.4f}). An interval that contains 0 means "
        "one season is not enough to separate the two models.",
        "",
        "### Feature-set ablation on validation",
        "",
        ev.to_markdown(ablation),
        "",
        f"## Test season {test_label} (models refit on {train_label} + {val_label})",
        "",
        "The test season was not used for any modelling decision.",
        "",
        ev.to_markdown(ev.comparison_table(test_results)),
        "",
        "### Draw rule",
        "",
        "Plain argmax never predicts a draw, because a draw is almost never the single most likely outcome "
        "(the bookmaker's prices never make it one either). The draw rule predicts a draw when the draw "
        "probability reaches a threshold chosen on the walk-forward seasons to maximise macro F1. "
        "Probabilities, log loss and the betting simulation are unchanged.",
        "",
        ev.to_markdown(pd.DataFrame([{
            "Model": name, "Threshold": rule["threshold"],
            "WF accuracy (argmax)": rule["argmax"]["accuracy"], "WF accuracy (rule)": rule["selected"]["accuracy"],
            "WF macro F1 (argmax)": rule["argmax"]["macro_f1"], "WF macro F1 (rule)": rule["selected"]["macro_f1"],
            "WF draws predicted": rule["selected"]["draw_share"],
        } for name, rule in draw_rule.items()])),
        "",
        "### Per-class performance (test)",
        "",
        ev.to_markdown(ev.per_class_table({k: test_results[k] for k in test_results if k.startswith(MODEL_NAMES)})),
        "",
        "### Confusion matrices (test)",
        "",
        cm_block(test_results, "Random Forest"),
        "",
        cm_block(test_results, "XGBoost"),
        "",
        "![Confusion matrices](confusion_matrices_test.png)",
        "",
        "### Probability quality (test)",
        "",
        "Brier is the multiclass Brier score (lower is better). ECE is the expected calibration "
        "error of the predicted class's probability.",
        "",
        ev.to_markdown(pd.DataFrame([
            {"Model": k, "Log Loss": test_results[k]["log_loss"], "Brier": test_results[k]["brier"],
             "ECE": test_results[k]["ece"]} for k in ("Class frequencies", *MODEL_NAMES)
        ])),
        "",
        "![Calibration](calibration_test.png)",
        "",
        f"## Feature importance (validation season {val_label})",
        "",
        "Permutation importance: how much validation log loss rises when a feature is shuffled. "
        "Built-in rank is the model's own impurity/gain ranking, shown for comparison.",
        "",
        *importance_tables,
        "",
        "![Feature importance](feature_importance.png)",
        "",
        "## Hyperparameters chosen by walk-forward log loss",
        "",
        "```json",
        json.dumps(best_params, indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 training pipeline.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--quick", action="store_true", help="Few tuning candidates (smoke test)")
    parser.add_argument("--force-download", action="store_true", help="Re-download raw CSVs")
    args = parser.parse_args()
    setup_logging()
    run(args.config, quick=args.quick, force_download=args.force_download)


if __name__ == "__main__":
    main()
