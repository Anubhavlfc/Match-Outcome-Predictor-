"""Phase 1 end-to-end pipeline.

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
8. Save both final models and write reports/model_comparison.md.
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
from src.models.split import season_split
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
    features = build_features(
        matches,
        form_window=feat_cfg["form_window"],
        venue_window=feat_cfg["venue_window"],
        venue_min_periods=feat_cfg["venue_min_periods"],
    )
    # The warm-up season only feeds rolling history; it is never a model row.
    features = features[features["season"] >= data_cfg["first_season"]].reset_index(drop=True)
    features.to_csv(processed_dir / "features.csv", index=False)
    use_diffs = feat_cfg["use_diff_features"]
    columns = feature_columns(include_diffs=use_diffs)
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

    ablation_rows = []
    for include_diffs in (False, True):
        cols = feature_columns(include_diffs=include_diffs)
        for name in MODEL_NAMES:
            model = _fit(name, split.train[cols], y_train, best_params[name], seed)
            m = ev.compute_metrics(y_val, model.predict_proba(split.validation[cols]))
            ablation_rows.append({
                "Model": name,
                "Features": f"individual + diffs ({len(cols)})" if include_diffs else f"individual only ({len(cols)})",
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

    selected = min(MODEL_NAMES, key=lambda name: val_results[name]["log_loss"])
    gaps = {
        "validation": ev.paired_bootstrap_logloss_diff(y_val.to_numpy(), val_probas["Random Forest"],
                                                        val_probas["XGBoost"], seed),
        "test": ev.paired_bootstrap_logloss_diff(y_test.to_numpy(), test_probas["Random Forest"],
                                                  test_probas["XGBoost"], seed),
    }

    # 8. Save models, metrics, figures, report --------------------------------
    save_random_forest(final_models["Random Forest"], models_dir / "random_forest.pkl")
    save_xgboost(final_models["XGBoost"], models_dir / "xgboost.json")
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": columns,
        "target_encoding": {str(k): v for k, v in TARGET_LABELS.items()},
        "trained_on_seasons": [season_label(s) for s in [*split_cfg["train_seasons"], split_cfg["validation_season"]]],
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
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    report = _render_report(config, matches, features, split, walk_forward, val_results, test_results,
                            ablation, importances, builtin, best_params, quick, selected, gaps)
    (reports_dir / "model_comparison.md").write_text(report)
    logger.info("Report written to %s", reports_dir / "model_comparison.md")
    print("\nValidation season\n" + ev.comparison_table(val_results).round(4).to_string(index=False))
    print("\nTest season\n" + ev.comparison_table(test_results).round(4).to_string(index=False))
    return metrics


def _render_report(config, matches, features, split, walk_forward, val_results, test_results,
                   ablation, importances, builtin, best_params, quick, selected, gaps) -> str:
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
        "# Phase 1 model comparison",
        "",
        "_Generated by `python -m src.pipeline`" + (" in --quick mode (reduced tuning)" if quick else "") + "._",
        "",
        "## Data",
        "",
        f"* {len(matches)} Premier League matches loaded ({season_label(matches['season'].min())} to "
        f"{season_label(matches['season'].max())}); the first season only warms up rolling features.",
        f"* {len(features)} model rows. Train {len(split.train)} ({train_label}), validation "
        f"{len(split.validation)} ({val_label}), test {len(split.test)} ({test_label}).",
        f"* Training outcome mix: Home {target_share.get(2, 0):.1%}, Draw {target_share.get(1, 0):.1%}, "
        f"Away {target_share.get(0, 0):.1%}.",
        f"* {len(feature_columns(config['features']['use_diff_features']))} features. No possession or xG "
        "(not in the source data).",
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
        "### Per-class performance (test)",
        "",
        ev.to_markdown(ev.per_class_table({k: test_results[k] for k in MODEL_NAMES})),
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
