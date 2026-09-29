"""Feature selection with walk-forward permutation importance.

    python -m src.models.feature_selection      # needs python -m src.pipeline first

The reference walkthrough used recursive feature elimination to go from
40+ features to 13. The equivalent here, done without touching the
validation or test seasons:

1. Rank features by permutation importance (increase in log loss when a
   feature is shuffled), averaged over the walk-forward folds inside the
   training seasons.
2. For several subset sizes k, keep the top-k features and measure mean
   walk-forward log loss.
3. Report the best k per model, then check that subset once on the
   validation season against the full feature set.

It reports; it does not change the production feature set. That stays a
decision in config.yaml.
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from src.features.build_features import feature_columns
from src.models.split import walk_forward_folds
from src.models.train_random_forest import build_random_forest
from src.models.train_xgboost import build_xgboost
from src.utils.config import load_config, resolve_path, setup_logging
from src.utils.plotting import ENTITY_COLORS, apply_style

logger = logging.getLogger("feature_selection")

SUBSET_SIZES = [4, 8, 12, 16, 24, 32]


def _build(name: str, params: dict, seed: int):
    return build_random_forest(params, seed) if name == "Random Forest" else build_xgboost(params, seed)


def walk_forward_ranking(train: pd.DataFrame, columns: list[str], name: str, params: dict,
                         folds: list[int], seed: int, n_repeats: int = 3) -> pd.Series:
    """Mean permutation importance per feature across walk-forward folds (training seasons only)."""
    scores = []
    for _, train_idx, val_idx in walk_forward_folds(train["season"], folds):
        model = _build(name, params, seed).fit(train.iloc[train_idx][columns], train.iloc[train_idx]["target"])
        result = permutation_importance(
            model, train.iloc[val_idx][columns], train.iloc[val_idx]["target"],
            scoring="neg_log_loss", n_repeats=n_repeats, random_state=seed, n_jobs=-1,
        )
        scores.append(result.importances_mean)
    return pd.Series(np.mean(scores, axis=0), index=columns).sort_values(ascending=False)


def walk_forward_logloss(train: pd.DataFrame, columns: list[str], name: str, params: dict,
                         folds: list[int], seed: int) -> float:
    losses = []
    for _, train_idx, val_idx in walk_forward_folds(train["season"], folds):
        model = _build(name, params, seed).fit(train.iloc[train_idx][columns], train.iloc[train_idx]["target"])
        proba = model.predict_proba(train.iloc[val_idx][columns])
        losses.append(log_loss(train.iloc[val_idx]["target"], proba, labels=[0, 1, 2]))
    return float(np.mean(losses))


def run(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    seed = config["random_state"]
    split_cfg = config["split"]
    features = pd.read_csv(resolve_path(config["data"]["processed_dir"]) / "features.csv", parse_dates=["date"])
    params = json.loads((resolve_path(config["output"]["reports_dir"]) / "metrics.json").read_text())["hyperparameters"]
    feat_cfg = config["features"]
    columns = feature_columns(feat_cfg["use_diff_features"], feat_cfg["groups"], base=feat_cfg["base_features"])
    folds = split_cfg["walk_forward_validation_seasons"]
    train = features[features["season"].isin(split_cfg["train_seasons"])].reset_index(drop=True)
    validation = features[features["season"] == split_cfg["validation_season"]]

    results = {}
    for name in ("Random Forest", "XGBoost"):
        ranking = walk_forward_ranking(train, columns, name, params[name], folds, seed)
        curve = []
        for k in [*[size for size in SUBSET_SIZES if size < len(columns)], len(columns)]:
            subset = list(ranking.index[:k])
            curve.append({"k": k, "walk_forward_log_loss": walk_forward_logloss(train, subset, name, params[name], folds, seed)})
            logger.info("%s top-%d: walk-forward log loss %.4f", name, k, curve[-1]["walk_forward_log_loss"])
        curve_df = pd.DataFrame(curve)
        best_k = int(curve_df.loc[curve_df["walk_forward_log_loss"].idxmin(), "k"])
        best_subset = list(ranking.index[:best_k])

        val_scores = {}
        for label, cols in (("all", columns), ("selected", best_subset)):
            model = _build(name, params[name], seed).fit(train[cols], train["target"])
            val_scores[label] = float(log_loss(validation["target"], model.predict_proba(validation[cols]), labels=[0, 1, 2]))
        results[name] = {
            "ranking": ranking.round(5).to_dict(),
            "curve": curve,
            "best_k": best_k,
            "selected_features": best_subset,
            "validation_log_loss_all": val_scores["all"],
            "validation_log_loss_selected": val_scores["selected"],
        }

    reports_dir = resolve_path(config["output"]["reports_dir"])
    (reports_dir / "feature_selection.json").write_text(json.dumps(results, indent=2))
    _plot(results, len(columns), reports_dir / "figures" / "feature_selection.png")
    return results


def _plot(results: dict, n_features: int, path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in results.items():
        curve = pd.DataFrame(r["curve"])
        ax.plot(curve["k"], curve["walk_forward_log_loss"], marker="o", color=ENTITY_COLORS[name], label=name)
    ax.set_xlabel("Number of features kept (ranked by walk-forward permutation importance)")
    ax.set_ylabel("Mean walk-forward log loss")
    ax.set_title("Feature selection: fewer features vs log loss (lower is better)")
    ax.set_xticks([*[size for size in SUBSET_SIZES if size < n_features], n_features])
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    setup_logging()
    for model_name, summary in run().items():
        print(f"{model_name}: best k={summary['best_k']}, validation log loss "
              f"{summary['validation_log_loss_selected']:.4f} (selected) vs {summary['validation_log_loss_all']:.4f} (all)")
