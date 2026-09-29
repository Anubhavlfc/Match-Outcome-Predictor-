"""Metrics, baselines, plots and the comparison report."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write files, never open windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.data.clean import TARGET_LABELS  # noqa: E402

logger = logging.getLogger(__name__)

CLASSES = [0, 1, 2]
CLASS_NAMES = [TARGET_LABELS[c] for c in CLASSES]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and the one-hot outcome (0 = perfect, 2 = worst)."""
    onehot = np.eye(3)[np.asarray(y_true)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Gap between confidence and hit rate of the predicted class, averaged over confidence bins."""
    y_true = np.asarray(y_true)
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray | None, y_pred: np.ndarray | None = None
                    ) -> dict[str, Any]:
    """All evaluation numbers for one model on one dataset.

    ``proba`` may be None for hard-label baselines; probability metrics are then NaN.
    """
    y_true = np.asarray(y_true)
    if y_pred is None:
        y_pred = proba.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=CLASSES)) if proba is not None else float("nan"),
        "brier": multiclass_brier(y_true, proba) if proba is not None else float("nan"),
        "ece": expected_calibration_error(y_true, proba) if proba is not None else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "predicted_share": {TARGET_LABELS[c]: float(np.mean(y_pred == c)) for c in CLASSES},
        "per_class": {
            TARGET_LABELS[c]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(CLASSES)
        },
    }
    return metrics


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def class_prior_proba(y_train: pd.Series, n_rows: int) -> np.ndarray:
    """Predict the training-set outcome frequencies for every match."""
    prior = np.bincount(np.asarray(y_train), minlength=3) / len(y_train)
    return np.tile(prior, (n_rows, 1))


def baseline_metrics(y_train: pd.Series, y_eval: pd.Series) -> dict[str, dict[str, Any]]:
    """Context for the models.

    * "Always Home Win": a hard prediction, so it has no log loss.
    * "Class frequencies": outputs the training outcome rates as probabilities.
      Its most likely class is the most common outcome (Home Win in every
      season here), so it also stands in for "always predict the most common
      outcome", with a meaningful log loss.
    """
    y_eval = np.asarray(y_eval)
    always_home = np.full(len(y_eval), 2)
    return {
        "Always Home Win": compute_metrics(y_eval, None, y_pred=always_home),
        "Class frequencies": compute_metrics(y_eval, class_prior_proba(y_train, len(y_eval))),
    }


def paired_bootstrap_logloss_diff(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray,
                                  random_state: int, n_boot: int = 2000) -> dict[str, float]:
    """Mean log-loss difference (A minus B) with a 95% bootstrap interval.

    Matches are resampled with replacement and both models are scored on the
    same resample, so the interval shows whether the gap between them is
    larger than the noise from a single season of results.
    """
    y_true = np.asarray(y_true)
    rows = np.arange(len(y_true))
    eps = 1e-15
    loss_a = -np.log(np.clip(proba_a[rows, y_true], eps, 1))
    loss_b = -np.log(np.clip(proba_b[rows, y_true], eps, 1))
    diff = loss_a - loss_b
    rng = np.random.default_rng(random_state)
    samples = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[samples].mean(axis=1)
    return {"mean": float(diff.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5))}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Macro F1": m["macro_f1"],
            "Log Loss": m["log_loss"],
            "Brier": m["brier"],
            "Draw recall": m["per_class"]["Draw"]["recall"],
        })
    return pd.DataFrame(rows)


def per_class_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        for cls in CLASS_NAMES:
            stats = m["per_class"][cls]
            rows.append({
                "Model": name, "Class": cls,
                "Precision": stats["precision"], "Recall": stats["recall"], "F1": stats["f1"],
                "Support": stats["support"], "Predicted share": m["predicted_share"][cls],
            })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Small markdown table writer (avoids an optional 'tabulate' dependency)."""
    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "n/a" if np.isnan(value) else float_format.format(value)
        return str(value)

    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *body])


# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------

def builtin_importance(model: Any, feature_names: list[str]) -> pd.Series:
    """Impurity importance (RF) or total-gain-normalized importance (XGBoost)."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def permutation_importance_logloss(model: Any, X: pd.DataFrame, y: pd.Series, random_state: int,
                                   n_repeats: int = 10) -> pd.DataFrame:
    """How much validation log loss worsens when one feature is shuffled.

    More trustworthy than impurity importance, which favours features with
    many distinct values. Positive = the model relies on the feature.
    """
    result = permutation_importance(
        model, X, y, scoring="neg_log_loss", n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return pd.DataFrame(
        {"importance": result.importances_mean, "std": result.importances_std}, index=X.columns
    ).sort_values("importance", ascending=False)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_confusion_matrices(results: dict[str, dict[str, Any]], path: Path, title: str) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3), CLASS_NAMES, rotation=30)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, probas: dict[str, np.ndarray], path: Path, title: str,
                     n_bins: int = 8) -> None:
    """Reliability curves: predicted probability vs observed frequency, per class."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        for name, proba in probas.items():
            p = proba[:, cls]
            hit = (y_true == cls).astype(float)
            bins = pd.qcut(p, q=n_bins, duplicates="drop")
            grouped = pd.DataFrame({"p": p, "hit": hit}).groupby(bins, observed=True).mean()
            ax.plot(grouped["p"], grouped["hit"], marker="o", label=name)
        ax.set_title(TARGET_LABELS[cls])
        ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(importances: dict[str, pd.DataFrame], path: Path, top_n: int = 15) -> None:
    fig, axes = plt.subplots(1, len(importances), figsize=(7 * len(importances), 6))
    axes = np.atleast_1d(axes)
    for ax, (name, table) in zip(axes, importances.items()):
        top = table.head(top_n).iloc[::-1]
        ax.barh(top.index, top["importance"], xerr=top["std"], color="#2b8a3e")
        ax.set_title(f"{name}: permutation importance (validation log loss)")
        ax.set_xlabel("Increase in log loss when shuffled")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
