"""Live performance of the saved predictions against real results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.clean import TARGET_LABELS
from src.models.predict import MODEL_PREFIXES
from src.tracking.history import as_bool, completed

CLASS_NAMES = [TARGET_LABELS[2], TARGET_LABELS[1], TARGET_LABELS[0]]  # Home, Draw, Away
_PROBA_SUFFIX = {TARGET_LABELS[2]: "home", TARGET_LABELS[1]: "draw", TARGET_LABELS[0]: "away"}


def _probabilities(done: pd.DataFrame, prefix: str) -> np.ndarray:
    return done[[f"{prefix}_{_PROBA_SUFFIX[c]}_probability" for c in CLASS_NAMES]].to_numpy(dtype=float)


def live_summary(history: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers for the live forward test."""
    done = completed(history)
    summary: dict[str, Any] = {
        "total_predictions": int(len(history)),
        "completed": int(len(done)),
        "pending": int(len(history) - len(done)),
        "models": {},
    }
    if done.empty:
        return summary

    actual_index = done["actual_result"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
    for prefix, name in MODEL_PREFIXES.items():
        correct = as_bool(done[f"{prefix}_correct"])
        proba = np.clip(_probabilities(done, prefix), 1e-15, 1)
        onehot = np.eye(3)[actual_index]
        by_predicted = {}
        for cls in CLASS_NAMES:
            picked = done[f"{prefix}_prediction"] == cls
            by_predicted[cls] = {
                "predicted": int(picked.sum()),
                "correct": int(correct[picked].sum()),
                "accuracy": float(correct[picked].mean()) if picked.any() else float("nan"),
            }
        by_actual = {}
        for cls in CLASS_NAMES:
            happened = done["actual_result"] == cls
            by_actual[cls] = {
                "matches": int(happened.sum()),
                "correctly_predicted": int(correct[happened].sum()),
            }
        summary["models"][name] = {
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "log_loss": float(-np.mean(np.log(proba[np.arange(len(done)), actual_index]))),
            "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
            "by_predicted_class": by_predicted,
            "by_actual_result": by_actual,
        }

    agree = as_bool(done["models_agree"])
    rf_correct = as_bool(done["rf_correct"])
    summary["agreement"] = {
        "rate": float(agree.mean()),
        "accuracy_when_agree": float(rf_correct[agree].mean()) if agree.any() else float("nan"),
        "matches_when_disagree": int((~agree).sum()),
    }
    summary["home_win_baseline_accuracy"] = float((done["actual_result"] == TARGET_LABELS[2]).mean())
    return summary


def accuracy_over_time(history: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Cumulative accuracy of each model by fixture date, grouped by period (weekly by default)."""
    done = completed(history)
    if done.empty:
        return pd.DataFrame(columns=["period_end", "matches", "rf_cumulative_accuracy", "xgb_cumulative_accuracy"])
    done["fixture_date"] = pd.to_datetime(done["fixture_date"])
    done = done.sort_values("fixture_date")
    grouped = done.groupby(pd.Grouper(key="fixture_date", freq=freq))
    table = pd.DataFrame({
        "matches": grouped.size(),
        "rf_correct": grouped["rf_correct"].apply(lambda s: as_bool(s).sum()),
        "xgb_correct": grouped["xgb_correct"].apply(lambda s: as_bool(s).sum()),
    })
    table = table[table["matches"] > 0]
    cumulative = table.cumsum()
    out = pd.DataFrame({
        "period_end": table.index.date,
        "matches": table["matches"].to_numpy(),
        "rf_period_accuracy": (table["rf_correct"] / table["matches"]).to_numpy(),
        "xgb_period_accuracy": (table["xgb_correct"] / table["matches"]).to_numpy(),
        "cumulative_matches": cumulative["matches"].to_numpy(),
        "rf_cumulative_accuracy": (cumulative["rf_correct"] / cumulative["matches"]).to_numpy(),
        "xgb_cumulative_accuracy": (cumulative["xgb_correct"] / cumulative["matches"]).to_numpy(),
    })
    return out


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Predictions saved: {summary['total_predictions']}  "
        f"(completed {summary['completed']}, awaiting result {summary['pending']})",
    ]
    if not summary["models"]:
        lines.append("No completed matches yet.")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"{'Model':<15}{'Correct':>9}{'Accuracy':>10}{'Log loss':>10}{'Brier':>8}")
    for name, m in summary["models"].items():
        lines.append(f"{name:<15}{m['correct']:>9}{m['accuracy']:>10.1%}{m['log_loss']:>10.3f}{m['brier']:>8.3f}")
    lines.append(f"{'Always Home':<15}{'':>9}{summary['home_win_baseline_accuracy']:>10.1%}")
    lines.append("")
    lines.append("Hit rate by predicted outcome (when the model picked X, how often X happened):")
    for name, m in summary["models"].items():
        parts = [f"{cls} {v['correct']}/{v['predicted']}" for cls, v in m["by_predicted_class"].items()]
        lines.append(f"  {name}: " + ", ".join(parts))
    a = summary["agreement"]
    lines.append("")
    lines.append(f"Models agreed on {a['rate']:.0%} of matches; accuracy when they agree {a['accuracy_when_agree']:.1%}")
    return "\n".join(lines)
