"""Turning probabilities into a single predicted outcome.

Why the models never predict a draw: a draw is almost never the single most
likely outcome. Even the bookmaker's own prices make a draw the favourite in
practically no Premier League match, because draws happen about 25% of the
time while the stronger side usually wins more often than that. Picking the
highest probability ("argmax") therefore never picks a draw.

That does not make the probabilities wrong. It only means argmax is a poor
rule if you want the predicted outcomes to include draws. The rule here
predicts a draw whenever the draw probability reaches a threshold, and
otherwise picks the more likely of home and away. The threshold is chosen
on walk-forward seasons (never on validation or test) to maximise macro F1,
the average of the per-outcome F1 scores, which rewards getting draws right
as well as wins. Accuracy usually drops a little; the report shows by how
much.

The probabilities themselves are unchanged, so log loss, calibration and the
betting simulation are unaffected.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

DRAW = 1
THRESHOLD_GRID = np.round(np.arange(0.24, 0.345, 0.005), 3)


def predict_outcome(proba: np.ndarray, draw_threshold: float | None = None) -> np.ndarray:
    """Predicted class per row (0 away, 1 draw, 2 home).

    ``draw_threshold=None`` is plain argmax.
    """
    proba = np.asarray(proba)
    if draw_threshold is None:
        return proba.argmax(axis=1)
    home_or_away = np.where(proba[:, 2] >= proba[:, 0], 2, 0)
    return np.where(proba[:, DRAW] >= draw_threshold, DRAW, home_or_away)


def threshold_curve(y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> list[dict]:
    """Accuracy, macro F1 and draw statistics for each candidate threshold."""
    y_true = np.asarray(y_true)
    rows = []
    for threshold in grid:
        pred = predict_outcome(proba, float(threshold))
        predicted_draw = pred == DRAW
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "draw_share": float(predicted_draw.mean()),
            "draw_precision": float((y_true[predicted_draw] == DRAW).mean()) if predicted_draw.any() else float("nan"),
            "draw_recall": float((pred[y_true == DRAW] == DRAW).mean()) if (y_true == DRAW).any() else float("nan"),
        })
    return rows


def choose_draw_threshold(y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> dict:
    """The threshold with the best macro F1 on the given (walk-forward) predictions.

    Returns threshold None (plain argmax) if no threshold beats argmax.
    """
    curve = threshold_curve(y_true, proba, grid)
    best = max(curve, key=lambda row: (row["macro_f1"], row["accuracy"]))
    pred = predict_outcome(proba)
    argmax = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "draw_share": float((pred == DRAW).mean()),
    }
    if argmax["macro_f1"] >= best["macro_f1"]:
        return {"threshold": None, "selected": argmax, "argmax": argmax, "curve": curve}
    return {"threshold": best["threshold"], "selected": best, "argmax": argmax, "curve": curve}
