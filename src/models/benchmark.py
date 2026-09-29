"""Compare model probabilities with the bookmaker's implied probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.odds import implied_probabilities, odds_matrix
from src.models.evaluate import compute_metrics


def bookmaker_metrics(frame: pd.DataFrame, closing: bool = False) -> dict[str, Any]:
    """The bookmaker scored as if it were one more model.

    ``frame`` has one row per match with ``target`` and the odds columns.
    """
    usable = frame.dropna(subset=[f"{'close_odds' if closing else 'odds'}_{o}" for o in ("home", "draw", "away")])
    proba = implied_probabilities(odds_matrix(usable, closing))
    metrics = compute_metrics(usable["target"].to_numpy(), proba)
    metrics["overround"] = float((1.0 / odds_matrix(usable, closing)).sum(axis=1).mean() - 1)
    return metrics


def probability_gap(model_proba: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    """How far the model's probabilities sit from the market's, on average."""
    market = implied_probabilities(odds_matrix(frame))
    gap = np.abs(model_proba - market)
    return {"mean_abs_gap": float(gap.mean()), "same_favourite": float(np.mean(model_proba.argmax(1) == market.argmax(1)))}
