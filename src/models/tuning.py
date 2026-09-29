"""Shared walk-forward hyperparameter search.

Every candidate is scored on expanding-window folds inside the training
seasons (train on seasons < N, validate on N). The validation and test
seasons are never touched here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler

from src.models.split import walk_forward_folds

logger = logging.getLogger(__name__)


def walk_forward_search(build_model: Callable[[dict[str, Any]], Any], param_space: dict[str, list[Any]],
                        X: pd.DataFrame, y: pd.Series, seasons: pd.Series, validation_seasons: list[int],
                        n_iter: int, random_state: int, name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Random search scored by mean walk-forward log loss (lower is better).

    Log loss is used rather than accuracy because the goal is good
    probabilities, and accuracy barely moves between reasonable settings.
    Returns (best_params, one row per candidate and fold).
    """
    candidates = list(ParameterSampler(param_space, n_iter=n_iter, random_state=random_state))
    records = []
    for i, params in enumerate(candidates):
        for season, train_idx, val_idx in walk_forward_folds(seasons, validation_seasons):
            model = build_model(params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[val_idx])
            records.append({
                "candidate": i,
                "params": params,
                "validation_season": season,
                "log_loss": log_loss(y.iloc[val_idx], proba, labels=[0, 1, 2]),
                "accuracy": float(np.mean(proba.argmax(axis=1) == y.iloc[val_idx].to_numpy())),
            })
        logger.info("%s candidate %d/%d done", name, i + 1, len(candidates))

    results = pd.DataFrame(records)
    summary = results.groupby("candidate")["log_loss"].mean()
    best = int(summary.idxmin())
    logger.info("%s best walk-forward log loss %.4f with %s", name, summary[best], candidates[best])
    return candidates[best], results
