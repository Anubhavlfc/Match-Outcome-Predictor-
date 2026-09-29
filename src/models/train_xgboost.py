"""XGBoost: small walk-forward tuning search, fitting and saving."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from src.models.tuning import walk_forward_search

logger = logging.getLogger(__name__)

# Shallow trees and a small learning rate: with noisy football outcomes,
# heavily regularised boosting generalizes better than deep trees.
PARAM_SPACE: dict[str, list[Any]] = {
    "n_estimators": [150, 300, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "min_child_weight": [1, 5, 10, 20],
    "gamma": [0, 0.5, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}


def build_xgboost(params: dict[str, Any], random_state: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="multi:softprob",  # returns a probability for each of the 3 classes
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def tune_xgboost(X: pd.DataFrame, y: pd.Series, seasons: pd.Series, validation_seasons: list[int],
                 n_iter: int, random_state: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Pick hyperparameters by mean walk-forward log loss."""
    return walk_forward_search(
        lambda params: build_xgboost(params, random_state),
        PARAM_SPACE, X, y, seasons, validation_seasons, n_iter, random_state, name="XGBoost",
    )


def train_xgboost(X: pd.DataFrame, y: pd.Series, params: dict[str, Any], random_state: int) -> XGBClassifier:
    model = build_xgboost(params, random_state)
    model.fit(X, y)
    return model


def save_xgboost(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    logger.info("Saved XGBoost to %s", path)


def load_xgboost(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model
