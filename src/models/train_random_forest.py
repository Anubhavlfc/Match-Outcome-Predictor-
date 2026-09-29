"""Random Forest: small walk-forward tuning search, fitting and saving."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.tuning import walk_forward_search

logger = logging.getLogger(__name__)

# Modest ranges: the dataset has only a few thousand rows, so deep, fully
# grown trees would mostly memorize noise.
PARAM_SPACE: dict[str, list[Any]] = {
    "n_estimators": [300, 500],
    "max_depth": [4, 6, 8, 12, None],
    "min_samples_split": [2, 10, 30],
    "min_samples_leaf": [5, 10, 20, 40],
    "max_features": ["sqrt", 0.3, 0.5],
    "class_weight": [None, "balanced"],
}


def build_random_forest(params: dict[str, Any], random_state: int) -> RandomForestClassifier:
    # No scaling or imputation: trees are scale-invariant, and scikit-learn
    # >= 1.4 routes missing values (e.g. a promoted team's empty form) natively.
    return RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)


def tune_random_forest(X: pd.DataFrame, y: pd.Series, seasons: pd.Series,
                       validation_seasons: list[int], n_iter: int, random_state: int
                       ) -> tuple[dict[str, Any], pd.DataFrame]:
    """Pick hyperparameters by mean walk-forward log loss."""
    return walk_forward_search(
        lambda params: build_random_forest(params, random_state),
        PARAM_SPACE, X, y, seasons, validation_seasons, n_iter, random_state, name="Random Forest",
    )


def train_random_forest(X: pd.DataFrame, y: pd.Series, params: dict[str, Any],
                        random_state: int) -> RandomForestClassifier:
    model = build_random_forest(params, random_state)
    model.fit(X, y)
    return model


def save_random_forest(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    logger.info("Saved Random Forest to %s", path)


def load_random_forest(path: Path) -> RandomForestClassifier:
    return joblib.load(path)
