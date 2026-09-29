"""Chronological splits.

Matches are never shuffled across time. Training data always ends before
validation data starts, the same way the model would be used in production.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SeasonSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _assert_chronological(earlier: pd.DataFrame, later: pd.DataFrame) -> None:
    if len(earlier) and len(later) and earlier["date"].max() >= later["date"].min():
        raise ValueError("Split is not chronological: earlier set overlaps the later set in time")


def season_split(features: pd.DataFrame, train_seasons: list[int], validation_season: int,
                 test_season: int) -> SeasonSplit:
    """Train / validation / test by whole seasons."""
    train = features[features["season"].isin(train_seasons)]
    validation = features[features["season"] == validation_season]
    test = features[features["season"] == test_season]
    _assert_chronological(train, validation)
    _assert_chronological(validation, test)
    return SeasonSplit(train.reset_index(drop=True), validation.reset_index(drop=True),
                       test.reset_index(drop=True))


def walk_forward_folds(seasons: pd.Series, validation_seasons: list[int]
                       ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Expanding-window folds over the rows of one frame.

    For each validation season N, train on every earlier season and validate
    on N. Yields (season, train_positions, validation_positions).
    """
    values = seasons.to_numpy()
    for season in validation_seasons:
        train_idx = np.flatnonzero(values < season)
        val_idx = np.flatnonzero(values == season)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Walk-forward fold for season {season} has no train or validation rows")
        yield season, train_idx, val_idx
