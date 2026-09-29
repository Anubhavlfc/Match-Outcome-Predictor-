import pandas as pd
import pytest

from src.models.split import season_split, walk_forward_folds


def _frame():
    return pd.DataFrame({
        "season": [2019, 2019, 2020, 2020, 2021, 2022],
        "date": pd.to_datetime(["2019-09-01", "2020-03-01", "2020-09-01", "2021-03-01", "2021-09-01", "2022-09-01"]),
        "target": [0, 1, 2, 0, 1, 2],
    })


def test_season_split_is_chronological():
    split = season_split(_frame(), [2019, 2020], 2021, 2022)
    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()
    assert len(split.train) == 4 and len(split.validation) == 1 and len(split.test) == 1


def test_overlapping_split_rejected():
    with pytest.raises(ValueError):
        season_split(_frame(), [2019, 2021], 2020, 2022)


def test_walk_forward_folds_expand_and_never_look_ahead():
    frame = _frame()
    folds = list(walk_forward_folds(frame["season"], [2020, 2021]))
    assert [season for season, _, _ in folds] == [2020, 2021]
    previous_size = 0
    for season, train_idx, val_idx in folds:
        assert frame["season"].iloc[train_idx].max() < season
        assert (frame["season"].iloc[val_idx] == season).all()
        assert len(train_idx) > previous_size
        previous_size = len(train_idx)
