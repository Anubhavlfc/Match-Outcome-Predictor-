"""Shared fixtures: a small synthetic league and the real match table (if cached)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import RESULT_TO_TARGET
from src.data.collect import raw_path
from src.utils.config import load_config, resolve_path

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton"]


def make_matches(n_seasons: int = 2, seed: int = 0) -> pd.DataFrame:
    """Double round-robin seasons for four teams, one match per team per week."""
    rng = np.random.default_rng(seed)
    pairs = list(itertools.permutations(TEAMS, 2))  # 12 fixtures per season
    # Six "weeks" of two matches each, so no team plays twice on one date.
    rounds = [
        [("Arsenal", "Chelsea"), ("Liverpool", "Everton")],
        [("Chelsea", "Liverpool"), ("Everton", "Arsenal")],
        [("Arsenal", "Liverpool"), ("Chelsea", "Everton")],
        [("Chelsea", "Arsenal"), ("Everton", "Liverpool")],
        [("Liverpool", "Chelsea"), ("Arsenal", "Everton")],
        [("Liverpool", "Arsenal"), ("Everton", "Chelsea")],
    ]
    assert sorted(p for r in rounds for p in r) == sorted(pairs)
    rows = []
    for season in range(2020, 2020 + n_seasons):
        start = pd.Timestamp(f"{season}-08-10")
        for week, fixtures in enumerate(rounds):
            for home, away in fixtures:
                hg, ag = rng.integers(0, 4, size=2)
                rows.append({
                    "season": season, "date": start + pd.Timedelta(days=7 * week),
                    "home_team": home, "away_team": away,
                    "home_goals": float(hg), "away_goals": float(ag),
                    "home_shots": float(rng.integers(5, 20)), "away_shots": float(rng.integers(5, 20)),
                    "home_shots_on_target": float(rng.integers(1, 8)),
                    "away_shots_on_target": float(rng.integers(1, 8)),
                    "home_corners": 5.0, "away_corners": 5.0,
                })
    df = pd.DataFrame(rows).sort_values(["date", "home_team"]).reset_index(drop=True)
    df["result"] = np.select([df.home_goals > df.away_goals, df.home_goals < df.away_goals], ["H", "A"], "D")
    df["target"] = df["result"].map(RESULT_TO_TARGET)
    df.insert(0, "match_id", np.arange(len(df)))
    return df


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    return make_matches()


@pytest.fixture(scope="session")
def real_matches() -> pd.DataFrame:
    """The real cleaned match table, built from cached raw CSVs (skips if absent)."""
    from src.data.clean import build_match_table

    config = load_config()
    raw_dir = resolve_path(config["data"]["raw_dir"])
    years = range(config["data"]["history_start_season"], config["data"]["last_season"] + 1)
    paths = {y: raw_path(raw_dir, y) for y in years}
    if not all(Path(p).exists() for p in paths.values()):
        pytest.skip("Raw data not downloaded; run python -m src.pipeline first")
    return build_match_table(paths)
