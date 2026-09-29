"""Bookmaker odds for benchmarking and the betting simulation.

Odds are never model inputs. They answer two questions: how good are our
probabilities compared with the market's, and would betting on them have
made money?

Source: football-data.co.uk's odds columns, republished by the
``premier-league-data`` project (the official site is unreachable from the
build environment). ``B365H/D/A`` are Bet365 prices collected before the
weekend's matches (every season); ``B365CH/CD/CA`` are closing prices
(from 2019/20).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.clean import DataValidationError, parse_dates
from src.utils.team_names import normalize_team_name

logger = logging.getLogger(__name__)

ODDS_COLUMNS = {
    "pre_match": ("{b}_1x2_home", "{b}_1x2_draw", "{b}_1x2_away"),
    "closing": ("{b}_1x2_home_close", "{b}_1x2_draw_close", "{b}_1x2_away_close"),
}
OUTPUT_COLUMNS = [
    "season", "date", "home_team", "away_team",
    "odds_home", "odds_draw", "odds_away",
    "close_odds_home", "close_odds_draw", "close_odds_away",
]


def _season_start(label: str) -> int:
    """'2015-16' -> 2015."""
    return int(str(label)[:4])


def download_odds(url: str, cache: Path, bookmaker: str, first_season: int, force: bool = False,
                  timeout: int = 120) -> pd.DataFrame:
    """Download the odds file once and keep a small E0 extract in ``cache``."""
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["date"])
    logger.info("Downloading odds from %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    from io import BytesIO
    raw = pd.read_csv(BytesIO(response.content), low_memory=False)

    raw = raw[raw["season"].map(_season_start) >= first_season].copy()
    pre = [c.format(b=bookmaker) for c in ODDS_COLUMNS["pre_match"]]
    close = [c.format(b=bookmaker) for c in ODDS_COLUMNS["closing"]]
    missing = [c for c in pre + close if c not in raw.columns]
    if missing:
        raise DataValidationError(f"Odds file is missing columns {missing}")
    odds = pd.DataFrame({
        "season": raw["season"].map(_season_start),
        "date": parse_dates(raw["date"].astype(str)),
        "home_team": raw["home_team"].map(normalize_team_name),
        "away_team": raw["away_team"].map(normalize_team_name),
    })
    for name, column in zip(OUTPUT_COLUMNS[4:7], pre):
        odds[name] = pd.to_numeric(raw[column], errors="coerce")
    for name, column in zip(OUTPUT_COLUMNS[7:], close):
        odds[name] = pd.to_numeric(raw[column], errors="coerce")
    cache.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(cache, index=False)
    return odds


def attach_odds(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join odds onto the match table by season and teams, checking dates agree.

    Each pairing happens once per season, so (season, home, away) is a key.
    A date disagreement would mean the two sources describe different
    matches, so it raises instead of silently joining.
    """
    joined = matches.merge(odds, on=["season", "home_team", "away_team"], how="left",
                           suffixes=("", "_odds"), validate="one_to_one")
    has_odds = joined["date_odds"].notna()
    mismatched = has_odds & (joined["date"] != joined["date_odds"])
    if mismatched.any():
        sample = joined.loc[mismatched, ["season", "home_team", "away_team", "date", "date_odds"]].head()
        raise DataValidationError(f"{int(mismatched.sum())} matches have different dates in the odds file:\n{sample}")
    missing = int((~has_odds).sum())
    if missing:
        logger.warning("%d matches have no odds row", missing)
    return joined.drop(columns=["date_odds"])


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Bookmaker probabilities with the margin removed.

    1/odds sums to more than 1 (the bookmaker's margin, or overround). Dividing
    by that sum is the simplest standard way to remove it. Columns follow the
    project's class order: away, draw, home.
    """
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def odds_matrix(frame: pd.DataFrame, closing: bool = False) -> np.ndarray:
    """Decimal odds as an (n, 3) array in class order (away, draw, home)."""
    prefix = "close_odds" if closing else "odds"
    return frame[[f"{prefix}_away", f"{prefix}_draw", f"{prefix}_home"]].to_numpy(dtype=float)
