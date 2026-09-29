"""Turn raw Football-Data.co.uk CSVs into one clean match table.

Output columns (one row per match):
    match_id, season, date, home_team, away_team,
    home_goals, away_goals, result, target,
    home_shots, away_shots, home_shots_on_target, away_shots_on_target,
    home_corners, away_corners

Possession and expected goals are not in these files. They are left out
rather than filled with invented values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.team_names import normalize_team_name

logger = logging.getLogger(__name__)

# Target encoding used everywhere in the project.
AWAY_WIN, DRAW, HOME_WIN = 0, 1, 2
TARGET_LABELS = {AWAY_WIN: "Away Win", DRAW: "Draw", HOME_WIN: "Home Win"}
RESULT_TO_TARGET = {"A": AWAY_WIN, "D": DRAW, "H": HOME_WIN}

# Football-Data column -> project column
RAW_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
}
REQUIRED_RAW = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
STAT_COLUMNS = [
    "home_goals", "away_goals",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
]


class DataValidationError(ValueError):
    """Raised when raw data fails a consistency check."""


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse Football-Data dates.

    The official site uses dd/mm/yy or dd/mm/yyyy; the GitHub mirror uses
    ISO yyyy-mm-dd. Each format is tried explicitly so a day and a month are
    never silently swapped.
    """
    text = values.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        missing = parsed.isna()
        if not missing.any():
            break
        parsed[missing] = pd.to_datetime(text[missing], format=fmt, errors="coerce")
    if parsed.isna().any():
        bad = text[parsed.isna()].unique()[:5]
        raise DataValidationError(f"Unparseable dates: {list(bad)}")
    return parsed


def load_raw_season(path: Path, start_year: int) -> pd.DataFrame:
    """Read one raw E0 CSV and return it with project column names."""
    raw = pd.read_csv(path, encoding="latin-1")
    raw = raw.dropna(how="all")  # official files sometimes end with blank rows
    missing = [col for col in REQUIRED_RAW if col not in raw.columns]
    if missing:
        raise DataValidationError(f"{path.name} is missing required columns {missing}")

    df = raw[[c for c in RAW_COLUMNS if c in raw.columns]].rename(columns=RAW_COLUMNS)
    for col in RAW_COLUMNS.values():
        if col not in df.columns:
            df[col] = np.nan
    df["season"] = start_year
    return df


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize names, types and the target, and validate consistency."""
    df = raw.copy()
    df["date"] = parse_dates(df["date"])
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    for col in STAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["result"] = df["result"].astype(str).str.strip().str.upper()
    if not df["result"].isin(RESULT_TO_TARGET).all():
        bad = df.loc[~df["result"].isin(RESULT_TO_TARGET), "result"].unique()
        raise DataValidationError(f"Unexpected result codes: {list(bad)}")

    # The recorded result must agree with the recorded score.
    implied = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] < df["away_goals"]],
        ["H", "A"],
        default="D",
    )
    mismatches = (implied != df["result"]).sum()
    if mismatches:
        raise DataValidationError(f"{mismatches} matches have a result that disagrees with the score")

    df["target"] = df["result"].map(RESULT_TO_TARGET).astype(int)

    duplicates = df.duplicated(subset=["season", "home_team", "away_team"]).sum()
    if duplicates:
        raise DataValidationError(f"{duplicates} duplicate fixtures found")

    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    df.insert(0, "match_id", np.arange(len(df)))
    return df[[
        "match_id", "season", "date", "home_team", "away_team",
        "home_goals", "away_goals", "result", "target",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners",
    ]]


def report_quality(df: pd.DataFrame) -> None:
    """Log per-season match counts and missing statistics."""
    for season, group in df.groupby("season"):
        n_missing = group[STAT_COLUMNS].isna().sum().sum()
        n_teams = pd.concat([group["home_team"], group["away_team"]]).nunique()
        level = logging.WARNING if len(group) != 380 or n_teams != 20 or n_missing else logging.INFO
        logger.log(level, "Season %s: %d matches, %d teams, %d missing stat values",
                   season, len(group), n_teams, n_missing)


def build_match_table(raw_paths: dict[int, Path]) -> pd.DataFrame:
    """Load, combine and clean every season file. ``raw_paths`` maps start year -> path."""
    frames = [load_raw_season(path, year) for year, path in sorted(raw_paths.items())]
    matches = clean_matches(pd.concat(frames, ignore_index=True))
    report_quality(matches)
    return matches
