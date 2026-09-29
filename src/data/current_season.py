"""Fixtures and results for the season in progress.

Two sources, combined:

* **Schedule and goals** from openfootball (``football.json`` on GitHub): every
  fixture with its kickoff time, plus the score once played. Scores only, no
  shots.
* **Full match statistics** from Football-Data.co.uk's current-season E0 file,
  when it can be downloaded. Where both sources have a match, the scores
  must agree, and the Football-Data row (with shots) is used.

openfootball writes most scores as ``{"ft": [h, a], "ht": [...]}`` but some
(all 0-0 so far) as a bare ``[h, a]`` list. Checked against all 380 matches of
2025/26 in Football-Data.co.uk, both forms are full-time scores.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.data.clean import DataValidationError, load_raw_season, parse_dates
from src.data.collect import DataCollectionError, download_season
from src.utils.team_names import normalize_team_name

logger = logging.getLogger(__name__)

SHOT_COLUMNS = ["home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target"]


@dataclass
class CurrentSeason:
    season: int
    played: pd.DataFrame      # finished matches in raw project schema (ready for clean_matches)
    fixtures: pd.DataFrame    # unplayed fixtures: season, date, kickoff, round, home_team, away_team
    results_source: str       # where the finished results came from
    shot_data_share: float    # share of finished matches that have shot statistics


def season_in_progress(today: date) -> int:
    """Premier League seasons start in August: Jul 2026-Jun 2027 is season 2026."""
    return today.year if today.month >= 7 else today.year - 1


def openfootball_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def parse_openfootball(payload: dict, season: int, tz: str) -> pd.DataFrame:
    """One row per fixture with kickoff time and (if played) the score."""
    zone = ZoneInfo(tz)
    rows = []
    for match in payload.get("matches", []):
        score = match.get("score")
        if isinstance(score, dict):
            full_time = score.get("ft")
        elif isinstance(score, list):
            full_time = score  # bare list form, see module docstring
        else:
            full_time = None
        if full_time is not None and len(full_time) != 2:
            raise DataValidationError(f"Unexpected openfootball score {score!r} in {match}")
        kickoff_time = match.get("time") or "00:00"
        kickoff = datetime.fromisoformat(f"{match['date']}T{kickoff_time}").replace(tzinfo=zone)
        rows.append({
            "season": season,
            "date": match["date"],
            "kickoff": kickoff,
            "round": match.get("round"),
            "home_team": normalize_team_name(match["team1"]),
            "away_team": normalize_team_name(match["team2"]),
            "home_goals": float(full_time[0]) if full_time else np.nan,
            "away_goals": float(full_time[1]) if full_time else np.nan,
        })
    schedule = pd.DataFrame(rows)
    if schedule.duplicated(["home_team", "away_team"]).any():
        raise DataValidationError("openfootball schedule has duplicate fixtures")
    return schedule


def fetch_schedule(season: int, url_template: str, cache_dir: Path, tz: str, refresh: bool = True,
                   timeout: int = 30) -> pd.DataFrame:
    """Download (or reuse the cached) openfootball file for a season."""
    label = openfootball_label(season)
    cache = cache_dir / f"openfootball_{label}.json"
    if refresh or not cache.exists():
        try:
            response = requests.get(url_template.format(label=label), timeout=timeout)
            response.raise_for_status()
            json.loads(response.content)  # validate before overwriting the cache
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(response.content)
            logger.info("Downloaded %s schedule from openfootball", label)
        except (requests.RequestException, ValueError) as exc:
            if not cache.exists():
                raise DataCollectionError(f"Could not download the {label} schedule: {exc}") from exc
            logger.warning("Schedule download failed (%s); using cached %s", exc, cache.name)
    return parse_openfootball(json.loads(cache.read_text()), season, tz)


def _football_data_results(season: int, sources: list[dict], raw_dir: Path, refresh: bool) -> pd.DataFrame | None:
    """Current-season Football-Data file with shots, or None if unavailable."""
    try:
        path = download_season(season, sources, raw_dir, force=refresh)
    except DataCollectionError as exc:
        logger.warning("No Football-Data.co.uk file for the current season (%s)", str(exc).splitlines()[0])
        return None
    return load_raw_season(path, season)


def load_current_season(config: dict, today: date, raw_dir: Path, refresh: bool = True) -> CurrentSeason:
    live = config["live"]
    season = season_in_progress(today)
    schedule = fetch_schedule(season, live["schedule_url_template"], raw_dir, live["timezone"], refresh)

    schedule_played = schedule[schedule["home_goals"].notna()].copy()
    schedule_played["result"] = np.select(
        [schedule_played["home_goals"] > schedule_played["away_goals"],
         schedule_played["home_goals"] < schedule_played["away_goals"]], ["H", "A"], default="D")
    for column in [*SHOT_COLUMNS, "home_corners", "away_corners"]:
        schedule_played[column] = np.nan

    football_data = _football_data_results(season, config["data"]["sources"], raw_dir, refresh)
    source = "openfootball (goals only)"
    played = schedule_played
    if football_data is not None and len(football_data):
        fd = football_data.copy()
        fd["home_team"] = fd["home_team"].map(normalize_team_name)
        fd["away_team"] = fd["away_team"].map(normalize_team_name)
        both = fd.merge(schedule_played, on=["home_team", "away_team"], suffixes=("", "_of"))
        disagree = both[(both["home_goals"] != both["home_goals_of"]) | (both["away_goals"] != both["away_goals_of"])]
        if len(disagree):
            raise DataValidationError(f"Football-Data and openfootball disagree on {len(disagree)} scores")
        keys = set(zip(fd["home_team"], fd["away_team"]))
        extra = schedule_played[[k not in keys for k in zip(schedule_played["home_team"], schedule_played["away_team"])]]
        played = pd.concat([fd, extra[fd.columns.intersection(extra.columns)]], ignore_index=True)
        source = "Football-Data.co.uk + openfootball" if len(extra) else "Football-Data.co.uk"

    played = played.copy()
    # Football-Data uses dd/mm/yyyy, openfootball ISO: normalize to ISO strings.
    played["date"] = parse_dates(played["date"].astype(str)).dt.strftime("%Y-%m-%d")
    played_keys = set(zip(played["home_team"], played["away_team"]))
    fixtures = schedule[[k not in played_keys for k in zip(schedule["home_team"], schedule["away_team"])]]
    fixtures = fixtures[["season", "date", "kickoff", "round", "home_team", "away_team"]].copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    shot_share = float(played[SHOT_COLUMNS].notna().all(axis=1).mean()) if len(played) else 0.0
    logger.info("Season %s: %d played (%s), %d fixtures left, shot data for %.0f%% of played matches",
                season, len(played), source, len(fixtures), 100 * shot_share)
    return CurrentSeason(season, played, fixtures.reset_index(drop=True), source, shot_share)
