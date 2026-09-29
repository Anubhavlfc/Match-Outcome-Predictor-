"""Assemble everything known right now: historical seasons plus the season in progress."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.data.clean import build_match_table, clean_matches, load_raw_season, report_quality
from src.data.collect import collect_seasons, raw_path
from src.data.current_season import CurrentSeason, load_current_season
from src.utils.config import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class LiveData:
    matches: pd.DataFrame        # clean table of every finished match, history + current season
    current: CurrentSeason


def load_live_data(config: dict, today: date, refresh: bool = True) -> LiveData:
    data_cfg = config["data"]
    raw_dir = resolve_path(data_cfg["raw_dir"])
    first, last = data_cfg["history_start_season"], data_cfg["last_season"]
    collect_seasons(first, last, data_cfg["sources"], raw_dir)  # completed seasons: cached, never re-downloaded

    current = load_current_season(config, today, raw_dir, refresh=refresh)
    if current.season <= last:
        # The configured history already covers this season; nothing to add.
        return LiveData(build_match_table({y: raw_path(raw_dir, y) for y in range(first, last + 1)}), current)

    frames = [load_raw_season(raw_path(raw_dir, year), year) for year in range(first, last + 1)]
    frames.append(current.played)
    matches = clean_matches(pd.concat(frames, ignore_index=True))
    report_quality(matches[matches["season"] == current.season])
    return LiveData(matches, current)
