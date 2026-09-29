from datetime import date

import numpy as np
import pandas as pd

from src.data.current_season import parse_openfootball, season_in_progress

PAYLOAD = {"matches": [
    {"round": "Matchday 1", "date": "2026-08-21", "time": "20:00", "team1": "Arsenal FC",
     "team2": "Coventry City FC", "score": {"ht": [2, 0], "ft": [3, 0]}},
    {"round": "Matchday 1", "date": "2026-08-22", "time": "15:00", "team1": "Hull City AFC",
     "team2": "AFC Bournemouth", "score": [0, 0]},
    {"round": "Matchday 2", "date": "2026-08-29", "time": "17:30", "team1": "Manchester United FC",
     "team2": "Nottingham Forest FC"},
]}


def test_season_in_progress():
    assert season_in_progress(date(2026, 9, 29)) == 2026
    assert season_in_progress(date(2027, 3, 1)) == 2026
    assert season_in_progress(date(2026, 7, 1)) == 2026


def test_parse_openfootball_scores_and_names():
    schedule = parse_openfootball(PAYLOAD, 2026, "Europe/London")
    assert list(schedule["home_team"]) == ["Arsenal", "Hull City", "Manchester United"]
    assert list(schedule["away_team"]) == ["Coventry City", "Bournemouth", "Nottingham Forest"]
    assert schedule.loc[0, ["home_goals", "away_goals"]].tolist() == [3.0, 0.0]
    # The bare-list form is a full-time score (verified against 2025/26 data).
    assert schedule.loc[1, ["home_goals", "away_goals"]].tolist() == [0.0, 0.0]
    assert np.isnan(schedule.loc[2, "home_goals"])


def test_kickoff_is_uk_local_time():
    schedule = parse_openfootball(PAYLOAD, 2026, "Europe/London")
    kickoff = schedule.loc[0, "kickoff"]
    # 20:00 BST on 21 Aug is 19:00 UTC.
    assert pd.Timestamp(kickoff).tz_convert("UTC").hour == 19
