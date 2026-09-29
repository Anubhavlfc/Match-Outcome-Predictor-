import pandas as pd
import pytest

from src.data.clean import DataValidationError, clean_matches, parse_dates


def _raw(**overrides):
    row = {"date": "2023-08-12", "home_team": "Arsenal", "away_team": "Nott'm Forest",
           "home_goals": 2, "away_goals": 1, "result": "H", "home_shots": 15, "away_shots": 6,
           "home_shots_on_target": 7, "away_shots_on_target": 2, "home_corners": 8,
           "away_corners": 3, "season": 2023}
    row.update(overrides)
    return pd.DataFrame([row])


def test_date_formats_parse_to_the_same_day():
    parsed = parse_dates(pd.Series(["2023-08-12", "12/08/2023", "12/08/23"]))
    assert (parsed == pd.Timestamp("2023-08-12")).all()


def test_target_encoding():
    assert clean_matches(_raw())["target"].iloc[0] == 2
    assert clean_matches(_raw(home_goals=1, away_goals=1, result="D"))["target"].iloc[0] == 1
    assert clean_matches(_raw(home_goals=0, away_goals=1, result="A"))["target"].iloc[0] == 0


def test_result_must_match_score():
    with pytest.raises(DataValidationError):
        clean_matches(_raw(home_goals=0, away_goals=1, result="H"))


def test_team_names_normalized():
    cleaned = clean_matches(_raw())
    assert cleaned["away_team"].iloc[0] == "Nottingham Forest"
