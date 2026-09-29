"""Data-leakage tests for the feature pipeline.

These are the most important tests in the project: if any of them fail,
the reported model performance can't be trusted.
"""

import numpy as np
import pandas as pd

from src.features.build_features import (
    POST_MATCH_COLUMNS,
    build_features,
    feature_columns,
    features_for_fixtures,
)

from tests.conftest import make_matches

ALL_FEATURES = feature_columns(include_diffs=True)


def _row(table, match_id):
    return table.loc[table["match_id"] == match_id, ALL_FEATURES].iloc[0]


def test_no_target_or_post_match_columns_in_features():
    for include_diffs in (True, False):
        columns = feature_columns(include_diffs)
        assert not set(columns) & set(POST_MATCH_COLUMNS)
        assert not {"match_id", "season", "date", "home_team", "away_team"} & set(columns)
        assert len(columns) == len(set(columns))


def test_feature_columns_exist_in_built_table(synthetic_matches):
    table = build_features(synthetic_matches)
    missing = set(ALL_FEATURES) - set(table.columns)
    assert not missing


def test_current_match_result_does_not_change_its_own_features(synthetic_matches):
    """If Liverpool scores 10 today, today's features must not move."""
    match_id = 20  # season 2, a mid-season match
    original = build_features(synthetic_matches)
    edited = synthetic_matches.copy()
    idx = edited["match_id"] == match_id
    edited.loc[idx, ["home_goals", "away_goals", "home_shots", "away_shots",
                     "home_shots_on_target", "away_shots_on_target"]] = [10, 0, 40, 0, 30, 0]
    edited.loc[idx, ["result", "target"]] = ["H", 2]
    changed = build_features(edited)

    pd.testing.assert_series_equal(_row(original, match_id), _row(changed, match_id))

    # ...but the home team's *next* match must see the new result.
    home_team = synthetic_matches.loc[idx, "home_team"].iloc[0]
    later = changed[(changed["match_id"] > match_id)
                    & ((changed["home_team"] == home_team) | (changed["away_team"] == home_team))]
    next_id = later["match_id"].min()
    assert not _row(original, next_id).equals(_row(changed, next_id))


def test_future_matches_do_not_change_past_features(synthetic_matches):
    """Rewriting everything after a cutoff must leave earlier features identical.

    This catches full-season aggregates leaking into early-season rows.
    """
    cutoff = synthetic_matches["date"].iloc[len(synthetic_matches) // 2]
    original = build_features(synthetic_matches)
    edited = synthetic_matches.copy()
    future = edited["date"] >= cutoff
    edited.loc[future, ["home_goals", "away_goals"]] = [9, 0]
    edited.loc[future, "home_shots"] = 50
    changed = build_features(edited)

    before = original["date"] <= cutoff  # the cutoff date itself is still pre-match for its own games
    pd.testing.assert_frame_equal(original.loc[before, ALL_FEATURES], changed.loc[before, ALL_FEATURES])


def test_last5_uses_exactly_the_previous_five_matches(synthetic_matches):
    table = build_features(synthetic_matches)
    team = "Liverpool"
    team_games = synthetic_matches[(synthetic_matches.home_team == team) | (synthetic_matches.away_team == team)]
    team_games = team_games.sort_values("date").reset_index(drop=True)

    def points_and_goals(match):
        is_home = match.home_team == team
        scored = match.home_goals if is_home else match.away_goals
        conceded = match.away_goals if is_home else match.home_goals
        points = 3 if scored > conceded else 1 if scored == conceded else 0
        return points, scored, conceded

    for position in range(len(team_games)):
        match = team_games.iloc[position]
        row = table[table.match_id == match.match_id].iloc[0]
        side = "home" if match.home_team == team else "away"
        if position < 5:
            assert np.isnan(row[f"{side}_last5_points"])
            continue
        previous = [points_and_goals(team_games.iloc[i]) for i in range(position - 5, position)]
        assert row[f"{side}_last5_points"] == sum(p for p, _, _ in previous)
        assert row[f"{side}_last5_goals_scored"] == sum(s for _, s, _ in previous)
        assert row[f"{side}_last5_goals_conceded"] == sum(c for _, _, c in previous)


def test_season_features_reset_each_season(synthetic_matches):
    table = build_features(synthetic_matches)
    for season, group in table.groupby("season"):
        first_date = group["date"].min()
        opening = group[group["date"] == first_date]
        assert (opening["home_season_games_played"] == 0).all()
        assert opening["home_season_points_per_game"].isna().all()
        assert opening["home_league_position"].isna().all()


def test_league_position_uses_only_earlier_dates(synthetic_matches):
    table = build_features(synthetic_matches)
    season = synthetic_matches[synthetic_matches.season == 2020]
    third_week = season["date"].drop_duplicates().sort_values().iloc[2]
    earlier = season[season["date"] < third_week]

    points = {}
    for m in earlier.itertuples():
        hp = 3 if m.home_goals > m.away_goals else 1 if m.home_goals == m.away_goals else 0
        ap = 3 if m.away_goals > m.home_goals else 1 if m.home_goals == m.away_goals else 0
        points[m.home_team] = points.get(m.home_team, 0) + hp
        points[m.away_team] = points.get(m.away_team, 0) + ap

    for row in table[table["date"] == third_week].itertuples():
        # A team's position can never be better than 1 + number of teams with more points.
        better = sum(p > points[row.home_team] for p in points.values())
        assert row.home_league_position >= 1 + better
        tied_or_better = sum(p >= points[row.home_team] for p in points.values())
        assert row.home_league_position <= tied_or_better


def test_promoted_team_form_restarts():
    """A team that misses a season starts with empty last-5 form when it returns."""
    matches = make_matches(n_seasons=3)
    # Remove Everton from the middle season entirely (pretend it was relegated).
    middle = matches["season"] == 2021
    everton = (matches["home_team"] == "Everton") | (matches["away_team"] == "Everton")
    matches = matches[~(middle & everton)].reset_index(drop=True)
    table = build_features(matches)
    return_season = table[(table.season == 2022) & ((table.home_team == "Everton") | (table.away_team == "Everton"))]
    first_back = return_season.sort_values("date").iloc[0]
    side = "home" if first_back.home_team == "Everton" else "away"
    assert np.isnan(first_back[f"{side}_last5_points"])


def test_prediction_features_match_training_features(real_matches):
    """Features built for an 'upcoming' fixture equal the training-table features.

    For a sample of real matches, rebuild features using only history before
    the match date plus the match as an unplayed fixture. Every feature must
    equal the value in the full training table. This proves both that no
    future information is used and that prediction-time features have the
    same schema and values as training-time features.
    """
    full = build_features(real_matches)
    sample = real_matches[real_matches["season"] >= 2016].sample(25, random_state=7)
    for match in sample.itertuples():
        fixture = pd.DataFrame([{
            "season": match.season, "date": match.date,
            "home_team": match.home_team, "away_team": match.away_team,
        }])
        rebuilt = features_for_fixtures(real_matches, fixture)
        assert list(rebuilt[ALL_FEATURES].columns) == ALL_FEATURES
        expected = full.loc[full["match_id"] == match.match_id, ALL_FEATURES].iloc[0].to_numpy(dtype=float)
        actual = rebuilt[ALL_FEATURES].iloc[0].to_numpy(dtype=float)
        assert np.allclose(expected, actual, equal_nan=True), (match.date, match.home_team, match.away_team)
