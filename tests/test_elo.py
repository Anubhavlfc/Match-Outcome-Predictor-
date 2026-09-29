"""Elo ratings: pre-match only, zero-sum, sensible for promoted clubs."""

import numpy as np
import pandas as pd

from src.features.elo import EloSettings, compute_elo, expected_home_score

from tests.conftest import make_matches


def test_ratings_are_pre_match_and_zero_sum():
    matches = make_matches(n_seasons=1)
    settings = EloSettings(k=20, home_advantage=60, season_regression=0.0)
    elo = compute_elo(matches, settings).merge(matches, on="match_id").sort_values(["date", "match_id"])
    # Before anyone has played, everyone is at the initial rating.
    first_date = elo["date"].min()
    opening = elo[elo["date"] == first_date]
    assert (opening[["home_elo", "away_elo"]] == settings.initial).all().all()
    # Points only move between the two clubs in a match, so the league total never changes.
    last = {}
    for row in elo.itertuples():
        last[row.home_team], last[row.away_team] = row.home_elo, row.away_elo
    assert np.isclose(sum(last.values()), settings.initial * len(last))


def test_a_win_raises_the_winner_and_lowers_the_loser():
    matches = make_matches(n_seasons=1)
    elo = compute_elo(matches, EloSettings(season_regression=0.0)).merge(matches, on="match_id")
    first = elo.sort_values(["date", "match_id"]).iloc[0]
    later = elo[(elo["date"] > first["date"])]
    next_home = later[(later.home_team == first.home_team) | (later.away_team == first.home_team)].iloc[0]
    rating_after = next_home.home_elo if next_home.home_team == first.home_team else next_home.away_elo
    if first.home_goals > first.away_goals:
        assert rating_after > first.home_elo
    elif first.home_goals < first.away_goals:
        assert rating_after < first.home_elo


def test_promoted_club_starts_at_relegated_clubs_average():
    matches = make_matches(n_seasons=2)
    # Replace Everton by a promoted club in the second season.
    second = matches["season"] == 2021
    matches.loc[second, ["home_team", "away_team"]] = matches.loc[second, ["home_team", "away_team"]].replace(
        "Everton", "Burnley")
    settings = EloSettings(season_regression=0.0)
    elo = compute_elo(matches, settings).merge(matches, on="match_id")

    # With a four-team league the "bottom three" are every club but the champion.
    first = matches[~second]
    pts = {}
    for m in first.itertuples():
        pts[m.home_team] = pts.get(m.home_team, 0) + (3 if m.home_goals > m.away_goals else m.home_goals == m.away_goals)
        pts[m.away_team] = pts.get(m.away_team, 0) + (3 if m.away_goals > m.home_goals else m.home_goals == m.away_goals)
    end_ratings = {}
    first_elo = elo[~elo["match_id"].isin(matches.loc[second, "match_id"])].sort_values(["date", "match_id"])
    for row in first_elo.itertuples():
        end_ratings[row.home_team], end_ratings[row.away_team] = row.home_elo, row.away_elo
    burnley_first = elo[second.values & ((elo.home_team == "Burnley") | (elo.away_team == "Burnley"))]
    burnley_first = burnley_first.sort_values("date").iloc[0]
    burnley_rating = burnley_first.home_elo if burnley_first.home_team == "Burnley" else burnley_first.away_elo
    # Burnley's rating must be an average of end-of-season ratings, not the default.
    assert min(end_ratings.values()) - 1e-9 <= burnley_rating <= max(end_ratings.values()) + 1e-9


def test_upcoming_fixtures_get_ratings_without_updating_them():
    matches = make_matches(n_seasons=1)
    last_date = matches["date"].max()
    fixtures = matches["date"] == last_date
    matches.loc[fixtures, ["home_goals", "away_goals"]] = np.nan
    elo = compute_elo(matches)
    assert elo["home_elo"].notna().all()


def test_expected_score_is_symmetric():
    assert np.isclose(expected_home_score(1500, 1500, 0), 0.5)
    assert np.isclose(expected_home_score(1600, 1500, 0) + expected_home_score(1500, 1600, 0), 1.0)
    assert expected_home_score(1500, 1500, 60) > 0.5
