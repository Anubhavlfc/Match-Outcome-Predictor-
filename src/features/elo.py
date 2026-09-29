"""Elo team-strength ratings, computed match by match from earlier results only.

Rolling form forgets everything older than a few games and restarts every
season. Elo is a single number per club that carries across seasons: after
each match the winner takes points from the loser, more for a surprise and
more for a big margin.

Leakage rule: a match's features hold both teams' ratings *before* that
match. Ratings change only after a team's own matches, and a team plays at
most once per date, so processing matches in date order is enough.

Between seasons every rating is pulled part of the way back to the mean,
and a newly promoted club starts at the average rating of the clubs that
went down the season before (the teams it replaces).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ELO_FEATURES = ["home_elo", "away_elo", "elo_diff"]


@dataclass(frozen=True)
class EloSettings:
    # Defaults chosen on training seasons 2003/04-2023/24 only (lowest squared
    # error of the expected home score); see config.yaml.
    k: float = 12.0                  # update size
    home_advantage: float = 60.0     # rating points added to the home side's expectation
    season_regression: float = 0.15  # share of the gap to the mean removed between seasons
    initial: float = 1500.0

    @classmethod
    def from_config(cls, config: dict | None) -> "EloSettings":
        return cls(**(config or {}))


def expected_home_score(home_elo: np.ndarray, away_elo: np.ndarray, home_advantage: float) -> np.ndarray:
    """Elo's expected score for the home side (win = 1, draw = 0.5, loss = 0)."""
    return 1.0 / (1.0 + 10 ** (-(np.asarray(home_elo) + home_advantage - np.asarray(away_elo)) / 400))


def _margin_multiplier(goal_difference: float) -> float:
    """World Football Elo margin factor: bigger wins move ratings further."""
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def _bottom_of_table(season_matches: pd.DataFrame, n: int = 3) -> list[str]:
    """The ``n`` lowest-placed teams of a finished season (points, goal difference, goals)."""
    played = season_matches.dropna(subset=["home_goals", "away_goals"])
    home = pd.DataFrame({"team": played["home_team"], "gf": played["home_goals"], "ga": played["away_goals"]})
    away = pd.DataFrame({"team": played["away_team"], "gf": played["away_goals"], "ga": played["home_goals"]})
    rows = pd.concat([home, away])
    rows["pts"] = np.select([rows["gf"] > rows["ga"], rows["gf"] == rows["ga"]], [3, 1], 0)
    table = rows.groupby("team").agg(pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    table["gd"] = table["gf"] - table["ga"]
    return list(table.sort_values(["pts", "gd", "gf"]).index[:n])


def compute_elo(matches: pd.DataFrame, settings: EloSettings = EloSettings()) -> pd.DataFrame:
    """Pre-match ratings for every row of ``matches``.

    ``matches`` needs match_id, season, date, home_team, away_team,
    home_goals, away_goals. Rows with missing goals (upcoming fixtures) get
    ratings but do not update them. Returns match_id, home_elo, away_elo.

    Everything a rating depends on comes from earlier matches: a club
    missing from the previous season is given, at its first match, the mean
    rating of the previous season's bottom three. That keeps the result the
    same whether the current season is complete or only partly played.
    """
    ordered = matches.sort_values(["date", "match_id"])
    ratings: dict[str, float] = {}
    previous_teams: set[str] = set()
    season_teams: set[str] = set()
    entry_rating = settings.initial
    current_season = None
    out_ids, out_home, out_away = [], [], []

    for row in ordered.itertuples(index=False):
        if row.season != current_season:
            if current_season is not None:
                finished = ordered[ordered["season"] == current_season]
                previous_teams = season_teams
                relegated = _bottom_of_table(finished)
            else:
                relegated = []
            current_season = row.season
            season_teams = set()
            for team in ratings:
                ratings[team] += settings.season_regression * (settings.initial - ratings[team])
            known = [ratings[t] for t in relegated if t in ratings]
            entry_rating = float(np.mean(known)) if known else settings.initial

        for team in (row.home_team, row.away_team):
            if team not in season_teams:
                season_teams.add(team)
                if team not in previous_teams:
                    ratings[team] = entry_rating

        home, away = ratings[row.home_team], ratings[row.away_team]
        out_ids.append(row.match_id)
        out_home.append(home)
        out_away.append(away)

        if pd.isna(row.home_goals) or pd.isna(row.away_goals):
            continue
        goal_difference = row.home_goals - row.away_goals
        actual = 1.0 if goal_difference > 0 else 0.5 if goal_difference == 0 else 0.0
        expected = expected_home_score(home, away, settings.home_advantage)
        change = settings.k * _margin_multiplier(goal_difference) * (actual - expected)
        ratings[row.home_team] = home + change
        ratings[row.away_team] = away - change

    return pd.DataFrame({"match_id": out_ids, "home_elo": out_home, "away_elo": out_away})


def elo_fit_error(matches: pd.DataFrame, settings: EloSettings, seasons: list[int]) -> float:
    """Mean squared error of Elo's expected home score on ``seasons``.

    Used to choose the Elo settings on training seasons only.
    """
    elo = compute_elo(matches, settings).merge(
        matches[["match_id", "season", "home_goals", "away_goals"]], on="match_id")
    elo = elo[elo["season"].isin(seasons) & elo["home_goals"].notna()]
    actual = np.sign(elo["home_goals"] - elo["away_goals"]) / 2 + 0.5
    expected = expected_home_score(elo["home_elo"], elo["away_elo"], settings.home_advantage)
    return float(np.mean((actual - expected) ** 2))
