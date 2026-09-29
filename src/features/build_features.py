"""Leakage-safe pre-match features.

The core rule: a match's features may only use matches that were finished
before that match kicked off.

How it is enforced
------------------
1. Every match is split into two "team rows" (one from each club's point of
   view) and the rows are sorted by team and date.
2. Every rolling or cumulative statistic is computed on ``.shift(1)`` of the
   team's own history, so the current match is always excluded from its own
   features.
3. League position is recomputed for each match date from matches played on
   strictly earlier dates, so neither the match itself nor same-day matches
   (whose kickoff order we don't know) leak in.
4. Results of the match being described (goals, shots, result, target) are
   never copied into the feature columns; ``feature_columns()`` is the single
   list the models are allowed to see.

Upcoming fixtures can be passed in with their goals/stats set to NaN. They
run through exactly the same code, which is what guarantees prediction-time
features match training-time features.

Possession is not available from Football-Data.co.uk, so possession
features are deliberately absent (see README).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.elo import ELO_FEATURES, EloSettings, compute_elo

logger = logging.getLogger(__name__)

# Columns that describe the outcome of the match itself. They must never be
# model inputs; tests assert none of them appear in feature_columns().
POST_MATCH_COLUMNS = [
    "home_goals", "away_goals", "result", "target",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
]

# Per-team features used by the models. Each appears once with a "home_"
# prefix (describing the home team) and once with "away_".
TEAM_FEATURES = [
    # Recent form: last 5 matches (NaN until a team has 5 matches in its current spell)
    "last5_points",
    "last5_goals_scored",
    "last5_goals_conceded",
    "last5_avg_shots",
    "last5_avg_shots_on_target",
    # Season to date (before this match)
    "season_games_played",
    "season_points_per_game",
    "season_goals_per_game",
    "season_goals_conceded_per_game",
    "season_goal_difference_per_game",
    "season_shots_per_game",
    "season_shots_on_target_per_game",
    "league_position",
]

# Venue-specific form: the home team's record at home, the away team's away.
HOME_VENUE_FEATURES = [
    "home_home_points_per_game",
    "home_home_win_rate",
    "home_home_goals_per_game",
    "home_home_goals_conceded_per_game",
]
AWAY_VENUE_FEATURES = [
    "away_away_points_per_game",
    "away_away_win_rate",
    "away_away_goals_per_game",
    "away_away_goals_conceded_per_game",
]

# Home minus away. Trees can learn these from the separate columns, but
# giving them explicitly may help; the pipeline measures whether it does.
DIFF_FEATURES = [
    "form_points_diff",
    "goal_difference_diff",
    "shots_diff",
    "shots_on_target_diff",
    "season_ppg_diff",
    "season_goal_difference_diff",
    "league_position_diff",
    "venue_ppg_diff",
]


# Optional groups added on top of the base set above. config.yaml's
# features.groups picks which ones the models use.
EWM_TEAM_FEATURES = ["ewm_points", "ewm_goal_diff", "ewm_sot_for", "ewm_sot_against"]
FEATURE_GROUPS: dict[str, list[str]] = {
    # Team strength that carries across seasons (src/features/elo.py).
    "elo": ELO_FEATURES,
    # Exponentially weighted form: recent matches count most, older ones fade
    # out instead of dropping off a 5-match cliff. Includes shots on target
    # conceded, the only defensive shot statistic in the data.
    "ewm": [f"home_{f}" for f in EWM_TEAM_FEATURES] + [f"away_{f}" for f in EWM_TEAM_FEATURES]
           + ["ewm_points_diff", "ewm_goal_diff_diff", "ewm_sot_balance_diff"],
    # Match context: rest, runs, last season's finish and head-to-head.
    "context": [
        "home_rest_days", "away_rest_days",
        "home_unbeaten_run", "away_unbeaten_run", "home_winless_run", "away_winless_run",
        "home_prev_season_position", "away_prev_season_position",
        "h2h_home_points_per_game", "h2h_draw_rate",
    ],
}


def feature_columns(include_diffs: bool = True, groups: list[str] | tuple[str, ...] = (),
                    base: bool = True) -> list[str]:
    """The exact, ordered list of model input columns.

    ``base`` includes the 42 original features (form, season to date, venue,
    and their differences when ``include_diffs``); ``groups`` adds optional
    groups from FEATURE_GROUPS.
    """
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups {sorted(unknown)}; choose from {sorted(FEATURE_GROUPS)}")
    columns: list[str] = []
    if base:
        columns += [f"home_{f}" for f in TEAM_FEATURES] + [f"away_{f}" for f in TEAM_FEATURES]
        columns += HOME_VENUE_FEATURES + AWAY_VENUE_FEATURES
        if include_diffs:
            columns += DIFF_FEATURES
    for group in FEATURE_GROUPS:
        if group in groups:
            columns += FEATURE_GROUPS[group]
    return columns


# --------------------------------------------------------------------------
# Team-level rows
# --------------------------------------------------------------------------

def matches_to_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team), from that team's perspective."""
    common = ["match_id", "season", "date"]
    home = matches[common].copy()
    home["team"] = matches["home_team"]
    home["opponent"] = matches["away_team"]
    home["is_home"] = 1
    home["goals_for"] = matches["home_goals"]
    home["goals_against"] = matches["away_goals"]
    home["shots_for"] = matches["home_shots"]
    home["sot_for"] = matches["home_shots_on_target"]
    home["sot_against"] = matches["away_shots_on_target"]

    away = matches[common].copy()
    away["team"] = matches["away_team"]
    away["opponent"] = matches["home_team"]
    away["is_home"] = 0
    away["goals_for"] = matches["away_goals"]
    away["goals_against"] = matches["home_goals"]
    away["shots_for"] = matches["away_shots"]
    away["sot_for"] = matches["away_shots_on_target"]
    away["sot_against"] = matches["home_shots_on_target"]

    rows = pd.concat([home, away], ignore_index=True)
    rows["played"] = rows["goals_for"].notna().astype(float)

    played = rows["played"] == 1
    rows["win"] = np.where(played, (rows["goals_for"] > rows["goals_against"]).astype(float), np.nan)
    rows["draw"] = np.where(played, (rows["goals_for"] == rows["goals_against"]).astype(float), np.nan)
    rows["loss"] = np.where(played, (rows["goals_for"] < rows["goals_against"]).astype(float), np.nan)
    rows["points"] = 3 * rows["win"] + rows["draw"]
    rows["goal_diff"] = rows["goals_for"] - rows["goals_against"]

    rows = rows.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    rows["spell"] = _premier_league_spells(rows)
    return rows


def _premier_league_spells(rows: pd.DataFrame) -> pd.Series:
    """Number each continuous run of Premier League seasons per team.

    A promoted team's last Premier League match can be several seasons old.
    Carrying that stale form into its first matches back would be misleading,
    so rolling form restarts when a team returns after at least one season away.
    """
    season_of_previous_row = rows.groupby("team")["season"].shift(1)
    new_spell = season_of_previous_row.isna() | (rows["season"] - season_of_previous_row > 1)
    return new_spell.astype(int).groupby(rows["team"]).cumsum()


def _lagged_rolling(rows: pd.DataFrame, keys: list[str], column: str, window: int,
                    min_periods: int, how: str) -> pd.Series:
    """Rolling statistic over each group's *previous* rows.

    ``shift(1)`` inside the group moves every value one match later, so the
    window ending at a match covers the matches before it, never the match
    itself.
    """
    previous = rows.groupby(keys)[column].shift(1)
    rolled = previous.groupby([rows[k] for k in keys]).rolling(window, min_periods=min_periods).agg(how)
    return rolled.reset_index(level=list(range(len(keys))), drop=True).reindex(rows.index)


def _lagged_cumsum(rows: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    """Sum of each group's previous rows: the running total minus the current row."""
    values = rows[column].fillna(0)
    return values.groupby([rows[k] for k in keys]).cumsum() - values


def add_form_features(rows: pd.DataFrame, window: int) -> pd.DataFrame:
    """Last-N-match form within the team's current Premier League spell.

    min_periods == window: a partial "last 5" sum would not be comparable
    with a full one, so it is left NaN (both models handle NaN natively).
    """
    keys = ["team", "spell"]
    out = rows
    out["last5_points"] = _lagged_rolling(rows, keys, "points", window, window, "sum")
    out["last5_wins"] = _lagged_rolling(rows, keys, "win", window, window, "sum")
    out["last5_draws"] = _lagged_rolling(rows, keys, "draw", window, window, "sum")
    out["last5_losses"] = _lagged_rolling(rows, keys, "loss", window, window, "sum")
    out["last5_goals_scored"] = _lagged_rolling(rows, keys, "goals_for", window, window, "sum")
    out["last5_goals_conceded"] = _lagged_rolling(rows, keys, "goals_against", window, window, "sum")
    out["last5_goal_difference"] = out["last5_goals_scored"] - out["last5_goals_conceded"]
    out["last5_avg_shots"] = _lagged_rolling(rows, keys, "shots_for", window, window, "mean")
    out["last5_avg_shots_on_target"] = _lagged_rolling(rows, keys, "sot_for", window, window, "mean")
    out["form_points_per_game"] = out["last5_points"] / window
    return out


def add_season_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date averages using only earlier matches in the same season."""
    keys = ["team", "season"]

    def per_game(column: str) -> pd.Series:
        # Divide by the number of earlier matches where this statistic was
        # actually recorded, so a source without shot data gives NaN rather
        # than a misleading 0 shots per game.
        total = _lagged_cumsum(rows, keys, column)
        recorded = _lagged_cumsum(rows.assign(_recorded=rows[column].notna().astype(float)), keys, "_recorded")
        return total / recorded.replace(0, np.nan)

    rows["season_games_played"] = _lagged_cumsum(rows, keys, "played")
    rows["season_points_per_game"] = per_game("points")
    rows["season_goals_per_game"] = per_game("goals_for")
    rows["season_goals_conceded_per_game"] = per_game("goals_against")
    rows["season_goal_difference_per_game"] = per_game("goal_diff")
    rows["season_shots_per_game"] = per_game("shots_for")
    rows["season_shots_on_target_per_game"] = per_game("sot_for")
    rows["season_win_rate"] = per_game("win")
    rows["season_draw_rate"] = per_game("draw")
    rows["season_loss_rate"] = per_game("loss")
    return rows


def add_venue_features(rows: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """Form over the team's last N matches at the same venue type (home or away)."""
    keys = ["team", "spell", "is_home"]
    rows["venue_points_per_game"] = _lagged_rolling(rows, keys, "points", window, min_periods, "mean")
    rows["venue_win_rate"] = _lagged_rolling(rows, keys, "win", window, min_periods, "mean")
    rows["venue_goals_per_game"] = _lagged_rolling(rows, keys, "goals_for", window, min_periods, "mean")
    rows["venue_goals_conceded_per_game"] = _lagged_rolling(rows, keys, "goals_against", window, min_periods, "mean")
    return rows


def add_league_position(rows: pd.DataFrame) -> pd.DataFrame:
    """League position before kickoff.

    For every match date D the table is built from matches of that season
    played on dates strictly before D (same-day matches are excluded because
    kickoff order is unknown). Teams are ranked by points, goal difference,
    then goals scored, among teams that have played at least one game; a
    team with no games yet gets NaN.
    """
    stats = ["points", "goal_diff", "goals_for", "played"]
    positions = []
    for season, season_rows in rows.groupby("season"):
        dates = np.sort(season_rows["date"].unique())
        played = season_rows[season_rows["played"] == 1]
        daily = played.groupby(["date", "team"])[stats].sum()
        # date x team running totals, shifted one date so each row holds the
        # table as it stood *before* that date.
        before = {
            stat: daily[stat].unstack().reindex(dates).fillna(0).cumsum().shift(1)
            for stat in stats
        }
        # One sortable number per team (points, then goal difference, then
        # goals scored); tied teams share the better position.
        sort_key = before["points"] * 1_000_000 + (before["goal_diff"] + 500) * 1_000 + before["goals_for"]
        sort_key = sort_key.where(before["played"] > 0)
        rank = sort_key.rank(axis=1, method="min", ascending=False)
        rank.index.name, rank.columns.name = "date", "team"
        positions.append(rank.stack().rename("league_position").reset_index())

    position_table = pd.concat(positions, ignore_index=True) if positions else None
    if position_table is None or position_table.empty:
        rows["league_position"] = np.nan
        return rows
    return rows.merge(position_table, on=["date", "team"], how="left")


def add_ewm_features(rows: pd.DataFrame, halflife: float, min_periods: int = 3) -> pd.DataFrame:
    """Exponentially weighted averages of earlier matches in the current spell.

    Missing values (a match without shot data) are skipped rather than
    counted as zero.
    """
    keys = ["team", "spell"]
    group_keys = [rows[k] for k in keys]
    for column, name in (("points", "ewm_points"), ("goal_diff", "ewm_goal_diff"),
                         ("sot_for", "ewm_sot_for"), ("sot_against", "ewm_sot_against")):
        previous = rows.groupby(keys)[column].shift(1)
        rows[name] = previous.groupby(group_keys).transform(
            lambda s: s.ewm(halflife=halflife, min_periods=min_periods, ignore_na=True).mean())
    return rows


def _lagged_run(rows: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    """Length of the run of earlier matches, ending just before this one, where ``column`` was 0.

    Example: column = "loss" gives the current unbeaten run.
    """
    previous = rows.groupby(keys)[column].shift(1)
    breaks = (previous != 0).astype(int)  # a loss, or no earlier match, ends the run
    block = breaks.groupby([rows[k] for k in keys]).cumsum()
    extends = (previous == 0).astype(int)
    return extends.groupby([rows[k] for k in keys] + [block]).cumsum()


def final_positions(rows: pd.DataFrame) -> pd.DataFrame:
    """Final league position of every team in every season with all matches played."""
    played = rows[rows["played"] == 1]
    table = played.groupby(["season", "team"])[["points", "goal_diff", "goals_for"]].sum()
    key = table["points"] * 1_000_000 + (table["goal_diff"] + 500) * 1_000 + table["goals_for"]
    position = key.groupby(level="season").rank(method="min", ascending=False)
    return position.rename("final_position").reset_index()


def add_context_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Rest days, unbeaten/winless runs, last season's finish."""
    by_team = rows.groupby("team")
    previous_date = by_team["date"].shift(1)
    previous_season = by_team["season"].shift(1)
    rest = (rows["date"] - previous_date).dt.days
    # Only within a season: the summer break is not "rest" in any useful sense.
    rows["rest_days"] = rest.where(previous_season == rows["season"])

    keys = ["team", "spell"]
    rows["unbeaten_run"] = _lagged_run(rows, keys, "loss")
    rows["winless_run"] = _lagged_run(rows, keys, "win")

    # Last season's finishing position; a club promoted into the league is
    # given 21 (below every Premier League club). NaN only when the previous
    # season is not in the data at all.
    finals = final_positions(rows)
    finals = finals.assign(season=finals["season"] + 1).rename(columns={"final_position": "prev_season_position"})
    rows = rows.merge(finals, on=["season", "team"], how="left")
    seasons_with_table = set(finals["season"])
    promoted = rows["prev_season_position"].isna() & rows["season"].isin(seasons_with_table)
    rows.loc[promoted, "prev_season_position"] = 21.0
    return rows


def add_head_to_head(rows: pd.DataFrame, window: int = 6, min_periods: int = 2) -> pd.DataFrame:
    """Points per game and draw rate in the last meetings between the same two clubs (any venue)."""
    keys = ["team", "opponent"]
    rows["h2h_points_per_game"] = _lagged_rolling(rows, keys, "points", window, min_periods, "mean")
    rows["h2h_draw_rate"] = _lagged_rolling(rows, keys, "draw", window, min_periods, "mean")
    return rows


def build_team_features(matches: pd.DataFrame, form_window: int = 5, venue_window: int = 10,
                        venue_min_periods: int = 3, ewm_halflife: float = 5.0) -> pd.DataFrame:
    """All per-team pre-match statistics, one row per (match, team)."""
    rows = matches_to_team_rows(matches)
    rows = add_form_features(rows, form_window)
    rows = add_season_features(rows)
    rows = add_venue_features(rows, venue_window, venue_min_periods)
    rows = add_ewm_features(rows, ewm_halflife)
    rows = add_head_to_head(rows)
    rows = add_context_features(rows)
    rows = add_league_position(rows)
    return rows


# --------------------------------------------------------------------------
# Match-level feature table
# --------------------------------------------------------------------------

def build_features(matches: pd.DataFrame, form_window: int = 5, venue_window: int = 10,
                   venue_min_periods: int = 3, ewm_halflife: float = 5.0,
                   elo: dict | None = None) -> pd.DataFrame:
    """Build the model-ready table: identifiers, all features, and the target.

    ``matches`` must have the clean-table columns. Rows whose goals are NaN
    are treated as unplayed fixtures: they get features but no target.
    Every feature group is always built; ``feature_columns`` decides which
    ones a model sees. ``elo`` holds EloSettings fields (defaults if None).
    """
    team_rows = build_team_features(matches, form_window, venue_window, venue_min_periods, ewm_halflife)

    per_team = TEAM_FEATURES + EWM_TEAM_FEATURES + [
        "last5_wins", "last5_draws", "last5_losses", "last5_goal_difference", "form_points_per_game",
        "season_win_rate", "season_draw_rate", "season_loss_rate",
        "rest_days", "unbeaten_run", "winless_run", "prev_season_position",
    ]
    venue = ["venue_points_per_game", "venue_win_rate", "venue_goals_per_game", "venue_goals_conceded_per_game"]

    home_rows = team_rows[team_rows["is_home"] == 1].set_index("match_id")
    away_rows = team_rows[team_rows["is_home"] == 0].set_index("match_id")

    home = home_rows[per_team].add_prefix("home_")
    away = away_rows[per_team].add_prefix("away_")
    home_venue = home_rows[venue].rename(columns=lambda c: c.replace("venue_", "home_home_"))
    away_venue = away_rows[venue].rename(columns=lambda c: c.replace("venue_", "away_away_"))

    ids = ["match_id", "season", "date", "home_team", "away_team"]
    outcome = [c for c in ["home_goals", "away_goals", "result", "target"] if c in matches.columns]
    table = matches[ids + outcome].set_index("match_id")
    table = table.join([home, away, home_venue, away_venue]).reset_index()

    table["form_points_diff"] = table["home_last5_points"] - table["away_last5_points"]
    table["goal_difference_diff"] = table["home_last5_goal_difference"] - table["away_last5_goal_difference"]
    table["shots_diff"] = table["home_last5_avg_shots"] - table["away_last5_avg_shots"]
    table["shots_on_target_diff"] = table["home_last5_avg_shots_on_target"] - table["away_last5_avg_shots_on_target"]
    table["season_ppg_diff"] = table["home_season_points_per_game"] - table["away_season_points_per_game"]
    table["season_goal_difference_diff"] = (
        table["home_season_goal_difference_per_game"] - table["away_season_goal_difference_per_game"]
    )
    # Positive means the home team is placed *higher* (smaller number).
    table["league_position_diff"] = table["away_league_position"] - table["home_league_position"]
    table["venue_ppg_diff"] = table["home_home_points_per_game"] - table["away_away_points_per_game"]

    table["ewm_points_diff"] = table["home_ewm_points"] - table["away_ewm_points"]
    table["ewm_goal_diff_diff"] = table["home_ewm_goal_diff"] - table["away_ewm_goal_diff"]
    table["ewm_sot_balance_diff"] = (
        (table["home_ewm_sot_for"] - table["home_ewm_sot_against"])
        - (table["away_ewm_sot_for"] - table["away_ewm_sot_against"])
    )
    # Head-to-head from the home side's point of view.
    h2h = home_rows[["h2h_points_per_game", "h2h_draw_rate"]].rename(
        columns={"h2h_points_per_game": "h2h_home_points_per_game"})
    table = table.set_index("match_id").join(h2h).reset_index()

    ratings = compute_elo(matches, EloSettings.from_config(elo))
    table = table.merge(ratings, on="match_id", how="left")
    table["elo_diff"] = table["home_elo"] - table["away_elo"]

    return table.sort_values(["date", "match_id"]).reset_index(drop=True)


def features_for_fixtures(history: pd.DataFrame, fixtures: pd.DataFrame, **feature_kwargs) -> pd.DataFrame:
    """Features for upcoming fixtures, computed by the same code as training.

    ``history`` is the clean match table of finished games; ``fixtures`` has
    at least season, date, home_team and away_team. Only history strictly
    before each fixture's date is used.
    """
    fixtures = fixtures.copy()
    for column in POST_MATCH_COLUMNS:
        if column not in ("result", "target"):
            fixtures[column] = np.nan
    start_id = int(history["match_id"].max()) + 1 if len(history) else 0
    fixtures["match_id"] = np.arange(start_id, start_id + len(fixtures))
    earliest = fixtures["date"].min()
    past = history[history["date"] < earliest]
    combined = pd.concat([past, fixtures], ignore_index=True)
    table = build_features(combined, **feature_kwargs)
    return table[table["match_id"].isin(fixtures["match_id"])].reset_index(drop=True)
