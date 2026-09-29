"""Exploratory charts for the final report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.odds import implied_probabilities, odds_matrix
from src.utils.config import season_label
from src.utils.plotting import ENTITY_COLORS, NEUTRAL, OUTCOME_COLORS, TEXT_SECONDARY, apply_style

RESULT_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def outcome_share_by_season(matches: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Share of home wins, draws and away wins per season."""
    apply_style()
    share = pd.crosstab(matches["season"], matches["result"], normalize="index").rename(columns=RESULT_NAMES)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [season_label(s) for s in share.index]
    for outcome in ("Home Win", "Draw", "Away Win"):
        ax.plot(x, share[outcome], marker="o", color=OUTCOME_COLORS[outcome], label=outcome)
    if 2020 in share.index:
        i = list(share.index).index(2020)
        ax.annotate("2020/21: no crowds", (i, share.loc[2020, "Home Win"]), xytext=(i + 0.3, 0.52),
                    fontsize=9, color=TEXT_SECONDARY, arrowprops={"arrowstyle": "-", "color": NEUTRAL})
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share of matches")
    ax.set_title("Outcome mix by season")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right", ncols=3)
    _save(fig, path)
    return share


def goals_by_season(matches: pd.DataFrame, path: Path) -> pd.DataFrame:
    apply_style()
    goals = matches.groupby("season")[["home_goals", "away_goals"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [season_label(s) for s in goals.index]
    ax.plot(x, goals["home_goals"], marker="o", color=OUTCOME_COLORS["Home Win"], label="Home team")
    ax.plot(x, goals["away_goals"], marker="o", color=OUTCOME_COLORS["Away Win"], label="Away team")
    ax.set_ylim(0, 2)
    ax.set_ylabel("Goals per match")
    ax.set_title("Goals per match by season")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    _save(fig, path)
    return goals


def outcome_by_position_gap(features: pd.DataFrame, path: Path) -> pd.DataFrame:
    """How often each outcome happens, by the pre-match league-position gap.

    Positive gap = the home team is placed higher before kickoff.
    """
    apply_style()
    frame = features.dropna(subset=["league_position_diff"])
    bins = [-20, -10, -5, -2, 1, 4, 9, 20]
    labels = ["≤ -10", "-9 to -5", "-4 to -2", "-1 to 1", "2 to 4", "5 to 9", "≥ 10"]
    frame = frame.assign(gap=pd.cut(frame["league_position_diff"], bins=bins, labels=labels))
    names = {2: "Home Win", 1: "Draw", 0: "Away Win"}
    table = pd.crosstab(frame["gap"], frame["target"].map(names), normalize="index")
    counts = frame["gap"].value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(8, 4))
    for outcome in ("Home Win", "Draw", "Away Win"):
        ax.plot(labels, table[outcome], marker="o", color=OUTCOME_COLORS[outcome], label=outcome)
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("League-position gap before kickoff (positive = home team placed higher)")
    ax.set_ylabel("Share of matches")
    ax.set_title("Pre-match table position is a strong signal")
    ax.legend(ncols=3, loc="upper left")
    _save(fig, path)
    return table.assign(matches=counts)


def bookmaker_calibration(frame: pd.DataFrame, path: Path, n_bins: int = 10) -> None:
    """Implied probability vs observed frequency for each outcome."""
    apply_style()
    proba = implied_probabilities(odds_matrix(frame))
    y = frame["target"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1)
    for cls, outcome in ((2, "Home Win"), (1, "Draw"), (0, "Away Win")):
        p = proba[:, cls]
        grouped = pd.DataFrame({"p": p, "hit": (y == cls).astype(float)}).groupby(
            pd.qcut(p, n_bins, duplicates="drop"), observed=True).mean()
        ax.plot(grouped["p"], grouped["hit"], marker="o", color=OUTCOME_COLORS[outcome], label=outcome)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Bookmaker implied probability (margin removed)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("The market is well calibrated")
    ax.legend(loc="upper left")
    _save(fig, path)


def log_loss_by_season(table: pd.DataFrame, path: Path) -> None:
    """Out-of-sample log loss per season for the models, the bookmaker and the baseline.

    ``table`` has columns season plus one column per entity.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [season_label(s) for s in table["season"]]
    for entity in ("Class frequencies", "Random Forest", "XGBoost", "Bookmaker"):
        if entity in table:
            ax.plot(x, table[entity], marker="o", color=ENTITY_COLORS[entity], label=entity,
                    linewidth=1.5 if entity == "Class frequencies" else 2)
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Out-of-sample log loss by season")
    ax.legend(ncols=2)
    _save(fig, path)


def cumulative_profit(bet_logs: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    """Running profit through a season for each betting strategy."""
    apply_style()
    colors = {"Random Forest value bets": ENTITY_COLORS["Random Forest"],
              "XGBoost value bets": ENTITY_COLORS["XGBoost"],
              "Bookmaker favourite": ENTITY_COLORS["Bookmaker"],
              "Always home": NEUTRAL}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    for name, bets in bet_logs.items():
        if bets.empty:
            continue
        ordered = bets.sort_values(["date", "match_id"])
        ax.plot(np.arange(1, len(ordered) + 1), ordered["profit"].cumsum(), color=colors.get(name, NEUTRAL),
                label=f"{name} ({len(ordered)} bets)")
    ax.set_xlabel("Bets placed, in date order")
    ax.set_ylabel("Cumulative profit ($)")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)
