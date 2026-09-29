"""Betting simulation ("Simulating Investment").

Flat stakes on decimal odds: a winning bet returns ``stake * odds`` (profit
``stake * (odds - 1)``); a losing bet loses the stake. By default bets are
placed at Bet365's pre-match prices, which a bettor could actually have
taken. ``odds_kind="max_odds"`` uses the best price listed across
bookmakers at the same time instead ("line shopping"), which is what a
bettor with several accounts would do.

Strategies
----------
* **Model pick**: back the model's most likely outcome in every match
  (the approach in the reference walkthrough).
* **Value bets**: back the outcome with the largest expected value
  ``p_model * odds - 1``, and only when that exceeds ``edge``. This is how a
  probability model would really be used against a market.
  ``max_odds`` optionally skips long prices, where a large apparent edge is
  more often a model error than a real mispricing.
* **Always home** and **bookmaker favourite**: baselines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.odds import implied_probabilities, odds_matrix

OUTCOME_NAMES = np.array(["Away Win", "Draw", "Home Win"])


def _settle(frame: pd.DataFrame, picks: np.ndarray, stake: float, odds_kind: str = "odds") -> pd.DataFrame:
    """One row per bet placed. ``picks`` holds a class index, or -1 for no bet."""
    odds = odds_matrix(frame, kind=odds_kind)
    placed = picks >= 0
    rows = np.flatnonzero(placed)
    chosen = picks[placed]
    price = odds[rows, chosen]
    won = frame["target"].to_numpy()[rows] == chosen
    bets = frame.iloc[rows][["match_id", "season", "date", "home_team", "away_team"]].copy()
    bets["pick"] = OUTCOME_NAMES[chosen]
    bets["odds"] = price
    bets["won"] = won
    bets["stake"] = stake
    bets["profit"] = np.where(won, stake * (price - 1), -stake)
    return bets.reset_index(drop=True)


def model_pick_bets(frame: pd.DataFrame, proba: np.ndarray, stake: float, odds_kind: str = "odds") -> pd.DataFrame:
    return _settle(frame, proba.argmax(axis=1), stake, odds_kind)


def value_bets(frame: pd.DataFrame, proba: np.ndarray, stake: float, edge: float, odds_kind: str = "odds",
               max_odds: float | None = None) -> pd.DataFrame:
    odds = odds_matrix(frame, kind=odds_kind)
    expected_value = proba * odds - 1
    if max_odds is not None:
        expected_value = np.where(odds <= max_odds, expected_value, -np.inf)
    best = expected_value.argmax(axis=1)
    picks = np.where(expected_value.max(axis=1) > edge, best, -1)
    return _settle(frame, picks, stake, odds_kind)


def always_home_bets(frame: pd.DataFrame, stake: float, odds_kind: str = "odds") -> pd.DataFrame:
    return _settle(frame, np.full(len(frame), 2), stake, odds_kind)


def favourite_bets(frame: pd.DataFrame, stake: float, odds_kind: str = "odds") -> pd.DataFrame:
    # The favourite is judged by Bet365's prices whichever prices the bet is settled at.
    return _settle(frame, implied_probabilities(odds_matrix(frame)).argmax(axis=1), stake, odds_kind)


def summarize(bets: pd.DataFrame, random_state: int = 42, n_boot: int = 5000) -> dict[str, Any]:
    """Profit, ROI and a bootstrap 95% interval for ROI (bets resampled with replacement)."""
    if bets.empty:
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"), "hit_rate": float("nan"),
                "average_odds": float("nan"), "roi_ci_low": float("nan"), "roi_ci_high": float("nan")}
    profit = bets["profit"].to_numpy()
    stake = bets["stake"].to_numpy()
    rng = np.random.default_rng(random_state)
    samples = rng.integers(0, len(bets), size=(n_boot, len(bets)))
    boot_roi = profit[samples].sum(axis=1) / stake[samples].sum(axis=1)
    return {
        "bets": int(len(bets)),
        "staked": float(stake.sum()),
        "profit": float(profit.sum()),
        "roi": float(profit.sum() / stake.sum()),
        "hit_rate": float(bets["won"].mean()),
        "average_odds": float(bets["odds"].mean()),
        "roi_ci_low": float(np.percentile(boot_roi, 2.5)),
        "roi_ci_high": float(np.percentile(boot_roi, 97.5)),
    }
