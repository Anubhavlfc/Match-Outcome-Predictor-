import numpy as np
import pandas as pd
import pytest

from src.data.clean import DataValidationError
from src.data.odds import attach_odds, implied_probabilities, odds_matrix
from src.models import betting


def _frame():
    # target: 2 = home win, 1 = draw, 0 = away win
    return pd.DataFrame({
        "match_id": [1, 2, 3],
        "season": [2025] * 3,
        "date": pd.to_datetime(["2025-08-16"] * 3),
        "home_team": ["A", "C", "E"],
        "away_team": ["B", "D", "F"],
        "target": [2, 1, 0],
        "odds_home": [2.0, 1.5, 3.0],
        "odds_draw": [3.5, 4.0, 3.2],
        "odds_away": [4.0, 6.0, 2.5],
    })


def test_implied_probabilities_remove_margin_and_keep_class_order():
    frame = _frame()
    proba = implied_probabilities(odds_matrix(frame))
    assert np.allclose(proba.sum(axis=1), 1.0)
    # Column order is away, draw, home: the 1.5 home favourite has the largest home probability.
    assert proba[1].argmax() == 2
    assert proba[2].argmax() == 0


def test_model_pick_settles_wins_and_losses():
    frame = _frame()
    picks = np.array([[0.1, 0.2, 0.7],    # home, wins at 2.0 -> +100
                      [0.1, 0.2, 0.7],    # home, but it was a draw -> -100
                      [0.6, 0.2, 0.2]])   # away, wins at 2.5 -> +150
    bets = betting.model_pick_bets(frame, picks, stake=100)
    assert bets["profit"].tolist() == [100.0, -100.0, 150.0]
    summary = betting.summarize(bets)
    assert summary["bets"] == 3 and summary["profit"] == 150.0
    assert summary["roi"] == pytest.approx(0.5)
    assert summary["roi_ci_low"] <= summary["roi"] <= summary["roi_ci_high"]


def test_value_bets_need_an_edge():
    frame = _frame()
    proba = np.array([[0.10, 0.20, 0.70],   # home EV = 0.7*2.0-1 = 0.40 -> bet home
                      [0.15, 0.20, 0.65],   # best EV: draw 0.2*4-1 = -0.2, home 0.65*1.5-1=-0.025 -> no bet
                      [0.44, 0.28, 0.28]])  # away EV = 0.44*2.5-1 = 0.10 -> bet only if edge < 0.10
    assert len(betting.value_bets(frame, proba, 100, edge=0.05)) == 2
    assert len(betting.value_bets(frame, proba, 100, edge=0.20)) == 1
    only = betting.value_bets(frame, proba, 100, edge=0.20)
    assert only["pick"].tolist() == ["Home Win"]


def test_attach_odds_rejects_date_mismatch():
    matches = _frame()[["match_id", "season", "date", "home_team", "away_team", "target"]]
    odds = _frame().drop(columns=["match_id", "target"])
    assert attach_odds(matches, odds)["odds_home"].notna().all()
    odds.loc[0, "date"] = pd.Timestamp("2025-09-01")
    with pytest.raises(DataValidationError):
        attach_odds(matches, odds)
