"""The draw rule turns probabilities into outcomes without touching the probabilities."""

import numpy as np

from src.models.decision import choose_draw_threshold, predict_outcome, threshold_curve


def test_argmax_when_no_threshold():
    proba = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
    assert predict_outcome(proba).tolist() == [2, 0, 1]


def test_draw_predicted_when_draw_probability_reaches_threshold():
    proba = np.array([[0.35, 0.29, 0.36], [0.30, 0.25, 0.45], [0.40, 0.28, 0.32]])
    assert predict_outcome(proba, 0.28).tolist() == [1, 2, 1]
    # Otherwise the more likely of home and away, even if the draw is second.
    assert predict_outcome(proba, 0.30).tolist() == [2, 2, 0]


def test_threshold_choice_improves_or_keeps_macro_f1():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([3, 2.5, 4], size=600)
    y = np.array([rng.choice(3, p=p) for p in proba])
    rule = choose_draw_threshold(y, proba)
    assert rule["selected"]["macro_f1"] >= rule["argmax"]["macro_f1"] - 1e-12
    assert len(rule["curve"]) == len(threshold_curve(y, proba))
