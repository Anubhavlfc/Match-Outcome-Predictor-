"""Prediction pipeline, history file and live tracker."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features, feature_columns
from src.live import select_fixtures
from src.models.predict import TrainedModels, predict_features, predict_fixtures
from src.models.train_random_forest import train_random_forest
from src.models.train_xgboost import train_xgboost
from src.tracking.history import (
    PredictionLockedError,
    load_history,
    record_predictions,
    save_history,
    update_results,
)
from src.tracking.performance import accuracy_over_time, live_summary

from tests.conftest import make_matches

SETTINGS = {"form_window": 5, "venue_window": 10, "venue_min_periods": 3}
UTC = timezone.utc


@pytest.fixture(scope="module")
def tiny_models():
    matches = make_matches(n_seasons=4, seed=3)
    table = build_features(matches, **SETTINGS)
    cols = feature_columns()
    rf = train_random_forest(table[cols], table["target"], {"n_estimators": 20, "max_depth": 3}, 0)
    xgb = train_xgboost(table[cols], table["target"], {"n_estimators": 20, "max_depth": 2}, 0)
    metadata = {"feature_columns": cols, "feature_settings": SETTINGS, "trained_at": "test"}
    return TrainedModels(rf, xgb, metadata), matches


def _next_fixtures(matches, kickoff):
    return pd.DataFrame([
        {"season": 2024, "date": pd.Timestamp(kickoff.date()), "kickoff": kickoff,
         "home_team": "Arsenal", "away_team": "Chelsea", "round": "Matchday 1"},
        {"season": 2024, "date": pd.Timestamp(kickoff.date()), "kickoff": kickoff,
         "home_team": "Liverpool", "away_team": "Everton", "round": "Matchday 1"},
    ])


def test_predictions_are_valid_probabilities(tiny_models):
    models, matches = tiny_models
    kickoff = pd.Timestamp("2024-08-10 15:00", tz="Europe/London")
    predictions = predict_fixtures(models, matches, _next_fixtures(matches, kickoff))
    assert len(predictions) == 2
    for prefix in ("rf", "xgb"):
        total = predictions[[f"{prefix}_home_probability", f"{prefix}_draw_probability",
                             f"{prefix}_away_probability"]].sum(axis=1)
        assert np.allclose(total, 1.0)
    assert set(predictions["rf_prediction"]) <= {"Home Win", "Draw", "Away Win"}
    assert "kickoff" in predictions.columns


def test_home_probability_maps_to_home_class(tiny_models):
    """Class 2 is Home Win; the home_probability column must come from that column of predict_proba."""
    models, matches = tiny_models
    table = build_features(matches, **SETTINGS).tail(3)
    out = predict_features(models, table)
    proba = models.random_forest.predict_proba(table[models.feature_columns])
    assert np.allclose(out["rf_home_probability"], proba[:, 2])
    assert np.allclose(out["rf_away_probability"], proba[:, 0])


def test_missing_feature_column_is_rejected(tiny_models):
    models, matches = tiny_models
    table = build_features(matches, **SETTINGS).tail(2).drop(columns=["home_last5_points"])
    with pytest.raises(ValueError):
        predict_features(models, table)


def test_prediction_features_equal_training_features(tiny_models):
    """Predicting a known match from earlier history gives exactly the training-table features."""
    models, matches = tiny_models
    full = build_features(matches, **SETTINGS)
    target = matches.iloc[-1]
    fixture = pd.DataFrame([{"season": target.season, "date": target.date,
                             "home_team": target.home_team, "away_team": target.away_team}])
    predicted = predict_fixtures(models, matches[matches["date"] < target.date], fixture)
    expected = predict_features(models, full[full["match_id"] == target.match_id])
    cols = [c for c in expected.columns if c.endswith("_probability")]
    assert np.allclose(predicted[cols].to_numpy(dtype=float), expected[cols].to_numpy(dtype=float))


def test_select_fixtures_only_next_match_within_horizon():
    now = datetime(2026, 10, 1, 12, tzinfo=UTC)
    k1, k2, k3 = (pd.Timestamp(now + timedelta(days=d)) for d in (3, 10, 30))
    fixtures = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "kickoff": k1},
        {"home_team": "C", "away_team": "A", "kickoff": k2},   # A's second upcoming match: not yet
        {"home_team": "D", "away_team": "E", "kickoff": k2},
        {"home_team": "F", "away_team": "G", "kickoff": k3},   # beyond the horizon
        {"home_team": "H", "away_team": "I", "kickoff": pd.Timestamp(now - timedelta(days=1))},  # in the past
    ])
    chosen = select_fixtures(fixtures, now, horizon_days=14)
    assert set(zip(chosen["home_team"], chosen["away_team"])) == {("A", "B"), ("D", "E")}


def _history_with_predictions(tiny_models, now):
    models, matches = tiny_models
    kickoff = pd.Timestamp("2024-08-10 15:00", tz="Europe/London")
    predictions = predict_fixtures(models, matches, _next_fixtures(matches, kickoff))
    history = load_history(pd.io.common.Path("/nonexistent/history.csv"))
    return record_predictions(history, predictions, now, "test", "unit test"), predictions


def test_record_before_kickoff_and_never_overwrite(tiny_models):
    now = datetime(2024, 8, 9, 12, tzinfo=UTC)
    (history, added, skipped), predictions = _history_with_predictions(tiny_models, now)
    assert (added, skipped) == (2, 0)
    assert history["actual_result"].isna().all()

    altered = predictions.copy()
    altered["rf_home_probability"] = 0.99
    history2, added2, skipped2 = record_predictions(history, altered, now, "test", "unit test")
    assert (added2, skipped2) == (0, 2)
    pd.testing.assert_series_equal(history["rf_home_probability"], history2["rf_home_probability"])


def test_recording_after_kickoff_is_refused(tiny_models):
    late = datetime(2024, 8, 10, 14, 0, tzinfo=UTC)  # 15:00 BST == 14:00 UTC: kickoff
    with pytest.raises(PredictionLockedError):
        _history_with_predictions(tiny_models, late)


def test_update_results_scores_predictions_without_touching_probabilities(tiny_models, tmp_path):
    now = datetime(2024, 8, 9, 12, tzinfo=UTC)
    (history, _, _), _ = _history_with_predictions(tiny_models, now)
    # Read the history back from disk first, as the live command does.
    saved = tmp_path / "saved.csv"
    save_history(history, saved)
    history = load_history(saved)
    results = pd.DataFrame([
        {"season": 2024, "home_team": "Arsenal", "away_team": "Chelsea", "home_goals": 2.0, "away_goals": 0.0,
         "result": "H"},
    ])
    later = datetime(2024, 8, 11, tzinfo=UTC)
    updated, n = update_results(history, results, later)
    assert n == 1
    row = updated[updated["home_team"] == "Arsenal"].iloc[0]
    assert row["actual_result"] == "Home Win"
    assert row["rf_correct"] == (row["rf_prediction"] == "Home Win")
    assert updated[updated["home_team"] == "Liverpool"]["actual_result"].isna().all()
    pd.testing.assert_frame_equal(updated.filter(like="_probability"), history.filter(like="_probability"))

    # Round trip through the CSV file and compute the live summary.
    path = tmp_path / "prediction_history.csv"
    save_history(updated, path)
    reloaded = load_history(path)
    summary = live_summary(reloaded)
    assert summary["total_predictions"] == 2 and summary["completed"] == 1 and summary["pending"] == 1
    rf = summary["models"]["Random Forest"]
    assert rf["correct"] == int(row["rf_correct"])
    assert rf["accuracy"] == float(row["rf_correct"])
    assert len(accuracy_over_time(reloaded)) == 1
