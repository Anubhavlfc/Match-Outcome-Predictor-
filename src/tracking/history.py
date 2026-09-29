"""The prediction history: a forward test that can't be edited after the fact.

Rules
-----
* A prediction is recorded only **before kickoff**. Trying to record one for
  a match that has started is refused.
* The first prediction for a fixture stands. Re-running ``predict`` never
  overwrites probabilities that were already saved.
* Filling in results only touches the actual-result columns.

The file is a plain CSV so it can be read by anyone and diffed in git (the
commit history is an independent record of when each prediction was made).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import TARGET_LABELS

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = [
    "prediction_id",
    "prediction_timestamp",       # UTC, when the prediction was saved
    "kickoff",                    # local kickoff time with UTC offset
    "fixture_date",
    "season",
    "round",
    "home_team",
    "away_team",
    "rf_home_probability", "rf_draw_probability", "rf_away_probability", "rf_prediction",
    "xgb_home_probability", "xgb_draw_probability", "xgb_away_probability", "xgb_prediction",
    "models_agree",
    "model_trained_at",           # which saved models made the prediction
    "results_source",
    "data_notes",                 # e.g. no_shot_data; empty when inputs were complete
    "actual_home_goals", "actual_away_goals",
    "actual_result",
    "rf_correct", "xgb_correct",
    "result_recorded_at",
]
PROBABILITY_COLUMNS = [c for c in HISTORY_COLUMNS if c.endswith("_probability")]
RESULT_CODES = {"H": TARGET_LABELS[2], "D": TARGET_LABELS[1], "A": TARGET_LABELS[0]}


class PredictionLockedError(ValueError):
    """Raised when a prediction would be saved at or after kickoff."""


def prediction_id(season: int, home_team: str, away_team: str) -> str:
    """Each pairing happens once per league season, so this identifies a fixture."""
    slug = lambda name: name.lower().replace(" ", "-").replace("'", "")  # noqa: E731
    return f"{season}-{slug(home_team)}-vs-{slug(away_team)}"


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.read_csv(path, dtype={"data_notes": str, "round": str})
    missing = set(HISTORY_COLUMNS) - set(history.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return history[HISTORY_COLUMNS]


def save_history(history: pd.DataFrame, path: Path) -> None:
    """Write via a temporary file so a crash never leaves a half-written history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    history[HISTORY_COLUMNS].to_csv(tmp, index=False, float_format="%.6f")
    os.replace(tmp, path)


def record_predictions(history: pd.DataFrame, predictions: pd.DataFrame, now: datetime,
                       model_trained_at: str, results_source: str, data_notes: dict[str, str] | None = None
                       ) -> tuple[pd.DataFrame, int, int]:
    """Append new predictions. Returns (history, n_added, n_already_recorded).

    ``predictions`` comes from ``predict_fixtures`` and must include ``kickoff``.
    ``now`` must be timezone-aware.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    data_notes = data_notes or {}
    existing = set(history["prediction_id"])
    new_rows, skipped = [], 0
    for row in predictions.itertuples(index=False):
        pid = prediction_id(row.season, row.home_team, row.away_team)
        if pid in existing:
            skipped += 1
            continue
        kickoff = pd.Timestamp(row.kickoff)
        if now >= kickoff:
            raise PredictionLockedError(
                f"{row.home_team} vs {row.away_team} kicked off at {kickoff}; predictions must be saved before kickoff"
            )
        new_rows.append({
            "prediction_id": pid,
            "prediction_timestamp": pd.Timestamp(now).tz_convert("UTC").isoformat(timespec="seconds"),
            "kickoff": kickoff.isoformat(),
            "fixture_date": pd.Timestamp(row.date).date().isoformat(),
            "season": row.season,
            "round": getattr(row, "round", None),
            "home_team": row.home_team,
            "away_team": row.away_team,
            **{c: getattr(row, c) for c in PROBABILITY_COLUMNS},
            "rf_prediction": row.rf_prediction,
            "xgb_prediction": row.xgb_prediction,
            "models_agree": bool(row.models_agree),
            "model_trained_at": model_trained_at,
            "results_source": results_source,
            "data_notes": data_notes.get(pid, ""),
        })
        existing.add(pid)
    if new_rows:
        history = pd.concat([history, pd.DataFrame(new_rows, columns=HISTORY_COLUMNS)], ignore_index=True)
    return history, len(new_rows), skipped


def update_results(history: pd.DataFrame, matches: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, int]:
    """Fill actual results for recorded predictions whose match has finished.

    ``matches`` is the clean match table. Only rows without a result are
    touched, and only the actual-result columns change.
    """
    history = history.copy()
    # A history read back from CSV has all-empty result columns typed as
    # float; make them object so text and booleans can be written.
    for column in ("actual_result", "rf_correct", "xgb_correct", "result_recorded_at"):
        history[column] = history[column].astype(object)
    finished = matches.assign(
        prediction_id=[prediction_id(s, h, a) for s, h, a in zip(matches["season"], matches["home_team"], matches["away_team"])]
    ).set_index("prediction_id")
    pending = history["actual_result"].isna() & history["prediction_id"].isin(finished.index)
    for index in history.index[pending]:
        match = finished.loc[history.at[index, "prediction_id"]]
        actual = RESULT_CODES[match["result"]]
        history.at[index, "actual_home_goals"] = match["home_goals"]
        history.at[index, "actual_away_goals"] = match["away_goals"]
        history.at[index, "actual_result"] = actual
        history.at[index, "rf_correct"] = history.at[index, "rf_prediction"] == actual
        history.at[index, "xgb_correct"] = history.at[index, "xgb_prediction"] == actual
        history.at[index, "result_recorded_at"] = pd.Timestamp(now).tz_convert("UTC").isoformat(timespec="seconds")
    return history, int(pending.sum())


def completed(history: pd.DataFrame) -> pd.DataFrame:
    return history[history["actual_result"].notna()].copy()


def as_bool(series: pd.Series) -> pd.Series:
    """CSV round-trips booleans as strings; normalize them."""
    return series.map(lambda v: v if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() == "true")
