"""Predict upcoming fixtures with both saved models.

For each fixture:
1. take every match finished before its date;
2. build both teams' features with the same code used in training
   (``features_for_fixtures``);
3. check the columns are exactly the saved models' feature schema;
4. run Random Forest and XGBoost and return each model's probabilities,
   plus a predicted outcome from the draw rule saved with the models
   (``src/models/decision.py``); older model files without one use argmax.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.clean import AWAY_WIN, DRAW, HOME_WIN, TARGET_LABELS
from src.features.build_features import features_for_fixtures
from src.models.decision import predict_outcome
from src.models.train_random_forest import load_random_forest
from src.models.train_xgboost import load_xgboost

logger = logging.getLogger(__name__)

# Short prefixes used in prediction tables and the history file.
MODEL_PREFIXES = {"rf": "Random Forest", "xgb": "XGBoost"}


@dataclass
class TrainedModels:
    random_forest: RandomForestClassifier
    xgboost: XGBClassifier
    metadata: dict[str, Any]

    @property
    def feature_columns(self) -> list[str]:
        return self.metadata["feature_columns"]

    @property
    def feature_settings(self) -> dict[str, int]:
        return self.metadata["feature_settings"]

    def draw_threshold(self, prefix: str) -> float | None:
        return self.metadata.get("draw_threshold", {}).get(MODEL_PREFIXES[prefix])

    def by_prefix(self) -> dict[str, Any]:
        return {"rf": self.random_forest, "xgb": self.xgboost}


def load_models(models_dir: Path) -> TrainedModels:
    paths = {
        "rf": models_dir / "random_forest.pkl",
        "xgb": models_dir / "xgboost.json",
        "meta": models_dir / "model_metadata.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files {missing}. Train first with: python -m src.pipeline")
    metadata = json.loads(paths["meta"].read_text())
    return TrainedModels(load_random_forest(paths["rf"]), load_xgboost(paths["xgb"]), metadata)


def build_fixture_features(history: pd.DataFrame, fixtures: pd.DataFrame, settings: dict[str, int]) -> pd.DataFrame:
    """Pre-match features for each fixture, using only matches before its date.

    Fixtures are processed one date at a time: a team plays at most once per
    date, and ``features_for_fixtures`` only uses history strictly before
    the date it is given.
    """
    frames = []
    for fixture_date, group in fixtures.groupby("date", sort=True):
        base = group[["season", "date", "home_team", "away_team"]]
        frames.append(features_for_fixtures(history, base, **settings))
    if not frames:
        return pd.DataFrame()
    features = pd.concat(frames, ignore_index=True)
    # Carry extra fixture columns (kickoff time, round) back onto the feature rows.
    extra = [c for c in fixtures.columns if c not in features.columns]
    return features.merge(fixtures[["date", "home_team", "away_team", *extra]],
                          on=["date", "home_team", "away_team"], how="left")


def predict_features(models: TrainedModels, features: pd.DataFrame) -> pd.DataFrame:
    """Both models' probabilities for rows that already have the feature columns."""
    columns = models.feature_columns
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"Feature rows are missing model columns: {sorted(missing)}")
    X = features[columns]  # exact training order

    out = features[["season", "date", "home_team", "away_team"]].copy()
    for prefix, model in models.by_prefix().items():
        proba = model.predict_proba(X)
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"{MODEL_PREFIXES[prefix]} probabilities do not sum to 1")
        out[f"{prefix}_home_probability"] = proba[:, HOME_WIN]
        out[f"{prefix}_draw_probability"] = proba[:, DRAW]
        out[f"{prefix}_away_probability"] = proba[:, AWAY_WIN]
        predicted = predict_outcome(proba, models.draw_threshold(prefix))
        out[f"{prefix}_prediction"] = [TARGET_LABELS[i] for i in predicted]
    out["models_agree"] = out["rf_prediction"] == out["xgb_prediction"]
    return out


def predict_fixtures(models: TrainedModels, history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Features + predictions for upcoming fixtures, plus any extra fixture columns."""
    features = build_fixture_features(history, fixtures, models.feature_settings)
    if features.empty:
        return features
    predictions = predict_features(models, features)
    extra = [c for c in features.columns if c in ("kickoff", "round")]
    return pd.concat([predictions, features[extra]], axis=1)


def outcome_name(label: str, home_team: str, away_team: str) -> str:
    return {"Home Win": f"{home_team} Win", "Away Win": f"{away_team} Win", "Draw": "Draw"}[label]


def format_prediction(row: pd.Series) -> str:
    """Readable block like the project brief's example output."""
    home, away = row["home_team"], row["away_team"]
    width = max(len(home), len(away), 4) + 6
    lines = [f"{home} vs {away}  ({pd.Timestamp(row['date']).date()})", ""]
    for prefix, name in MODEL_PREFIXES.items():
        lines += [
            name,
            f"  {home + ' Win:':<{width}} {row[f'{prefix}_home_probability']:6.1%}",
            f"  {'Draw:':<{width}} {row[f'{prefix}_draw_probability']:6.1%}",
            f"  {away + ' Win:':<{width}} {row[f'{prefix}_away_probability']:6.1%}",
            f"  Predicted: {outcome_name(row[f'{prefix}_prediction'], home, away)}",
            "",
        ]
    lines.append("Models agree" if row["models_agree"] else "Models disagree")
    return "\n".join(lines)
