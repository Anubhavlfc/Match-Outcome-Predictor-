"""Why some public projects report ~70% accuracy: a reproduction.

    python -m src.analysis.in_play_reference

stogaja/Football-Match-Outcome-Predictor reports ~70% with Random Forest and
logistic regression. Its inputs are the half-time score plus the full-match
shots, shots on target and red cards of the match being predicted, with a
random 80/20 split. That is an in-play model: it predicts at half-time with
second-half statistics already known. None of those inputs exist before
kickoff, so the number cannot be compared with a pre-match model.

This script reruns that setup on our Premier League data (2009/10 to
2018/19, the seasons that project used) and then removes the same-match
information step by step, so the report can show where the accuracy comes
from. It is never used by the pre-match models.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data.collect import raw_path
from src.utils.config import load_config, resolve_path, setup_logging

logger = logging.getLogger("in_play_reference")

SEASONS = range(2009, 2019)
FEATURE_SETS = {
    "Their inputs: team ids, half-time score, same-match shots, shots on target, red cards":
        ["home_id", "away_id", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HR", "AR"],
    "Half-time score only": ["HTHG", "HTAG"],
    "Same-match shots on target and red cards only": ["HST", "AST", "HR", "AR"],
    "Team ids only (the only pre-match information in their inputs)": ["home_id", "away_id"],
}


def run(config_path: str | None = None, seed: int = 0) -> dict:
    config = load_config(config_path)
    raw_dir = resolve_path(config["data"]["raw_dir"])
    frames = []
    for season in SEASONS:
        frame = pd.read_csv(raw_path(raw_dir, season), encoding="latin-1").dropna(how="all")
        frames.append(frame.assign(season=season))
    data = pd.concat(frames, ignore_index=True)
    data["home_id"] = LabelEncoder().fit_transform(data["HomeTeam"])
    data["away_id"] = LabelEncoder().fit_transform(data["AwayTeam"])
    data = data.dropna(subset=["HTHG", "HTAG", "HS", "AS", "HST", "AST", "HR", "AR", "FTR"])

    rows = []
    for name, columns in FEATURE_SETS.items():
        X, y = data[columns], data["FTR"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X_train, y_train)
        lr = LogisticRegression(max_iter=2000).fit(X_train, y_train)
        rows.append({
            "inputs": name,
            "random_forest_accuracy": float(rf.score(X_test, y_test)),
            "logistic_regression_accuracy": float(lr.score(X_test, y_test)),
        })
        logger.info("%s: RF %.3f, LR %.3f", name, rows[-1]["random_forest_accuracy"],
                    rows[-1]["logistic_regression_accuracy"])
    result = {
        "source": "https://github.com/stogaja/Football-Match-Outcome-Predictor",
        "seasons": "2009/10 to 2018/19",
        "matches": int(len(data)),
        "split": "random 80/20, as in the original",
        "results": rows,
        "draw_share": float(np.mean(data["FTR"] == "D")),
    }
    path = resolve_path(config["output"]["reports_dir"]) / "in_play_reference.json"
    path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", path)
    return result


if __name__ == "__main__":
    setup_logging()
    run()
