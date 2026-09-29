"""Live predictions and tracking.

    python -m src.live predict          # refresh data, fill results, predict and save upcoming fixtures
    python -m src.live predict --dry-run
    python -m src.live update           # refresh data and fill in finished results
    python -m src.live report           # live accuracy so far
    python -m src.live match "Liverpool" "Man City"   # one-off prediction, not saved

``predict`` also runs ``update`` first, so a single scheduled command keeps
the history current.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.update import LiveData, load_live_data
from src.models.predict import TrainedModels, format_prediction, load_models, predict_fixtures
from src.tracking.history import (
    load_history,
    prediction_id,
    record_predictions,
    save_history,
    update_results,
)
from src.tracking.performance import accuracy_over_time, format_summary, live_summary
from src.utils.config import load_config, resolve_path, setup_logging
from src.utils.team_names import normalize_team_name

logger = logging.getLogger("live")


def select_fixtures(fixtures: pd.DataFrame, now: datetime, horizon_days: int) -> pd.DataFrame:
    """Fixtures that can be predicted fairly right now.

    A fixture is eligible when it kicks off after ``now`` and within the
    horizon, and it is the *next* upcoming match for both teams. Predicting a
    team's match after next would miss the result of the match in between.
    """
    upcoming = fixtures[fixtures["kickoff"] > now].sort_values("kickoff")
    team_next = {}
    for row in upcoming.itertuples():
        team_next.setdefault(row.home_team, row.kickoff)
        team_next.setdefault(row.away_team, row.kickoff)
    within = upcoming["kickoff"] <= now + timedelta(days=horizon_days)
    is_next = [team_next[h] == k and team_next[a] == k
               for h, a, k in zip(upcoming["home_team"], upcoming["away_team"], upcoming["kickoff"])]
    return upcoming[within & pd.Series(is_next, index=upcoming.index)]


def stale_fixtures(fixtures: pd.DataFrame, now: datetime, stale_hours: int) -> pd.DataFrame:
    """Past fixtures still without a result: postponed, or the results feed is behind."""
    return fixtures[fixtures["kickoff"] < now - timedelta(hours=stale_hours)]


def data_notes_for(selected: pd.DataFrame, data: LiveData, stale: pd.DataFrame) -> dict[str, str]:
    """Flag predictions whose inputs are known to be incomplete."""
    stale_teams = set(stale["home_team"]) | set(stale["away_team"])
    notes = {}
    for row in selected.itertuples():
        flags = []
        if data.current.shot_data_share < 1.0:
            flags.append("no_shot_data" if data.current.shot_data_share == 0 else "partial_shot_data")
        if {row.home_team, row.away_team} & stale_teams:
            flags.append("missing_earlier_result")
        notes[prediction_id(row.season, row.home_team, row.away_team)] = ";".join(flags)
    return notes


def _refresh(config: dict, now: datetime) -> tuple[LiveData, pd.DataFrame, int]:
    data = load_live_data(config, now.date())
    path = resolve_path(config["live"]["history_path"])
    history, n_updated = update_results(load_history(path), data.matches, now)
    if n_updated:
        save_history(history, path)
        logger.info("Recorded %d new results in %s", n_updated, path)
    return data, history, n_updated


def cmd_update(config: dict, now: datetime) -> None:
    _, history, n_updated = _refresh(config, now)
    print(f"Results recorded this run: {n_updated}. Awaiting results: {int(history['actual_result'].isna().sum())}.")


def cmd_predict(config: dict, now: datetime, dry_run: bool, horizon_days: int | None) -> None:
    live = config["live"]
    data, history, _ = _refresh(config, now)
    models = load_models(resolve_path(config["output"]["models_dir"]))

    stale = stale_fixtures(data.current.fixtures, now, live["stale_result_hours"])
    if len(stale):
        logger.warning("%d past fixtures have no result yet (postponed, or the feed is behind): %s", len(stale),
                       ", ".join(f"{r.home_team} v {r.away_team}" for r in stale.itertuples()))

    selected = select_fixtures(data.current.fixtures, now, horizon_days or live["horizon_days"])
    if selected.empty:
        print("No fixtures to predict in the horizon.")
        return
    predictions = predict_fixtures(models, data.matches, selected)
    for _, row in predictions.iterrows():
        print(format_prediction(row), end="\n\n")

    if data.current.shot_data_share < 1.0:
        print(f"Note: shot statistics are available for {data.current.shot_data_share:.0%} of this season's "
              f"matches ({data.current.results_source}); shot features are missing for those matches.")
    if dry_run:
        print("Dry run: nothing saved.")
        return
    history, added, skipped = record_predictions(
        history, predictions, now, models.metadata["trained_at"], data.current.results_source,
        data_notes_for(selected, data, stale),
    )
    save_history(history, resolve_path(live["history_path"]))
    print(f"Saved {added} new predictions ({skipped} already recorded, left unchanged).")


def cmd_report(config: dict) -> None:
    history = load_history(resolve_path(config["live"]["history_path"]))
    print(format_summary(live_summary(history)))
    over_time = accuracy_over_time(history)
    if not over_time.empty:
        print("\nAccuracy over time (weekly):")
        print(over_time.round(3).to_string(index=False))


def cmd_match(config: dict, now: datetime, home: str, away: str, date: str | None) -> None:
    """One-off prediction for any pairing, not saved to the history."""
    data = load_live_data(config, now.date())
    models: TrainedModels = load_models(resolve_path(config["output"]["models_dir"]))
    fixture_date = pd.Timestamp(date) if date else pd.Timestamp(now.date()) + pd.Timedelta(days=1)
    fixture = pd.DataFrame([{
        "season": data.current.season, "date": fixture_date,
        "home_team": normalize_team_name(home), "away_team": normalize_team_name(away),
    }])
    history = data.matches[data.matches["date"] < fixture_date]
    prediction = predict_fixtures(models, history, fixture)
    print(format_prediction(prediction.iloc[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Premier League predictions and tracking.")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p_predict = sub.add_parser("predict", help="Predict and save upcoming fixtures")
    p_predict.add_argument("--dry-run", action="store_true", help="Print predictions without saving")
    p_predict.add_argument("--days", type=int, default=None, help="Horizon in days (default from config)")
    sub.add_parser("update", help="Fill in results for finished matches")
    sub.add_parser("report", help="Show live performance")
    p_match = sub.add_parser("match", help="One-off prediction (not saved)")
    p_match.add_argument("home")
    p_match.add_argument("away")
    p_match.add_argument("--date", default=None, help="Fixture date YYYY-MM-DD (default: tomorrow)")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    now = datetime.now(timezone.utc)
    if args.command == "predict":
        cmd_predict(config, now, args.dry_run, args.days)
    elif args.command == "update":
        cmd_update(config, now)
    elif args.command == "report":
        cmd_report(config)
    elif args.command == "match":
        cmd_match(config, now, args.home, args.away, args.date)


if __name__ == "__main__":
    main()
