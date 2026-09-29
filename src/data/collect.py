"""Download raw Premier League season CSVs.

Each season is one Football-Data.co.uk "E0" file with one row per match.
Files are cached in data/raw/ so the pipeline is reproducible offline and
the source is not hit on every run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from src.utils.config import season_code, season_label

logger = logging.getLogger(__name__)


class DataCollectionError(RuntimeError):
    """Raised when a season could not be downloaded from any source."""


def raw_path(raw_dir: Path, start_year: int, league_code: str = "E0") -> Path:
    return raw_dir / f"{league_code}_{season_code(start_year)}.csv"


def download_season(
    start_year: int,
    sources: list[dict[str, str]],
    raw_dir: Path,
    force: bool = False,
    timeout: int = 30,
    league_code: str = "E0",
) -> Path:
    """Download one season, trying each configured source in order.

    Returns the cached path. An existing file is reused unless ``force``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_path(raw_dir, start_year, league_code)
    if target.exists() and not force:
        logger.debug("Using cached %s", target.name)
        return target

    errors: list[str] = []
    for source in sources:
        url = source["url_template"].format(code=season_code(start_year))
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            errors.append(f"{source['name']}: {exc}")
            logger.warning("Could not fetch %s from %s (%s)", season_label(start_year), source["name"], exc)
            continue
        # Sanity check: an E0 file always has these header fields.
        head = response.content[:500].decode("latin-1")
        if "HomeTeam" not in head or "FTHG" not in head:
            errors.append(f"{source['name']}: response is not an E0 CSV")
            continue
        target.write_bytes(response.content)
        logger.info("Downloaded %s from %s", season_label(start_year), source["name"])
        return target

    raise DataCollectionError(
        f"Could not download season {season_label(start_year)}:\n  " + "\n  ".join(errors)
    )


def collect_seasons(
    first_season: int,
    last_season: int,
    sources: list[dict[str, str]],
    raw_dir: Path,
    force: bool = False,
    league_code: str = "E0",
) -> list[Path]:
    """Make sure every season in [first_season, last_season] is cached locally."""
    return [
        download_season(year, sources, raw_dir, force=force, league_code=league_code)
        for year in range(first_season, last_season + 1)
    ]
