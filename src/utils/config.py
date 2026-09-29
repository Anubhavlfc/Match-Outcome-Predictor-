"""Configuration loading and small shared helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Relative paths inside it resolve against the repo root."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str | Path) -> Path:
    """Turn a config path into an absolute path under the project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def season_code(start_year: int) -> str:
    """2015 -> '1516' (the code Football-Data.co.uk uses in its URLs)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2015 -> '2015/16' (human-readable season name)."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
