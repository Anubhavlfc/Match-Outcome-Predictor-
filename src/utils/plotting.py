"""Shared chart style so every figure in the report looks like one system.

Colors come from a colorblind-checked categorical palette, assigned in a
fixed order and tied to the entity (Random Forest is always blue, XGBoost
always orange, the bookmaker always aqua).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#9a9892"        # baselines and reference lines

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # slots 1-3: blue, orange, aqua
ENTITY_COLORS = {
    "Random Forest": SERIES[0],
    "XGBoost": SERIES[1],
    "Bookmaker": SERIES[2],
    "Always Home Win": NEUTRAL,
    "Class frequencies": NEUTRAL,
}
OUTCOME_COLORS = {"Home Win": SERIES[0], "Draw": SERIES[1], "Away Win": SERIES[2]}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10,
        "text.color": TEXT,
    })
