from __future__ import annotations

import matplotlib as mpl

# Greyscale palette for the paper figures: journal print and photocopies drop colour,
# so sex/pair-type are encoded redundantly through shade and marker shape rather than
# hue alone. Kept here so every figure script draws from one definition.
SEX_ORDER = ["Female", "Male"]
SEX_GREY = {"Female": "#404040", "Male": "#a6a6a6"}
SEX_MARKER = {"Female": "o", "Male": "s"}

# Sex including the "Unknown" class (unsexed cubs) - used for descriptive figures
# over the full dataset, as opposed to the sex-restricted spatial analysis.
SEX3_ORDER = ["Female", "Male", "Unknown"]
SEX3_GREY = {"Female": "#262626", "Male": "#a6a6a6", "Unknown": "#5c5c5c"}

PAIR_ORDER = ["Female-Female", "Female-Male", "Male-Male"]
PAIR_GREY = {"Female-Female": "#262626", "Female-Male": "#808080", "Male-Male": "#adadad"}
PAIR_MARKER = {"Female-Female": "o", "Female-Male": "s", "Male-Male": "^"}


def set_pub_style() -> None:
    """
    Set Matplotlib rcParams for publication-ready figures.

    Notes:
      - Uses vector-friendly fonts and sizes
      - Saves clean axes and consistent linewidths
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,  # TrueType fonts in PDF
            "ps.fonttype": 42,
        }
    )