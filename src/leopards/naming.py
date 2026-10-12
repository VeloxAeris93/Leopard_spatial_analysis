"""Leopard name and sex helpers shared across scripts.

``ensure_primary_leopard`` / ``infer_sex`` / ``norm_name`` were duplicated (with
drift) across ~10 scripts. The looser matchers used only by the image<->sighting
matcher (16a) are kept distinct as ``normalize_name`` / ``infer_sex_loose`` to
preserve that step's behavior.
"""

from __future__ import annotations

import re

import pandas as pd


# Spelling fixes applied when matching image folder names to sighting names (16a).
NAME_REPLACEMENTS = {
    "maun rd": "maun road",
    "mojelemane": "mojeremane",
    "moretwla": "moretlwa",
}


# Names later confirmed to be the same leopard: former name -> name used going forward.
# Photo folders and location records filed under the former name belong to the new one.
MERGED_IDENTITIES = {
    "Morobe": "Tswaletswale",  # confirmed the same female, Oct 2026
}


def ensure_primary_leopard(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a ``primary_leopard`` column, falling back to ``Leopard``.

    The cleaned CSV sometimes only has ``Leopard``; this standardizes the name
    without re-running the cleaning step.

    Raises:
        KeyError: if neither column exists.
    """
    if "primary_leopard" in df.columns:
        return df
    if "Leopard" in df.columns:
        df = df.copy()
        df["primary_leopard"] = df["Leopard"]
        return df
    raise KeyError("Neither 'primary_leopard' nor 'Leopard' columns exist in sightings_clean.csv")


def infer_sex(name: str) -> str:
    """Infer sex from names like 'Mapula Female' / 'Mogotho Male'.

    Returns 'Female', 'Male', or 'Unknown'. Uses word-boundary matching.
    """
    n = str(name).strip().lower()
    if re.search(r"\bfemale\b", n):
        return "Female"
    if re.search(r"\bmale\b", n):
        return "Male"
    return "Unknown"


def norm_name(value: object) -> str:
    """Collapse a name to lowercase alphanumerics (used for fuzzy matching, 26/30)."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalize_name(value: object) -> str | None:
    """16a-specific normalization: lowercase, strip role words, apply spelling fixes."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("_", " ")
    text = re.sub(r"\b(male|female)\b", "", text)
    for source, target in NAME_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def infer_sex_loose(value: object) -> str | None:
    """16a-specific sex inference: substring match, returns None when unknown."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "female" in text:
        return "Female"
    if "male" in text:
        return "Male"
    return None
