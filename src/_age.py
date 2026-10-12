"""Age-class lookup, used to restrict the spatial analyses to territory holders.

Age class lives in `leopard_lineage_master.xlsx` (column "Estimated age"), not in the
image metadata, so every script that applies the inclusion criteria has to join it in
from here.

Caveat worth carrying into the Methods: the lineage label is a single static
attribute per individual, not a time-varying state. An animal recorded as a subadult
over a two-year detection history will have matured within that period, so excluding
it removes its adult locations too. This is conservative for a question about
established territories, but it is an approximation, not an age-at-detection filter.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from _paths import LINEAGE_XLSX as _LINEAGE_XLSX

LINEAGE_XLSX = _LINEAGE_XLSX

# Cubs and subadults have not established territories, so pairwise overlap between
# them and resident adults does not speak to territorial exclusivity. Cubs never meet
# the location threshold in the current archive, but they are named here so the rule
# holds if that changes.
NON_TERRITORIAL_AGE_CLASSES = frozenset({"cub", "subadult"})


def _key(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _normalise(label: object) -> str:
    text = re.sub(r"\s+", " ", str(label).strip().lower())
    return "" if text in {"", "nan", "none", "?"} else text


def age_class_lookup() -> dict[str, str]:
    """Map normalised leopard name -> lowercase age class ('' when unknown)."""
    if not LINEAGE_XLSX.exists():
        raise FileNotFoundError(f"Cannot find: {LINEAGE_XLSX}")

    lineage = pd.read_excel(LINEAGE_XLSX, sheet_name="Lineage")
    names = lineage.get("leopard_name", pd.Series(dtype="object"))
    ages = lineage.get("Estimated age", pd.Series(dtype="object"))
    return {_key(n): _normalise(a) for n, a in zip(names, ages) if _key(n)}


def is_territorial(name: object, lookup: dict[str, str] | None = None) -> bool:
    """True when the individual is old enough to hold a territory.

    Unknown age classes are treated as territorial: the archive contains adults whose
    age was never recorded, and dropping them would silently shrink the sample on a
    missing-data artefact rather than on biology.
    """
    lookup = age_class_lookup() if lookup is None else lookup
    return lookup.get(_key(name), "") not in NON_TERRITORIAL_AGE_CLASSES


def annotate(frame: pd.DataFrame, name_column: str = "leopard_name") -> pd.DataFrame:
    """Add `age_class` and `is_territorial_age` columns to a frame of individuals."""
    lookup = age_class_lookup()
    out = frame.copy()
    out["age_class"] = out[name_column].map(lambda n: lookup.get(_key(n), ""))
    out["is_territorial_age"] = ~out["age_class"].isin(NON_TERRITORIAL_AGE_CLASSES)
    return out
