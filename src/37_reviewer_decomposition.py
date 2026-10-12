"""Decompose the difference between the submitted analysis and the revised one.

The submitted analysis reported a significant difference in pairwise overlap between
pair types. The revision changes four things at once, so this script varies them one
at a time to show which change is responsible for any shift in the result. It exists
to make the response to the reviewer auditable rather than asking them to take the
new numbers on trust.

Steps, cumulative, on the pooled 2014-2026 data the paper used:

  A  as submitted        100% MCP, non-zero dyads only, km^2, Kruskal-Wallis / Mann-Whitney
  B  + 95% MCP           the estimator the manuscript claims
  C  + prop_overlap      size-normalised response (reviewer's polygon-size point)
  D  + zeros included    every dyad, not only those that already overlap
  E  + permutation       node-label permutation instead of rank tests
  F  + threshold >=8     the relaxed inclusion criterion used in the revision
  G  + adults only       cubs and subadults excluded (not territory holders)
  H  + dated only        events carrying the archive export date excluded (src/_dates.py)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _age import age_class_lookup, is_territorial
from _inclusion import MIN_UNIQUE_DATES, MIN_UNIQUE_POINTS, SUBMITTED_MIN_UNIQUE_POINTS
from _mcp import mcp, polygon_from_ring

IN_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.normalized_events.csv"
DATED_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.dated_events.csv"
OUT_PATH = Path("outputs") / "tables" / "reviewer_decomposition.csv"

PAIR_TYPES = ["Female-Female", "Female-Male", "Male-Male"]
FOCAL = ("Female-Female", "Female-Male")  # the manuscript's headline contrast
N_PERMUTATIONS = 10_000
SEED = 20260821


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna() if str(v) != "Unknown"]
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "Unknown"


def build_pairs(
    percent: float,
    min_points: int = SUBMITTED_MIN_UNIQUE_POINTS,
    exclude_non_territorial: bool = False,
    path: Path = IN_PATH,
) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["sex"] = df.get("sex").fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df["gps_lat"], errors="coerce")
    df["gps_lon"] = pd.to_numeric(df["gps_lon"], errors="coerce")
    df["date"] = pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True)
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", "date"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(_resolve_sex)

    age_lookup = age_class_lookup()
    polys, sexes = {}, {}
    for (name, sex), group in df.groupby(["leopard_name", "sex"]):
        if sex not in ("Female", "Male"):
            continue
        n_points = len(set(zip(group["gps_lat"], group["gps_lon"])))
        if n_points < min_points or group["date"].nunique() < MIN_UNIQUE_DATES:
            continue
        if exclude_non_territorial and not is_territorial(name, age_lookup):
            continue
        hull = mcp(list(zip(group["gps_lat"], group["gps_lon"])), percent=percent)
        if hull["ring_latlon"]:
            polys[name] = polygon_from_ring(hull["ring_latlon"])
            sexes[name] = sex

    rows = []
    for a, b in combinations(sorted(polys), 2):
        inter = polys[a].intersection(polys[b])
        inter_km2 = (inter.area / 1e6) if not inter.is_empty else 0.0
        area_a, area_b = polys[a].area / 1e6, polys[b].area / 1e6
        rows.append(
            {
                "leopard_a": a,
                "sex_a": sexes[a],
                "leopard_b": b,
                "sex_b": sexes[b],
                "pair_sex": "-".join(sorted([sexes[a], sexes[b]])),
                "overlap_km2": inter_km2,
                "prop_overlap": 2.0 * inter_km2 / (area_a + area_b),
            }
        )
    return pd.DataFrame(rows)


def _perm_p(pairs: pd.DataFrame, response: str) -> float:
    """Two-sided permutation p for the focal contrast, relabelling individuals.

    The test statistic is the rank-biserial correlation, i.e. the same rank ordering
    Mann-Whitney uses in steps A-D, so only the null distribution changes between the
    rank-test and permutation rows. A median difference cannot be used here: once
    zero-overlap dyads are included more than half of dyads are exactly zero, so both
    group medians are 0 and the statistic degenerates to 0 regardless of the data.
    """
    values = pairs[response].to_numpy(float)
    types = pairs["pair_sex"].reset_index(drop=True)

    def diff(t: pd.Series) -> float:
        a = values[(t == FOCAL[0]).to_numpy()]
        b = values[(t == FOCAL[1]).to_numpy()]
        if len(a) == 0 or len(b) == 0:
            return np.nan
        u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
        return float(2.0 * u / (len(a) * len(b)) - 1.0)

    observed = diff(types)
    ind = (
        pd.concat(
            [
                pairs[["leopard_a", "sex_a"]].rename(columns={"leopard_a": "n", "sex_a": "s"}),
                pairs[["leopard_b", "sex_b"]].rename(columns={"leopard_b": "n", "sex_b": "s"}),
            ]
        )
        .drop_duplicates()
        .sort_values("n")
    )
    names, sx = ind["n"].tolist(), ind["s"].to_numpy()
    a_arr, b_arr = pairs["leopard_a"].to_numpy(), pairs["leopard_b"].to_numpy()

    rng = np.random.default_rng(SEED)
    ge = 0
    for _ in range(N_PERMUTATIONS):
        lookup = dict(zip(names, rng.permutation(sx)))
        perm = pd.Series(["-".join(sorted([lookup[a], lookup[b]])) for a, b in zip(a_arr, b_arr)])
        d = diff(perm)
        if not np.isnan(d) and abs(d) >= abs(observed) - 1e-12:
            ge += 1
    return (ge + 1) / (N_PERMUTATIONS + 1)


def _mw_p(pairs: pd.DataFrame, response: str) -> tuple[float, float, float, int, int]:
    a = pairs.loc[pairs["pair_sex"] == FOCAL[0], response].to_numpy(float)
    b = pairs.loc[pairs["pair_sex"] == FOCAL[1], response].to_numpy(float)
    test = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(test.pvalue), float(np.median(a)), float(np.median(b)), len(a), len(b)


def main() -> None:
    pairs_100 = build_pairs(100.0)
    pairs_95 = build_pairs(95.0)
    pairs_95_relaxed = build_pairs(95.0, min_points=MIN_UNIQUE_POINTS)
    pairs_95_adults = build_pairs(95.0, min_points=MIN_UNIQUE_POINTS, exclude_non_territorial=True)
    pairs_95_dated = build_pairs(95.0, min_points=MIN_UNIQUE_POINTS, exclude_non_territorial=True, path=DATED_PATH)

    rows: list[dict[str, object]] = []

    def record(step: str, change: str, pairs: pd.DataFrame, response: str, inference: str,
               estimator: str, nonzero_only: bool, min_points: int) -> None:
        p_mw, med_a, med_b, n_a, n_b = _mw_p(pairs, response)
        p = _perm_p(pairs, response) if inference == "permutation" else p_mw
        rows.append(
            {
                "step": step,
                "cumulative_change": change,
                "estimator": estimator,
                "response": response,
                "dyads": "non-zero only" if nonzero_only else "all dyads",
                "age_classes": "adults only" if pairs is pairs_95_adults or pairs is pairs_95_dated else "all ages",
                "min_unique_points": min_points,
                "inference": inference,
                "n_FF": n_a,
                "n_FM": n_b,
                "median_FF": med_a,
                "median_FM": med_b,
                "p_focal_FF_vs_FM": p,
            }
        )

    nz100 = pairs_100[pairs_100["overlap_km2"] > 0]
    nz95 = pairs_95[pairs_95["overlap_km2"] > 0]

    record("A", "as submitted", nz100, "overlap_km2", "mann_whitney", "100% MCP", True, SUBMITTED_MIN_UNIQUE_POINTS)
    record("B", "+ 95% MCP", nz95, "overlap_km2", "mann_whitney", "95% MCP", True, SUBMITTED_MIN_UNIQUE_POINTS)
    record("C", "+ size-normalised response", nz95, "prop_overlap", "mann_whitney", "95% MCP", True, SUBMITTED_MIN_UNIQUE_POINTS)
    record("D", "+ zero-overlap dyads included", pairs_95, "prop_overlap", "mann_whitney", "95% MCP", False, SUBMITTED_MIN_UNIQUE_POINTS)
    record("E", "+ node-label permutation", pairs_95, "prop_overlap", "permutation", "95% MCP", False, SUBMITTED_MIN_UNIQUE_POINTS)
    record("F", f"+ inclusion threshold >={MIN_UNIQUE_POINTS} points", pairs_95_relaxed, "prop_overlap",
           "permutation", "95% MCP", False, MIN_UNIQUE_POINTS)
    record("G", "+ cubs and subadults excluded", pairs_95_adults, "prop_overlap",
           "permutation", "95% MCP", False, MIN_UNIQUE_POINTS)
    record("H", "+ archive-dated events excluded", pairs_95_dated, "prop_overlap",
           "permutation", "95% MCP", False, MIN_UNIQUE_POINTS)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    pd.set_option("display.width", 240)
    print(out.to_string(index=False))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
