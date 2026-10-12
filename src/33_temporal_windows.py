"""Temporally explicit MCPs and pairwise overlap.

Reviewer point: polygons pooled over 2014-2026 can contain territory shifts,
dispersal and population turnover, so the intersection of two long-term polygons is
not evidence that the two animals used the space at the same time.

This script rebuilds 95% MCPs *within* defined time windows and recomputes every
pairwise intersection inside each window, so overlap is only ever measured between
locations recorded during the same period.

All dyads are emitted, including those with zero overlap. Filtering to non-zero
overlap changes the question from "do pair types differ in their tendency to
overlap" to "how much do they overlap once they already do", and the absence of
overlap is itself the signal that speaks to territorial exclusivity.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import pandas as pd
from scipy import stats

from _mcp import geojson_feature, mcp, polygon_from_ring
from _stats import bootstrap_median_diff, cliffs_delta
from _age import annotate
from _inclusion import MIN_UNIQUE_DATES, MIN_UNIQUE_POINTS


IN_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.dated_events.csv"
OUT_TABLES = Path("outputs") / "tables"
OUT_FIGS = Path("outputs") / "figures"

MCP_PERCENT = 95.0


# Calendar years are too thinly sampled to stand alone (2024: 15 individuals at the
# >=5-point mark but only 8 at the inclusion threshold, 2025: 15, 2026: 6). A
# 24-month contemporary window is the shortest span that retains a workable sample
# while keeping every polygon contemporaneous.
WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "contemporary_24mo": ("2024-06-01", "2026-05-25"),
    "recent_12mo": ("2025-06-01", "2026-05-25"),
    "pooled": (None, None),
}
PRIMARY_WINDOW = "contemporary_24mo"


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna() if str(v) != "Unknown"]
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "Unknown"


def load_events() -> pd.DataFrame:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {IN_PATH.resolve()}")

    df = pd.read_csv(IN_PATH, low_memory=False)
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df["date"] = pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True)
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", "date"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(_resolve_sex)
    return df


def build_window_individuals(events: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    sub = events
    if start is not None:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end is not None:
        sub = sub[sub["date"] <= pd.Timestamp(end)]

    rows: list[dict[str, object]] = []
    for (name, sex), group in sub.groupby(["leopard_name", "sex"], dropna=False):
        pts = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
        hull = mcp(pts, percent=MCP_PERCENT)
        rows.append(
            {
                "leopard_name": name,
                "sex": sex,
                "n_events": int(len(group)),
                "n_unique_points": int(hull["n_unique_points"]),
                "n_unique_dates": int(group["date"].nunique()),
                "first_seen": group["date"].min(),
                "last_seen": group["date"].max(),
                "n_points_dropped": int(hull["n_points_dropped"]),
                "mcp95_km2": hull["area_km2"],
                "mcp100_km2": mcp(pts, percent=100.0)["area_km2"],
                "ring_latlon": hull["ring_latlon"],
            }
        )

    out = annotate(pd.DataFrame(rows))
    out["include"] = (
        (out["n_unique_points"] >= MIN_UNIQUE_POINTS)
        & (out["n_unique_dates"] >= MIN_UNIQUE_DATES)
        & out["mcp95_km2"].notna()
        & out["sex"].isin(["Female", "Male"])
        # Cubs and subadults are not territory holders.
        & out["is_territorial_age"]
    )
    return out


def build_window_pairs(individuals: pd.DataFrame) -> pd.DataFrame:
    kept = individuals[individuals["include"]].copy()
    polys = {row.leopard_name: polygon_from_ring(row.ring_latlon) for row in kept.itertuples()}
    meta = kept.set_index("leopard_name")

    rows: list[dict[str, object]] = []
    for a, b in combinations(sorted(polys), 2):
        poly_a, poly_b = polys[a], polys[b]
        inter = poly_a.intersection(poly_b)
        inter_km2 = (inter.area / 1_000_000.0) if not inter.is_empty else 0.0
        union = poly_a.union(poly_b)
        union_km2 = (union.area / 1_000_000.0) if not union.is_empty else float("nan")
        area_a = poly_a.area / 1_000_000.0
        area_b = poly_b.area / 1_000_000.0

        # Shared detection window: how long both animals were on record at once.
        overlap_start = max(meta.at[a, "first_seen"], meta.at[b, "first_seen"])
        overlap_end = min(meta.at[a, "last_seen"], meta.at[b, "last_seen"])
        codetect_days = int((overlap_end - overlap_start).days)

        rows.append(
            {
                "leopard_a": a,
                "sex_a": meta.at[a, "sex"],
                "area_a_km2": area_a,
                "leopard_b": b,
                "sex_b": meta.at[b, "sex"],
                "area_b_km2": area_b,
                "pair_sex": "-".join(sorted([meta.at[a, "sex"], meta.at[b, "sex"]])),
                "overlaps": int(inter_km2 > 0),
                "overlap_km2": inter_km2,
                "union_km2": union_km2,
                "jaccard_overlap": (inter_km2 / union_km2) if union_km2 else float("nan"),
                # Size-normalised response: symmetric, and not inflated by the fact
                # that larger polygons simply have more opportunity to intersect.
                "prop_overlap": 2.0 * inter_km2 / (area_a + area_b) if (area_a + area_b) else float("nan"),
                "pct_of_a_overlapped": (inter_km2 / area_a * 100.0) if area_a else float("nan"),
                "pct_of_b_overlapped": (inter_km2 / area_b * 100.0) if area_b else float("nan"),
                "mean_polygon_area_km2": (area_a + area_b) / 2.0,
                "temporal_cooccurrence_days": codetect_days,
            }
        )

    return pd.DataFrame(rows).sort_values(["overlap_km2"], ascending=False).reset_index(drop=True)


def build_window_sex_stats(individuals: pd.DataFrame, window: str) -> dict[str, object]:
    """Male vs female 95% MCP area within one window."""
    kept = individuals[individuals["include"]]
    female = kept.loc[kept["sex"] == "Female", "mcp95_km2"].to_numpy(float)
    male = kept.loc[kept["sex"] == "Male", "mcp95_km2"].to_numpy(float)
    if len(female) == 0 or len(male) == 0:
        return {"window": window}

    test = stats.mannwhitneyu(male, female, alternative="two-sided")
    med_diff, ci_lo, ci_hi = bootstrap_median_diff(male, female)
    return {
        "window": window,
        "n_female": int(len(female)),
        "n_male": int(len(male)),
        "median_female_km2": float(pd.Series(female).median()),
        "median_male_km2": float(pd.Series(male).median()),
        "mean_female_km2": float(pd.Series(female).mean()),
        "mean_male_km2": float(pd.Series(male).mean()),
        "mann_whitney_u": float(test.statistic),
        "p_value": float(test.pvalue),
        "cliffs_delta_male_minus_female": cliffs_delta(male, female),
        "median_diff_male_minus_female_km2": med_diff,
        "median_diff_ci95_low_km2": ci_lo,
        "median_diff_ci95_high_km2": ci_hi,
    }


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    events = load_events()
    summaries: list[dict[str, object]] = []
    sex_stats: list[dict[str, object]] = []

    for label, (start, end) in WINDOWS.items():
        individuals = build_window_individuals(events, start, end)
        pairs = build_window_pairs(individuals)
        kept = individuals[individuals["include"]]
        sex_stats.append(build_window_sex_stats(individuals, label))

        features = [
            geojson_feature(
                row.ring_latlon,
                {
                    "leopard_name": row.leopard_name,
                    "sex": row.sex,
                    "window": label,
                    "mcp_percent": MCP_PERCENT,
                    "n_unique_points": int(row.n_unique_points),
                    "n_points_dropped": int(row.n_points_dropped),
                    "home_range_km2_mcp": float(row.mcp95_km2),
                },
            )
            for row in kept.itertuples()
        ]

        ind_out = individuals.drop(columns=["ring_latlon"]).copy()
        for column in ("first_seen", "last_seen"):
            ind_out[column] = pd.to_datetime(ind_out[column]).dt.date.astype("string")

        window_points = events.copy()
        if start is not None:
            window_points = window_points[window_points["date"] >= pd.Timestamp(start)]
        if end is not None:
            window_points = window_points[window_points["date"] <= pd.Timestamp(end)]
        window_points = window_points[window_points["leopard_name"].isin(set(kept["leopard_name"]))]
        points_path = OUT_TABLES / f"merged_leopard_images_metadata.window_subset__{label}.csv"
        window_points.drop(columns=["date"]).to_csv(points_path, index=False)
        print(f"Saved -> {points_path}")

        ind_path = OUT_TABLES / f"merged_image_window_individuals__{label}.csv"
        pair_path = OUT_TABLES / f"merged_image_overlap_pairs__{label}.csv"
        geo_path = OUT_FIGS / f"merged_image_home_ranges_mcp95__{label}.geojson"

        ind_out.to_csv(ind_path, index=False)
        pairs.to_csv(pair_path, index=False)
        with open(geo_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

        summaries.append(
            {
                "window": label,
                "is_primary": label == PRIMARY_WINDOW,
                "start": start or str(events["date"].min().date()),
                "end": end or str(events["date"].max().date()),
                "n_individuals": int(len(kept)),
                "n_female": int((kept["sex"] == "Female").sum()),
                "n_male": int((kept["sex"] == "Male").sum()),
                "n_dyads": int(len(pairs)),
                "n_dyads_overlapping": int(pairs["overlaps"].sum()) if len(pairs) else 0,
                "min_cooccurrence_days": int(pairs["temporal_cooccurrence_days"].min()) if len(pairs) else 0,
                "median_cooccurrence_days": float(pairs["temporal_cooccurrence_days"].median()) if len(pairs) else 0.0,
            }
        )
        print(f"Saved -> {ind_path}")
        print(f"Saved -> {pair_path}")
        print(f"Saved -> {geo_path}")

    summary_path = OUT_TABLES / "merged_image_window_summary.csv"
    sex_stats_path = OUT_TABLES / "merged_image_window_sex_stats.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(sex_stats).to_csv(sex_stats_path, index=False)
    print(f"Saved -> {summary_path}")
    print(f"Saved -> {sex_stats_path}")

    pd.set_option("display.width", 240)
    print(pd.DataFrame(summaries).to_string(index=False))
    print()
    print(pd.DataFrame(sex_stats).to_string(index=False))


if __name__ == "__main__":
    main()
