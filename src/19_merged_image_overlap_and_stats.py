from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

from _mcp import polygon_from_ring
from _stats import bootstrap_median_diff, cliffs_delta


SUMMARY_PATH = Path("outputs") / "tables" / "merged_image_home_ranges_mcp_summary.csv"
GEOJSON_PATH = Path("outputs") / "figures" / "merged_image_home_ranges_mcp95.geojson"

OUT_TABLES = Path("outputs") / "tables"
OUT_OVERLAP_MATRIX = OUT_TABLES / "merged_image_home_range_overlap_km2.csv"
OUT_OVERLAP_PAIRS = OUT_TABLES / "merged_image_home_range_overlap_pairs.csv"
OUT_OVERLAP_BY_PAIR = OUT_TABLES / "merged_image_home_range_overlap_by_pair_type.csv"
OUT_SEX_STATS = OUT_TABLES / "merged_image_home_range_sex_stats.csv"
OUT_SPARSE_MALES = OUT_TABLES / "merged_image_home_range_sparse_male_drivers.csv"


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {SUMMARY_PATH.resolve()}")
    df = pd.read_csv(SUMMARY_PATH)
    return df


def build_sparse_male_drivers(summary: pd.DataFrame) -> pd.DataFrame:
    out = (
        summary.dropna(subset=["home_range_km2_mcp"])
        .query("sex == 'Male' and 2 <= n_unique_dates <= 4")
        .sort_values("home_range_km2_mcp", ascending=False)
        .reset_index(drop=True)
    )
    return out[
        ["leopard_name", "n_images", "n_unique_dates", "n_unique_points", "home_range_km2_mcp"]
    ]


def build_sex_stats(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.dropna(subset=["home_range_km2_mcp"]).copy()
    base = base[base["sex"].isin(["Female", "Male"])].copy()

    subsets = {
        "all_polygons": base,
        "unique_dates_ge_3": base[base["n_unique_dates"] >= 3],
        "unique_dates_ge_5": base[base["n_unique_dates"] >= 5],
        "unique_dates_ge_10": base[base["n_unique_dates"] >= 10],
        "unique_points_ge_10": base[base["n_unique_points"] >= 10],
        "unique_points_ge_20": base[base["n_unique_points"] >= 20],
    }

    rows: list[dict[str, object]] = []
    for label, sub in subsets.items():
        female = sub.loc[sub["sex"] == "Female", "home_range_km2_mcp"].to_numpy(float)
        male = sub.loc[sub["sex"] == "Male", "home_range_km2_mcp"].to_numpy(float)
        if len(female) == 0 or len(male) == 0:
            continue

        u = stats.mannwhitneyu(male, female, alternative="two-sided")
        delta = cliffs_delta(male, female)
        med_diff, ci_lo, ci_hi = bootstrap_median_diff(male, female)
        rows.append(
            {
                "subset": label,
                "n_male": int(len(male)),
                "n_female": int(len(female)),
                "median_male_km2": float(np.median(male)),
                "median_female_km2": float(np.median(female)),
                "mean_male_km2": float(np.mean(male)),
                "mean_female_km2": float(np.mean(female)),
                "mann_whitney_u": float(u.statistic),
                "p_value": float(u.pvalue),
                "cliffs_delta_male_minus_female": float(delta),
                "median_diff_male_minus_female_km2": float(med_diff),
                "median_diff_ci95_low_km2": float(ci_lo),
                "median_diff_ci95_high_km2": float(ci_hi),
            }
        )
    return pd.DataFrame(rows)


def build_overlap_outputs(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {GEOJSON_PATH.resolve()}")

    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        fc = json.load(f)

    polygons: dict[str, object] = {}
    polygon_sex: dict[str, str] = {}
    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        name = str(props.get("leopard_name", props.get("leopard", "Unknown")))
        sex = str(props.get("sex", "Unknown"))
        ring = [(float(pair[1]), float(pair[0])) for pair in feat["geometry"]["coordinates"][0]]
        poly = polygon_from_ring(ring)
        if poly.is_valid and not poly.is_empty:
            polygons[name] = poly
            polygon_sex[name] = sex

    names = sorted(polygons)
    matrix = []
    pair_rows: list[dict[str, object]] = []

    for a in names:
        row = []
        for b in names:
            if a == b:
                row.append(polygons[a].area / 1_000_000.0)
            else:
                inter = polygons[a].intersection(polygons[b])
                row.append((inter.area / 1_000_000.0) if not inter.is_empty else 0.0)
        matrix.append(row)

    overlap_matrix = pd.DataFrame(matrix, index=names, columns=names)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            poly_a = polygons[a]
            poly_b = polygons[b]
            inter = poly_a.intersection(poly_b)
            inter_km2 = (inter.area / 1_000_000.0) if not inter.is_empty else 0.0
            union = poly_a.union(poly_b)
            union_km2 = union.area / 1_000_000.0 if not union.is_empty else np.nan
            area_a = poly_a.area / 1_000_000.0
            area_b = poly_b.area / 1_000_000.0
            sex_pair = "-".join(sorted([polygon_sex[a], polygon_sex[b]]))
            pair_rows.append(
                {
                    "leopard_a": a,
                    "sex_a": polygon_sex[a],
                    "area_a_km2": area_a,
                    "leopard_b": b,
                    "sex_b": polygon_sex[b],
                    "area_b_km2": area_b,
                    "pair_sex": sex_pair,
                    "overlaps": int(inter_km2 > 0),
                    "overlap_km2": inter_km2,
                    "mean_polygon_area_km2": (area_a + area_b) / 2.0,
                    "prop_overlap": 2.0 * inter_km2 / (area_a + area_b) if (area_a + area_b) else np.nan,
                    "union_km2": union_km2,
                    "jaccard_overlap": (inter_km2 / union_km2) if union_km2 else np.nan,
                    "pct_of_a_overlapped": (inter_km2 / area_a * 100.0) if area_a else np.nan,
                    "pct_of_b_overlapped": (inter_km2 / area_b * 100.0) if area_b else np.nan,
                }
            )

    overlap_pairs = pd.DataFrame(pair_rows).sort_values(
        ["overlap_km2", "jaccard_overlap"], ascending=[False, False]
    ).reset_index(drop=True)

    overlap_by_pair = (
        overlap_pairs.groupby("pair_sex", dropna=False)
        .agg(
            n_pairs=("pair_sex", "size"),
            n_overlapping_pairs=("overlap_km2", lambda s: int((s > 0).sum())),
            mean_overlap_km2=("overlap_km2", "mean"),
            median_overlap_km2=("overlap_km2", "median"),
            mean_jaccard=("jaccard_overlap", "mean"),
            median_jaccard=("jaccard_overlap", "median"),
        )
        .reset_index()
        .sort_values("pair_sex")
    )

    return overlap_matrix, overlap_pairs, overlap_by_pair


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()

    sparse_males = build_sparse_male_drivers(summary)
    sex_stats = build_sex_stats(summary)
    overlap_matrix, overlap_pairs, overlap_by_pair = build_overlap_outputs(summary)

    sparse_males.to_csv(OUT_SPARSE_MALES, index=False)
    sex_stats.to_csv(OUT_SEX_STATS, index=False)
    overlap_matrix.to_csv(OUT_OVERLAP_MATRIX)
    overlap_pairs.to_csv(OUT_OVERLAP_PAIRS, index=False)
    overlap_by_pair.to_csv(OUT_OVERLAP_BY_PAIR, index=False)

    print(f"Saved -> {OUT_SPARSE_MALES}")
    print(f"Saved -> {OUT_SEX_STATS}")
    print(f"Saved -> {OUT_OVERLAP_MATRIX}")
    print(f"Saved -> {OUT_OVERLAP_PAIRS}")
    print(f"Saved -> {OUT_OVERLAP_BY_PAIR}")


if __name__ == "__main__":
    main()
