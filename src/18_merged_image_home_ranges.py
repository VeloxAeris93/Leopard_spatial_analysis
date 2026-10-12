from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _mcp import geojson_feature, mcp
from _plot_style import set_pub_style


IN_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.spatial_subset.csv"
OUT_TABLES = Path("outputs") / "tables"
OUT_FIGS = Path("outputs") / "figures"

OUT_SUMMARY = OUT_TABLES / "merged_image_home_ranges_mcp_summary.csv"
OUT_STATS_BY_SEX = OUT_TABLES / "merged_image_home_range_stats_by_sex.csv"
OUT_SENSITIVITY = OUT_TABLES / "merged_image_home_range_sensitivity.csv"
OUT_GEOJSON_95 = OUT_FIGS / "merged_image_home_ranges_mcp95.geojson"
OUT_GEOJSON_100 = OUT_FIGS / "merged_image_home_ranges_mcp.geojson"

FIG_TOP = OUT_FIGS / "merged_image_home_ranges_top15.png"
FIG_BY_SEX = OUT_FIGS / "merged_image_home_ranges_by_sex.png"

# The paper reports 95% MCPs; the 100% hull is retained as a sensitivity estimator.
PRIMARY_PERCENT = 95.0


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna() if str(v) != "Unknown"]
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "Unknown"


def _load_data() -> pd.DataFrame:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {IN_PATH.resolve()}")

    df = pd.read_csv(IN_PATH, low_memory=False)
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["leopard_name"] = df.get("leopard_name", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["image_date"] = pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True)
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(_resolve_sex)
    return df


def build_home_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    features_95: list[dict[str, object]] = []
    features_100: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for leopard, group in df.groupby("leopard_name", dropna=False):
        sex = str(group["sex"].iloc[0]) if not group["sex"].empty else "Unknown"
        pts_ll = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))

        hull_95 = mcp(pts_ll, percent=PRIMARY_PERCENT)
        hull_100 = mcp(pts_ll, percent=100.0)

        summary_rows.append(
            {
                "leopard_name": leopard,
                "sex": sex,
                "n_images": int(len(group)),
                "n_unique_dates": int(group["image_date"].nunique()),
                "n_unique_points": int(hull_95["n_unique_points"]),
                "first_seen": group["image_date"].min(),
                "last_seen": group["image_date"].max(),
                "centroid_lat": float(group["gps_lat"].mean()),
                "centroid_lon": float(group["gps_lon"].mean()),
                "n_points_dropped": int(hull_95["n_points_dropped"]),
                "pct_points_dropped": hull_95["pct_points_dropped"],
                "mcp95_km2": hull_95["area_km2"],
                "mcp100_km2": hull_100["area_km2"],
                # Primary estimator, kept under the historical column name so
                # downstream scripts read the 95% MCP by default.
                "home_range_km2_mcp": hull_95["area_km2"],
            }
        )

        props = {
            "leopard_name": leopard,
            "sex": sex,
            "n_images": int(len(group)),
            "n_unique_dates": int(group["image_date"].nunique()),
            "n_unique_points": int(hull_95["n_unique_points"]),
        }
        if hull_95["ring_latlon"]:
            features_95.append(
                geojson_feature(
                    hull_95["ring_latlon"],
                    {
                        **props,
                        "mcp_percent": PRIMARY_PERCENT,
                        "n_points_dropped": int(hull_95["n_points_dropped"]),
                        "home_range_km2_mcp": float(hull_95["area_km2"]),
                    },
                )
            )
        if hull_100["ring_latlon"]:
            features_100.append(
                geojson_feature(
                    hull_100["ring_latlon"],
                    {**props, "mcp_percent": 100.0, "home_range_km2_mcp": float(hull_100["area_km2"])},
                )
            )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["home_range_km2_mcp", "n_images"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
    summary["first_seen"] = pd.to_datetime(summary["first_seen"], errors="coerce").dt.date.astype("string")
    summary["last_seen"] = pd.to_datetime(summary["last_seen"], errors="coerce").dt.date.astype("string")

    return (
        summary,
        {"type": "FeatureCollection", "features": features_95},
        {"type": "FeatureCollection", "features": features_100},
    )


def build_stats_by_sex(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.dropna(subset=["home_range_km2_mcp"])
        .groupby("sex", dropna=False)
        .agg(
            n_individuals=("leopard_name", "nunique"),
            median_km2=("home_range_km2_mcp", "median"),
            mean_km2=("home_range_km2_mcp", "mean"),
            min_km2=("home_range_km2_mcp", "min"),
            max_km2=("home_range_km2_mcp", "max"),
            median_unique_points=("n_unique_points", "median"),
            median_unique_dates=("n_unique_dates", "median"),
        )
        .reset_index()
        .sort_values("sex")
    )


def build_sensitivity_table(summary: pd.DataFrame) -> pd.DataFrame:
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
        for estimator in ("mcp95_km2", "mcp100_km2"):
            females = sub.loc[sub["sex"] == "Female", estimator].dropna()
            males = sub.loc[sub["sex"] == "Male", estimator].dropna()
            rows.append(
                {
                    "subset": label,
                    "estimator": estimator,
                    "n_female": int(len(females)),
                    "n_male": int(len(males)),
                    "median_female_km2": float(females.median()) if len(females) else None,
                    "median_male_km2": float(males.median()) if len(males) else None,
                    "mean_female_km2": float(females.mean()) if len(females) else None,
                    "mean_male_km2": float(males.mean()) if len(males) else None,
                    "median_diff_male_minus_female_km2": (
                        float(males.median() - females.median()) if len(females) and len(males) else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_top_home_ranges(summary: pd.DataFrame) -> None:
    top = summary.dropna(subset=["home_range_km2_mcp"]).head(15).sort_values("home_range_km2_mcp")
    plt.figure(figsize=(8.4, 5.8))
    plt.barh(top["leopard_name"], top["home_range_km2_mcp"])
    plt.xlabel("95% MCP area (km$^2$)")
    plt.title("Top 15 merged-image 95% MCP areas")
    plt.tight_layout()
    plt.savefig(FIG_TOP, dpi=220)
    plt.close()


def plot_home_ranges_by_sex(summary: pd.DataFrame) -> None:
    plot_df = summary.dropna(subset=["home_range_km2_mcp"]).copy()
    order = [sex for sex in ["Female", "Male", "Unknown"] if sex in plot_df["sex"].unique()]
    data = [plot_df.loc[plot_df["sex"] == sex, "home_range_km2_mcp"].to_numpy() for sex in order]

    plt.figure(figsize=(6.2, 4.8))
    plt.boxplot(data, tick_labels=order)
    plt.ylabel("95% MCP area (km$^2$)")
    plt.title("Merged-image 95% MCP areas by sex")
    plt.tight_layout()
    plt.savefig(FIG_BY_SEX, dpi=220)
    plt.close()


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    set_pub_style()

    df = _load_data()
    summary, geojson_95, geojson_100 = build_home_ranges(df)
    stats_by_sex = build_stats_by_sex(summary)
    sensitivity = build_sensitivity_table(summary)

    summary.to_csv(OUT_SUMMARY, index=False)
    stats_by_sex.to_csv(OUT_STATS_BY_SEX, index=False)
    sensitivity.to_csv(OUT_SENSITIVITY, index=False)
    with open(OUT_GEOJSON_95, "w", encoding="utf-8") as f:
        json.dump(geojson_95, f)
    with open(OUT_GEOJSON_100, "w", encoding="utf-8") as f:
        json.dump(geojson_100, f)

    plot_top_home_ranges(summary)
    plot_home_ranges_by_sex(summary)

    for path in (OUT_SUMMARY, OUT_STATS_BY_SEX, OUT_SENSITIVITY, OUT_GEOJSON_95, OUT_GEOJSON_100, FIG_TOP, FIG_BY_SEX):
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
