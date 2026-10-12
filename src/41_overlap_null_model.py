"""Is observed range overlap lower than expected if leopards placed ranges at random?

About half of all dyads overlap, but that figure has no benchmark on its own. This
script builds one. Each individual's 95% MCP is kept fixed in size and shape and
relocated at random - rotated by a uniform angle and translated so that its centroid
falls uniformly inside the area where guides could have recorded a leopard. Overlap
is recomputed for every dyad, and the process is repeated N_ITERATIONS times. The
result is the distribution of overlap expected if individuals positioned their ranges
independently of one another. Observed overlap in the lower tail of that distribution
is evidence of spatial avoidance, which territoriality predicts.

The test pools all individuals rather than splitting them by sex, so the small number
of males does not limit it. Pair-type results are reported as well, from the same
simulations.

The accessible area is the region in which a polygon centroid may be placed:

  sampled_1000m  (primary)  NG31 within 1 km of any GPS-referenced leopard encounter,
                            all individuals and all years. This is the area the drives
                            demonstrably reached.
  sampled_500m              as above, 500 m.
  roads2013_1000m           NG31 within 1 km of the 2013 game-drive road file.
  roads2013_2000m           as above, 2 km.

The 2013 road file predates much of the network driven in 2024-2026: a quarter of
2024 encounters lie more than 1 km from it, and several observed MCP centroids fall
outside its 1 km buffer. A null region that is too small packs random polygons into
the core and inflates expected overlap, which would manufacture an avoidance result,
so the road-based regions are reported as sensitivity analyses only.

Usage:
    python src/41_overlap_null_model.py                      # contemporary 24-month window
    python src/41_overlap_null_model.py recent_12mo
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import LineString, Point, shape
from shapely.ops import polygonize, transform, unary_union

from _ng31_boundary import load_ng31_lonlat_lines
from _paths import NG31_GEOJSON, ROADS_GEOJSON
from _plot_style import PAIR_GREY, PAIR_ORDER, set_pub_style

WINDOW = sys.argv[1] if len(sys.argv) > 1 else "contemporary_24mo"
PRIMARY_REGION = "sampled_1000m"
N_ITERATIONS = 2_000
SEED = 20261005

TABLES = Path("outputs") / "tables"
FIGS = Path("outputs") / "figures"
MCP_PATH = FIGS / f"merged_image_home_ranges_mcp95__{WINDOW}.geojson"
# Locations are genuine even where the date is not, so the sampled area uses every
# GPS-referenced encounter, including those excluded from the time windows.
EVENTS_PATH = TABLES / "merged_leopard_images_metadata.normalized_events.csv"
ROADS_PATH = ROADS_GEOJSON
NG31_PATH = NG31_GEOJSON

OUT_SUMMARY = TABLES / f"overlap_null_model_summary__{WINDOW}.csv"
OUT_ITER = TABLES / f"overlap_null_model_iterations__{WINDOW}.csv"
OUT_REGIONS = TABLES / f"overlap_null_model_regions__{WINDOW}.csv"
FIG_STEM = FIGS / f"fig6_overlap_null_model__{WINDOW}"

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32734", always_xy=True).transform


def _ng31() -> shapely.Polygon:
    lines = [LineString(line) for line in load_ng31_lonlat_lines(NG31_PATH)]
    return transform(_TO_UTM, max(polygonize(unary_union(lines)), key=lambda p: p.area))


def _roads() -> shapely.Geometry:
    features = json.loads(ROADS_PATH.read_text(encoding="utf-8"))["features"]
    parts = []
    for feature in features:
        geom = feature["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        parts += [LineString([c[:2] for c in line]) for line in lines]
    return transform(_TO_UTM, unary_union(parts))


def _encounters() -> shapely.Geometry:
    df = pd.read_csv(EVENTS_PATH, low_memory=False)
    lon = pd.to_numeric(df["gps_lon"], errors="coerce")
    lat = pd.to_numeric(df["gps_lat"], errors="coerce")
    ok = lon.notna() & lat.notna()
    x, y = _TO_UTM(lon[ok].to_numpy(), lat[ok].to_numpy())
    return shapely.multipoints(np.column_stack([x, y]))


def build_regions() -> dict[str, shapely.Geometry]:
    ng31 = _ng31()
    encounters = _encounters()
    roads = _roads()
    return {
        "sampled_1000m": encounters.buffer(1000).intersection(ng31),
        "sampled_500m": encounters.buffer(500).intersection(ng31),
        "roads2013_1000m": roads.buffer(1000).intersection(ng31),
        "roads2013_2000m": roads.buffer(2000).intersection(ng31),
    }


def load_polygons() -> pd.DataFrame:
    features = json.loads(MCP_PATH.read_text(encoding="utf-8"))["features"]
    rows = []
    for f in features:
        geom = transform(_TO_UTM, shape(f["geometry"]))
        if geom.is_empty or geom.area <= 0:
            continue
        rows.append({"leopard_name": f["properties"]["leopard_name"], "sex": f["properties"]["sex"], "geometry": geom})
    return pd.DataFrame(rows).sort_values("leopard_name").reset_index(drop=True)


def _dyads(polys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ia, ib = map(np.array, zip(*combinations(range(len(polys)), 2)))
    sex = polys["sex"].to_numpy()
    pair = np.array(["-".join(sorted([sex[a], sex[b]])) for a, b in zip(ia, ib)])
    return ia, ib, pair


def overlap_stats(geoms: np.ndarray, areas: np.ndarray, ia: np.ndarray, ib: np.ndarray, pair: np.ndarray) -> dict[str, float]:
    inter = shapely.area(shapely.intersection(geoms[ia], geoms[ib]))
    prop = 2.0 * inter / (areas[ia] + areas[ib])
    overlaps = inter > 0
    out = {"all__pct_overlapping": 100 * overlaps.mean(), "all__mean_prop_overlap": 100 * prop.mean()}
    for p in PAIR_ORDER:
        m = pair == p
        if m.any():
            out[f"{p}__pct_overlapping"] = 100 * overlaps[m].mean()
            out[f"{p}__mean_prop_overlap"] = 100 * prop[m].mean()
    return out


def _sample_points(region: shapely.Geometry, n: int, rng: np.random.Generator) -> np.ndarray:
    shapely.prepare(region)
    minx, miny, maxx, maxy = region.bounds
    out = np.empty((0, 2))
    while len(out) < n:
        cand = np.column_stack([rng.uniform(minx, maxx, 4 * n), rng.uniform(miny, maxy, 4 * n)])
        keep = shapely.contains_xy(region, cand[:, 0], cand[:, 1])
        out = np.vstack([out, cand[keep]])
    return out[:n]


def simulate(polys: pd.DataFrame, region: shapely.Geometry, rng: np.random.Generator) -> pd.DataFrame:
    ia, ib, pair = _dyads(polys)
    # Centre every polygon on the origin once; each iteration then only rotates and shifts.
    centred = [affinity.translate(g, -g.centroid.x, -g.centroid.y) for g in polys["geometry"]]
    areas = np.array([g.area for g in centred])
    rows = []
    for _ in range(N_ITERATIONS):
        angles = rng.uniform(0.0, 360.0, len(centred))
        centres = _sample_points(region, len(centred), rng)
        geoms = np.array(
            [affinity.translate(affinity.rotate(g, a, origin=(0, 0)), x, y) for g, a, (x, y) in zip(centred, angles, centres)],
            dtype=object,
        )
        rows.append(overlap_stats(geoms, areas, ia, ib, pair))
    return pd.DataFrame(rows)


def summarise(observed: dict[str, float], sims: pd.DataFrame, region_name: str) -> pd.DataFrame:
    rows = []
    for key, obs in observed.items():
        group, metric = key.split("__")
        null = sims[key].to_numpy()
        rows.append(
            {
                "window": WINDOW,
                "region": region_name,
                "group": group,
                "metric": metric,
                "observed": obs,
                "null_mean": null.mean(),
                "null_p05": np.percentile(null, 5),
                "null_p50": np.percentile(null, 50),
                "null_p95": np.percentile(null, 95),
                "observed_percentile": 100 * (null < obs).mean() + 50 * (null == obs).mean(),
                "p_lower": ((null <= obs).sum() + 1) / (len(null) + 1),
                "p_upper": ((null >= obs).sum() + 1) / (len(null) + 1),
                "n_iterations": len(null),
            }
        )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, sims: pd.DataFrame, n_polys: int, n_dyads: int) -> None:
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    panels = [("all__pct_overlapping", "Dyads that overlap (%)"), ("all__mean_prop_overlap", "Mean proportional overlap (%)")]
    for ax, (key, label), tag in zip(axes, panels, "AB"):
        group, metric = key.split("__")
        row = summary[(summary["region"] == PRIMARY_REGION) & (summary["group"] == group) & (summary["metric"] == metric)].iloc[0]
        ax.hist(sims[key], bins=30, color="#bdbdbd", edgecolor="white", linewidth=0.4)
        ax.axvline(row["observed"], color="black", linewidth=1.4)
        ax.axvline(row["null_p05"], color="#636363", linestyle="--", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Simulations")
        ax.set_title(f"({tag}) observed {row['observed']:.1f}; null 5th percentile {row['null_p05']:.1f}\n"
                     f"p(lower) = {row['p_lower']:.3f}, p(upper) = {row['p_upper']:.3f}", loc="left", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{n_polys} individuals, {n_dyads} dyads, {len(sims):,} random placements", fontsize=8.5, y=0.99)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_STEM.with_suffix(f".{ext}"))
    plt.close(fig)


def main() -> None:
    polys = load_polygons()
    ia, ib, pair = _dyads(polys)
    geoms = polys["geometry"].to_numpy()
    areas = np.array([g.area for g in geoms])
    observed = overlap_stats(geoms, areas, ia, ib, pair)

    regions = build_regions()
    region_rows = []
    for name, region in regions.items():
        inside = sum(region.contains(g.centroid) for g in geoms)
        region_rows.append({"window": WINDOW, "region": name, "area_km2": region.area / 1e6,
                            "observed_centroids_inside": inside, "n_polygons": len(geoms)})
    pd.DataFrame(region_rows).to_csv(OUT_REGIONS, index=False)

    rng = np.random.default_rng(SEED)
    summaries, iterations, primary_sims = [], [], None
    for name, region in regions.items():
        print(f"{WINDOW} / {name}: {N_ITERATIONS:,} iterations over {region.area / 1e6:.1f} km2 ...", flush=True)
        sims = simulate(polys, region, rng)
        summaries.append(summarise(observed, sims, name))
        iterations.append(sims.assign(region=name, iteration=np.arange(len(sims))))
        if name == PRIMARY_REGION:
            primary_sims = sims

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    pd.concat(iterations, ignore_index=True).to_csv(OUT_ITER, index=False)
    plot(summary, primary_sims, len(polys), len(ia))

    pd.set_option("display.width", 220)
    print(pd.DataFrame(region_rows).round(1).to_string(index=False))
    print(summary.round(3).to_string(index=False))
    print(f"Saved -> {OUT_SUMMARY}\nSaved -> {OUT_ITER}\nSaved -> {OUT_REGIONS}\nSaved -> {FIG_STEM}.png")


if __name__ == "__main__":
    main()
