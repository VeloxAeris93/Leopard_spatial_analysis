from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
import numpy as np
import pandas as pd

from _floodplain import load_floodplain_polygons
from _ng31_boundary import load_ng31_lonlat_lines
from _paths import NG31_GEOJSON
from _plot_style import SEX_GREY, SEX_MARKER, set_pub_style


# Figure 3 shows the contemporary-window 95% MCPs, so it must plot the same
# window's locations - not the pooled 2014-2026 set.
WINDOW = "contemporary_24mo"
POINTS_PATH = Path("outputs") / "tables" / f"merged_leopard_images_metadata.window_subset__{WINDOW}.csv"
GEOJSON_PATH = Path("outputs") / "figures" / f"merged_image_home_ranges_mcp95__{WINDOW}.geojson"
NG31_PATH = NG31_GEOJSON

OUT_FIGS = Path("outputs") / "figures"
FIG_PNG = OUT_FIGS / "new_fig_merged_subset_home_ranges_map_osm_vector_gray.png"
FIG_PDF = OUT_FIGS / "new_fig_merged_subset_home_ranges_map_osm_vector_gray.pdf"
FIG_SVG = OUT_FIGS / "new_fig_merged_subset_home_ranges_map_osm_vector_gray.svg"

WEB_MERCATOR_RADIUS_M = 6_378_137.0

# A raster tile basemap converted to greyscale reads as almost flat white: pale web
# map styles (CartoDB Positron and similar) separate land/water/park mainly by hue,
# and a plain luminance conversion collapses those near-identical grey values. This
# map is drawn as a clean vector map instead - a plain land tint, the OSM
# water/wetland layer, and the NG31 concession outline for context - which is what
# actually stays legible once colour is stripped out.
LAND_FACE = "#fbfbf9"
FLOODPLAIN_FACE = "#c9c9c9"
FLOODPLAIN_EDGE = "#8c8c8c"
BOUNDARY_COLOR = "#9a9a9a"

CHITABE_CAMP_LAT = -19.52708
CHITABE_CAMP_LON = 23.37879


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna() if str(v) != "Unknown"]
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "Unknown"


def _load_points() -> pd.DataFrame:
    df = pd.read_csv(POINTS_PATH)
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(_resolve_sex)
    df["x_m"], df["y_m"] = _lonlat_to_web_mercator(df["gps_lon"], df["gps_lat"])
    return df[df["sex"].isin(["Female", "Male"])].copy()


def _load_polygons() -> list[dict[str, object]]:
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        fc = json.load(f)

    features = []
    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        sex = str(props.get("sex", "Unknown"))
        if sex not in {"Female", "Male"}:
            continue
        coords = feat["geometry"]["coordinates"][0]
        features.append(
            {
                "leopard_name": str(props.get("leopard_name", "Unknown")),
                "sex": sex,
                "coords_lonlat": coords,
                "n_unique_points": props.get("n_unique_points"),
            }
        )
    return features


def _lonlat_to_web_mercator(lon: pd.Series | np.ndarray | float, lat: pd.Series | np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    x = WEB_MERCATOR_RADIUS_M * np.deg2rad(lon_arr)
    y = WEB_MERCATOR_RADIUS_M * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_arr) / 2.0))
    return x, y


def _web_mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = np.rad2deg(x / WEB_MERCATOR_RADIUS_M)
    lat = np.rad2deg(2.0 * np.arctan(np.exp(y / WEB_MERCATOR_RADIUS_M)) - np.pi / 2.0)
    return float(lon), float(lat)


def _load_floodplain_mercator_rings(x_min: float, x_max: float, y_min: float, y_max: float) -> list[list[tuple[float, float]]]:
    """Floodplain polygon rings in web-mercator metres, clipped to the map extent."""
    from shapely.geometry import box

    lon_min, lat_min = _web_mercator_to_lonlat(x_min, y_min)
    lon_max, lat_max = _web_mercator_to_lonlat(x_max, y_max)
    view = box(lon_min, lat_min, lon_max, lat_max)

    rings: list[list[tuple[float, float]]] = []
    for poly in load_floodplain_polygons():
        if not poly.intersects(view):
            continue
        clipped = poly.intersection(view)
        geoms = clipped.geoms if clipped.geom_type == "MultiPolygon" else [clipped]
        for geom in geoms:
            if geom.is_empty or geom.geom_type != "Polygon":
                continue
            xs, ys = _lonlat_to_web_mercator(
                np.array([c[0] for c in geom.exterior.coords]),
                np.array([c[1] for c in geom.exterior.coords]),
            )
            rings.append(list(zip(xs, ys)))
    return rings


def _load_boundary_mercator_lines() -> list[tuple[np.ndarray, np.ndarray]]:
    lines = []
    for ring in load_ng31_lonlat_lines(NG31_PATH):
        lons = np.array([c[0] for c in ring])
        lats = np.array([c[1] for c in ring])
        xs, ys = _lonlat_to_web_mercator(lons, lats)
        lines.append((xs, ys))
    return lines


def _set_map_extent(ax: plt.Axes, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def _add_scalebar(ax: plt.Axes, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    length_m = 2_000.0
    x0 = x_min + (x_max - x_min) * 0.05
    y0 = y_min + (y_max - y_min) * 0.05
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=2.0, solid_capstyle="butt", zorder=7)
    ax.plot([x0, x0], [y0 - 120.0, y0 + 120.0], color="black", linewidth=1.1, zorder=7)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 120.0, y0 + 120.0], color="black", linewidth=1.1, zorder=7)
    ax.text(x0 + length_m / 2.0, y0 + 190.0, "2 km", ha="center", va="bottom", fontsize=8)


def _add_north_arrow(ax: plt.Axes, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    x = x_max - (x_max - x_min) * 0.07
    y = y_max - (y_max - y_min) * 0.08
    ax.annotate("N", xy=(x, y), xytext=(x, y - (y_max - y_min) * 0.07), ha="center", va="center", fontsize=9, arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="black"))


def _label_positions(points: pd.DataFrame) -> pd.DataFrame:
    return (
        points.groupby(["leopard_name", "sex"], dropna=False)
        .agg(label_x=("x_m", "mean"), label_y=("y_m", "mean"))
        .reset_index()
    )


def _add_camp_marker(ax: plt.Axes) -> None:
    camp_x, camp_y = _lonlat_to_web_mercator(CHITABE_CAMP_LON, CHITABE_CAMP_LAT)
    ax.scatter([camp_x], [camp_y], s=44, c="black", marker="o", edgecolors="white", linewidths=0.8, zorder=7)
    ax.text(camp_x + 350.0, camp_y + 350.0, "Chitabe Camp", fontsize=7, color="black", ha="left", va="bottom", zorder=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.9))


def _build_panel_legend_handles(sex: str) -> list:
    """Per-panel key: georeferenced records (this panel's marker) + floodplain swatch.

    Each panel already encodes sex through its A/B letter and the leopard names shown,
    so the record marker is labelled generically rather than repeating "female"/"male".
    """
    return [
        Line2D(
            [0], [0], linestyle="none", marker=SEX_MARKER[sex], markersize=5,
            markerfacecolor=SEX_GREY[sex], markeredgecolor="white",
            label="Georeferenced leopard records",
        ),
        Patch(facecolor=FLOODPLAIN_FACE, edgecolor=FLOODPLAIN_EDGE, linewidth=0.6, label="Seasonal floodplain"),
    ]


def main() -> None:
    set_pub_style()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    points = _load_points()
    polygons = _load_polygons()
    x_min = float(points["x_m"].min())
    x_max = float(points["x_m"].max())
    y_min = float(points["y_m"].min())
    y_max = float(points["y_m"].max())
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.10
    padded_bounds = (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    floodplain_rings = _load_floodplain_mercator_rings(*padded_bounds)
    boundary_lines = _load_boundary_mercator_lines()

    panel_letters = {"Female": "A", "Male": "B"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 7.2), sharex=True, sharey=True)
    for ax, sex in zip(axes, ["Female", "Male"]):
        sub_pts = points[points["sex"] == sex].copy()
        sub_poly = [feat for feat in polygons if feat["sex"] == sex]
        grey = SEX_GREY[sex]

        ax.set_facecolor(LAND_FACE)

        for xs, ys in boundary_lines:
            ax.plot(xs, ys, color=BOUNDARY_COLOR, linewidth=0.6, zorder=1)

        for ring in floodplain_rings:
            ax.add_patch(
                MplPolygon(
                    ring, closed=True, facecolor=FLOODPLAIN_FACE, edgecolor=FLOODPLAIN_EDGE,
                    linewidth=0.4, alpha=0.85, zorder=1,
                )
            )

        for feat in sub_poly:
            lons = [coord[0] for coord in feat["coords_lonlat"]]
            lats = [coord[1] for coord in feat["coords_lonlat"]]
            xs, ys = _lonlat_to_web_mercator(lons, lats)
            patch = MplPolygon(
                list(zip(xs, ys)), closed=True, facecolor=grey, edgecolor="black",
                linewidth=0.5, alpha=0.22, zorder=2,
            )
            ax.add_patch(patch)

        ax.scatter(
            sub_pts["x_m"], sub_pts["y_m"], s=12, marker=SEX_MARKER[sex],
            facecolor=grey, alpha=0.85, linewidths=0.3, edgecolors="white", zorder=4,
        )

        # Each range is labelled with the number of locations it rests on, since some
        # polygons are built from eight points and others from fifty.
        n_points = {f["leopard_name"]: f["n_unique_points"] for f in sub_poly}
        for row in _label_positions(sub_pts).itertuples(index=False):
            n = n_points.get(row.leopard_name)
            label = f"{row.leopard_name} ({int(n)})" if n is not None else row.leopard_name
            ax.text(row.label_x, row.label_y, label, fontsize=7, color="black", ha="center", va="center", zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.9))

        _set_map_extent(ax, *padded_bounds)
        _add_scalebar(ax, *padded_bounds)
        _add_north_arrow(ax, *padded_bounds)
        _add_camp_marker(ax)
        # Panel identifier only - matches "female (A) and male (B)" in the figure
        # caption. Descriptive titles are not set here; they belong in the caption.
        ax.text(
            0.03, 0.97, panel_letters[sex], transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", zorder=7,
            bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6, pad=2.5),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend sits inside the axes, not anchored below them, so it is part of the
        # already-allocated plot area and can never be squashed against the figure
        # edge regardless of how tight_layout apportions the margins.
        ax.legend(
            handles=_build_panel_legend_handles(sex), loc="lower right",
            frameon=True, framealpha=0.88, edgecolor="none", fontsize=7.5,
            handlelength=1.5, columnspacing=1.2, borderpad=0.6,
        )

    fig.tight_layout()

    fig.savefig(FIG_PNG)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)

    print(f"Saved -> {FIG_PNG}")
    print(f"Saved -> {FIG_PDF}")
    print(f"Saved -> {FIG_SVG}")


if __name__ == "__main__":
    main()
