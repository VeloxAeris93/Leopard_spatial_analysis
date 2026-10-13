from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import requests
import shapefile
from shapely.geometry import Point, Polygon, shape
from shapely.geometry.base import BaseGeometry

from _paths import NATURAL_EARTH_COUNTRIES, NG31_EDIT_SHP
from _plot_style import set_pub_style


POINTS_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.dated_events.csv"
OUT_FIGS = Path("outputs") / "figures"
FIG_PNG = OUT_FIGS / "new_fig_study_area_ng31_osm_vector_gray.png"
FIG_PDF = OUT_FIGS / "new_fig_study_area_ng31_osm_vector_gray.pdf"
FIG_SVG = OUT_FIGS / "new_fig_study_area_ng31_osm_vector_gray.svg"

CACHE_DIR = Path("outputs") / "cache"
TILE_CACHE_DIR = CACHE_DIR / "tiles" / "carto_light_nolabels"
FALLBACK_CACHE_DIR = CACHE_DIR / "tiles" / "osm_grey"
BOUNDARY_CACHE_PATH = CACHE_DIR / "ne_110m_admin_0_countries.geojson"
NG31_PATH = NG31_EDIT_SHP

WEB_MERCATOR_RADIUS_M = 6_378_137.0
WEB_MERCATOR_LIMIT_M = WEB_MERCATOR_RADIUS_M * np.pi
TILE_SIZE_PX = 256
BASEMAP_URLS = [
    "https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "https://b.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "https://c.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "https://d.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
]
BOUNDARY_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
BASEMAP_ATTRIBUTION = "Basemap: clean greyscale OSM-style reference | Inset boundaries: Natural Earth"

CHITABE_CAMP_LAT = -19.52708
CHITABE_CAMP_LON = 23.37879


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


def _xy_to_tile_indices(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_clamped = max(min(lat, 85.05112878), -85.05112878)
    n_tiles = 2**zoom
    x_tile = int((lon + 180.0) / 360.0 * n_tiles)
    y_tile = int((1.0 - np.log(np.tan(np.deg2rad(lat_clamped)) + 1.0 / np.cos(np.deg2rad(lat_clamped))) / np.pi) / 2.0 * n_tiles)
    return x_tile, y_tile


def _tile_bounds_mercator(x_tile: int, y_tile: int, zoom: int) -> tuple[float, float, float, float]:
    tiles_per_axis = 2**zoom
    tile_span_m = (2.0 * WEB_MERCATOR_LIMIT_M) / tiles_per_axis
    min_x = -WEB_MERCATOR_LIMIT_M + x_tile * tile_span_m
    max_x = min_x + tile_span_m
    max_y = WEB_MERCATOR_LIMIT_M - y_tile * tile_span_m
    min_y = max_y - tile_span_m
    return min_x, max_x, min_y, max_y


def _choose_basemap_zoom(bounds_lonlat: tuple[float, float, float, float], max_tiles: int = 25) -> int:
    lon_min, lon_max, lat_min, lat_max = bounds_lonlat
    for zoom in range(12, 8, -1):
        x0, y1 = _xy_to_tile_indices(lon_min, lat_min, zoom)
        x1, y0 = _xy_to_tile_indices(lon_max, lat_max, zoom)
        n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
        if n_tiles <= max_tiles:
            return zoom
    return 8


def _fetch_tile_image(x_tile: int, y_tile: int, zoom: int) -> Image.Image:
    tile_dir = TILE_CACHE_DIR / str(zoom) / str(x_tile)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f"{y_tile}.png"
    if tile_path.exists():
        return Image.open(tile_path).convert("RGB")

    last_error: Exception | None = None
    for url_template in BASEMAP_URLS:
        try:
            response = requests.get(
                url_template.format(z=zoom, x=x_tile, y=y_tile),
                timeout=20,
                headers={"User-Agent": "Codex NG31 clean OSM-style study-area figure builder/1.0"},
            )
            response.raise_for_status()
            tile = Image.open(BytesIO(response.content)).convert("RGB")
            tile = ImageEnhance.Contrast(tile).enhance(1.02)
            tile = ImageEnhance.Brightness(tile).enhance(1.01)
            tile.save(tile_path)
            return tile
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        fallback_tile_path = FALLBACK_CACHE_DIR / str(zoom) / str(x_tile) / f"{y_tile}.png"
        if fallback_tile_path.exists():
            return Image.open(fallback_tile_path).convert("RGB")
        for delta in (1, -1, 2, -2):
            neighbor = TILE_CACHE_DIR / str(zoom) / str(x_tile + delta) / f"{y_tile}.png"
            if neighbor.exists():
                return Image.open(neighbor).convert("RGB")
            fallback_neighbor = FALLBACK_CACHE_DIR / str(zoom) / str(x_tile + delta) / f"{y_tile}.png"
            if fallback_neighbor.exists():
                return Image.open(fallback_neighbor).convert("RGB")
        print(f"Warning: using blank fallback tile for z={zoom}, x={x_tile}, y={y_tile} after clean-basemap fetch failures: {last_error}")
        return Image.new("RGB", (TILE_SIZE_PX, TILE_SIZE_PX), color="#eef1eb")
    raise RuntimeError("No clean-basemap URL templates were available.")


def _fetch_basemap(bounds_mercator: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    min_x, max_x, min_y, max_y = bounds_mercator
    lon_min, lat_min = _web_mercator_to_lonlat(min_x, min_y)
    lon_max, lat_max = _web_mercator_to_lonlat(max_x, max_y)
    zoom = _choose_basemap_zoom((lon_min, lon_max, lat_min, lat_max))
    x0, y1 = _xy_to_tile_indices(lon_min, lat_min, zoom)
    x1, y0 = _xy_to_tile_indices(lon_max, lat_max, zoom)

    width_tiles = x1 - x0 + 1
    height_tiles = y1 - y0 + 1
    mosaic = Image.new("RGB", (width_tiles * TILE_SIZE_PX, height_tiles * TILE_SIZE_PX))

    for x_tile in range(x0, x1 + 1):
        for y_tile in range(y0, y1 + 1):
            tile = _fetch_tile_image(x_tile, y_tile, zoom)
            mosaic.paste(tile, ((x_tile - x0) * TILE_SIZE_PX, (y_tile - y0) * TILE_SIZE_PX))

    extent_min_x, _, _, extent_max_y = _tile_bounds_mercator(x0, y0, zoom)
    _, extent_max_x, extent_min_y, _ = _tile_bounds_mercator(x1, y1, zoom)
    return np.asarray(mosaic), (extent_min_x, extent_max_x, extent_min_y, extent_max_y)


def _load_points() -> pd.DataFrame:
    df = pd.read_csv(POINTS_PATH)
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df = df.dropna(subset=["gps_lat", "gps_lon"]).copy()
    df["x_m"], df["y_m"] = _lonlat_to_web_mercator(df["gps_lon"], df["gps_lat"])
    return df


def _load_world_boundaries() -> dict:
    BOUNDARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BOUNDARY_CACHE_PATH.exists() and NATURAL_EARTH_COUNTRIES.exists():
        BOUNDARY_CACHE_PATH.write_bytes(NATURAL_EARTH_COUNTRIES.read_bytes())
    if not BOUNDARY_CACHE_PATH.exists():
        response = requests.get(BOUNDARY_URL, timeout=30, headers={"User-Agent": "Codex NG31 study-area figure builder/1.0"})
        response.raise_for_status()
        BOUNDARY_CACHE_PATH.write_bytes(response.content)

    with open(BOUNDARY_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_ng31_polygon(path: Path = NG31_PATH) -> Polygon:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find concession boundary file: {path.resolve()}")

    reader = shapefile.Reader(str(path))
    if not reader.shapes():
        raise ValueError(f"No shapes found in: {path.resolve()}")

    points = reader.shapes()[0].points
    if len(points) < 3:
        raise ValueError(f"Boundary line has too few points in: {path.resolve()}")

    # The concession is stored as a traced boundary line; close it into a polygon.
    ring = [(float(lon), float(lat)) for lon, lat in points]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    polygon = Polygon(ring)
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError(f"Unable to build a valid polygon from: {path.resolve()}")
    return polygon


def _filter_points_to_polygon(points: pd.DataFrame, polygon: Polygon) -> tuple[pd.DataFrame, pd.DataFrame]:
    inside_mask = [
        polygon.contains(Point(float(lon), float(lat))) or polygon.touches(Point(float(lon), float(lat)))
        for lon, lat in zip(points["gps_lon"], points["gps_lat"])
    ]
    inside = points.loc[inside_mask].copy()
    outside = points.loc[[not flag for flag in inside_mask]].copy()
    return inside, outside


def _plot_geom_lonlat(ax: plt.Axes, geom: BaseGeometry, facecolor: str, edgecolor: str, linewidth: float, alpha: float = 1.0) -> None:
    if geom.geom_type == "Polygon":
        geoms = [geom]
    elif geom.geom_type == "MultiPolygon":
        geoms = list(geom.geoms)
    else:
        return

    for poly in geoms:
        xy = np.asarray(poly.exterior.coords)
        patch = MplPolygon(xy, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        ax.add_patch(patch)


def _add_scalebar(ax: plt.Axes, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    length_m = 5_000.0
    x0 = x_min + (x_max - x_min) * 0.05
    y0 = y_min + (y_max - y_min) * 0.05
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=2.2, solid_capstyle="butt", zorder=7)
    ax.plot([x0, x0], [y0 - 220.0, y0 + 220.0], color="black", linewidth=1.2, zorder=7)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 220.0, y0 + 220.0], color="black", linewidth=1.2, zorder=7)
    ax.text(x0 + length_m / 2.0, y0 + 320.0, "5 km", ha="center", va="bottom", fontsize=8)


def _add_north_arrow(ax: plt.Axes, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    x = x_max - (x_max - x_min) * 0.07
    y = y_max - (y_max - y_min) * 0.08
    ax.annotate("N", xy=(x, y), xytext=(x, y - (y_max - y_min) * 0.08), ha="center", va="center", fontsize=9, arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="black"))


def _build_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#444444", linewidth=0, marker="o", markersize=4, markerfacecolor="#444444", label="Georeferenced leopard records"),
        Line2D([0], [0], color="#cc5a2e", linewidth=1.8, label="NG/31 concession boundary"),
        Line2D([0], [0], color="black", linewidth=0, marker="o", markersize=6, markerfacecolor="black", markeredgecolor="white", label="Chitabe Camp"),
        Line2D([0], [0], color="#7f8c8d", linewidth=0, marker="s", markersize=7, markerfacecolor="#7f8c8d", label="Botswana"),
    ]


def main() -> None:
    set_pub_style()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    points_all = _load_points()
    world = _load_world_boundaries()
    ng31 = _load_ng31_polygon()
    points, outside_points = _filter_points_to_polygon(points_all, ng31)
    if points.empty:
        raise RuntimeError("No sighting points fall inside the NG/31 concession boundary.")

    x_min = float(points["x_m"].min())
    x_max = float(points["x_m"].max())
    y_min = float(points["y_m"].min())
    y_max = float(points["y_m"].max())
    boundary_x, boundary_y = _lonlat_to_web_mercator(*zip(*ng31.exterior.coords))
    x_min = min(x_min, float(np.min(boundary_x)))
    x_max = max(x_max, float(np.max(boundary_x)))
    y_min = min(y_min, float(np.min(boundary_y)))
    y_max = max(y_max, float(np.max(boundary_y)))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad_left = x_span * 0.16
    x_pad_right = x_span * 0.06
    y_pad = y_span * 0.08
    main_bounds = (x_min - x_pad_left, x_max + x_pad_right, y_min - y_pad, y_max + y_pad)
    basemap_image, basemap_extent = _fetch_basemap(main_bounds)

    fig = plt.figure(figsize=(10.8, 8.2))
    ax_main = fig.add_axes([0.05, 0.10, 0.76, 0.82])
    ax_inset = fig.add_axes([0.80, 0.62, 0.18, 0.24])

    ax_main.imshow(basemap_image, extent=basemap_extent, origin="upper", alpha=0.98, zorder=0)
    ax_main.scatter(points["x_m"], points["y_m"], s=8, c="#444444", alpha=0.42, linewidths=0, zorder=3)
    ax_main.plot(boundary_x, boundary_y, color="#cc5a2e", linewidth=1.8, zorder=4)
    camp_x, camp_y = _lonlat_to_web_mercator(CHITABE_CAMP_LON, CHITABE_CAMP_LAT)
    ax_main.scatter([camp_x], [camp_y], s=48, c="black", edgecolors="white", linewidths=0.8, zorder=5)
    ax_main.text(
        camp_x + 520.0,
        camp_y + 360.0,
        "Chitabe Camp",
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.0),
    )
    ax_main.text(
        float(np.mean(boundary_x)),
        float(np.max(boundary_y)) - (y_max - y_min) * 0.08,
        "NG/31 concession",
        fontsize=10,
        fontweight="bold",
        color="#1f2b3a",
        ha="center",
        va="center",
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.0),
    )

    ax_main.set_xlim(main_bounds[0], main_bounds[1])
    ax_main.set_ylim(main_bounds[2], main_bounds[3])
    ax_main.set_aspect("equal")
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    _add_scalebar(ax_main, *main_bounds)
    _add_north_arrow(ax_main, *main_bounds)

    africa_geoms: list[BaseGeometry] = []
    botswana_geom: BaseGeometry | None = None
    for feature in world.get("features", []):
        props = feature.get("properties", {})
        continent = props.get("CONTINENT")
        iso3 = props.get("ADM0_A3") or props.get("ISO_A3") or props.get("ADM0_A3_US")
        geom = shape(feature["geometry"])
        if continent == "Africa":
            africa_geoms.append(geom)
        if iso3 == "BWA":
            botswana_geom = geom

    for geom in africa_geoms:
        _plot_geom_lonlat(ax_inset, geom, facecolor="#e3e6e8", edgecolor="#aab3b8", linewidth=0.35)
    if botswana_geom is not None:
        _plot_geom_lonlat(ax_inset, botswana_geom, facecolor="#7f8c8d", edgecolor="#566168", linewidth=0.7)

    ax_inset.scatter([CHITABE_CAMP_LON], [CHITABE_CAMP_LAT], s=20, c="black", edgecolors="white", linewidths=0.5, zorder=5)
    ax_inset.text(
        CHITABE_CAMP_LON + 3.2,
        CHITABE_CAMP_LAT + 2.6,
        "Study site",
        fontsize=7,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.8),
    )
    ax_inset.text(24.0, -22.0, "Botswana", fontsize=7, color="#2f3a40", ha="center", va="center")
    ax_inset.set_xlim(-20, 55)
    ax_inset.set_ylim(-36, 38)
    ax_inset.set_aspect("equal")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_visible(False)
    ax_inset.set_title("Location within Africa", fontsize=9)

    fig.legend(handles=_build_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=True, framealpha=0.92, fontsize=8, handlelength=1.8, columnspacing=1.4)
    fig.text(0.06, 0.02, BASEMAP_ATTRIBUTION, ha="left", va="bottom", fontsize=6, color="#666666")

    fig.savefig(FIG_PNG)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)

    print(f"Mapped {len(points)} in-boundary sighting points; excluded {len(outside_points)} outside NG/31.")
    print(f"Saved -> {FIG_PNG}")
    print(f"Saved -> {FIG_PDF}")
    print(f"Saved -> {FIG_SVG}")


if __name__ == "__main__":
    main()
