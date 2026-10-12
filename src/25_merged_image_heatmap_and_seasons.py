from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import requests

from _plot_style import set_pub_style


IN_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.dated_events.csv"

OUT_TABLES = Path("outputs") / "tables"
OUT_FIGS = Path("outputs") / "figures"

OUT_MONTH = OUT_TABLES / "merged_image_counts_by_monthname.csv"
OUT_SEASON = OUT_TABLES / "merged_image_counts_by_botswana_season.csv"
OUT_MONTH_SEX = OUT_TABLES / "merged_image_counts_by_monthname_and_sex.csv"
OUT_SEASON_SEX = OUT_TABLES / "merged_image_counts_by_botswana_season_and_sex.csv"

FIG_HEATMAP_PNG = OUT_FIGS / "merged_image_detection_heatmap.png"
FIG_HEATMAP_PDF = OUT_FIGS / "merged_image_detection_heatmap.pdf"
FIG_HEATMAP_SVG = OUT_FIGS / "merged_image_detection_heatmap.svg"
FIG_HEATMAP_CARTO_PNG = OUT_FIGS / "merged_image_detection_heatmap_osm_vector_gray.png"
FIG_HEATMAP_CARTO_PDF = OUT_FIGS / "merged_image_detection_heatmap_osm_vector_gray.pdf"
FIG_HEATMAP_CARTO_SVG = OUT_FIGS / "merged_image_detection_heatmap_osm_vector_gray.svg"
FIG_MONTH_PNG = OUT_FIGS / "merged_image_counts_by_monthname.png"
FIG_MONTH_PDF = OUT_FIGS / "merged_image_counts_by_monthname.pdf"
FIG_MONTH_SVG = OUT_FIGS / "merged_image_counts_by_monthname.svg"
FIG_SEASON_PNG = OUT_FIGS / "merged_image_counts_by_botswana_season.png"
FIG_SEASON_PDF = OUT_FIGS / "merged_image_counts_by_botswana_season.pdf"
FIG_SEASON_SVG = OUT_FIGS / "merged_image_counts_by_botswana_season.svg"

CACHE_DIR = Path("outputs") / "cache" / "tiles" / "osm_grey"
CARTO_CACHE_DIR = Path("outputs") / "cache" / "tiles" / "carto_light_nolabels"
WEB_MERCATOR_RADIUS_M = 6_378_137.0
WEB_MERCATOR_LIMIT_M = WEB_MERCATOR_RADIUS_M * np.pi
TILE_SIZE_PX = 256
BASEMAP_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
CARTO_BASEMAP_URLS = [
    "https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "https://b.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "https://c.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "https://d.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
]
BASEMAP_ATTRIBUTION = "Basemap: OpenStreetMap contributors"
CARTO_BASEMAP_ATTRIBUTION = "Basemap: clean greyscale OSM-style reference"
CHITABE_CAMP_LAT = -19.52708
CHITABE_CAMP_LON = 23.37879

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
SEASON_ORDER = ["Wet season", "Dry season"]


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
    y_tile = int(
        (1.0 - np.log(np.tan(np.deg2rad(lat_clamped)) + 1.0 / np.cos(np.deg2rad(lat_clamped))) / np.pi) / 2.0 * n_tiles
    )
    return x_tile, y_tile


def _tile_bounds_mercator(x_tile: int, y_tile: int, zoom: int) -> tuple[float, float, float, float]:
    tiles_per_axis = 2**zoom
    tile_span_m = (2.0 * WEB_MERCATOR_LIMIT_M) / tiles_per_axis
    min_x = -WEB_MERCATOR_LIMIT_M + x_tile * tile_span_m
    max_x = min_x + tile_span_m
    max_y = WEB_MERCATOR_LIMIT_M - y_tile * tile_span_m
    min_y = max_y - tile_span_m
    return min_x, max_x, min_y, max_y


def _choose_basemap_zoom(bounds_lonlat: tuple[float, float, float, float], max_tiles: int = 25, max_zoom: int = 14) -> int:
    lon_min, lon_max, lat_min, lat_max = bounds_lonlat
    for zoom in range(max_zoom, 8, -1):
        x0, y1 = _xy_to_tile_indices(lon_min, lat_min, zoom)
        x1, y0 = _xy_to_tile_indices(lon_max, lat_max, zoom)
        n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
        if n_tiles <= max_tiles:
            return zoom
    return 8


def _fetch_tile_image(
    x_tile: int,
    y_tile: int,
    zoom: int,
    *,
    cache_root: Path,
    url_template: str | list[str],
    user_agent: str,
) -> Image.Image:
    tile_dir = cache_root / str(zoom) / str(x_tile)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f"{y_tile}.png"

    if tile_path.exists():
        return Image.open(tile_path).convert("RGB")

    templates = [url_template] if isinstance(url_template, str) else url_template
    last_error: Exception | None = None
    for template in templates:
        try:
            response = requests.get(
                template.format(z=zoom, x=x_tile, y=y_tile),
                timeout=20,
                headers={"User-Agent": user_agent},
            )
            response.raise_for_status()
            tile = Image.open(BytesIO(response.content)).convert("RGB")
            tile = ImageOps.grayscale(tile).convert("RGB")
            tile = ImageEnhance.Contrast(tile).enhance(1.15)
            tile = ImageEnhance.Brightness(tile).enhance(1.04)
            tile.save(tile_path)
            return tile
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        fallback_tile_path = CACHE_DIR / str(zoom) / str(x_tile) / f"{y_tile}.png"
        if fallback_tile_path.exists():
            return Image.open(fallback_tile_path).convert("RGB")
        for delta in (1, -1, 2, -2):
            neighbor = cache_root / str(zoom) / str(x_tile + delta) / f"{y_tile}.png"
            if neighbor.exists():
                return Image.open(neighbor).convert("RGB")
            fallback_neighbor = CACHE_DIR / str(zoom) / str(x_tile + delta) / f"{y_tile}.png"
            if fallback_neighbor.exists():
                return Image.open(fallback_neighbor).convert("RGB")
        print(f"Warning: using blank fallback tile for z={zoom}, x={x_tile}, y={y_tile} after CARTO fetch failures: {last_error}")
        return Image.new("RGB", (TILE_SIZE_PX, TILE_SIZE_PX), color="#eef1eb")
    raise RuntimeError("No basemap URL templates were provided.")


def _fetch_basemap(
    bounds_mercator: tuple[float, float, float, float],
    *,
    cache_root: Path,
    url_template: str | list[str],
    user_agent: str,
    max_zoom: int = 14,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    min_x, max_x, min_y, max_y = bounds_mercator
    lon_min, lat_min = _web_mercator_to_lonlat(min_x, min_y)
    lon_max, lat_max = _web_mercator_to_lonlat(max_x, max_y)

    zoom = _choose_basemap_zoom((lon_min, lon_max, lat_min, lat_max), max_zoom=max_zoom)
    x0, y1 = _xy_to_tile_indices(lon_min, lat_min, zoom)
    x1, y0 = _xy_to_tile_indices(lon_max, lat_max, zoom)

    width_tiles = x1 - x0 + 1
    height_tiles = y1 - y0 + 1
    mosaic = Image.new("RGB", (width_tiles * TILE_SIZE_PX, height_tiles * TILE_SIZE_PX))

    for x_tile in range(x0, x1 + 1):
        for y_tile in range(y0, y1 + 1):
            tile = _fetch_tile_image(
                x_tile,
                y_tile,
                zoom,
                cache_root=cache_root,
                url_template=url_template,
                user_agent=user_agent,
            )
            mosaic.paste(tile, ((x_tile - x0) * TILE_SIZE_PX, (y_tile - y0) * TILE_SIZE_PX))

    extent_min_x, _, _, extent_max_y = _tile_bounds_mercator(x0, y0, zoom)
    _, extent_max_x, extent_min_y, _ = _tile_bounds_mercator(x1, y1, zoom)
    return np.asarray(mosaic), (extent_min_x, extent_max_x, extent_min_y, extent_max_y)


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna() if str(v) != "Unknown"]
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "Unknown"


def _season_from_month(month: int) -> str:
    if month in {11, 12, 1, 2, 3}:
        return "Wet season"
    return "Dry season"


def _load_data() -> pd.DataFrame:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {IN_PATH.resolve()}")

    df = pd.read_csv(IN_PATH)
    df["image_date"] = pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True)
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    if "leopard_name" in df.columns:
        df["sex"] = df.groupby("leopard_name")["sex"].transform(_resolve_sex)

    df = df.dropna(subset=["image_date"]).copy()
    df["month_num"] = df["image_date"].dt.month
    df["month_name"] = pd.Categorical(df["image_date"].dt.month_name(), categories=MONTH_ORDER, ordered=True)
    df["botswana_season"] = pd.Categorical(df["month_num"].map(_season_from_month), categories=SEASON_ORDER, ordered=True)

    spatial = df.dropna(subset=["gps_lat", "gps_lon"]).copy()
    spatial["x_m"], spatial["y_m"] = _lonlat_to_web_mercator(spatial["gps_lon"], spatial["gps_lat"])
    return df, spatial


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
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - (y_max - y_min) * 0.08),
        ha="center",
        va="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="black"),
    )


def build_month_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_year_month = (
        df.assign(year=df["image_date"].dt.year.astype(int))
        .groupby(["year", "month_num", "month_name"], observed=True)
        .size()
        .reset_index(name="n_records")
        .sort_values(["year", "month_num"])
    )
    month = (
        by_year_month.groupby("month_num", observed=True)
        .agg(
            n_records=("n_records", "sum"),
            n_years_with_records=("year", "nunique"),
            mean_records_per_year=("n_records", "mean"),
            median_records_per_year=("n_records", "median"),
        )
        .reset_index()
        .sort_values("month_num")
    )
    month["month_name"] = pd.Categorical(
        [MONTH_ORDER[int(month_num) - 1] for month_num in month["month_num"]],
        categories=MONTH_ORDER,
        ordered=True,
    )
    month_sex = (
        df.groupby(["month_num", "month_name", "sex"], observed=True)
        .size()
        .reset_index(name="n_records")
        .sort_values(["month_num", "sex"])
    )
    return month, month_sex


def build_season_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    season = (
        df.groupby("botswana_season", observed=False)
        .size()
        .reset_index(name="n_records")
        .sort_values("botswana_season")
    )
    season["percent_records"] = 100.0 * season["n_records"] / season["n_records"].sum()

    season_sex = (
        df.groupby(["botswana_season", "sex"], observed=False)
        .size()
        .reset_index(name="n_records")
        .sort_values(["botswana_season", "sex"])
    )
    return season, season_sex


def plot_heatmap(spatial: pd.DataFrame) -> None:
    x_min = float(spatial["x_m"].min())
    x_max = float(spatial["x_m"].max())
    y_min = float(spatial["y_m"].min())
    y_max = float(spatial["y_m"].max())
    x_pad = (x_max - x_min) * 0.10
    y_pad = (y_max - y_min) * 0.12
    bounds = (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    basemap_image, basemap_extent = _fetch_basemap(
        bounds,
        cache_root=CACHE_DIR,
        url_template=BASEMAP_URL,
        user_agent="Codex merged image heatmap builder/1.0",
    )

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.imshow(basemap_image, extent=basemap_extent, origin="upper", alpha=0.96, zorder=0)
    hb = ax.hexbin(
        spatial["x_m"],
        spatial["y_m"],
        gridsize=35,
        mincnt=1,
        cmap="inferno",
        linewidths=0,
        alpha=0.72,
        zorder=3,
    )
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Number of leopard sightings")
    camp_x, camp_y = _lonlat_to_web_mercator(CHITABE_CAMP_LON, CHITABE_CAMP_LAT)
    ax.scatter([camp_x], [camp_y], s=44, c="black", edgecolors="white", linewidths=0.8, zorder=5)
    ax.text(
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
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _add_scalebar(ax, *bounds)
    _add_north_arrow(ax, *bounds)
    fig.text(0.02, 0.02, BASEMAP_ATTRIBUTION, ha="left", va="bottom", fontsize=6, color="#666666")
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    fig.savefig(FIG_HEATMAP_PNG)
    fig.savefig(FIG_HEATMAP_PDF)
    fig.savefig(FIG_HEATMAP_SVG)
    plt.close(fig)

    try:
        carto_image, carto_extent = _fetch_basemap(
            bounds,
            cache_root=CARTO_CACHE_DIR,
            url_template=CARTO_BASEMAP_URLS,
            user_agent="Codex merged image heatmap CARTO builder/1.0",
            max_zoom=12,
        )
        fig, ax = plt.subplots(figsize=(7.8, 7.2))
        ax.imshow(carto_image, extent=carto_extent, origin="upper", alpha=0.98, zorder=0)
        hb = ax.hexbin(
            spatial["x_m"],
            spatial["y_m"],
            gridsize=35,
            mincnt=1,
            cmap="inferno",
            linewidths=0,
            alpha=0.72,
            zorder=3,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Number of leopard sightings")
        camp_x, camp_y = _lonlat_to_web_mercator(CHITABE_CAMP_LON, CHITABE_CAMP_LAT)
        ax.scatter([camp_x], [camp_y], s=44, c="black", edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(
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
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        _add_scalebar(ax, *bounds)
        _add_north_arrow(ax, *bounds)
        fig.text(0.02, 0.02, CARTO_BASEMAP_ATTRIBUTION, ha="left", va="bottom", fontsize=6, color="#666666")
        fig.tight_layout(rect=[0, 0.02, 1, 1])

        fig.savefig(FIG_HEATMAP_CARTO_PNG)
        fig.savefig(FIG_HEATMAP_CARTO_PDF)
        fig.savefig(FIG_HEATMAP_CARTO_SVG)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: skipped clean OSM-style heatmap variant because tile fetch failed: {exc}")


def plot_month_counts(month: pd.DataFrame, month_sex: pd.DataFrame) -> None:
    del month_sex
    avg_per_year = (
        month.set_index("month_name")["mean_records_per_year"]
        .reindex(MONTH_ORDER)
        .fillna(0.0)
    )
    years_present = (
        month.set_index("month_name")["n_years_with_records"]
        .reindex(MONTH_ORDER)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(MONTH_ORDER, avg_per_year.to_numpy(), color="#7e8f68", width=0.76)
    ax.set_title("Average merged leopard detections by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average detections per year")
    ax.tick_params(axis="x", rotation=45)
    for bar, years in zip(bars, years_present):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(avg_per_year.max() * 0.02, 0.08),
            f"n={int(years)}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#4d5b43",
        )
    ax.text(
        0.01,
        0.98,
        "Standardized as the mean number of detections per sampled year in each month.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()

    fig.savefig(FIG_MONTH_PNG)
    fig.savefig(FIG_MONTH_PDF)
    fig.savefig(FIG_MONTH_SVG)
    plt.close(fig)


def plot_season_counts(season: pd.DataFrame, season_sex: pd.DataFrame) -> None:
    del season_sex
    totals = (
        season.groupby("botswana_season", observed=False)["n_records"]
        .sum()
        .reindex(SEASON_ORDER)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    colors = ["#7e8f68", "#b99254"]
    bars = ax.bar(SEASON_ORDER, totals.to_numpy(), color=colors, width=0.62)
    ax.set_ylim(0, max(totals) * 1.16)
    for bar, total, pct in zip(bars, totals, season["percent_records"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            total + max(totals) * 0.022,
            f"{int(total)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Total merged leopard detections by Botswana season", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Detections")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    fig.savefig(FIG_SEASON_PNG)
    fig.savefig(FIG_SEASON_PDF)
    fig.savefig(FIG_SEASON_SVG)
    plt.close(fig)


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CARTO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    set_pub_style()

    df, spatial = _load_data()
    month, month_sex = build_month_tables(df)
    season, season_sex = build_season_tables(df)

    month.to_csv(OUT_MONTH, index=False)
    month_sex.to_csv(OUT_MONTH_SEX, index=False)
    season.to_csv(OUT_SEASON, index=False)
    season_sex.to_csv(OUT_SEASON_SEX, index=False)

    plot_heatmap(spatial)
    plot_month_counts(month, month_sex)
    plot_season_counts(season, season_sex)

    print(f"Saved -> {OUT_MONTH}")
    print(f"Saved -> {OUT_MONTH_SEX}")
    print(f"Saved -> {OUT_SEASON}")
    print(f"Saved -> {OUT_SEASON_SEX}")
    print(f"Saved -> {FIG_HEATMAP_PNG}")
    print(f"Saved -> {FIG_HEATMAP_PDF}")
    print(f"Saved -> {FIG_HEATMAP_SVG}")
    print(f"Saved -> {FIG_HEATMAP_CARTO_PNG}")
    print(f"Saved -> {FIG_HEATMAP_CARTO_PDF}")
    print(f"Saved -> {FIG_HEATMAP_CARTO_SVG}")
    print(f"Saved -> {FIG_MONTH_PNG}")
    print(f"Saved -> {FIG_MONTH_PDF}")
    print(f"Saved -> {FIG_MONTH_SVG}")
    print(f"Saved -> {FIG_SEASON_PNG}")
    print(f"Saved -> {FIG_SEASON_PDF}")
    print(f"Saved -> {FIG_SEASON_SVG}")


if __name__ == "__main__":
    main()
