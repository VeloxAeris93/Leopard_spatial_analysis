"""Chitabe leopards: spatial overlap analysis.

Companion code for "Patterns of spatial overlap of leopards in the Okavango Delta"
(Pretorius, Hamman, Keith & Somers). One script, run top to bottom, rebuilds every
table, figure and number in the manuscript from the raw records:

    python analysis.py

Raw inputs are read from data/ (not included in the repository, see README.md).
Intermediate tables are written to outputs/tables/ after each step so that every stage
can be inspected; figures go to outputs/figures/ and the Supplementary Tables to
outputs/.

Steps
     1  Photo metadata          GPS position and date of every photo
     2  Sightings log           match each photo to the nearest sighting of its leopard
     3  Batch identities        add the 27 May 2026 batch, named from the manual list
     4  Encounter events        repeat photos of one leopard, one day, within 250 m -> one event
     5  Camera summary          cameras used (Table S1)
     6  Date validity           drop events dated only by the archive export day
     7  Seasonality             detections by wet and dry season
     8  Inclusion               individuals with enough data for a range estimate
     9  Pooled home ranges      95% and 100% MCPs over the whole study period
    10  Time windows            95% MCPs and pairwise overlap within each window
    11  Overlap models          two-part models with node-label permutation tests
    12  Null model              observed overlap against randomly placed ranges (Figure 6)
    13  Power                   smallest detectable sex difference in range size
    14  Reviewer decomposition  the submitted analysis rebuilt step by step into the revised one
    15  Detection table         one row per leopard (Table S2)
    16  Figures                 Figures 1-5, S1, S2
    17  Manuscript numbers      every number quoted in the manuscript
    18  Supplementary Tables    Tables S1-S5 (Word and Markdown)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
from io import BytesIO
from itertools import combinations
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from xml.sax.saxutils import escape
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageEnhance
from pyproj import Transformer
import requests
from scipy import stats
from scipy.stats import mannwhitneyu, norm
import shapefile
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, transform, unary_union
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


# =============================================================================
# Settings
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REFERENCE = ROOT / "reference"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGS = OUTPUTS / "figures"
CACHE = OUTPUTS / "cache"

# Raw inputs (see README.md for their contents).
PHOTO_ZIP = DATA / "Chitabe Leopards.zip"            # one folder per leopard
BATCH_ZIP = DATA / "2026-05-27.zip"                  # later batch, no per-leopard folders
FILE_NAMES_LIST = DATA / "file_names_list.csv"       # manual identities for the batch photos
SIGHTINGS_XLSX = DATA / "Chitabe Leopards.xlsx"      # guides' sightings log
LINEAGE_XLSX = DATA / "leopard_lineage_master.xlsx"  # sex, estimated age, family links
ROADS_GEOJSON = DATA / "chitabe_roads_2013.geojson"  # 2013 game-drive roads
NG31_GEOJSON = DATA / "ng31.geojson"                 # NG31 concession boundary
NG31_EDIT_SHP = DATA / "ng31_edit.shp"               # edited outline drawn on Figure 1

# Saved reference data, so the figures do not depend on live servers.
OSM_FLOODPLAIN = REFERENCE / "osm_floodplain_ng31_2026-08-25.json.gz"  # (c) OSM contributors, ODbL
NATURAL_EARTH = REFERENCE / "ne_110m_admin_0_countries.geojson"        # public domain

BATCH_ZIP_NAME = BATCH_ZIP.name

# Names later confirmed to be one leopard: former name -> name used going forward.
MERGED_IDENTITIES = {"Morobe": "Tswaletswale"}

# The archive was exported from Lightroom with capture dates stripped; photos whose only
# date is the export day keep their location but cannot be placed in time (step 6).
ARCHIVE_EXPORT_DATE = pd.Timestamp("2026-03-16")
STUDY_START = "2014-08-10"  # first photograph in the archive

HIGH_CONFIDENCE_DISTANCE_M = 50.0  # photo-to-sighting match counted as reliable
EVENT_DISTANCE_M = 250.0           # photos of one leopard on one day within this form one event

# Inclusion for range estimates (step 8). At 8 locations the 95% MCP's peripheral removal
# drops one point (12.5%), so the threshold and the estimator are reported together.
MIN_UNIQUE_POINTS = 8
MIN_UNIQUE_DATES = 3
SUBMITTED_MIN_UNIQUE_POINTS = 10  # criterion of the original submission (step 14)
NON_TERRITORIAL_AGE_CLASSES = frozenset({"cub", "subadult"})

MCP_PERCENT = 95.0
WINDOWS: dict[str, tuple[str | None, str | None]] = {
    "contemporary_24mo": ("2024-06-01", "2026-05-25"),
    "recent_12mo": ("2025-06-01", "2026-05-25"),
    "pooled": (None, None),
}
PRIMARY_WINDOW = "contemporary_24mo"

PAIR_ORDER = ["Female-Female", "Female-Male", "Male-Male"]
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 20260821
NULL_ITERATIONS = 2_000
NULL_SEED = 20261005
NULL_PRIMARY_REGION = "sampled_1000m"
POWER_SIMULATIONS = 5_000
POWER_SEED = 20261005

CHITABE_CAMP_LAT = -19.52708
CHITABE_CAMP_LON = 23.37879

UTM_34S = "EPSG:32734"
WGS84 = "EPSG:4326"

SOFTWARE_VERSIONS = {
    "python": "3.14.2",
    "pandas": "3.0.1",
    "numpy": "2.4.2",
    "scipy": "1.17.1",
    "statsmodels": "0.14.6",
    "shapely": "2.1.2",
    "pyproj": "3.7.2",
    "matplotlib": "3.10.8",
    "projection": "EPSG:32734 (UTM 34S)",
}


def table(name: str) -> Path:
    return TABLES / name


def read_table(name: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(table(name), **kwargs)


# =============================================================================
# Shared helpers
# =============================================================================

# --- names and sex ------------------------------------------------------------

# Spelling differences between photo folders and the sightings log.
NAME_REPLACEMENTS = {"maun rd": "maun road", "mojelemane": "mojeremane", "moretwla": "moretlwa"}


def name_key(value: object) -> str:
    """Lowercase alphanumerics only, for joining names across tables."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def display_name(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text.replace("_", " "))


def match_name(value: object) -> str | None:
    """Name used to match photos to sightings: no sex words, spelling fixes applied."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("_", " ")
    text = re.sub(r"\b(male|female)\b", "", text)
    for source, target in NAME_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def sex_from_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "female" in text:
        return "Female"
    if "male" in text:
        return "Male"
    return None


def resolve_sex(series: pd.Series) -> str:
    """One sex per leopard: the single known value among its records, else Unknown."""
    values = [str(v).strip().title() for v in series.dropna() if str(v).strip() and str(v).strip().title() != "Unknown"]
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "Unknown"


# --- lineage register -----------------------------------------------------------

def read_lineage() -> pd.DataFrame:
    return pd.read_excel(LINEAGE_XLSX, sheet_name="Lineage")


def age_class_lookup() -> dict[str, str]:
    """Leopard name key -> lowercase age class ('' when unknown)."""
    lineage = read_lineage()

    def normalise(label: object) -> str:
        text = re.sub(r"\s+", " ", str(label).strip().lower())
        return "" if text in {"", "nan", "none", "?"} else text

    return {name_key(n): normalise(a) for n, a in zip(lineage["leopard_name"], lineage["Estimated age"]) if name_key(n)}


def is_territorial(name: object, lookup: dict[str, str]) -> bool:
    """Old enough to hold a territory. Unknown age counts as territorial, so adults whose
    age was never recorded are not dropped on missing data. The age class is a single
    label per leopard, not age at each detection."""
    return lookup.get(name_key(name), "") not in NON_TERRITORIAL_AGE_CLASSES


def annotate_age(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = age_class_lookup()
    out = frame.copy()
    out["age_class"] = out["leopard_name"].map(lambda n: lookup.get(name_key(n), ""))
    out["is_territorial_age"] = ~out["age_class"].isin(NON_TERRITORIAL_AGE_CLASSES)
    return out


# --- dates ----------------------------------------------------------------------

def parse_image_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True, format="mixed").dt.normalize()


def is_archive_export_date(df: pd.DataFrame) -> pd.Series:
    """Only records whose date came from the EXIF save time AND fall on the export day."""
    source = df.get("image_datetime_source", pd.Series("", index=df.index)).astype("string")
    return parse_image_date(df).eq(ARCHIVE_EXPORT_DATE) & source.eq("exif_datetime")


# --- geometry -------------------------------------------------------------------

_TO_UTM = Transformer.from_crs(WGS84, UTM_34S, always_xy=True)
_TO_WGS84 = Transformer.from_crs(UTM_34S, WGS84, always_xy=True)
WEB_MERCATOR_RADIUS_M = 6_378_137.0


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2 - lat1) / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2.0) ** 2
    return 2.0 * 6_371_000.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def to_web_mercator(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    x = WEB_MERCATOR_RADIUS_M * np.deg2rad(lon_arr)
    y = WEB_MERCATOR_RADIUS_M * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_arr) / 2.0))
    return x, y


def from_web_mercator(x: float, y: float) -> tuple[float, float]:
    lon = np.rad2deg(x / WEB_MERCATOR_RADIUS_M)
    lat = np.rad2deg(2.0 * np.arctan(np.exp(y / WEB_MERCATOR_RADIUS_M)) - np.pi / 2.0)
    return float(lon), float(lat)


def load_ng31_lines(path: Path = NG31_GEOJSON) -> list[list[tuple[float, float]]]:
    """NG31 boundary as (lon, lat) line strings."""
    lines: list[list[tuple[float, float]]] = []
    for feature in json.loads(path.read_text(encoding="utf-8")).get("features", []):
        geometry = feature.get("geometry") or {}
        kind, coords = geometry.get("type"), geometry.get("coordinates", [])
        rings = {"LineString": [coords], "MultiLineString": coords, "Polygon": coords,
                 "MultiPolygon": [ring for poly in coords for ring in poly]}.get(kind, [])
        lines += [[(float(lon), float(lat)) for lon, lat, *_ in ring] for ring in rings]
    if not lines:
        raise ValueError(f"No usable boundary geometry in {path}")
    return lines


# --- minimum convex polygons ------------------------------------------------------

def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotone-chain convex hull of (x, y) points."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def n_points_to_drop(n: int, percent: float) -> int:
    """Peripheral locations removed for a `percent` MCP: ceil(n * (100 - percent) / 100),
    in integer arithmetic so 0.05 * 60 does not round up to 4."""
    if percent >= 100:
        return 0
    numerator = n * (100 - percent)
    if numerator != int(numerator):
        return int(math.ceil(numerator / 100.0))
    return -(-int(numerator) // 100)


def mcp(points_latlon: list[tuple[float, float]], percent: float = MCP_PERCENT) -> dict[str, object]:
    """`percent` MCP from (lat, lon) pairs, in UTM 34S. Peripheral removal drops the
    unique locations furthest from their mean centre (computed once, not iteratively)."""
    unique = sorted(set((float(lat), float(lon)) for lat, lon in points_latlon))
    n_unique = len(unique)
    xy = [tuple(float(v) for v in _TO_UTM.transform(lon, lat)) for lat, lon in unique]
    n_drop = n_points_to_drop(n_unique, percent)
    n_retained = n_unique - n_drop

    result: dict[str, object] = {
        "n_unique_points": n_unique,
        "n_points_dropped": n_drop,
        "pct_points_dropped": (100.0 * n_drop / n_unique) if n_unique else None,
        "area_km2": None,
        "ring_latlon": None,
    }
    if n_retained < 3:
        return result
    cx = sum(p[0] for p in xy) / n_unique
    cy = sum(p[1] for p in xy) / n_unique
    kept = sorted(xy, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))[:n_retained]
    hull = _hull(kept)
    if len(hull) < 3:
        return result

    ring = [tuple(reversed([float(v) for v in _TO_WGS84.transform(x, y)])) for x, y in hull]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    area = sum(hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1] for i in range(len(hull)))
    result["area_km2"] = abs(area) / 2.0 / 1_000_000.0
    result["ring_latlon"] = ring
    return result


def polygon_from_ring(ring_latlon: list[tuple[float, float]]) -> Polygon:
    """Shapely polygon in UTM 34S metres."""
    return Polygon([_TO_UTM.transform(lon, lat) for lat, lon in ring_latlon])


def geojson_feature(ring_latlon: list[tuple[float, float]], props: dict[str, object]) -> dict[str, object]:
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in ring_latlon]]}}


# --- statistics -------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) - P(x < y)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(1, -1)
    return float((x > y).mean() - (x < y).mean())


def bootstrap_median_diff(x: np.ndarray, y: np.ndarray, n: int = 20_000, seed: int = 42) -> tuple[float, float, float]:
    """Median difference (x - y) with a percentile bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1) - np.median(rng.choice(y, size=(n, len(y)), replace=True), axis=1)
    return float(np.median(x) - np.median(y)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def pair_label(sex_a: str, sex_b: str) -> str:
    return "-".join(sorted([sex_a, sex_b]))


# --- figures --------------------------------------------------------------------

# Greyscale with redundant marker shapes, so groups stay distinguishable in print.
SEX_GREY = {"Female": "#404040", "Male": "#a6a6a6"}
SEX_MARKER = {"Female": "o", "Male": "s"}
SEX3_ORDER = ["Female", "Male", "Unknown"]
SEX3_GREY = {"Female": "#262626", "Male": "#a6a6a6", "Unknown": "#5c5c5c"}
PAIR_GREY = {"Female-Female": "#262626", "Female-Male": "#808080", "Male-Male": "#adadad"}
PAIR_MARKER = {"Female-Female": "o", "Female-Male": "s", "Male-Male": "^"}


def figure_style() -> None:
    """Fresh publication style for each figure (300 dpi, TrueType fonts in PDF)."""
    mpl.rc_file_defaults()
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGS / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  Saved figure {stem}")


def add_scalebar(ax: plt.Axes, bounds: tuple[float, float, float, float], length_m: float, tick_m: float,
                 label_m: float, linewidths: tuple[float, float]) -> None:
    x_min, x_max, y_min, y_max = bounds
    x0 = x_min + (x_max - x_min) * 0.05
    y0 = y_min + (y_max - y_min) * 0.05
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", linewidth=linewidths[0], solid_capstyle="butt", zorder=7)
    for x in (x0, x0 + length_m):
        ax.plot([x, x], [y0 - tick_m, y0 + tick_m], color="black", linewidth=linewidths[1], zorder=7)
    ax.text(x0 + length_m / 2.0, y0 + label_m, f"{length_m / 1000:.0f} km", ha="center", va="bottom", fontsize=8)


def add_north_arrow(ax: plt.Axes, bounds: tuple[float, float, float, float], length: float) -> None:
    x_min, x_max, y_min, y_max = bounds
    x = x_max - (x_max - x_min) * 0.07
    y = y_max - (y_max - y_min) * 0.08
    ax.annotate("N", xy=(x, y), xytext=(x, y - (y_max - y_min) * length), ha="center", va="center", fontsize=9,
                arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="black"))


# =============================================================================
# Step 1. Photo metadata
# =============================================================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
GENERIC_NOTE_TEXTS = {"digital camera", "olympus digital camera"}
NOTE_TAG_PATTERNS = (
    ("cubs", re.compile(r"\bcubs?\b", flags=re.IGNORECASE)),
    ("feeding", re.compile(r"\b(feed(?:ing)?|eat(?:ing)?)\b", flags=re.IGNORECASE)),
    ("kill", re.compile(r"\b(kill(?:s|ed|ing)?|carcass|carcase|prey)\b", flags=re.IGNORECASE)),
    ("drinking", re.compile(r"\bdrink(?:ing)?\b", flags=re.IGNORECASE)),
    ("fighting", re.compile(r"\b(fight(?:ing|s|ed)?|fought)\b", flags=re.IGNORECASE)),
    ("resting", re.compile(r"\b(rest(?:ing)?|sleep(?:ing)?|nap(?:ping)?|lying|laying)\b", flags=re.IGNORECASE)),
    ("walking", re.compile(r"\b(walk(?:ing)?|mov(?:e|ing|es|ed)|stroll(?:ing)?)\b", flags=re.IGNORECASE)),
    ("hunting", re.compile(r"\b(hunt(?:ing)?|stalk(?:ing)?)\b", flags=re.IGNORECASE)),
    ("chasing", re.compile(r"\b(chase|chased|chasing|pursue|pursued|pursuing)\b", flags=re.IGNORECASE)),
    ("playing", re.compile(r"\bplay(?:ing|ed|s)?\b", flags=re.IGNORECASE)),
    ("mating", re.compile(r"\b(mat(?:e|es|ed|ing)|copulat(?:e|es|ed|ing))\b", flags=re.IGNORECASE)),
    ("grooming", re.compile(r"\b(groom(?:ing)?|lick(?:ing)?)\b", flags=re.IGNORECASE)),
)
EXIFTOOL_TAGS = [
    "-EXIF:GPSLatitude", "-EXIF:GPSLongitude", "-XMP:GPSLatitude", "-XMP:GPSLongitude",
    "-Composite:GPSLatitude", "-Composite:GPSLongitude", "-EXIF:DateTimeOriginal", "-EXIF:DateTimeDigitized",
    "-EXIF:DateTime", "-EXIF:UserComment", "-EXIF:ImageDescription", "-IPTC:Caption-Abstract", "-XMP:CreateDate",
    "-XMP:Description", "-XMP:ModifyDate", "-XMP:Subject", "-XMP:Title", "-XMP:Keywords", "-File:FileModifyDate",
]


def find_exiftool() -> str:
    path = os.environ.get("EXIFTOOL") or shutil.which("exiftool")
    if not path or not Path(path).exists() and not shutil.which(path):
        raise SystemExit("ExifTool not found. Install it (exiftool.org) and put it on PATH, or set EXIFTOOL.")
    return path


def _safe_str(x: Any) -> str | None:
    if x is None:
        return None
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None


def _dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            out.append(value)
    return out


def _note_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                value = value.decode(encoding)
                break
            except Exception:
                continue
        else:
            value = value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, tuple, set)):
        parts = _dedupe(_note_str(item) for item in value)
        return "; ".join(parts) if parts else None
    text = _safe_str(value)
    if text is None:
        return None
    text = text.replace("\x00", " ")
    text = re.sub(r"^(ASCII|UNICODE|JIS)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ;|")
    return text or None


def _is_meaningful_note(text: str | None) -> bool:
    if not text or text.casefold() in GENERIC_NOTE_TEXTS:
        return False
    return len(re.findall(r"\b[A-Z]{1,4}\[[^\]]+\]", text)) < 3


def _note_source_name(key: str) -> str | None:
    return {
        "usercomment": "exif_user_comment", "imagedescription": "exif_image_description",
        "caption-abstract": "iptc_caption_abstract", "captionabstract": "iptc_caption_abstract",
        "description": "xmp_description", "subject": "xmp_subject", "title": "xmp_title",
        "keywords": "keywords", "xpkeywords": "keywords",
    }.get(key.split(":")[-1].lower())


def _filename_note_text(rel_path: str) -> str | None:
    stem = re.sub(r"[_\-.]+", " ", Path(rel_path).stem)
    stem = re.sub(r"\b(female|male|fem)\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return None
    matches = _dedupe(m.group(0) for _, pattern in NOTE_TAG_PATTERNS for m in [pattern.search(stem)] if m is not None)
    return "; ".join(matches) if matches else None


def _collect_notes(rel_path: str, exif: dict[str, Any], exiftool: dict[str, Any]) -> tuple[str | None, ...]:
    texts: list[str] = []
    sources: list[str] = []

    def add(source: str, value: Any) -> None:
        text = _note_str(value)
        if _is_meaningful_note(text):
            texts.append(text)
            sources.append(source)

    add("exif_user_comment", exif.get("exif_UserComment"))
    add("exif_image_description", exif.get("exif_ImageDescription"))
    for key, value in exiftool.items():
        source = _note_source_name(key)
        if source is not None:
            add(source, value)
    texts, sources = _dedupe(texts), _dedupe(sources)

    filename_note = _filename_note_text(rel_path)
    if filename_note:
        sources = _dedupe([*sources, "filename"])
    metadata_note = "; ".join(texts) if texts else None
    haystack = " | ".join(t for t in (metadata_note, filename_note) if t)
    tags = [tag for tag, pattern in NOTE_TAG_PATTERNS if pattern.search(haystack)] if haystack else []
    return metadata_note, filename_note, "; ".join(tags) if tags else None, "; ".join(sources) if sources else None


def _parse_datetime(value: Any) -> datetime | None:
    s = _safe_str(value)
    if not s:
        return None
    candidates = [s]
    if re.match(r"^\d{4}:\d{2}:\d{2} ", s):
        candidates.append(s.replace(":", "-", 2))
    for text in candidates:
        normalized = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue
    return None


def _parse_gps_timestamp(date_value: Any, time_value: Any) -> datetime | None:
    date_text = _safe_str(date_value)
    if not date_text:
        return None
    normalized_date = date_text.replace("-", ":")
    if not re.match(r"^\d{4}:\d{2}:\d{2}$", normalized_date):
        return None
    time_text = _safe_str(time_value)
    if time_text:
        match = re.search(r"(\d+(?:\.\d+)?)", time_text)
        if isinstance(time_value, (list, tuple)) and len(time_value) >= 3:
            try:
                seconds_float = float(time_value[2])
                seconds = int(seconds_float)
                return datetime.strptime(normalized_date, "%Y:%m:%d").replace(
                    hour=int(float(time_value[0])), minute=int(float(time_value[1])), second=seconds,
                    microsecond=int(round((seconds_float - seconds) * 1_000_000)))
            except Exception:
                pass
        if match and ":" in time_text:
            parsed = _parse_datetime(f"{normalized_date} {time_text}")
            if parsed is not None:
                return parsed
    try:
        return datetime.strptime(normalized_date, "%Y:%m:%d")
    except ValueError:
        return None


def _datetime_from_filename(rel_path: str) -> datetime | None:
    stem = Path(Path(rel_path).name).stem
    for pattern in (
        r"(?P<date>\d{4}-\d{2}-\d{2})[ _-](?:at[ _-])?(?P<time>\d{2})[.\-:](?P<minute>\d{2})[.\-:](?P<second>\d{2})",
        r"(?P<date>\d{4}-\d{2}-\d{2})",
    ):
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            base = datetime.strptime(match.group("date"), "%Y-%m-%d")
        except ValueError:
            continue
        if {"time", "minute", "second"}.issubset(match.groupdict()):
            try:
                return base.replace(hour=int(match.group("time")), minute=int(match.group("minute")), second=int(match.group("second")))
            except ValueError:
                return base
        return base
    return None


def _leopard_from_folder(rel_path: str) -> tuple[str | None, str | None]:
    """Leopard name and sex from the top folder, e.g. 'Mapula female/' -> ('Mapula', 'Female')."""
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    label = parts[0] if len(parts) > 1 else None
    if not label:
        return None, None
    low = label.lower()
    sex = "Female" if "female" in low else "Male" if "male" in low else None
    name = re.sub(r"\b(male|female)\b", "", label, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", name).strip()).strip()
    return (name or None), sex


def _exif_to_dict(img: Image.Image) -> dict[str, Any]:
    """EXIF tags flattened to exif_<Tag>, with GPS sub-tags as exif_gps_<Tag>."""
    out: dict[str, Any] = {}
    try:
        exif = img.getexif()
    except Exception:
        exif = None
    if not exif:
        return out
    gps_tag_id = {v: k for k, v in ExifTags.TAGS.items()}.get("GPSInfo")
    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        out[f"exif_{tag_name}"] = value
        if tag_name != "GPSInfo":
            continue
        try:
            gps_ifd = exif.get_ifd(gps_tag_id) if gps_tag_id is not None else None
        except Exception:
            gps_ifd = None
        source = gps_ifd if isinstance(gps_ifd, dict) else value if isinstance(value, dict) else {}
        for gps_key, gps_val in source.items():
            out[f"exif_gps_{ExifTags.GPSTAGS.get(gps_key, str(gps_key))}"] = gps_val
    return out


def _rational_to_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        pass
    try:
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            return float(num) / float(den) if float(den) != 0 else None
    except Exception:
        return None
    return None


def _dms_to_degrees(dms: Any, ref: Any) -> float | None:
    try:
        if not isinstance(dms, (list, tuple)) or len(dms) != 3:
            return None
        parts = [_rational_to_float(v) for v in dms]
        if any(p is None for p in parts):
            return None
        dd = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
        return -dd if str(ref).upper() in ("S", "W") else dd
    except Exception:
        return None


def _lat_lon_from_exif(exif: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = lon = None
    if "exif_gps_GPSLatitude" in exif and "exif_gps_GPSLatitudeRef" in exif:
        lat = _dms_to_degrees(exif["exif_gps_GPSLatitude"], exif["exif_gps_GPSLatitudeRef"])
    if "exif_gps_GPSLongitude" in exif and "exif_gps_GPSLongitudeRef" in exif:
        lon = _dms_to_degrees(exif["exif_gps_GPSLongitude"], exif["exif_gps_GPSLongitudeRef"])
    return lat, lon


def _read_with_exiftool(exiftool: str, raw: bytes, suffix: str) -> dict[str, Any]:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        proc = subprocess.run([exiftool, "-j", "-n", "-G1", *EXIFTOOL_TAGS, str(tmp_path)],
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        payload = json.loads(proc.stdout)
        if not payload or not isinstance(payload[0], dict):
            return {}
        data = dict(payload[0])
        data.pop("SourceFile", None)
        return data
    except Exception:
        return {}
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _lat_lon_from_exiftool(data: dict[str, Any]) -> tuple[float | None, float | None]:
    def first(keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key in data:
                value = _rational_to_float(data.get(key))
                if value is not None:
                    return value
        return None
    return (first(("Composite:GPSLatitude", "EXIF:GPSLatitude", "XMP:GPSLatitude")),
            first(("Composite:GPSLongitude", "EXIF:GPSLongitude", "XMP:GPSLongitude")))


def _choose_timestamp(rel_path: str, exif: dict[str, Any], exiftool: dict[str, Any], zip_time: str | None) -> tuple[str | None, str | None]:
    """Most trustworthy date available: GPS stamp, then filename, then EXIF/XMP dates,
    then file times. Which one was used is kept, because step 6 depends on it."""
    gps_dt = _parse_gps_timestamp(exif.get("exif_gps_GPSDateStamp"), exif.get("exif_gps_GPSTimeStamp"))
    if gps_dt is not None:
        return gps_dt.isoformat(sep=" "), "exif_gps_datestamp"
    filename_dt = _datetime_from_filename(rel_path)
    if filename_dt is not None:
        return filename_dt.isoformat(sep=" "), "filename_datetime"
    for source, value in (
        ("exif_datetime_original", exif.get("exif_DateTimeOriginal")),
        ("exif_datetime_digitized", exif.get("exif_DateTimeDigitized")),
        ("exiftool_datetime_original", exiftool.get("EXIF:DateTimeOriginal")),
        ("exiftool_datetime_digitized", exiftool.get("EXIF:DateTimeDigitized")),
        ("exiftool_xmp_create_date", exiftool.get("XMP:CreateDate")),
        ("exif_datetime", exif.get("exif_DateTime")),
        ("exiftool_datetime", exiftool.get("EXIF:DateTime")),
        ("exiftool_xmp_modify_date", exiftool.get("XMP:ModifyDate")),
        ("zip_modified_time", zip_time),
        ("file_modify_time", exiftool.get("File:FileModifyDate")),
    ):
        if _parse_datetime(value) is not None:
            return _safe_str(value), source
    return None, None


def _gps_status(lat: Any, lon: Any, exif: dict[str, Any], exiftool: dict[str, Any]) -> str:
    if lat is not None and lon is not None:
        return "gps_found"
    if any(k.startswith("exif_gps_") for k in exif):
        return "incomplete_exif_gps"
    if any("GPS" in k for k in exiftool):
        return "incomplete_exiftool_gps"
    if exif.get("exif_GPSInfo") is not None:
        return "gps_pointer_without_coordinates"
    return "no_gps_metadata"


def extract_photo_rows(zip_path: Path, exiftool: str) -> list[dict[str, Any]]:
    """One metadata row per image. GPS is read with Pillow, and with ExifTool where
    Pillow finds none."""
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            rel_path = info.filename
            ext = Path(rel_path).suffix.lower()
            if info.is_dir() or ext not in IMAGE_EXTS:
                continue
            leopard, sex = _leopard_from_folder(rel_path)
            raw = z.read(info)
            try:
                zip_time = datetime(*info.date_time).isoformat(sep=" ")
            except Exception:
                zip_time = None

            width = height = lat = lon = gps_source = None
            exif: dict[str, Any] = {}
            exiftool_data: dict[str, Any] = {}
            try:
                with Image.open(io.BytesIO(raw)) as img:
                    width, height = img.size
                    exif = _exif_to_dict(img)
                    lat, lon = _lat_lon_from_exif(exif)
                    if lat is not None and lon is not None:
                        gps_source = "exif"
            except Exception:
                pass
            if lat is None or lon is None:
                exiftool_data = _read_with_exiftool(exiftool, raw, ext)
                et_lat, et_lon = _lat_lon_from_exiftool(exiftool_data)
                if et_lat is not None and et_lon is not None:
                    lat, lon, gps_source = et_lat, et_lon, "exiftool"

            image_datetime, datetime_source = _choose_timestamp(rel_path, exif, exiftool_data, zip_time)
            parsed = _parse_datetime(image_datetime)
            note_text, filename_note, note_tags, note_sources = _collect_notes(rel_path, exif, exiftool_data)
            path_parts = rel_path.replace("\\", "/")
            row: dict[str, Any] = {
                "zip_file": str(zip_path),
                "relative_path": rel_path,
                "folder_top": path_parts.split("/")[0] if "/" in path_parts else "",
                "file_name": Path(rel_path).name,
                "file_ext": ext,
                "file_size_bytes": info.file_size,
                "zip_modified_time": zip_time,
                "leopard_name": leopard,
                "sex": sex,
                "image_width_px": width,
                "image_height_px": height,
                "exif_datetime_original": _safe_str(exif.get("exif_DateTimeOriginal")),
                "exif_datetime": _safe_str(exif.get("exif_DateTime")),
                "image_datetime": image_datetime,
                "image_date": parsed.date().isoformat() if parsed else None,
                "image_time": parsed.strftime("%H:%M:%S") if parsed else None,
                "image_datetime_source": datetime_source,
                "exif_make": _safe_str(exif.get("exif_Make")),
                "exif_model": _safe_str(exif.get("exif_Model")),
                "exif_lens_model": _safe_str(exif.get("exif_LensModel")),
                "exif_user_comment": _safe_str(exif.get("exif_UserComment")),
                "metadata_note_text": note_text,
                "filename_note_text": filename_note,
                "note_tags": note_tags,
                "note_sources": note_sources,
                "gps_lat": lat,
                "gps_lon": lon,
                "gps_source": gps_source,
                "gps_status": _gps_status(lat, lon, exif, exiftool_data),
            }
            existing = {k.lower() for k in row}
            for k, v in exif.items():
                if k.lower() not in existing:
                    row[k] = _safe_str(v) if not isinstance(v, (int, float)) else v
            rows.append(row)
    return rows


def with_gps(df: pd.DataFrame) -> pd.DataFrame:
    return df[pd.to_numeric(df["gps_lat"], errors="coerce").notna() & pd.to_numeric(df["gps_lon"], errors="coerce").notna()].copy()


def step_photo_metadata() -> None:
    exiftool = find_exiftool()
    photos = pd.DataFrame(extract_photo_rows(PHOTO_ZIP, exiftool))

    # Sex from the lineage register where the folder name gives none.
    lineage = read_lineage()
    sex_by_key = {}
    for name, sex in zip(lineage["leopard_name"], lineage["sex"]):
        key, sex = name_key(name) if _safe_str(name) else None, _safe_str(sex)
        if key and sex and sex.title() in {"Female", "Male"}:
            sex_by_key[key] = sex.title()
    keys = photos["leopard_name"].map(lambda n: name_key(n) if _safe_str(n) else None)
    lineage_sex = keys.map(sex_by_key)
    current = photos["sex"].astype("string")
    missing = current.isna() | current.str.strip().eq("")
    photos.loc[missing & lineage_sex.notna(), "sex"] = lineage_sex[missing & lineage_sex.notna()]

    photos.to_csv(table("leopard_images_metadata.csv"), index=False, encoding="utf-8")
    with_gps(photos).to_csv(table("leopard_images_metadata.filtered.csv"), index=False, encoding="utf-8")

    batch = pd.DataFrame(extract_photo_rows(BATCH_ZIP, exiftool))
    with_gps(batch).to_csv(table("leopard_images_metadata.batches.csv"), index=False, encoding="utf-8")
    print(f"  {len(photos)} photos ({len(with_gps(photos))} with GPS); batch {len(batch)} ({len(with_gps(batch))} with GPS)")


# =============================================================================
# Step 2. Match photos to the sightings log
# =============================================================================

def load_sightings() -> pd.DataFrame:
    sightings = pd.read_excel(SIGHTINGS_XLSX, sheet_name="Sheet1")
    sightings["Lat"] = pd.to_numeric(sightings.get("Lat"), errors="coerce")
    sightings["Long"] = pd.to_numeric(sightings.get("Long"), errors="coerce")
    sightings["name_norm"] = sightings.get("Leopard").map(match_name)
    sightings["sheet_sex"] = sightings.get("Leopard").map(sex_from_label)
    sightings["sheet_date"] = pd.to_datetime(dict(year=sightings["Year"], month=sightings["Month"], day=sightings["Day"]), errors="coerce")
    return sightings


def match_to_sightings(photos: pd.DataFrame, sightings: pd.DataFrame) -> pd.DataFrame:
    """Copy the nearest same-leopard sighting's fields onto each photo; matches within
    50 m count as reliable (match_use_for_analysis)."""
    by_name = {name: group.dropna(subset=["Lat", "Long"]).reset_index(drop=True)
               for name, group in sightings.groupby("name_norm", dropna=True)}
    rows: list[dict[str, object]] = []
    for row in photos.itertuples(index=False):
        out = row._asdict()
        out.update({"sheet_match_distance_m": None, "sheet_match_quality": "unmatched", "match_use_for_analysis": False,
                    "sheet_date": None, "sheet_guide": None, "sheet_behaviour": None, "sheet_notes": None,
                    "sheet_other_leopard": None, "sheet_leopard_2": None, "sheet_total_leopards": None})
        candidates = by_name.get(out.get("name_norm"))
        if candidates is not None and not candidates.empty and pd.notna(out.get("gps_lat")) and pd.notna(out.get("gps_lon")):
            distances = candidates.apply(lambda c: distance_m(float(out["gps_lat"]), float(out["gps_lon"]), float(c["Lat"]), float(c["Long"])), axis=1)
            best_idx = distances.idxmin()
            best, best_distance = candidates.loc[best_idx], float(distances.loc[best_idx])
            quality = "exact" if best_distance <= 5 else "near" if best_distance <= 50 else "loose" if best_distance <= 250 else "far"
            out.update({
                "sheet_match_distance_m": round(best_distance, 2),
                "sheet_match_quality": quality,
                "match_use_for_analysis": best_distance <= HIGH_CONFIDENCE_DISTANCE_M,
                "sheet_date": best["sheet_date"].date().isoformat() if pd.notna(best["sheet_date"]) else None,
                "sheet_guide": best.get("Guide"),
                "sheet_behaviour": best.get("Behaviour"),
                "sheet_notes": best.get("Notes"),
                "sheet_other_leopard": best.get("Other_Leopard"),
                "sheet_leopard_2": best.get("Leopard_2"),
                "sheet_total_leopards": best.get("Total_leopards"),
            })
            if (not out.get("sex")) and pd.notna(best.get("sheet_sex")):
                out["sex"] = best.get("sheet_sex")
        rows.append(out)
    return pd.DataFrame(rows).drop(columns=["name_norm"], errors="ignore")


def step_sightings_match() -> None:
    photos = read_table("leopard_images_metadata.filtered.csv")
    photos["gps_lat"] = pd.to_numeric(photos.get("gps_lat"), errors="coerce")
    photos["gps_lon"] = pd.to_numeric(photos.get("gps_lon"), errors="coerce")
    photos["name_norm"] = photos.get("leopard_name").map(match_name)
    combined = match_to_sightings(photos, load_sightings())
    combined.to_csv(table("leopard_images_metadata_combined.csv"), index=False)
    print(f"  {combined['sheet_match_distance_m'].notna().sum()} photos matched to a sighting, "
          f"{combined['match_use_for_analysis'].fillna(False).sum()} within {HIGH_CONFIDENCE_DISTANCE_M:.0f} m")


# =============================================================================
# Step 3. Batch photos and their manual identities
# =============================================================================

# Spellings in file_names_list.csv -> lineage names, and sexes where the lineage has none.
LIST_NAME_ALIASES = {
    "rose": "Rose", "montle": "Montle", "lesedi": "Lesedi", "mojelemane": "Mojeremane", "mojeremane": "Mojeremane",
    "puelo": "Puelo", "mooka": "Mooka", "maun rd femle": "Maun road", "maun rd female": "Maun road",
    "maun road": "Maun road", "maun road female": "Maun road", "mongololo": "Mongololo", "rradikolobe": "Rradikolobe",
    "river road": "River road", "mfana": "Mfana", "mfana male": "Mfana", "rradiphala": "Rradiphala",
    "kigelia ma e": "Kigelia", "kigelia male": "Kigelia", "kigelia": "Kigelia", "morula male": "Morula",
    "morula": "Morula", "tswaletswale": "Tswaletswale",
}
LIST_SEX_BY_NAME = {
    "Rose": "Female", "Mojeremane": "Male", "Mooka": "Female", "Maun road": "Female", "Mongololo": "Female",
    "Rradikolobe": "Male", "River road": "Female", "Mfana": "Male", "Rradiphala": "Male", "Kigelia": "Male",
    "Morula": "Male", "Tswaletswale": "Female",
}
IDENTITY_COLUMNS = ["identity_source", "identity_inference_confidence", "identity_reference_distance_m",
                    "identity_reference_match_count", "identity_reference_second_choice", "identity_reference_second_distance_m"]


def _clean_token(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("?", " ").replace("_", " ")).strip().lower()


def _lineage_sex_by_alias() -> dict[str, str]:
    sex_map: dict[str, str] = {}
    lineage = read_lineage()
    for raw_name, raw_sex in zip(lineage["leopard_name"], lineage["sex"]):
        lowered = _clean_token(str(raw_name or "").strip()).replace(" female", "").replace(" male", "").strip()
        if not lowered:
            continue
        sex = str(raw_sex or "").strip().title()
        if sex in {"Female", "Male"}:
            sex_map[LIST_NAME_ALIASES.get(lowered, lowered.title())] = sex
    return sex_map


def _photo_key(text: object) -> str:
    """File label key tolerant of I/1, L/1 and O/0 confusions in hand-typed lists."""
    folded = re.sub(r"[^A-Z0-9]+", "", Path(str(text).strip()).stem.upper())
    return folded.translate(str.maketrans({"I": "1", "L": "1", "O": "0"}))


def load_photo_identities(lineage_sex: dict[str, str]) -> dict[str, dict[str, object]]:
    """file_names_list.csv: photo -> leopards in it (the first is the primary identity)."""
    identities: dict[str, dict[str, object]] = {}
    with FILE_NAMES_LIST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row or not str(row[0]).strip():
                continue
            names: list[str] = []
            sexes: list[str | None] = []
            for cell in row[1:]:
                for token in re.split(r",|&", str(cell or "").strip()):
                    lowered = _clean_token(token.strip())
                    if not lowered:
                        continue
                    sex = "Female" if "female" in lowered else "Male" if "male" in lowered else None
                    name = LIST_NAME_ALIASES.get(lowered.replace(" female", "").replace(" male", "").strip(),
                                                 lowered.replace(" female", "").replace(" male", "").strip().title())
                    if name not in names:
                        names.append(name)
                        sexes.append(lineage_sex.get(name) or LIST_SEX_BY_NAME.get(name, sex))
            if names:
                identities[_photo_key(row[0])] = {"all_names": names, "primary_name": names[0], "primary_sex": sexes[0] or "Unknown"}
    return identities


def apply_photo_identities(df: pd.DataFrame, identities: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Name batch photos from the list and mark them as manually identified and usable."""
    work = df.copy()
    keys = work["file_name"].map(_photo_key)
    mask = keys.isin(identities) & work["zip_file"].fillna("").astype(str).str.endswith(BATCH_ZIP_NAME)
    if not mask.any():
        return work
    found = keys[mask].map(identities)
    work.loc[mask, "leopard_name"] = found.map(lambda item: item["primary_name"])
    work.loc[mask, "sex"] = found.map(lambda item: item["primary_sex"])
    if "metadata_note_text" in work.columns:
        merged = []
        for current, note in zip(work.loc[mask, "metadata_note_text"].fillna("").astype(str).str.strip(),
                                 found.map(lambda item: f"Filename identities: {', '.join(item['all_names'])}")):
            merged.append(note if not current else current if note.lower() in current.lower() else f"{current} | {note}")
        work.loc[mask, "metadata_note_text"] = pd.Series(merged, index=found.index, dtype="object")
    updates = {
        "identity_source": "file_names_list", "identity_inference_confidence": "manual",
        "identity_reference_distance_m": pd.NA, "identity_reference_match_count": pd.NA,
        "identity_reference_second_choice": pd.NA, "identity_reference_second_distance_m": pd.NA,
        "sheet_match_quality": "manual", "match_use_for_analysis": True,
    }
    for column, value in updates.items():
        if column in work.columns:
            work.loc[mask, column] = value
    if "sheet_other_leopard" in work.columns:
        work.loc[mask, "sheet_other_leopard"] = found.map(lambda item: "Yes" if len(item["all_names"]) > 1 else "No")
    if "sheet_leopard_2" in work.columns:
        work.loc[mask, "sheet_leopard_2"] = found.map(lambda item: ", ".join(item["all_names"][1:]) if len(item["all_names"]) > 1 else pd.NA)
    if "sheet_total_leopards" in work.columns:
        work.loc[mask, "sheet_total_leopards"] = found.map(lambda item: len(item["all_names"]))
    return work


def one_row_per_leopard(df: pd.DataFrame, identities: dict[str, dict[str, object]], lineage_sex: dict[str, str]) -> pd.DataFrame:
    """A batch photo showing several leopards becomes one record per leopard."""
    keys = df["file_name"].map(_photo_key)
    mask = keys.isin(identities) & df["zip_file"].fillna("").astype(str).str.endswith(BATCH_ZIP_NAME)
    extra: list[dict[str, object]] = []
    for idx, row in df.loc[mask].iterrows():
        names = list(identities[keys.at[idx]]["all_names"])
        for name in names[1:]:
            clone = row.copy()
            clone["leopard_name"] = name
            clone["sex"] = lineage_sex.get(name) or LIST_SEX_BY_NAME.get(name, "Unknown")
            clone["sheet_other_leopard"] = "Yes"
            companions = [n for n in names if n != name]
            clone["sheet_leopard_2"] = ", ".join(companions) if companions else pd.NA
            clone["sheet_total_leopards"] = len(names)
            extra.append(clone.to_dict())
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True) if extra else df


def backfill_sheet_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Photos carrying the archive export date take the date of a reliable sighting match."""
    out = df.copy()
    backfill = (
        out["match_use_for_analysis"].fillna(False)
        & out["sheet_date"].notna()
        & out["image_date"].astype("string").eq(ARCHIVE_EXPORT_DATE.date().isoformat())
        & out["image_datetime_source"].astype("string").eq("exif_datetime")
    )
    out.loc[backfill, "image_date"] = out.loc[backfill, "sheet_date"]
    out.loc[backfill, "image_time"] = pd.NA
    out.loc[backfill, "image_datetime"] = out.loc[backfill, "sheet_date"]
    out.loc[backfill, "image_datetime_source"] = "sheet_date_match"
    return out


def step_batch_identities() -> None:
    lineage_sex = _lineage_sex_by_alias()
    identities = load_photo_identities(lineage_sex)

    combined = read_table("leopard_images_metadata_combined.csv", low_memory=False)
    position = combined.columns.get_loc("sheet_match_distance_m")
    for offset, column in enumerate(IDENTITY_COLUMNS):
        combined.insert(position + offset, column, pd.NA)
    combined["identity_source"] = "path"

    # Batch photos have no leopard folder, so step 2 never saw them: name them from the
    # list, then match each to its own leopard's sightings as step 2 does for the rest.
    batch = read_table("leopard_images_metadata.batches.csv", low_memory=False)
    batch = batch[batch["zip_file"].fillna("").astype(str).str.endswith(BATCH_ZIP_NAME)].copy()
    for column in ("leopard_name", "sex", "metadata_note_text"):
        batch[column] = batch[column].astype("object")
    batch = apply_photo_identities(batch, identities)
    batch["gps_lat"] = pd.to_numeric(batch["gps_lat"], errors="coerce")
    batch["gps_lon"] = pd.to_numeric(batch["gps_lon"], errors="coerce")
    batch["name_norm"] = batch["leopard_name"].map(match_name)
    batch = match_to_sightings(batch, load_sightings())

    combined = pd.concat([combined, batch.reindex(columns=combined.columns)], ignore_index=True)
    combined = apply_photo_identities(combined, identities)
    corrected = one_row_per_leopard(backfill_sheet_dates(combined), identities, lineage_sex)

    combined.to_csv(table("leopard_images_metadata_combined.csv"), index=False)
    corrected.to_csv(table("merged_leopard_images_metadata.filtered.corrected.csv"), index=False)
    print(f"  {len(batch)} batch photos named from the list; {len(corrected)} photo records in all")


# =============================================================================
# Step 4. Encounter events
# =============================================================================

def _local_metres(lat: float, lon: float, lat0: float) -> tuple[float, float]:
    """Equirectangular metres; adequate at the 250 m scale of event grouping."""
    return lon * 111_320.0 * math.cos(math.radians(lat0)), lat * 111_132.0


def _clusters(group: pd.DataFrame) -> list[list[int]]:
    """Single-linkage groups of points within EVENT_DISTANCE_M of each other."""
    coords = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
    if len(coords) <= 1:
        return [list(range(len(coords)))]
    lat0 = float(group["gps_lat"].mean())
    pts = [_local_metres(lat, lon, lat0) for lat, lon in coords]
    parent = list(range(len(pts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) <= EVENT_DISTANCE_M:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for idx in range(len(pts)):
        groups.setdefault(find(idx), []).append(idx)
    return list(groups.values())


def _max_pair_distance(group: pd.DataFrame) -> float:
    coords = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
    if len(coords) <= 1:
        return 0.0
    lat0 = float(group["gps_lat"].mean())
    pts = [_local_metres(lat, lon, lat0) for lat, lon in coords]
    return float(max(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                     for i in range(len(pts)) for j in range(i + 1, len(pts))))


def _join_unique(series: pd.Series, split: bool = False) -> str | None:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna():
        for part in (str(value).split(";") if split else [str(value)]):
            text = part.strip()
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                values.append(text)
    return ("; " if split else " | ").join(values) if values else None


def step_encounter_events() -> None:
    df = read_table("merged_leopard_images_metadata.filtered.corrected.csv")
    df["leopard_name"] = df["leopard_name"].fillna("Unknown").replace(MERGED_IDENTITIES)
    df["sex"] = df["sex"].fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df["gps_lat"], errors="coerce")
    df["gps_lon"] = pd.to_numeric(df["gps_lon"], errors="coerce")
    df["image_date_parsed"] = pd.to_datetime(df["image_date"], errors="coerce", dayfirst=True)
    df["image_time_parsed"] = pd.to_datetime(df["image_time"], errors="coerce", format="%H:%M:%S")
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", "image_date_parsed"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(resolve_sex)

    rows: list[dict[str, object]] = []
    for (_, image_date), group in df.groupby(["leopard_name", "image_date_parsed"], sort=True, dropna=False):
        group = group.reset_index(drop=True)
        for cluster_number, indices in enumerate(_clusters(group), start=1):
            cluster = group.iloc[indices].copy()
            rep = cluster.sort_values(["image_time_parsed", "file_name"], na_position="last").iloc[0]
            row = rep.to_dict()
            row.update({
                "event_id": f"EVT{len(rows) + 1:05d}",
                "event_cluster_number": cluster_number,
                "n_images_in_event": int(len(cluster)),
                "n_unique_raw_points_in_event": int(len(set(zip(cluster["gps_lat"].astype(float), cluster["gps_lon"].astype(float))))),
                "event_max_pair_distance_m": round(_max_pair_distance(cluster), 2),
                "file_name": str(rep.get("file_name", "") or ""),
                "representative_file_name": str(rep.get("file_name", "") or ""),
                "source_file_names": "; ".join(cluster["file_name"].astype(str).tolist()),
                "gps_lat": float(cluster["gps_lat"].median()),
                "gps_lon": float(cluster["gps_lon"].median()),
                "image_date": pd.Timestamp(image_date).strftime("%d/%m/%Y"),
                "image_time": rep["image_time_parsed"].strftime("%H:%M:%S") if pd.notna(rep.get("image_time_parsed")) else None,
                "metadata_note_text": _join_unique(cluster.get("metadata_note_text", pd.Series(dtype="object"))),
                "note_tags": _join_unique(cluster.get("note_tags", pd.Series(dtype="object")), split=True),
            })
            if "exif_ImageDescription" in cluster.columns:
                row["exif_ImageDescription"] = _join_unique(cluster["exif_ImageDescription"])
            rows.append(row)

    events = pd.DataFrame(rows).drop(columns=["image_date_parsed", "image_time_parsed"], errors="ignore")
    events.to_csv(table("merged_leopard_images_metadata.normalized_events.csv"), index=False)
    print(f"  {len(df)} photo records -> {len(events)} encounter events")


# =============================================================================
# Step 5. Camera summary (Table S1)
# =============================================================================

def step_camera_summary() -> None:
    df = read_table("merged_leopard_images_metadata.normalized_events.csv")
    df["leopard_name"] = df["leopard_name"].fillna("Unknown")
    for column in ("exif_make", "exif_model"):
        df[column] = df.get(column, pd.Series(index=df.index, dtype="object")).fillna("Unknown").replace("", "Unknown")
    cameras = (
        df.groupby(["exif_make", "exif_model"], dropna=False)
        .agg(n_images=("file_name", "size"), n_individuals=("leopard_name", "nunique"))
        .reset_index()
        .sort_values(["n_images", "exif_make", "exif_model"], ascending=[False, True, True])
    )
    cameras.to_csv(table("merged_leopard_images_metadata_camera_summary.csv"), index=False)


# =============================================================================
# Step 6. Date validity
# =============================================================================

def step_date_validity() -> None:
    raw = read_table("leopard_images_metadata.csv", low_memory=False)
    merged = read_table("merged_leopard_images_metadata.filtered.corrected.csv", low_memory=False)
    events = read_table("merged_leopard_images_metadata.normalized_events.csv", low_memory=False)

    def source_counts(df: pd.DataFrame, stage: str) -> pd.DataFrame:
        counts = (df.assign(archive_export_date=is_archive_export_date(df))
                  .groupby(["image_datetime_source", "archive_export_date"], dropna=False).size()
                  .rename("n_records").reset_index())
        counts.insert(0, "stage", stage)
        counts.insert(1, "n_stage_total", len(df))
        return counts

    pd.concat([source_counts(raw, "extracted_images"), source_counts(merged, "gps_images_after_sheet_matching"),
               source_counts(events, "encounter_events")], ignore_index=True).to_csv(table("date_source_counts.csv"), index=False)

    bad = is_archive_export_date(events)
    audit = events.loc[bad, ["event_id", "leopard_name", "sex", "image_date", "image_datetime", "image_datetime_source",
                             "n_images_in_event", "gps_lat", "gps_lon", "file_name"]].copy()
    audit["reason"] = f"EXIF date equals archive export date {ARCHIVE_EXPORT_DATE.date()}"
    audit.to_csv(table("date_audit_archive_export.csv"), index=False)
    events.loc[~bad].to_csv(table("merged_leopard_images_metadata.dated_events.csv"), index=False)
    print(f"  excluded {int(bad.sum())} of {len(events)} events dated {ARCHIVE_EXPORT_DATE.date()} from EXIF")


def read_events(name: str = "merged_leopard_images_metadata.dated_events.csv", date_column: str = "date") -> pd.DataFrame:
    """Events with numeric coordinates, parsed dates and one sex per leopard."""
    df = read_table(name, low_memory=False)
    df["sex"] = df["sex"].fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df["gps_lat"], errors="coerce")
    df["gps_lon"] = pd.to_numeric(df["gps_lon"], errors="coerce")
    df[date_column] = pd.to_datetime(df["image_date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", date_column]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(resolve_sex)
    return df


# =============================================================================
# Step 7. Seasonality
# =============================================================================

def step_seasonality() -> None:
    df = read_table("merged_leopard_images_metadata.dated_events.csv")
    df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce", dayfirst=True)
    df["sex"] = df["sex"].fillna("Unknown")
    df["sex"] = df.groupby("leopard_name")["sex"].transform(resolve_sex)
    df = df.dropna(subset=["image_date"]).copy()
    # Botswana wet season: November to March.
    season = df["image_date"].dt.month.map(lambda m: "Wet season" if m in {11, 12, 1, 2, 3} else "Dry season")
    df["botswana_season"] = pd.Categorical(season, categories=["Wet season", "Dry season"], ordered=True)

    counts = df.groupby("botswana_season", observed=False).size().reset_index(name="n_records").sort_values("botswana_season")
    counts["percent_records"] = 100.0 * counts["n_records"] / counts["n_records"].sum()
    by_sex = (df.groupby(["botswana_season", "sex"], observed=False).size().reset_index(name="n_records")
              .sort_values(["botswana_season", "sex"]))
    counts.to_csv(table("merged_image_counts_by_botswana_season.csv"), index=False)
    by_sex.to_csv(table("merged_image_counts_by_botswana_season_and_sex.csv"), index=False)


# =============================================================================
# Step 8. Inclusion
# =============================================================================

def step_inclusion() -> None:
    df = read_table("merged_leopard_images_metadata.dated_events.csv")
    df["sex"] = df["sex"].fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df["gps_lat"], errors="coerce")
    df["gps_lon"] = pd.to_numeric(df["gps_lon"], errors="coerce")
    df["image_date_parsed"] = pd.to_datetime(df["image_date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", "image_date_parsed"]).copy()

    # Dates outside the study period mean a camera clock error, not an encounter.
    impossible = (df["image_date_parsed"] > pd.Timestamp.today().normalize()) | (df["image_date_parsed"] < pd.Timestamp(STUDY_START))
    rejected = df.loc[impossible, ["leopard_name", "image_date", "image_datetime_source", "file_name"]].copy()
    rejected["reason"] = "date outside study period"
    rejected.to_csv(table("date_audit_rejected.csv"), index=False)
    df = df.loc[~impossible].copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(resolve_sex)

    individuals = (
        df.groupby(["leopard_name", "sex"], dropna=False)
        .agg(n_images=("file_name", "size"), n_unique_dates=("image_date_parsed", "nunique"),
             first_seen=("image_date_parsed", "min"), last_seen=("image_date_parsed", "max"))
        .reset_index()
    )
    unique_points = df.groupby("leopard_name").apply(
        lambda g: int(len(set(zip(g["gps_lat"].astype(float), g["gps_lon"].astype(float)))))).rename("n_unique_points").reset_index()
    individuals = annotate_age(individuals.merge(unique_points, on="leopard_name", how="left"))
    individuals["include_spatial_subset"] = (
        (individuals["n_unique_points"] >= MIN_UNIQUE_POINTS)
        & (individuals["n_unique_dates"] >= MIN_UNIQUE_DATES)
        & individuals["is_territorial_age"]
    )
    for column in ("first_seen", "last_seen"):
        individuals[column] = individuals[column].dt.date.astype("string")

    keep = set(individuals.loc[individuals["include_spatial_subset"], "leopard_name"])
    df[df["leopard_name"].isin(keep)].drop(columns=["image_date_parsed"]).to_csv(
        table("merged_leopard_images_metadata.spatial_subset.csv"), index=False)
    individuals.to_csv(table("merged_leopard_images_metadata.spatial_subset_individuals.csv"), index=False)
    print(f"  {len(keep)} individuals meet the inclusion criteria over the whole study period")


# =============================================================================
# Step 9. Pooled home ranges
# =============================================================================

def step_pooled_home_ranges() -> None:
    df = read_table("merged_leopard_images_metadata.spatial_subset.csv", low_memory=False)
    df["sex"] = df["sex"].fillna("Unknown")
    df["leopard_name"] = df["leopard_name"].fillna("Unknown")
    df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce", dayfirst=True)
    df["gps_lat"] = pd.to_numeric(df["gps_lat"], errors="coerce")
    df["gps_lon"] = pd.to_numeric(df["gps_lon"], errors="coerce")
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(resolve_sex)

    rows = []
    for leopard, group in df.groupby("leopard_name", dropna=False):
        points = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
        hull_95, hull_100 = mcp(points, 95.0), mcp(points, 100.0)
        rows.append({
            "leopard_name": leopard,
            "sex": str(group["sex"].iloc[0]),
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
            "home_range_km2_mcp": hull_95["area_km2"],
        })
    summary = pd.DataFrame(rows).sort_values(["home_range_km2_mcp", "n_images"], ascending=[False, False], na_position="last").reset_index(drop=True)
    for column in ("first_seen", "last_seen"):
        summary[column] = pd.to_datetime(summary[column], errors="coerce").dt.date.astype("string")
    summary.to_csv(table("merged_image_home_ranges_mcp_summary.csv"), index=False)


# =============================================================================
# Step 10. Time windows
# =============================================================================

def window_individuals(events: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    sub = events
    if start is not None:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end is not None:
        sub = sub[sub["date"] <= pd.Timestamp(end)]
    rows = []
    for (name, sex), group in sub.groupby(["leopard_name", "sex"], dropna=False):
        points = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
        hull = mcp(points, MCP_PERCENT)
        rows.append({
            "leopard_name": name, "sex": sex, "n_events": int(len(group)),
            "n_unique_points": int(hull["n_unique_points"]), "n_unique_dates": int(group["date"].nunique()),
            "first_seen": group["date"].min(), "last_seen": group["date"].max(),
            "n_points_dropped": int(hull["n_points_dropped"]), "mcp95_km2": hull["area_km2"],
            "mcp100_km2": mcp(points, 100.0)["area_km2"], "ring_latlon": hull["ring_latlon"],
        })
    out = annotate_age(pd.DataFrame(rows))
    out["include"] = (
        (out["n_unique_points"] >= MIN_UNIQUE_POINTS)
        & (out["n_unique_dates"] >= MIN_UNIQUE_DATES)
        & out["mcp95_km2"].notna()
        & out["sex"].isin(["Female", "Male"])
        & out["is_territorial_age"]
    )
    return out


def window_pairs(individuals: pd.DataFrame) -> pd.DataFrame:
    """Every pair of included leopards, overlapping or not."""
    kept = individuals[individuals["include"]].copy()
    polys = {row.leopard_name: polygon_from_ring(row.ring_latlon) for row in kept.itertuples()}
    meta = kept.set_index("leopard_name")
    rows = []
    for a, b in combinations(sorted(polys), 2):
        inter = polys[a].intersection(polys[b])
        inter_km2 = (inter.area / 1e6) if not inter.is_empty else 0.0
        union = polys[a].union(polys[b])
        union_km2 = (union.area / 1e6) if not union.is_empty else float("nan")
        area_a, area_b = polys[a].area / 1e6, polys[b].area / 1e6
        # How long both animals were on record at once.
        shared_start = max(meta.at[a, "first_seen"], meta.at[b, "first_seen"])
        shared_end = min(meta.at[a, "last_seen"], meta.at[b, "last_seen"])
        rows.append({
            "leopard_a": a, "sex_a": meta.at[a, "sex"], "area_a_km2": area_a,
            "leopard_b": b, "sex_b": meta.at[b, "sex"], "area_b_km2": area_b,
            "pair_sex": pair_label(meta.at[a, "sex"], meta.at[b, "sex"]),
            "overlaps": int(inter_km2 > 0),
            "overlap_km2": inter_km2,
            "union_km2": union_km2,
            "jaccard_overlap": (inter_km2 / union_km2) if union_km2 else float("nan"),
            # Size-normalised: not inflated by larger polygons simply having more room to meet.
            "prop_overlap": 2.0 * inter_km2 / (area_a + area_b) if (area_a + area_b) else float("nan"),
            "pct_of_a_overlapped": (inter_km2 / area_a * 100.0) if area_a else float("nan"),
            "pct_of_b_overlapped": (inter_km2 / area_b * 100.0) if area_b else float("nan"),
            "mean_polygon_area_km2": (area_a + area_b) / 2.0,
            "temporal_cooccurrence_days": int((shared_end - shared_start).days),
        })
    return pd.DataFrame(rows).sort_values(["overlap_km2"], ascending=False).reset_index(drop=True)


def window_sex_stats(individuals: pd.DataFrame, window: str) -> dict[str, object]:
    kept = individuals[individuals["include"]]
    female = kept.loc[kept["sex"] == "Female", "mcp95_km2"].to_numpy(float)
    male = kept.loc[kept["sex"] == "Male", "mcp95_km2"].to_numpy(float)
    if len(female) == 0 or len(male) == 0:
        return {"window": window}
    test = stats.mannwhitneyu(male, female, alternative="two-sided")
    med_diff, ci_lo, ci_hi = bootstrap_median_diff(male, female)
    return {
        "window": window, "n_female": int(len(female)), "n_male": int(len(male)),
        "median_female_km2": float(pd.Series(female).median()), "median_male_km2": float(pd.Series(male).median()),
        "mean_female_km2": float(pd.Series(female).mean()), "mean_male_km2": float(pd.Series(male).mean()),
        "mann_whitney_u": float(test.statistic), "p_value": float(test.pvalue),
        "cliffs_delta_male_minus_female": cliffs_delta(male, female),
        "median_diff_male_minus_female_km2": med_diff, "median_diff_ci95_low_km2": ci_lo, "median_diff_ci95_high_km2": ci_hi,
    }


def step_time_windows() -> None:
    """Pooled polygons can contain territory shifts and turnover, so overlap of two
    long-term polygons is not evidence of simultaneous use. Ranges and overlap are
    therefore rebuilt within each window; the 24-month window is the primary analysis."""
    events = read_events()
    summaries, sex_stats = [], []
    for label, (start, end) in WINDOWS.items():
        individuals = window_individuals(events, start, end)
        pairs = window_pairs(individuals)
        kept = individuals[individuals["include"]]
        sex_stats.append(window_sex_stats(individuals, label))

        features = [geojson_feature(row.ring_latlon, {
            "leopard_name": row.leopard_name, "sex": row.sex, "window": label, "mcp_percent": MCP_PERCENT,
            "n_unique_points": int(row.n_unique_points), "n_points_dropped": int(row.n_points_dropped),
            "home_range_km2_mcp": float(row.mcp95_km2)}) for row in kept.itertuples()]
        with open(FIGS / f"merged_image_home_ranges_mcp95__{label}.geojson", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

        points = events
        if start is not None:
            points = points[points["date"] >= pd.Timestamp(start)]
        if end is not None:
            points = points[points["date"] <= pd.Timestamp(end)]
        points[points["leopard_name"].isin(set(kept["leopard_name"]))].drop(columns=["date"]).to_csv(
            table(f"merged_leopard_images_metadata.window_subset__{label}.csv"), index=False)

        out = individuals.drop(columns=["ring_latlon"]).copy()
        for column in ("first_seen", "last_seen"):
            out[column] = pd.to_datetime(out[column]).dt.date.astype("string")
        out.to_csv(table(f"merged_image_window_individuals__{label}.csv"), index=False)
        pairs.to_csv(table(f"merged_image_overlap_pairs__{label}.csv"), index=False)

        summaries.append({
            "window": label, "is_primary": label == PRIMARY_WINDOW,
            "start": start or str(events["date"].min().date()), "end": end or str(events["date"].max().date()),
            "n_individuals": int(len(kept)), "n_female": int((kept["sex"] == "Female").sum()),
            "n_male": int((kept["sex"] == "Male").sum()), "n_dyads": int(len(pairs)),
            "n_dyads_overlapping": int(pairs["overlaps"].sum()) if len(pairs) else 0,
            "min_cooccurrence_days": int(pairs["temporal_cooccurrence_days"].min()) if len(pairs) else 0,
            "median_cooccurrence_days": float(pairs["temporal_cooccurrence_days"].median()) if len(pairs) else 0.0,
        })
        print(f"  {label}: {len(kept)} individuals, {len(pairs)} dyads")
    pd.DataFrame(summaries).to_csv(table("merged_image_window_summary.csv"), index=False)
    pd.DataFrame(sex_stats).to_csv(table("merged_image_window_sex_stats.csv"), index=False)


# =============================================================================
# Step 11. Overlap models
# =============================================================================

def _design(pair_types: pd.Series, log_area: np.ndarray, with_pair_type: bool) -> np.ndarray:
    columns = [np.ones(len(pair_types)), log_area]
    if with_pair_type:  # Female-Female is the reference level
        columns += [(pair_types == level).to_numpy(float) for level in PAIR_ORDER[1:]]
    return np.column_stack(columns)


def _deviance(y: np.ndarray, X: np.ndarray) -> float:
    try:
        return float(sm.GLM(y, X, family=sm.families.Binomial()).fit().deviance)
    except Exception:
        return float("nan")


def _lr_statistic(y: np.ndarray, pair_types: pd.Series, log_area: np.ndarray) -> float:
    """Likelihood ratio for pair type, over and above polygon size."""
    return _deviance(y, _design(pair_types, log_area, False)) - _deviance(y, _design(pair_types, log_area, True))


def _group_stat(values: np.ndarray, pair_types: pd.Series, label: str, how: str) -> float:
    sel = (pair_types == label).to_numpy()
    if not sel.any():
        return float("nan")
    return float(np.mean(values[sel]) if how == "mean" else np.median(values[sel]))


def _individuals_and_sexes(pairs: pd.DataFrame, columns: tuple[str, str] = ("name", "sex")) -> pd.DataFrame:
    name, sex = columns
    return (pd.concat([pairs[["leopard_a", "sex_a"]].rename(columns={"leopard_a": name, "sex_a": sex}),
                       pairs[["leopard_b", "sex_b"]].rename(columns={"leopard_b": name, "sex_b": sex})])
            .drop_duplicates().sort_values(name))


def permutation_tests(pairs: pd.DataFrame, response: str, how: str, part: str, window: str) -> list[dict[str, object]]:
    """Node-label permutation: sexes are shuffled across individuals and every dyad's
    pair type rebuilt, which respects each animal appearing in many dyads."""
    y = pairs[response].to_numpy(float)
    log_area = np.log(pairs["mean_polygon_area_km2"].to_numpy(float))
    pair_types = pairs["pair_sex"].reset_index(drop=True)
    dyad_a, dyad_b = pairs["leopard_a"].to_numpy(), pairs["leopard_b"].to_numpy()
    ind = _individuals_and_sexes(pairs)
    individuals, sexes = ind["name"].tolist(), ind["sex"].to_numpy()
    contrasts = [("Female-Female", "Female-Male"), ("Female-Female", "Male-Male"), ("Female-Male", "Male-Male")]

    obs_lr = _lr_statistic(y, pair_types, log_area)
    observed = {(a, b): _group_stat(y, pair_types, a, how) - _group_stat(y, pair_types, b, how) for a, b in contrasts}
    rng = np.random.default_rng(PERMUTATION_SEED)
    lr_ge, n_valid = 0, 0
    contrast_ge = {key: 0 for key in observed}
    for _ in range(N_PERMUTATIONS):
        lookup = dict(zip(individuals, rng.permutation(sexes)))
        perm = pd.Series([pair_label(lookup[a], lookup[b]) for a, b in zip(dyad_a, dyad_b)])
        lr = _lr_statistic(y, perm, log_area)
        if np.isnan(lr):
            continue
        n_valid += 1
        lr_ge += lr >= obs_lr - 1e-12
        for (a, b), obs in observed.items():
            diff = _group_stat(y, perm, a, how) - _group_stat(y, perm, b, how)
            if not np.isnan(diff) and abs(diff) >= abs(obs) - 1e-12:
                contrast_ge[(a, b)] += 1

    # Add-one correction: a permutation p-value is never exactly zero.
    p_lr = (lr_ge + 1) / (n_valid + 1)
    rows: list[dict[str, object]] = [{
        "window": window, "part": part, "test": "omnibus_pair_type_LR", "group_a": None, "group_b": None,
        "n_dyads": int(len(pairs)), "statistic": obs_lr, "p_permutation": p_lr, "p_permutation_adjusted": p_lr,
        "n_permutations": n_valid,
    }]
    contrast_rows, raw_p = [], []
    for (a, b), obs in observed.items():
        p = (contrast_ge[(a, b)] + 1) / (n_valid + 1)
        raw_p.append(p)
        contrast_rows.append({
            "window": window, "part": part, "test": f"contrast_{how}_difference", "group_a": a, "group_b": b,
            "n_a": int((pair_types == a).sum()), "n_b": int((pair_types == b).sum()),
            "stat_a": _group_stat(y, pair_types, a, how), "stat_b": _group_stat(y, pair_types, b, how),
            "statistic": obs, "p_permutation": p, "n_permutations": n_valid,
        })
    for row, p_adj in zip(contrast_rows, multipletests(raw_p, method="holm")[1]):
        row["p_permutation_adjusted"] = float(p_adj)
    return rows + contrast_rows


def overlap_descriptives(pairs: pd.DataFrame, window: str) -> pd.DataFrame:
    out = pairs.groupby("pair_sex").agg(
        n_dyads=("overlaps", "size"),
        n_overlapping=("overlaps", "sum"),
        median_overlap_km2_nonzero=("overlap_km2", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else np.nan),
        median_prop_overlap_nonzero=("prop_overlap", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else np.nan),
        mean_polygon_area_km2=("mean_polygon_area_km2", "mean"),
        median_cooccurrence_days=("temporal_cooccurrence_days", "median"),
    ).reset_index()
    out.insert(0, "window", window)
    out["pct_overlapping"] = 100.0 * out["n_overlapping"] / out["n_dyads"]
    out["median_pct_overlap_nonzero"] = 100.0 * out["median_prop_overlap_nonzero"]
    return out


def step_overlap_models() -> None:
    """Part 1: does a dyad overlap at all (every dyad, zeros included)? Part 2: given
    overlap, how much (size-normalised)? Both by pair type, with permutation inference."""
    results, descriptives = [], []
    for window in WINDOWS:
        pairs = read_table(f"merged_image_overlap_pairs__{window}.csv")
        pairs = pairs[pairs["pair_sex"].isin(PAIR_ORDER)].reset_index(drop=True)
        descriptives.append(overlap_descriptives(pairs, window))
        results += permutation_tests(pairs, "overlaps", "mean", "part1_any_overlap", window)
        nonzero = pairs[pairs["overlaps"] == 1].reset_index(drop=True)
        if len(nonzero) >= 10:
            results += permutation_tests(nonzero, "prop_overlap", "median", "part2_extent_given_overlap", window)
        print(f"  {window}: {len(pairs)} dyads, {int(pairs['overlaps'].sum())} overlapping")
    pd.DataFrame(results).to_csv(table("overlap_model_results.csv"), index=False)
    pd.concat(descriptives, ignore_index=True).to_csv(table("overlap_descriptives_by_pair_type.csv"), index=False)


# =============================================================================
# Step 12. Null model (Figure 6)
# =============================================================================

def _null_regions() -> dict[str, BaseGeometry]:
    """Areas within which a range centroid may be placed. The primary one is NG31 within
    1 km of any GPS-referenced encounter (every individual and year, since locations are
    genuine even where dates are not). The 2013 road file misses much of the network
    driven in 2024-2026, so road-based regions are sensitivity analyses only."""
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32734", always_xy=True).transform
    ng31 = transform(to_utm, max(polygonize(unary_union([LineString(l) for l in load_ng31_lines()])), key=lambda p: p.area))
    events = read_table("merged_leopard_images_metadata.normalized_events.csv", low_memory=False)
    lon, lat = pd.to_numeric(events["gps_lon"], errors="coerce"), pd.to_numeric(events["gps_lat"], errors="coerce")
    ok = lon.notna() & lat.notna()
    x, y = to_utm(lon[ok].to_numpy(), lat[ok].to_numpy())
    encounters = shapely.multipoints(np.column_stack([x, y]))
    road_parts = []
    for feature in json.loads(ROADS_GEOJSON.read_text(encoding="utf-8"))["features"]:
        geom = feature["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        road_parts += [LineString([c[:2] for c in line]) for line in lines]
    roads = transform(to_utm, unary_union(road_parts))
    return {
        "sampled_1000m": encounters.buffer(1000).intersection(ng31),
        "sampled_500m": encounters.buffer(500).intersection(ng31),
        "roads2013_1000m": roads.buffer(1000).intersection(ng31),
        "roads2013_2000m": roads.buffer(2000).intersection(ng31),
    }


def _null_overlap(geoms: np.ndarray, areas: np.ndarray, ia: np.ndarray, ib: np.ndarray, pair: np.ndarray) -> dict[str, float]:
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


def _random_points(region: BaseGeometry, n: int, rng: np.random.Generator) -> np.ndarray:
    shapely.prepare(region)
    minx, miny, maxx, maxy = region.bounds
    out = np.empty((0, 2))
    while len(out) < n:
        cand = np.column_stack([rng.uniform(minx, maxx, 4 * n), rng.uniform(miny, maxy, 4 * n)])
        out = np.vstack([out, cand[shapely.contains_xy(region, cand[:, 0], cand[:, 1])]])
    return out[:n]


def step_null_model() -> None:
    """Each 95% MCP keeps its size and shape but is rotated at random and moved so its
    centroid falls at random in the region; overlap is recomputed, 2,000 times per region.
    Observed overlap in the lower tail of that distribution would indicate avoidance."""
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32734", always_xy=True).transform
    features = json.loads((FIGS / f"merged_image_home_ranges_mcp95__{PRIMARY_WINDOW}.geojson").read_text(encoding="utf-8"))["features"]
    polys = pd.DataFrame([{"leopard_name": f["properties"]["leopard_name"], "sex": f["properties"]["sex"],
                           "geometry": transform(to_utm, shape(f["geometry"]))} for f in features])
    polys = polys[polys["geometry"].map(lambda g: not g.is_empty and g.area > 0)].sort_values("leopard_name").reset_index(drop=True)
    ia, ib = map(np.array, zip(*combinations(range(len(polys)), 2)))
    sex = polys["sex"].to_numpy()
    pair = np.array([pair_label(sex[a], sex[b]) for a, b in zip(ia, ib)])
    geoms = polys["geometry"].to_numpy()
    observed = _null_overlap(geoms, np.array([g.area for g in geoms]), ia, ib, pair)

    regions = _null_regions()
    pd.DataFrame([{"window": PRIMARY_WINDOW, "region": name, "area_km2": region.area / 1e6,
                   "observed_centroids_inside": sum(region.contains(g.centroid) for g in geoms), "n_polygons": len(geoms)}
                  for name, region in regions.items()]).to_csv(table(f"overlap_null_model_regions__{PRIMARY_WINDOW}.csv"), index=False)

    centred = [affinity.translate(g, -g.centroid.x, -g.centroid.y) for g in polys["geometry"]]
    areas = np.array([g.area for g in centred])
    rng = np.random.default_rng(NULL_SEED)
    summaries, primary_sims = [], None
    for name, region in regions.items():
        print(f"  {name}: {NULL_ITERATIONS:,} random placements over {region.area / 1e6:.1f} km2")
        sims = []
        for _ in range(NULL_ITERATIONS):
            angles = rng.uniform(0.0, 360.0, len(centred))
            centres = _random_points(region, len(centred), rng)
            placed = np.array([affinity.translate(affinity.rotate(g, a, origin=(0, 0)), x, y)
                               for g, a, (x, y) in zip(centred, angles, centres)], dtype=object)
            sims.append(_null_overlap(placed, areas, ia, ib, pair))
        sims = pd.DataFrame(sims)
        for key, obs in observed.items():
            group, metric = key.split("__")
            null = sims[key].to_numpy()
            summaries.append({
                "window": PRIMARY_WINDOW, "region": name, "group": group, "metric": metric, "observed": obs,
                "null_mean": null.mean(), "null_p05": np.percentile(null, 5), "null_p50": np.percentile(null, 50),
                "null_p95": np.percentile(null, 95),
                "observed_percentile": 100 * (null < obs).mean() + 50 * (null == obs).mean(),
                "p_lower": ((null <= obs).sum() + 1) / (len(null) + 1),
                "p_upper": ((null >= obs).sum() + 1) / (len(null) + 1),
                "n_iterations": len(null),
            })
        if name == NULL_PRIMARY_REGION:
            primary_sims = sims
    summary = pd.DataFrame(summaries)
    summary.to_csv(table(f"overlap_null_model_summary__{PRIMARY_WINDOW}.csv"), index=False)

    figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (key, label), tag in zip(axes, [("all__pct_overlapping", "Dyads that overlap (%)"),
                                             ("all__mean_prop_overlap", "Mean proportional overlap (%)")], "AB"):
        group, metric = key.split("__")
        row = summary[(summary["region"] == NULL_PRIMARY_REGION) & (summary["group"] == group) & (summary["metric"] == metric)].iloc[0]
        ax.hist(primary_sims[key], bins=30, color="#bdbdbd", edgecolor="white", linewidth=0.4)
        ax.axvline(row["observed"], color="black", linewidth=1.4)
        ax.axvline(row["null_p05"], color="#636363", linestyle="--", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Simulations")
        ax.set_title(f"({tag}) observed {row['observed']:.1f}; null 5th percentile {row['null_p05']:.1f}\n"
                     f"p(lower) = {row['p_lower']:.3f}, p(upper) = {row['p_upper']:.3f}", loc="left", fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{len(polys)} individuals, {len(ia)} dyads, {len(primary_sims):,} random placements", fontsize=8.5, y=0.99)
    fig.tight_layout()
    save_figure(fig, f"fig6_overlap_null_model__{PRIMARY_WINDOW}")


# =============================================================================
# Step 13. Power of the sex comparison
# =============================================================================

def step_power() -> None:
    """Both main results are null, so report how large a sex difference the Mann-Whitney
    test could detect: unit normals separated by a shift, raised until the test rejects
    in 80% of simulations at alpha 0.05."""
    shifts = np.round(np.arange(0.1, 3.01, 0.05), 2)
    rng = np.random.default_rng(POWER_SEED)

    def power(n_female: int, n_male: int, shift: float) -> float:
        female = rng.normal(0.0, 1.0, (POWER_SIMULATIONS, n_female))
        male = rng.normal(shift, 1.0, (POWER_SIMULATIONS, n_male))
        return float((mannwhitneyu(female, male, axis=1, alternative="two-sided").pvalue < 0.05).mean())

    rows = []
    for row in read_table("merged_image_window_sex_stats.csv").itertuples():
        n_f, n_m = int(row.n_female), int(row.n_male)
        detectable = next((s for s in shifts if power(n_f, n_m, s) >= 0.80), np.nan)
        rows.append({
            "window": row.window, "n_female": n_f, "n_male": n_m, "alpha": 0.05, "target_power": 0.80,
            "detectable_shift_sd": detectable, "detectable_cliffs_delta": 2 * norm.cdf(detectable / np.sqrt(2)) - 1,
            "observed_cliffs_delta": row.cliffs_delta_male_minus_female, "n_simulations": POWER_SIMULATIONS,
        })
    pd.DataFrame(rows).to_csv(table("sex_test_power.csv"), index=False)


# =============================================================================
# Step 14. Reviewer decomposition
# =============================================================================

def decomposition_pairs(percent: float, min_points: int = SUBMITTED_MIN_UNIQUE_POINTS, adults_only: bool = False,
                        name: str = "merged_leopard_images_metadata.normalized_events.csv") -> pd.DataFrame:
    df = read_events(name)
    lookup = age_class_lookup()
    polys, sexes = {}, {}
    for (leopard, sex), group in df.groupby(["leopard_name", "sex"]):
        if sex not in ("Female", "Male"):
            continue
        if len(set(zip(group["gps_lat"], group["gps_lon"]))) < min_points or group["date"].nunique() < MIN_UNIQUE_DATES:
            continue
        if adults_only and not is_territorial(leopard, lookup):
            continue
        hull = mcp(list(zip(group["gps_lat"], group["gps_lon"])), percent)
        if hull["ring_latlon"]:
            polys[leopard], sexes[leopard] = polygon_from_ring(hull["ring_latlon"]), sex
    rows = []
    for a, b in combinations(sorted(polys), 2):
        inter = polys[a].intersection(polys[b])
        inter_km2 = (inter.area / 1e6) if not inter.is_empty else 0.0
        rows.append({"leopard_a": a, "sex_a": sexes[a], "leopard_b": b, "sex_b": sexes[b],
                     "pair_sex": pair_label(sexes[a], sexes[b]), "overlap_km2": inter_km2,
                     "prop_overlap": 2.0 * inter_km2 / (polys[a].area / 1e6 + polys[b].area / 1e6)})
    return pd.DataFrame(rows)


def _focal_rank_permutation_p(pairs: pd.DataFrame, response: str) -> float:
    """Two-sided permutation p for female-female vs female-male, relabelling individuals.
    The statistic is the rank-biserial correlation (Mann-Whitney's ordering), so only the
    null distribution changes from the rank-test steps."""
    values = pairs[response].to_numpy(float)

    def statistic(types: pd.Series) -> float:
        a = values[(types == "Female-Female").to_numpy()]
        b = values[(types == "Female-Male").to_numpy()]
        if len(a) == 0 or len(b) == 0:
            return np.nan
        return float(2.0 * stats.mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)) - 1.0)

    observed = statistic(pairs["pair_sex"].reset_index(drop=True))
    ind = _individuals_and_sexes(pairs, ("n", "s"))
    names, sx = ind["n"].tolist(), ind["s"].to_numpy()
    a_arr, b_arr = pairs["leopard_a"].to_numpy(), pairs["leopard_b"].to_numpy()
    rng = np.random.default_rng(PERMUTATION_SEED)
    ge = 0
    for _ in range(N_PERMUTATIONS):
        lookup = dict(zip(names, rng.permutation(sx)))
        d = statistic(pd.Series([pair_label(lookup[a], lookup[b]) for a, b in zip(a_arr, b_arr)]))
        if not np.isnan(d) and abs(d) >= abs(observed) - 1e-12:
            ge += 1
    return (ge + 1) / (N_PERMUTATIONS + 1)


def step_reviewer_decomposition() -> None:
    """Changes applied one at a time to the focal female-female vs female-male contrast,
    from the submitted analysis (A) to the revised one (H)."""
    pairs_100 = decomposition_pairs(100.0)
    pairs_95 = decomposition_pairs(95.0)
    pairs_relaxed = decomposition_pairs(95.0, MIN_UNIQUE_POINTS)
    pairs_adults = decomposition_pairs(95.0, MIN_UNIQUE_POINTS, adults_only=True)
    pairs_dated = decomposition_pairs(95.0, MIN_UNIQUE_POINTS, adults_only=True, name="merged_leopard_images_metadata.dated_events.csv")

    rows = []

    def record(step: str, change: str, pairs: pd.DataFrame, response: str, inference: str, estimator: str,
               nonzero_only: bool, min_points: int, adults: bool) -> None:
        a = pairs.loc[pairs["pair_sex"] == "Female-Female", response].to_numpy(float)
        b = pairs.loc[pairs["pair_sex"] == "Female-Male", response].to_numpy(float)
        p = _focal_rank_permutation_p(pairs, response) if inference == "permutation" else float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rows.append({
            "step": step, "cumulative_change": change, "estimator": estimator, "response": response,
            "dyads": "non-zero only" if nonzero_only else "all dyads", "age_classes": "adults only" if adults else "all ages",
            "min_unique_points": min_points, "inference": inference, "n_FF": len(a), "n_FM": len(b),
            "median_FF": float(np.median(a)), "median_FM": float(np.median(b)), "p_focal_FF_vs_FM": p,
        })

    nz100, nz95 = pairs_100[pairs_100["overlap_km2"] > 0], pairs_95[pairs_95["overlap_km2"] > 0]
    s = SUBMITTED_MIN_UNIQUE_POINTS
    record("A", "as submitted", nz100, "overlap_km2", "mann_whitney", "100% MCP", True, s, False)
    record("B", "+ 95% MCP", nz95, "overlap_km2", "mann_whitney", "95% MCP", True, s, False)
    record("C", "+ size-normalised response", nz95, "prop_overlap", "mann_whitney", "95% MCP", True, s, False)
    record("D", "+ zero-overlap dyads included", pairs_95, "prop_overlap", "mann_whitney", "95% MCP", False, s, False)
    record("E", "+ node-label permutation", pairs_95, "prop_overlap", "permutation", "95% MCP", False, s, False)
    record("F", f"+ inclusion threshold >={MIN_UNIQUE_POINTS} points", pairs_relaxed, "prop_overlap", "permutation", "95% MCP", False, MIN_UNIQUE_POINTS, False)
    record("G", "+ cubs and subadults excluded", pairs_adults, "prop_overlap", "permutation", "95% MCP", False, MIN_UNIQUE_POINTS, True)
    record("H", "+ archive-dated events excluded", pairs_dated, "prop_overlap", "permutation", "95% MCP", False, MIN_UNIQUE_POINTS, True)
    pd.DataFrame(rows).to_csv(table("reviewer_decomposition.csv"), index=False)


# =============================================================================
# Step 15. Detection table (Table S2)
# =============================================================================

def step_detection_table() -> None:
    """One row per leopard in the lineage register or the photo record."""
    events = read_table("merged_leopard_images_metadata.dated_events.csv")
    events["leopard_name"] = events["leopard_name"].map(display_name)
    events["sex"] = events["sex"].fillna("Unknown")
    events["date"] = pd.to_datetime(events["image_date"], errors="coerce", dayfirst=True, format="mixed")
    events["gps_lat"] = pd.to_numeric(events["gps_lat"], errors="coerce")
    events["gps_lon"] = pd.to_numeric(events["gps_lon"], errors="coerce")
    events = events[events["leopard_name"].ne("")].copy()

    detected = events.groupby("leopard_name", dropna=False).agg(
        sex_events=("sex", resolve_sex), first_detection=("date", "min"), last_detection=("date", "max"),
        n_encounter_events=("event_id", "nunique"), n_unique_dates=("date", "nunique")).reset_index()
    locations = events.dropna(subset=["gps_lat", "gps_lon"]).groupby("leopard_name").apply(
        lambda g: int(len(set(zip(g["gps_lat"].astype(float), g["gps_lon"].astype(float)))))).rename("n_unique_locations").reset_index()
    years = events.dropna(subset=["date"]).assign(year=lambda d: d["date"].dt.year).groupby("leopard_name")["year"].nunique().rename("n_years_detected").reset_index()
    detected = detected.merge(locations, on="leopard_name", how="left").merge(years, on="leopard_name", how="left")
    detected[["n_unique_locations", "n_years_detected"]] = detected[["n_unique_locations", "n_years_detected"]].fillna(0).astype(int)
    detected["individual_id"] = detected["leopard_name"].map(name_key)

    lineage = read_lineage()
    lineage["leopard_name"] = lineage["leopard_name"].map(display_name)
    lineage["sex_lineage"] = lineage["sex"].fillna("").astype(str).str.strip().str.title()
    lineage["age_class"] = lineage["Estimated age"].map(lambda v: "" if not str(v).strip() or str(v).strip().lower() == "nan" else re.sub(r"\s+", " ", str(v).strip()))
    lineage = lineage[lineage["leopard_name"].ne("")].copy()
    lineage["individual_id"] = lineage["leopard_name"].map(name_key)
    lineage = lineage[["individual_id", "leopard_name", "sex_lineage", "age_class"]].drop_duplicates(subset=["individual_id"], keep="first")

    pooled = read_table("merged_image_home_ranges_mcp_summary.csv", usecols=["leopard_name", "mcp95_km2", "mcp100_km2", "n_points_dropped"])
    pooled["individual_id"] = pooled["leopard_name"].map(display_name).map(name_key)
    pooled = pooled[["individual_id", "mcp95_km2", "mcp100_km2", "n_points_dropped"]].drop_duplicates(subset=["individual_id"], keep="first")
    subset = read_table("merged_leopard_images_metadata.spatial_subset_individuals.csv", usecols=["leopard_name", "include_spatial_subset"])
    subset["individual_id"] = subset["leopard_name"].map(display_name).map(name_key)
    subset["include_spatial_subset"] = subset["include_spatial_subset"].fillna(False).astype(bool)
    subset = subset[["individual_id", "include_spatial_subset"]].drop_duplicates(subset=["individual_id"], keep="first")
    window_columns = {"n_unique_points": "w24_n_unique_locations", "n_unique_dates": "w24_n_unique_dates",
                      "mcp95_km2": "w24_mcp95_km2", "mcp100_km2": "w24_mcp100_km2", "include": "w24_included"}
    window = read_table(f"merged_image_window_individuals__{PRIMARY_WINDOW}.csv", usecols=["leopard_name", *window_columns])
    window["individual_id"] = window["leopard_name"].map(display_name).map(name_key)
    window = window.rename(columns=window_columns).drop(columns=["leopard_name"])

    t = (detected.merge(lineage, on="individual_id", how="outer", suffixes=("_events", "_lineage"))
         .merge(pooled, on="individual_id", how="left").merge(subset, on="individual_id", how="left")
         .merge(window, on="individual_id", how="left"))
    t["leopard_name"] = t["leopard_name_events"].fillna(t["leopard_name_lineage"]).fillna("")
    t["sex"] = t["sex_lineage"].where(t["sex_lineage"].isin(["Female", "Male"]), t["sex_events"]).fillna("Unknown")
    t["age_class"] = t["age_class"].fillna("")
    t["include_spatial_subset"] = t["include_spatial_subset"].fillna(False).astype(bool)
    for column in ("n_encounter_events", "n_unique_dates", "n_unique_locations", "n_years_detected"):
        t[column] = t[column].fillna(0).astype(int)
    for column in ("first_detection", "last_detection"):
        t[column] = t[column].map(lambda v: "" if pd.isna(v) else pd.Timestamp(v).date().isoformat())
    for column in ("mcp95_km2", "mcp100_km2"):
        t[column] = pd.to_numeric(t[column], errors="coerce").round(6)
    t["n_points_dropped"] = pd.to_numeric(t["n_points_dropped"], errors="coerce").astype("Int64")

    def status(row: pd.Series) -> str:
        if bool(row.get("include_spatial_subset", False)):
            return "Included"
        n_dates, n_locations = int(row.get("n_unique_dates", 0) or 0), int(row.get("n_unique_locations", 0) or 0)
        if int(row.get("n_encounter_events", 0) or 0) == 0:
            return "Excluded: no detections"
        age = str(row.get("age_class", "")).strip().lower()
        if age in NON_TERRITORIAL_AGE_CLASSES:
            return f"Excluded: {age} (territory not established)"  # biological, whatever the sample
        if n_locations < MIN_UNIQUE_POINTS and n_dates < MIN_UNIQUE_DATES:
            return f"Excluded: <{MIN_UNIQUE_POINTS} unique locations and <{MIN_UNIQUE_DATES} unique dates"
        if n_locations < MIN_UNIQUE_POINTS:
            return f"Excluded: <{MIN_UNIQUE_POINTS} unique locations"
        return f"Excluded: <{MIN_UNIQUE_DATES} unique dates"

    t["inclusion_status"] = t.apply(status, axis=1)

    # Leopards photographed only on undatable (archive-export) records were seen, so they
    # are listed separately rather than as "no detections".
    all_events = read_table("merged_leopard_images_metadata.normalized_events.csv", usecols=["leopard_name", "sex"])
    all_events["leopard_name"] = all_events["leopard_name"].map(display_name)
    all_events = all_events[all_events["leopard_name"].ne("")]
    all_events["individual_id"] = all_events["leopard_name"].map(name_key)
    undated = all_events[~all_events["individual_id"].isin(t.loc[t["n_encounter_events"] > 0, "individual_id"])]
    undated = undated.groupby(["individual_id", "leopard_name"], as_index=False)["sex"].agg(resolve_sex)
    t = t[~t["individual_id"].isin(undated["individual_id"])]
    extra = undated.assign(age_class="", first_detection="", last_detection="", inclusion_status="Excluded: only undatable (archive-export) records")
    for column in ("n_encounter_events", "n_unique_dates", "n_unique_locations", "n_years_detected"):
        extra[column] = 0
    t = pd.concat([t, extra], ignore_index=True)
    for column in ("w24_n_unique_locations", "w24_n_unique_dates"):
        t[column] = pd.to_numeric(t[column], errors="coerce").astype("Int64")
    for column in ("w24_mcp95_km2", "w24_mcp100_km2"):
        t[column] = pd.to_numeric(t[column], errors="coerce").round(6)
    t["w24_included"] = t["w24_included"].astype("boolean").fillna(False)

    t = t.sort_values(["n_encounter_events", "n_unique_dates", "n_unique_locations", "leopard_name"],
                      ascending=[False, False, False, True]).drop_duplicates(subset=["individual_id"], keep="first")
    columns = ["individual_id", "leopard_name", "sex", "age_class", "first_detection", "last_detection",
               "n_encounter_events", "n_unique_dates", "n_unique_locations", "n_years_detected", "mcp95_km2",
               "mcp100_km2", "n_points_dropped", "inclusion_status", *window_columns.values()]
    t[columns].sort_values(["n_encounter_events", "n_unique_dates", "leopard_name"], ascending=[False, False, True]).reset_index(drop=True).to_csv(
        table("merged_leopard_individual_detection_table.csv"), index=False)


# =============================================================================
# Step 16. Figures
# =============================================================================

def figure1_study_area() -> None:
    """Study area: NG31, every dated encounter inside it, and an Africa inset."""
    tile_dir = CACHE / "tiles" / "carto_light_nolabels"

    def tile(x: int, y: int, zoom: int) -> Image.Image:
        path = tile_dir / str(zoom) / str(x) / f"{y}.png"
        if path.exists():
            return Image.open(path).convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        for server in "abcd":
            try:
                response = requests.get(f"https://{server}.basemaps.cartocdn.com/light_nolabels/{zoom}/{x}/{y}.png",
                                        timeout=20, headers={"User-Agent": "chitabe-leopard-analysis/1.0 (research use)"})
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = ImageEnhance.Brightness(ImageEnhance.Contrast(img).enhance(1.02)).enhance(1.01)
                img.save(path)
                return img
            except Exception as exc:  # noqa: BLE001 - try the next server
                last = exc
        raise RuntimeError(f"Could not download background map tile {zoom}/{x}/{y}: {last}")

    def tile_index(lon: float, lat: float, zoom: int) -> tuple[int, int]:
        lat = max(min(lat, 85.05112878), -85.05112878)
        n = 2 ** zoom
        return (int((lon + 180.0) / 360.0 * n),
                int((1.0 - np.log(np.tan(np.deg2rad(lat)) + 1.0 / np.cos(np.deg2rad(lat))) / np.pi) / 2.0 * n))

    def tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
        limit = WEB_MERCATOR_RADIUS_M * np.pi
        span = 2.0 * limit / 2 ** zoom
        min_x, max_y = -limit + x * span, limit - y * span
        return min_x, min_x + span, max_y - span, max_y

    reader = shapefile.Reader(str(NG31_EDIT_SHP))
    ring = [(float(lon), float(lat)) for lon, lat in reader.shapes()[0].points]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    ng31 = Polygon(ring)

    points = read_table("merged_leopard_images_metadata.dated_events.csv")
    points["gps_lat"] = pd.to_numeric(points["gps_lat"], errors="coerce")
    points["gps_lon"] = pd.to_numeric(points["gps_lon"], errors="coerce")
    points = points.dropna(subset=["gps_lat", "gps_lon"]).copy()
    points["x_m"], points["y_m"] = to_web_mercator(points["gps_lon"], points["gps_lat"])
    inside = [ng31.contains(Point(lon, lat)) or ng31.touches(Point(lon, lat)) for lon, lat in zip(points["gps_lon"], points["gps_lat"])]
    points = points.loc[inside].copy()

    bx, by = to_web_mercator(*zip(*ng31.exterior.coords))
    x_min, x_max = min(float(points["x_m"].min()), float(np.min(bx))), max(float(points["x_m"].max()), float(np.max(bx)))
    y_min, y_max = min(float(points["y_m"].min()), float(np.min(by))), max(float(points["y_m"].max()), float(np.max(by)))
    bounds = (x_min - (x_max - x_min) * 0.16, x_max + (x_max - x_min) * 0.06, y_min - (y_max - y_min) * 0.08, y_max + (y_max - y_min) * 0.08)

    lon_min, lat_min = from_web_mercator(bounds[0], bounds[2])
    lon_max, lat_max = from_web_mercator(bounds[1], bounds[3])
    zoom = 8
    for z in range(12, 8, -1):
        x0, y1 = tile_index(lon_min, lat_min, z)
        x1, y0 = tile_index(lon_max, lat_max, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= 25:
            zoom = z
            break
    x0, y1 = tile_index(lon_min, lat_min, zoom)
    x1, y0 = tile_index(lon_max, lat_max, zoom)
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            mosaic.paste(tile(x, y, zoom), ((x - x0) * 256, (y - y0) * 256))
    extent = (tile_bounds(x0, y0, zoom)[0], tile_bounds(x1, y1, zoom)[1], tile_bounds(x1, y1, zoom)[2], tile_bounds(x0, y0, zoom)[3])

    figure_style()
    fig = plt.figure(figsize=(10.8, 8.2))
    ax = fig.add_axes([0.05, 0.10, 0.76, 0.82])
    inset = fig.add_axes([0.80, 0.62, 0.18, 0.24])
    ax.imshow(np.asarray(mosaic), extent=extent, origin="upper", alpha=0.98, zorder=0)
    ax.scatter(points["x_m"], points["y_m"], s=8, c="#444444", alpha=0.42, linewidths=0, zorder=3)
    ax.plot(bx, by, color="#cc5a2e", linewidth=1.8, zorder=4)
    camp_x, camp_y = to_web_mercator(CHITABE_CAMP_LON, CHITABE_CAMP_LAT)
    ax.scatter([camp_x], [camp_y], s=48, c="black", edgecolors="white", linewidths=0.8, zorder=5)
    ax.text(camp_x + 520.0, camp_y + 360.0, "Chitabe Camp", fontsize=8, color="black", ha="left", va="bottom", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.0))
    ax.text(float(np.mean(bx)), float(np.max(by)) - (y_max - y_min) * 0.08, "NG/31 concession", fontsize=10, fontweight="bold",
            color="#1f2b3a", ha="center", va="center", zorder=6, bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.0))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_scalebar(ax, bounds, 5_000.0, 220.0, 320.0, (2.2, 1.2))
    add_north_arrow(ax, bounds, 0.08)

    botswana = None
    for feature in json.loads(NATURAL_EARTH.read_text(encoding="utf-8")).get("features", []):
        props = feature.get("properties", {})
        geom = shape(feature["geometry"])
        if props.get("CONTINENT") == "Africa":
            for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom] if geom.geom_type == "Polygon" else []):
                inset.add_patch(MplPolygon(np.asarray(poly.exterior.coords), closed=True, facecolor="#e3e6e8", edgecolor="#aab3b8", linewidth=0.35))
        if (props.get("ADM0_A3") or props.get("ISO_A3") or props.get("ADM0_A3_US")) == "BWA":
            botswana = geom
    if botswana is not None:
        for poly in (botswana.geoms if botswana.geom_type == "MultiPolygon" else [botswana]):
            inset.add_patch(MplPolygon(np.asarray(poly.exterior.coords), closed=True, facecolor="#7f8c8d", edgecolor="#566168", linewidth=0.7))
    inset.scatter([CHITABE_CAMP_LON], [CHITABE_CAMP_LAT], s=20, c="black", edgecolors="white", linewidths=0.5, zorder=5)
    inset.text(CHITABE_CAMP_LON + 3.2, CHITABE_CAMP_LAT + 2.6, "Study site", fontsize=7, ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.8))
    inset.text(24.0, -22.0, "Botswana", fontsize=7, color="#2f3a40", ha="center", va="center")
    inset.set_xlim(-20, 55)
    inset.set_ylim(-36, 38)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_title("Location within Africa", fontsize=9)

    fig.legend(handles=[
        Line2D([0], [0], color="#444444", linewidth=0, marker="o", markersize=4, markerfacecolor="#444444", label="Georeferenced leopard records"),
        Line2D([0], [0], color="#cc5a2e", linewidth=1.8, label="NG/31 concession boundary"),
        Line2D([0], [0], color="black", linewidth=0, marker="o", markersize=6, markerfacecolor="black", markeredgecolor="white", label="Chitabe Camp"),
        Line2D([0], [0], color="#7f8c8d", linewidth=0, marker="s", markersize=7, markerfacecolor="#7f8c8d", label="Botswana"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=True, framealpha=0.92, fontsize=8, handlelength=1.8, columnspacing=1.4)
    fig.text(0.06, 0.02, "Basemap: clean greyscale OSM-style reference | Inset boundaries: Natural Earth", ha="left", va="bottom", fontsize=6, color="#666666")
    save_figure(fig, "new_fig_study_area_ng31_osm_vector_gray")


def figure2_seasonal() -> None:
    season_sex = read_table("merged_image_counts_by_botswana_season_and_sex.csv")
    pivot = (season_sex.pivot(index="botswana_season", columns="sex", values="n_records")
             .reindex(["Wet season", "Dry season"]).reindex(columns=SEX3_ORDER).fillna(0))
    totals = pivot.sum(axis=1)
    figure_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    bottom = np.zeros(len(pivot))
    for sex in SEX3_ORDER:
        ax.bar(pivot.index, pivot[sex].to_numpy(), bottom=bottom, color=SEX3_GREY[sex], edgecolor="black", linewidth=0.7, width=0.62, label=sex)
        bottom += pivot[sex].to_numpy()
    for i, total in enumerate(totals):
        ax.text(i, total + totals.max() * 0.015, f"{int(total)}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Detections")
    ax.set_ylim(0, totals.max() * 1.1)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, handlelength=1.4, columnspacing=1.2)
    fig.tight_layout()
    save_figure(fig, "fig2_seasonal_detections_by_sex")


def figure3_home_range_map() -> None:
    """Primary-window 95% MCPs and locations, female (A) and male (B), over the seasonal
    floodplain (OpenStreetMap water and wetland, from the saved snapshot)."""
    points = read_table(f"merged_leopard_images_metadata.window_subset__{PRIMARY_WINDOW}.csv")
    points["gps_lat"] = pd.to_numeric(points["gps_lat"], errors="coerce")
    points["gps_lon"] = pd.to_numeric(points["gps_lon"], errors="coerce")
    points["sex"] = points["sex"].fillna("Unknown")
    points = points.dropna(subset=["leopard_name", "gps_lat", "gps_lon"]).copy()
    points["sex"] = points.groupby("leopard_name")["sex"].transform(resolve_sex)
    points["x_m"], points["y_m"] = to_web_mercator(points["gps_lon"], points["gps_lat"])
    points = points[points["sex"].isin(["Female", "Male"])].copy()

    polygons = []
    for feat in json.loads((FIGS / f"merged_image_home_ranges_mcp95__{PRIMARY_WINDOW}.geojson").read_text(encoding="utf-8")).get("features", []):
        props = feat.get("properties", {})
        if str(props.get("sex", "Unknown")) in {"Female", "Male"}:
            polygons.append({"leopard_name": str(props.get("leopard_name", "Unknown")), "sex": str(props["sex"]),
                             "coords": feat["geometry"]["coordinates"][0], "n_unique_points": props.get("n_unique_points")})

    x_min, x_max = float(points["x_m"].min()), float(points["x_m"].max())
    y_min, y_max = float(points["y_m"].min()), float(points["y_m"].max())
    bounds = (x_min - (x_max - x_min) * 0.08, x_max + (x_max - x_min) * 0.08, y_min - (y_max - y_min) * 0.10, y_max + (y_max - y_min) * 0.10)

    # Floodplain polygons from the OSM snapshot: closed ways, and closed relation members
    # (open arcs are not stitched; that produced spurious polygons spanning the box).
    osm = json.loads(gzip.decompress(OSM_FLOODPLAIN.read_bytes()).decode("utf-8")).get("elements", [])
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in osm if el["type"] == "node"}
    ways = {el["id"]: el.get("nodes", []) for el in osm if el["type"] == "way"}

    def ring_of(way_id: int) -> list[tuple[float, float]]:
        return [nodes[n] for n in ways.get(way_id, []) if n in nodes]

    def closed(coords: list[tuple[float, float]]) -> Polygon | None:
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        try:
            poly = Polygon(coords)
        except Exception:  # noqa: BLE001
            return None
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if (not poly.is_empty and poly.area > 0) else None

    def water(el: dict) -> bool:
        return el.get("tags", {}).get("natural") in ("water", "wetland")

    floodplain: list[Polygon] = [p for el in osm if el["type"] == "way" and water(el) for p in [closed(ring_of(el["id"]))] if p is not None]
    for rel in osm:
        if rel["type"] != "relation" or not water(rel):
            continue
        outers, inners = [], []
        for member in rel.get("members", []):
            coords = ring_of(member["ref"]) if member.get("type") == "way" else []
            if len(coords) < 3 or coords[0] != coords[-1]:
                continue
            poly = closed(coords)
            if poly is not None:
                (inners if member.get("role") == "inner" else outers).append(poly)
        for outer in outers:
            for inner in inners:
                if outer.contains(inner.representative_point()):
                    outer = outer.difference(inner)
            floodplain.append(outer)
    floodplain = [g for p in floodplain if not p.is_empty for g in (p.geoms if p.geom_type == "MultiPolygon" else [p])]

    lon_min, lat_min = from_web_mercator(bounds[0], bounds[2])
    lon_max, lat_max = from_web_mercator(bounds[1], bounds[3])
    view = box(lon_min, lat_min, lon_max, lat_max)
    rings = []
    for poly in floodplain:
        if not poly.intersects(view):
            continue
        clipped = poly.intersection(view)
        for geom in (clipped.geoms if clipped.geom_type == "MultiPolygon" else [clipped]):
            if not geom.is_empty and geom.geom_type == "Polygon":
                xs, ys = to_web_mercator(np.array([c[0] for c in geom.exterior.coords]), np.array([c[1] for c in geom.exterior.coords]))
                rings.append(list(zip(xs, ys)))
    boundary = [to_web_mercator(np.array([c[0] for c in line]), np.array([c[1] for c in line])) for line in load_ng31_lines()]

    figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 7.2), sharex=True, sharey=True)
    for ax, sex, letter in zip(axes, ["Female", "Male"], "AB"):
        sub_pts = points[points["sex"] == sex].copy()
        sub_poly = [p for p in polygons if p["sex"] == sex]
        ax.set_facecolor("#fbfbf9")
        for xs, ys in boundary:
            ax.plot(xs, ys, color="#9a9a9a", linewidth=0.6, zorder=1)
        for ring in rings:
            ax.add_patch(MplPolygon(ring, closed=True, facecolor="#c9c9c9", edgecolor="#8c8c8c", linewidth=0.4, alpha=0.85, zorder=1))
        for feat in sub_poly:
            xs, ys = to_web_mercator([c[0] for c in feat["coords"]], [c[1] for c in feat["coords"]])
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True, facecolor=SEX_GREY[sex], edgecolor="black", linewidth=0.5, alpha=0.22, zorder=2))
        ax.scatter(sub_pts["x_m"], sub_pts["y_m"], s=12, marker=SEX_MARKER[sex], facecolor=SEX_GREY[sex], alpha=0.85,
                   linewidths=0.3, edgecolors="white", zorder=4)
        # Each range carries its number of locations: some rest on 8 points, others on 50.
        n_points = {f["leopard_name"]: f["n_unique_points"] for f in sub_poly}
        labels = sub_pts.groupby(["leopard_name", "sex"], dropna=False).agg(label_x=("x_m", "mean"), label_y=("y_m", "mean")).reset_index()
        for row in labels.itertuples(index=False):
            n = n_points.get(row.leopard_name)
            ax.text(row.label_x, row.label_y, f"{row.leopard_name} ({int(n)})" if n is not None else row.leopard_name, fontsize=7,
                    color="black", ha="center", va="center", zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.9))
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        add_scalebar(ax, bounds, 2_000.0, 120.0, 190.0, (2.0, 1.1))
        add_north_arrow(ax, bounds, 0.07)
        camp_x, camp_y = to_web_mercator(CHITABE_CAMP_LON, CHITABE_CAMP_LAT)
        ax.scatter([camp_x], [camp_y], s=44, c="black", marker="o", edgecolors="white", linewidths=0.8, zorder=7)
        ax.text(camp_x + 350.0, camp_y + 350.0, "Chitabe Camp", fontsize=7, color="black", ha="left", va="bottom", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.9))
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold", zorder=7,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6, pad=2.5))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(handles=[
            Line2D([0], [0], linestyle="none", marker=SEX_MARKER[sex], markersize=5, markerfacecolor=SEX_GREY[sex],
                   markeredgecolor="white", label="Georeferenced leopard records"),
            Patch(facecolor="#c9c9c9", edgecolor="#8c8c8c", linewidth=0.6, label="Seasonal floodplain"),
        ], loc="lower right", frameon=True, framealpha=0.88, edgecolor="none", fontsize=7.5, handlelength=1.5, columnspacing=1.2, borderpad=0.6)
    fig.tight_layout()
    save_figure(fig, "new_fig_merged_subset_home_ranges_map_osm_vector_gray")


def _primary_permutation_p(models: pd.DataFrame, part: str) -> float | None:
    row = models[(models["window"] == PRIMARY_WINDOW) & (models["part"] == part) & (models["test"] == "omnibus_pair_type_LR")]
    return float(row["p_permutation"].iloc[0]) if len(row) else None


def _box_with_points(ax: plt.Axes, data: list[np.ndarray], labels: list[str], groups: list[str], grey: dict[str, str],
                     marker: dict[str, str], seed: int, jitter: float, markersize: float, edgewidth: float, alpha: float) -> None:
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, median, g in zip(bp["boxes"], bp["medians"], groups):
        patch.set(facecolor="white", edgecolor=grey[g], linewidth=1.1)
        median.set(color=grey[g], linewidth=1.3)
    rng = np.random.default_rng(seed)
    for i, (arr, g) in enumerate(zip(data, groups), start=1):
        ax.plot(np.full_like(arr, i, dtype=float) + rng.normal(0.0, jitter, size=len(arr)), arr, marker=marker[g], linestyle="none",
                markersize=markersize, markerfacecolor=grey[g], markeredgecolor="black", markeredgewidth=edgewidth, alpha=alpha)


def figure4_range_by_sex() -> None:
    individuals = read_table(f"merged_image_window_individuals__{PRIMARY_WINDOW}.csv")
    kept = individuals[individuals["include"]]
    row = read_table("merged_image_window_sex_stats.csv").set_index("window").loc[PRIMARY_WINDOW]
    female = kept.loc[kept["sex"] == "Female", "mcp95_km2"].to_numpy(float)
    male = kept.loc[kept["sex"] == "Male", "mcp95_km2"].to_numpy(float)
    figure_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    _box_with_points(ax, [female, male], [f"Female\n(n={len(female)})", f"Male\n(n={len(male)})"], ["Female", "Male"],
                     SEX_GREY, SEX_MARKER, 42, 0.045, 4.2, 0.4, 0.9)
    ax.set_ylabel("95% MCP area (km$^2$)")
    ax.set_xlabel(
        f"Mann-Whitney U = {row['mann_whitney_u']:.0f}, p = {row['p_value']:.3f}   |   "
        f"Cliff's delta = {row['cliffs_delta_male_minus_female']:.3f}\n"
        f"Median difference (male - female) = {row['median_diff_male_minus_female_km2']:.2f} km$^2$ "
        f"(95% CI [{row['median_diff_ci95_low_km2']:.2f}, {row['median_diff_ci95_high_km2']:.2f}])", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig4_mcp95_by_sex_contemporary")


def figure5_overlap() -> None:
    pairs = read_table(f"merged_image_overlap_pairs__{PRIMARY_WINDOW}.csv")
    models = read_table("overlap_model_results.csv")
    figure_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 5.2))

    # A. How often each pair type overlaps at all, with Wilson intervals.
    counts = pairs.groupby("pair_sex")["overlaps"].agg(["sum", "size"]).reindex(PAIR_ORDER)
    props = counts["sum"] / counts["size"]
    lo, hi = proportion_confint(counts["sum"], counts["size"], method="wilson")
    ax_a.bar(range(len(PAIR_ORDER)), props * 100.0, yerr=np.vstack([props - lo, hi - props]) * 100.0,
             color=[PAIR_GREY[p] for p in PAIR_ORDER], edgecolor="black", linewidth=0.8, width=0.62, capsize=4, error_kw={"linewidth": 0.9})
    ax_a.set_xticks(range(len(PAIR_ORDER)))
    ax_a.set_xticklabels([p.replace("-", "–") for p in PAIR_ORDER])
    ax_a.set_ylabel("Dyads that overlap at all (%)")
    ax_a.set_ylim(0, 100)
    for i in range(len(PAIR_ORDER)):
        ax_a.text(i, props.iloc[i] * 100.0 + (hi.iloc[i] - props.iloc[i]) * 100.0 + 3,
                  f"{int(counts['sum'].iloc[i])}/{int(counts['size'].iloc[i])}", ha="center", va="bottom", fontsize=8)
    p1 = _primary_permutation_p(models, "part1_any_overlap")
    ax_a.set_xlabel("A. Propensity to overlap" + (f"\npermutation p = {p1:.3f}" if p1 is not None else ""), fontsize=8.5)

    # B. How much, among dyads that do overlap.
    nonzero = pairs[pairs["overlaps"] == 1]
    data = [nonzero.loc[nonzero["pair_sex"] == p, "prop_overlap"].to_numpy(float) * 100.0 for p in PAIR_ORDER]
    _box_with_points(ax_b, data, [f"{p.replace('-', chr(8211))}\n(n={len(d)})" for p, d in zip(PAIR_ORDER, data)], PAIR_ORDER,
                     PAIR_GREY, PAIR_MARKER, 123, 0.05, 3.8, 0.3, 0.85)
    ax_b.set_ylabel("Shared area (% of mean polygon area)")
    p2 = _primary_permutation_p(models, "part2_extent_given_overlap")
    ax_b.set_xlabel("B. Extent, given overlap" + (f"\npermutation p = {p2:.3f}" if p2 is not None else ""), fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "fig5_overlap_by_pair_type_revised")


def figure_s1_window_concordance() -> None:
    def keyed(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["dyad"] = ["|".join(sorted([a, b])) for a, b in zip(out["leopard_a"], out["leopard_b"])]
        return out[["dyad", "pair_sex", "prop_overlap"]]

    merged = keyed(read_table(f"merged_image_overlap_pairs__{PRIMARY_WINDOW}.csv")).merge(
        keyed(read_table("merged_image_overlap_pairs__pooled.csv")), on="dyad", suffixes=("_window", "_pooled"))
    rho = merged["prop_overlap_window"].corr(merged["prop_overlap_pooled"], method="spearman")
    figure_style()
    fig, ax = plt.subplots(figsize=(5.8, 5.9))
    for pair in PAIR_ORDER:
        sub = merged[merged["pair_sex_window"] == pair]
        ax.plot(sub["prop_overlap_pooled"] * 100.0, sub["prop_overlap_window"] * 100.0, marker=PAIR_MARKER[pair], linestyle="none",
                markersize=4.4, markerfacecolor=PAIR_GREY[pair], markeredgecolor="black", markeredgewidth=0.3, alpha=0.85, label=pair.replace("-", "–"))
    limit = max(merged[["prop_overlap_window", "prop_overlap_pooled"]].max()) * 100.0 * 1.08
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=0.8, color="black")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_ylabel("Proportional overlap, contemporary window (%)")
    ax.set_xlabel(f"Proportional overlap, pooled 2014-2026 (%)\n{len(merged)} shared dyads, Spearman rho = {rho:.2f}", fontsize=8.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figS1_window_concordance")


def figure_s2_mcp_estimators() -> None:
    df = read_table("merged_image_home_ranges_mcp_summary.csv").dropna(subset=["mcp95_km2", "mcp100_km2"]).sort_values("mcp100_km2")
    y = np.arange(len(df))
    figure_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.hlines(y, df["mcp95_km2"], df["mcp100_km2"], color="#999999", linewidth=1.2, zorder=1)
    ax.scatter(df["mcp100_km2"], y, s=32, facecolor="white", edgecolor="black", linewidth=1.0, marker="o", label="100% MCP", zorder=2)
    ax.scatter(df["mcp95_km2"], y, s=32, facecolor="black", edgecolor="black", linewidth=0.6, marker="o", label="95% MCP", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}  (-{int(n)})" for name, n in zip(df["leopard_name"], df["n_points_dropped"])])
    ax.set_xlabel("MCP area (km$^2$)\nLabel shows locations removed by 95% peripheral removal; pooled 2014-2026 subset", fontsize=8.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "figS2_mcp95_vs_mcp100")


def step_figures() -> None:
    figure1_study_area()
    figure2_seasonal()
    figure3_home_range_map()
    figure4_range_by_sex()
    figure5_overlap()
    figure_s1_window_concordance()
    figure_s2_mcp_estimators()


# =============================================================================
# Step 17. Manuscript numbers
# =============================================================================

def _jsonable(value: object) -> object:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return None if pd.isna(value) else value


def step_manuscript_numbers() -> None:
    """Every number the manuscript quotes, from this one run."""
    read = lambda name: read_table(name, low_memory=False)  # noqa: E731
    raw = read("leopard_images_metadata.csv")
    merged = read("merged_leopard_images_metadata.filtered.corrected.csv")
    all_events = read("merged_leopard_images_metadata.normalized_events.csv")
    events = read("merged_leopard_images_metadata.dated_events.csv")
    detection = read("merged_leopard_individual_detection_table.csv")
    windows = read("merged_image_window_summary.csv")
    sex_stats = read("merged_image_window_sex_stats.csv")
    mcp_summary = read("merged_image_home_ranges_mcp_summary.csv")

    dates = pd.to_datetime(events["image_date"], errors="coerce", dayfirst=True)
    detected = detection[detection["n_encounter_events"] > 0]
    sex_by_individual = detected.set_index("leopard_name")["sex"]  # from Table S2, so text and supplement agree
    primary_sex = sex_stats[sex_stats["window"] == PRIMARY_WINDOW].iloc[0]

    data = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "_source": "leopard_images_metadata.csv, merged_leopard_images_metadata.*",
            "n_image_records_extracted": int(len(raw)),
            "n_image_records_with_gps": int(raw["gps_lat"].notna().sum()),
            "n_gps_image_records_merged_archive": int(len(merged)),
            "n_encounter_events_before_date_exclusion": int(len(all_events)),
            "n_individuals_before_date_exclusion": int(all_events["leopard_name"].nunique()),
            "n_encounter_events": int(len(events)),
            "n_individuals_photographed": int(len(detected)),
            "n_female_individuals": int((sex_by_individual == "Female").sum()),
            "n_male_individuals": int((sex_by_individual == "Male").sum()),
            "n_unknown_sex_individuals": int((~sex_by_individual.isin(["Female", "Male"])).sum()),
            "first_detection": str(dates.min().date()),
            "last_detection": str(dates.max().date()),
            "study_span_days": int((dates.max() - dates.min()).days),
            "n_unique_sampling_dates": int(dates.nunique()),
            "n_individuals_in_catalogue_incl_no_detections": int(len(detection)),
        },
        "date_sources": {"_source": "date_source_counts.csv", "rows": read("date_source_counts.csv").to_dict(orient="records")},
        "seasonality": {"_source": "merged_image_counts_by_botswana_season.csv", **{
            str(row.botswana_season).lower().replace(" ", "_"): {"n_records": int(row.n_records), "percent": round(float(row.percent_records), 1)}
            for row in read("merged_image_counts_by_botswana_season.csv").itertuples()}},
        "spatial_subset": {"_source": "merged_image_window_summary.csv", "min_unique_points": MIN_UNIQUE_POINTS,
                           "min_unique_dates": MIN_UNIQUE_DATES, "mcp_percent": 95, "windows": windows.to_dict(orient="records"),
                           "primary_window": PRIMARY_WINDOW},
        "range_size_by_sex": {"_source": "merged_image_window_sex_stats.csv",
                              "primary_window": {k: (round(float(primary_sex[k]), 4) if isinstance(primary_sex[k], float) else primary_sex[k]) for k in primary_sex.index},
                              "all_windows": sex_stats.to_dict(orient="records")},
        "sex_test_power": {"_source": "sex_test_power.csv", "rows": read("sex_test_power.csv").to_dict(orient="records")},
        "mcp_estimator_comparison": {
            "_source": "merged_image_home_ranges_mcp_summary.csv",
            "median_mcp95_km2": round(float(mcp_summary["mcp95_km2"].median()), 3),
            "median_mcp100_km2": round(float(mcp_summary["mcp100_km2"].median()), 3),
            "median_pct_area_reduction": round(float((100 * (1 - mcp_summary["mcp95_km2"] / mcp_summary["mcp100_km2"])).median()), 1),
            "max_points_dropped": int(mcp_summary["n_points_dropped"].max()),
        },
        "overlap": {"_source": "overlap_descriptives_by_pair_type.csv, overlap_model_results.csv",
                    "descriptives": read("overlap_descriptives_by_pair_type.csv").to_dict(orient="records"),
                    "model_results": read("overlap_model_results.csv").to_dict(orient="records")},
        "overlap_null_model": {"_source": f"overlap_null_model_summary__{PRIMARY_WINDOW}.csv",
                               "regions": read(f"overlap_null_model_regions__{PRIMARY_WINDOW}.csv").to_dict(orient="records"),
                               "results": read(f"overlap_null_model_summary__{PRIMARY_WINDOW}.csv").to_dict(orient="records")},
        "reviewer_decomposition": {"_source": "reviewer_decomposition.csv", "steps": read("reviewer_decomposition.csv").to_dict(orient="records")},
        "software": {"_source": "requirements.txt / venv", **SOFTWARE_VERSIONS},
    }
    table("manuscript_numbers.json").write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")

    ds = data["dataset"]
    frame = lambda rows, digits: pd.DataFrame(rows).round(digits).to_markdown(index=False)  # noqa: E731
    lines = [
        "# Manuscript numbers", "", f"Build: {data['build_timestamp_utc']}", "",
        "Every number quoted in the manuscript appears below. Regenerate with `python analysis.py`.", "",
        "## Dataset", "",
        f"- {ds['n_image_records_extracted']:,} image records extracted, {ds['n_image_records_with_gps']:,} with valid GPS (original archive)",
        f"- {ds['n_gps_image_records_merged_archive']:,} GPS-referenced image records in the merged archive",
        f"- {ds['n_encounter_events_before_date_exclusion']:,} encounter events of {ds['n_individuals_before_date_exclusion']} individuals before date exclusion",
        f"- {ds['n_encounter_events']:,} dated encounter events retained for analysis",
        f"- {ds['n_individuals_photographed']} individually identified leopards ({ds['n_female_individuals']} females, "
        f"{ds['n_male_individuals']} males, {ds['n_unknown_sex_individuals']} unknown sex)",
        f"- {ds['first_detection']} to {ds['last_detection']} ({ds['study_span_days']:,} days, {ds['n_unique_sampling_dates']} unique sampling dates)",
        "", "## Date sources", "", pd.DataFrame(data["date_sources"]["rows"]).to_markdown(index=False), "", "## Seasonality", "",
    ]
    lines += [f"- {k.replace('_', ' ').title()}: {v['n_records']} ({v['percent']}%)" for k, v in data["seasonality"].items() if not k.startswith("_")]
    lines += ["", "## Windows", "", pd.DataFrame(data["spatial_subset"]["windows"]).to_markdown(index=False),
              "", "## Range size by sex", "", frame(data["range_size_by_sex"]["all_windows"], 4),
              "", "## Sex-test power (80%, alpha 0.05)", "", frame(data["sex_test_power"]["rows"], 3),
              "", "## Overlap descriptives", "", frame(data["overlap"]["descriptives"], 4),
              "", "## Overlap models (permutation inference)", "", frame(data["overlap"]["model_results"], 4),
              "", "## Overlap null model (random placement)", "", frame(data["overlap_null_model"]["regions"], 2), "",
              frame(data["overlap_null_model"]["results"], 4),
              "", "## Decomposition: submitted analysis to revised analysis", "", frame(data["reviewer_decomposition"]["steps"], 4)]
    m = data["mcp_estimator_comparison"]
    lines += ["", "## MCP estimator", "", f"- Median 95% MCP: {m['median_mcp95_km2']} km2", f"- Median 100% MCP: {m['median_mcp100_km2']} km2",
              f"- Median area reduction from peripheral removal: {m['median_pct_area_reduction']}%",
              f"- Most locations dropped for any individual: {m['max_points_dropped']}", "", "## Software", ""]
    lines += [f"- {k}: {v}" for k, v in data["software"].items() if not k.startswith("_")]
    table("manuscript_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Step 18. Supplementary Tables
# =============================================================================

def supplementary_tables() -> dict[str, pd.DataFrame]:
    cameras = read_table("merged_leopard_images_metadata_camera_summary.csv")
    s1 = (cameras[cameras["exif_make"] != "Unknown"]
          .rename(columns={"exif_make": "Make", "exif_model": "Model", "n_images": "Images", "n_individuals": "Individuals"})
          [["Make", "Model", "Images", "Individuals"]].sort_values(["Make", "Model"]))

    s2_columns = {
        "leopard_name": "Leopard", "sex": "Sex", "age_class": "Age class", "first_detection": "First detection",
        "last_detection": "Last detection", "n_encounter_events": "Encounter events", "n_unique_dates": "Unique dates",
        "n_unique_locations": "Unique locations", "n_years_detected": "Years detected", "mcp95_km2": "95% MCP (km2)",
        "mcp100_km2": "100% MCP (km2)", "n_points_dropped": "Locations removed", "inclusion_status": "Status",
        "w24_n_unique_locations": "2024-26 unique locations", "w24_n_unique_dates": "2024-26 unique dates",
        "w24_mcp95_km2": "2024-26 95% MCP (km2)", "w24_mcp100_km2": "2024-26 100% MCP (km2)", "w24_included": "2024-26 included",
    }
    s2 = read_table("merged_leopard_individual_detection_table.csv").rename(columns=s2_columns)
    s2["2024-26 included"] = s2["2024-26 included"].map({True: "Yes", False: ""})
    for column in ("95% MCP (km2)", "100% MCP (km2)", "2024-26 95% MCP (km2)", "2024-26 100% MCP (km2)"):
        s2[column] = pd.to_numeric(s2[column], errors="coerce").round(1)
    s2 = s2[list(s2_columns.values())]

    windows = read_table("merged_image_window_summary.csv")
    sex = read_table("merged_image_window_sex_stats.csv")
    desc = read_table("overlap_descriptives_by_pair_type.csv")
    models = read_table("overlap_model_results.csv")
    labels = {"contemporary_24mo": "24 months", "recent_12mo": "12 months", "pooled": "Pooled"}
    s3_rows = []
    for window in windows["window"]:
        w = windows[windows["window"] == window].iloc[0]
        s = sex[sex["window"] == window].iloc[0]
        d = desc[desc["window"] == window].set_index("pair_sex")
        p = {part: models[(models["window"] == window) & (models["part"] == part) & (models["test"] == "omnibus_pair_type_LR")]
             for part in ("part1_any_overlap", "part2_extent_given_overlap")}
        s3_rows.append({
            "Timeframe": labels.get(window, window),
            "Period": f"{w['start']} to {w['end']}",
            "Individuals (F/M)": f"{int(w['n_individuals'])} ({int(w['n_female'])}/{int(w['n_male'])})",
            "Dyads": int(w["n_dyads"]),
            "Median 95% MCP F (km2)": round(float(s["median_female_km2"]), 2),
            "Median 95% MCP M (km2)": round(float(s["median_male_km2"]), 2),
            "Mann-Whitney U statistic": round(float(s["mann_whitney_u"]), 0),
            "P-value": round(float(s["p_value"]), 3),
            "FF overlapping (%)": round(float(d.at["Female-Female", "pct_overlapping"]), 1),
            "FM overlapping (%)": round(float(d.at["Female-Male", "pct_overlapping"]), 1),
            "MM overlapping (%)": round(float(d.at["Male-Male", "pct_overlapping"]), 1),
            "Part 1: chance of overlap permutation p-value": round(float(p["part1_any_overlap"]["p_permutation"].iloc[0]), 3) if len(p["part1_any_overlap"]) else None,
            "FF prop. overlap (%)": round(100 * float(d.at["Female-Female", "median_prop_overlap_nonzero"]), 1),
            "FM prop. overlap (%)": round(100 * float(d.at["Female-Male", "median_prop_overlap_nonzero"]), 1),
            "MM prop. overlap (%)": round(100 * float(d.at["Male-Male", "median_prop_overlap_nonzero"]), 1),
            "Part 2: extent of overlap permutation p-value": round(float(p["part2_extent_given_overlap"]["p_permutation"].iloc[0]), 3) if len(p["part2_extent_given_overlap"]) else None,
        })

    s4 = read_table("reviewer_decomposition.csv").rename(columns={
        "step": "Step", "cumulative_change": "Cumulative change", "estimator": "Estimator", "response": "Response",
        "dyads": "Dyads included", "inference": "Inference", "n_FF": "n FF", "n_FM": "n FM", "median_FF": "Median FF",
        "median_FM": "Median FM", "p_focal_FF_vs_FM": "p (FF vs FM)"}).round(4)

    regions = read_table(f"overlap_null_model_regions__{PRIMARY_WINDOW}.csv").set_index("region")
    region_label = {"sampled_1000m": "Encounters + 1 km (primary)", "sampled_500m": "Encounters + 500 m",
                    "roads2013_1000m": "2013 roads + 1 km", "roads2013_2000m": "2013 roads + 2 km"}
    metric_label = {"pct_overlapping": "Dyads overlapping (%)", "mean_prop_overlap": "Mean proportional overlap (%)"}
    group_label = {"all": "All", "Female-Female": "FF", "Female-Male": "FM", "Male-Male": "MM"}
    s5 = pd.DataFrame([{
        "Null area": region_label.get(r.region, r.region),
        "Area (km2)": round(float(regions.loc[r.region, "area_km2"]), 1),
        "Observed centroids inside": f"{int(regions.loc[r.region, 'observed_centroids_inside'])}/{int(regions.loc[r.region, 'n_polygons'])}",
        "Dyads": group_label[r.group], "Metric": metric_label[r.metric], "Observed": round(r.observed, 1),
        "Null median": round(r.null_p50, 1), "Null 5th-95th percentile": f"{r.null_p05:.1f}-{r.null_p95:.1f}",
        "Observed percentile": round(r.observed_percentile, 1), "p (lower)": round(r.p_lower, 3), "p (higher)": round(r.p_upper, 3),
    } for r in read_table(f"overlap_null_model_summary__{PRIMARY_WINDOW}.csv").itertuples()])
    return {"S1": s1, "S2": s2, "S3": pd.DataFrame(s3_rows), "S4": s4, "S5": s5}


CAPTIONS = {
    "S1": "Supplementary Table S1. Camera makes and models used to photograph leopards in the Chitabe archive. Records without EXIF make/model information are omitted.",
    "S2": "Supplementary Table S2. Individually identified leopards: detection history, sampling metrics, MCP areas under both estimators, the number of locations removed by 95% peripheral removal, and inclusion status for the spatial analyses, from dated encounter events. The five columns headed 2024-26 describe each individual within the primary 2024-06-01 to 2026-05-25 window used for the range and overlap analyses. Individuals known from the lineage record but without georeferenced photographic detections are listed with the status \"Excluded: no detections\"; individuals photographed only on images whose sole date was the archive export date (16 March 2026) are listed as \"Excluded: only undatable (archive-export) records\".",
    "S3": "Supplementary Table S3. Home ranges and pairwise overlap statistics recalculated separately over three timeframes, to test whether the results are sensitive to the timeframe used. The 24 months row is the primary analysis, including individuals detected within the most recent, temporally overlapping 2-year period; the 12 months row restricts the analysis to the final year; the pooled row repeats the analysis using every dated record across the full study period as a sensitivity check, and is interpreted as an accumulated space-use footprint rather than evidence of simultaneous occupation. For each timeframe, the table reports the number of individuals and dyads (all possible pairs) included; median 95% minimum convex polygon (MCP) home-range size for females and males, with the Mann-Whitney U statistic and p-value testing whether home-range size differs between the sexes; and overlap statistics for female-female (FF), female-male (FM) and male-male (MM) pairs. Overlapping (%) gives the percentage of all dyads of that pair type whose home ranges share any area at all; prop. overlap (%) is the median percentage of combined home-range area shared, calculated only among the dyads that do overlap. Part 1 permutation p-values test whether the probability of overlapping at all differs among the three pair types; Part 2 permutation p-values test whether the extent of overlap, given that a pair already overlaps, differs among pair types. All permutation p-values are from 10,000 node-label permutations, which account for each individual contributing to multiple pairwise comparisons.",
    "S4": "Supplementary Table S4. Effect of each analytical choice on the focal female-female versus female-male overlap comparison, applied cumulatively to the pooled data used in the original submission. Step A reproduces the submitted analysis; step G is the revised analysis before the date correction, and step H additionally excludes encounter events whose only date was the archive export date.",
    "S5": "Supplementary Table S5. Observed pairwise overlap in the 2024-06-01 to 2026-05-25 window compared with overlap expected under random placement of ranges. Each individual's 95% MCP was rotated by a random angle and relocated so that its centroid fell at a random position within the null area, 2,000 times per area. Null areas are NG31 within the stated distance of any GPS-referenced leopard encounter (all individuals and years) or of the 2013 game-drive road map; the road map does not cover the full network driven in 2024-2026, so road-based areas are sensitivity analyses. Observed centroids inside gives how many of the observed range centroids lie within each area. p (lower) is the one-sided probability of overlap as low as observed under random placement (evidence of avoidance if below 0.05); p (higher) the probability of overlap as high as observed.",
}


def _docx_paragraph(text: object, bold: bool = False, size_half_points: int | None = None) -> str:
    ppr = f'<w:pPr><w:rPr><w:sz w:val="{size_half_points}"/></w:rPr></w:pPr>' if size_half_points else ""
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{escape(str(text))}</w:t></w:r></w:p>'


def _docx_table(frame: pd.DataFrame, total_width: int = 15000) -> str:
    n = len(frame.columns)
    widths = [total_width // n] * n
    widths[-1] = total_width - widths[0] * (n - 1)

    def cell(text: object, width: int, header: bool = False) -> str:
        shading = '<w:shd w:val="clear" w:color="auto" w:fill="EFEFEF"/>' if header else ""
        return f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shading}</w:tcPr>{_docx_paragraph(text, header)}</w:tc>'

    borders = "".join(f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="999999"/>' for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
    header = "<w:tr><w:trPr><w:tblHeader/></w:trPr>" + "".join(cell(c, w, True) for c, w in zip(frame.columns, widths)) + "</w:tr>"
    body = "".join("<w:tr>" + "".join(cell("" if pd.isna(v) else v, w) for v, w in zip(row, widths)) + "</w:tr>" for row in frame.itertuples(index=False))
    return (f'<w:tbl><w:tblPr><w:tblW w:w="{total_width}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders></w:tblPr>'
            f'<w:tblGrid>{"".join(f"<w:gridCol w:w=\"{w}\"/>" for w in widths)}</w:tblGrid>{header}{body}</w:tbl>')


def write_docx(path: Path, body: str) -> None:
    """Minimal landscape .docx: paragraphs and bordered tables."""
    ns = "http://schemas.openxmlformats.org"
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{ns}/wordprocessingml/2006/main"><w:body>{body}'
        '<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="850" w:right="850" w:bottom="850" '
        'w:left="850" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>')
    parts = {
        "[Content_Types].xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="{ns}/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>',
        "_rels/.rels": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{ns}/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{ns}/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": document,
        "word/_rels/document.xml.rels": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{ns}/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{ns}/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        "word/styles.xml": f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="{ns}/wordprocessingml/2006/main">'
            '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults></w:styles>',
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in parts.items():
            zf.writestr(name, content)


def step_supplementary_tables() -> None:
    tables = supplementary_tables()
    markdown = ["# Supplementary Tables", ""]
    body = [_docx_paragraph("Supplementary Tables", bold=True, size_half_points=28)]
    for key, frame in tables.items():
        markdown += [CAPTIONS[key], "", frame.to_markdown(index=False), "", ""]
        body += [_docx_paragraph(CAPTIONS[key], bold=True), _docx_table(frame), _docx_paragraph("")]
    (OUTPUTS / "supplementary_tables.md").write_text("\n".join(markdown), encoding="utf-8")
    write_docx(OUTPUTS / "Supplementary_Tables.docx", "".join(body))


# =============================================================================
# Run
# =============================================================================

STEPS = [
    ("Photo metadata", step_photo_metadata),
    ("Sightings log", step_sightings_match),
    ("Batch identities", step_batch_identities),
    ("Encounter events", step_encounter_events),
    ("Camera summary", step_camera_summary),
    ("Date validity", step_date_validity),
    ("Seasonality", step_seasonality),
    ("Inclusion", step_inclusion),
    ("Pooled home ranges", step_pooled_home_ranges),
    ("Time windows", step_time_windows),
    ("Overlap models", step_overlap_models),
    ("Null model", step_null_model),
    ("Power", step_power),
    ("Reviewer decomposition", step_reviewer_decomposition),
    ("Detection table", step_detection_table),
    ("Figures", step_figures),
    ("Manuscript numbers", step_manuscript_numbers),
    ("Supplementary Tables", step_supplementary_tables),
]


def main() -> None:
    missing = [p.name for p in (PHOTO_ZIP, BATCH_ZIP, FILE_NAMES_LIST, SIGHTINGS_XLSX, LINEAGE_XLSX, ROADS_GEOJSON, NG31_GEOJSON, NG31_EDIT_SHP) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing input files in {DATA}: {', '.join(missing)} (see README.md)")
    for folder in (TABLES, FIGS, CACHE):
        folder.mkdir(parents=True, exist_ok=True)
    for number, (name, step) in enumerate(STEPS, start=1):
        print(f"[{number:2d}/{len(STEPS)}] {name}", flush=True)
        step()
    print(f"Done. Numbers: {table('manuscript_numbers.md')}")


if __name__ == "__main__":
    main()
