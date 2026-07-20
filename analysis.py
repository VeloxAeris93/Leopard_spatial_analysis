"""
Chitabe Leopards: spatial ecology analysis pipeline.

Companion code for the Chitabe leopard spatial-ecology paper. Reproduces the
full analysis from raw inputs (camera/guide photo archive, sightings log,
lineage table) through to the tables and figures reported in the paper:

    photo archive + sightings log
        -> image metadata extraction (EXIF/GPS)
        -> sightings cleaning
        -> match images to sightings (identity + location correction)
        -> normalize repeat photos into discrete detection "events"
        -> temporal summaries (monthly / seasonal / yearly)
        -> spatial subset (individuals with enough data for a home range)
        -> minimum convex polygon (MCP) home ranges
        -> pairwise home-range overlap
        -> sex-based and pair-type statistical comparisons
        -> per-individual detection table

Run with: python analysis.py
Inputs are read from ./data (not included in this repository - see README.md
for the expected files and columns). Outputs are written to ./outputs.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from scipy import stats
from shapely.geometry import Polygon
from statsmodels.stats.multitest import multipletests

# ============================================================================
# 1. Configuration
# ============================================================================

DATA_DIR = Path("data")
PHOTO_ARCHIVE_PATH = DATA_DIR / "leopard_photos.zip"
SIGHTINGS_LOG_PATH = DATA_DIR / "sightings_log.xlsx"
LINEAGE_PATH = DATA_DIR / "leopard_lineage.xlsx"

OUT_TABLES = Path("outputs") / "tables"
OUT_FIGURES = Path("outputs") / "figures"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# Study-area bounding box used to sanity-check sightings GPS (generous window
# around the Okavango Delta / Chitabe concession).
STUDY_AREA_LAT = (-20.5, -18.0)
STUDY_AREA_LON = (22.0, 24.8)

# A photo <-> sightings-log match this close (metres) is treated as the same
# encounter, so the sightings-log date/behaviour/guide notes can be attached
# to the photo record.
HIGH_CONFIDENCE_MATCH_DISTANCE_M = 50.0

# Some images in the source archive had their real capture date overwritten
# with the archive-preparation date during zip creation. Rows still carrying
# this exact date, sourced only from plain EXIF DateTime (i.e. no better GPS
# or filename date was available), are corrected using the matched
# sightings-log date instead. Set to None to disable this correction for a
# different archive.
SUSPICIOUS_ARCHIVE_DATE = "2026-03-16"

# Repeat photos of the same leopard on the same day, taken within this
# distance of one another, are collapsed into a single detection "event"
# so the analysis reflects independent encounters rather than shutter count.
EVENT_CLUSTER_THRESHOLD_M = 250.0

# An individual needs at least this many unique GPS points and unique dates
# before its home range is considered robust enough to include in the main
# spatial comparison.
MIN_UNIQUE_POINTS = 10
MIN_UNIQUE_DATES = 3

# Botswana wet/dry season definition used for seasonal summaries.
WET_SEASON_MONTHS = {11, 12, 1, 2, 3}
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

SEX_COLORS = {"Female": "#7e8f68", "Male": "#b99254"}
PAIR_COLORS = {"Female-Female": "#2b7bba", "Female-Male": "#7d6b91", "Male-Male": "#d97a1b"}


def set_pub_style() -> None:
    """Matplotlib rcParams for consistent, publication-ready figures."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ============================================================================
# 2. Shared helpers: geometry and statistics
# ============================================================================

@dataclass(frozen=True)
class Pt:
    x: float
    y: float


def meters_projection(lat: float, lon: float, lat0: float) -> Pt:
    """Local equirectangular projection (metres) centred on latitude lat0.

    Good enough for the small (tens-of-km) extent of a single home range;
    avoids pulling in a full projection/CRS library for this step.
    """
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    return Pt(x=lon * m_per_deg_lon, y=lat * m_per_deg_lat)


def inv_meters_projection(p: Pt, lat0: float) -> tuple[float, float]:
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    return p.y / m_per_deg_lat, p.x / m_per_deg_lon  # (lat, lon)


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def convex_hull(points: list[Pt]) -> list[Pt]:
    """Andrew's monotone chain convex hull."""
    pts = sorted(set((p.x, p.y) for p in points))
    pts = [Pt(x, y) for x, y in pts]
    if len(pts) <= 1:
        return pts

    lower: list[Pt] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Pt] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def polygon_area_sq_m(poly: list[Pt]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y
    return abs(area) / 2.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2.0 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def resolve_sex(series: pd.Series) -> str:
    """Collapse a group's sex labels to a single value, or 'Unknown' if mixed."""
    values = [str(v) for v in series.dropna() if str(v).strip() and str(v) != "Unknown"]
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "Unknown"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    return float((x > y).mean() - (x < y).mean())


def bootstrap_median_diff(x: np.ndarray, y: np.ndarray, n: int = 20_000, seed: int = 42) -> tuple[float, float, float]:
    """Median(x) - median(y) with a 95% percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    diffs = np.empty(n, dtype=float)
    for i in range(n):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.median(xb) - np.median(yb)
    return float(np.median(x) - np.median(y)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ============================================================================
# 3. Stage: extract image metadata (EXIF GPS + date) from the photo archive
# ============================================================================

def _infer_leopard_and_sex_from_path(rel_path: str) -> tuple[str | None, str | None]:
    """Leopard name + sex from the archive's top-level folder name, e.g.
    'Chikara Female/IMG_001.jpg' -> ('Chikara', 'Female')."""
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    if len(parts) <= 1:
        return None, None
    label = parts[0]
    low = label.lower()
    sex = "Female" if "female" in low else ("Male" if "male" in low else None)
    name = re.sub(r"\b(male|female)\b", "", label, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return (name or None), sex


def _exif_to_dict(img: Image.Image) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        exif = img.getexif()
    except Exception:
        return out
    if not exif:
        return out

    tag_map = {v: k for k, v in ExifTags.TAGS.items()}
    gps_tag_id = tag_map.get("GPSInfo")
    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if tag_name == "GPSInfo":
            try:
                gps_ifd = exif.get_ifd(gps_tag_id) if gps_tag_id is not None else {}
            except Exception:
                gps_ifd = {}
            for gps_key, gps_val in gps_ifd.items():
                out[f"gps_{ExifTags.GPSTAGS.get(gps_key, str(gps_key))}"] = gps_val
        else:
            out[tag_name] = value
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
        pass
    return None


def _gps_dms_to_dd(dms: Any, ref: Any) -> float | None:
    try:
        if not isinstance(dms, (list, tuple)) or len(dms) != 3:
            return None
        deg = _rational_to_float(dms[0])
        minutes = _rational_to_float(dms[1])
        seconds = _rational_to_float(dms[2])
        if deg is None or minutes is None or seconds is None:
            return None
        dd = deg + minutes / 60.0 + seconds / 3600.0
        return -dd if str(ref).upper() in ("S", "W") else dd
    except Exception:
        return None


def _extract_lat_lon(exif_flat: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = lon = None
    if "gps_GPSLatitude" in exif_flat and "gps_GPSLatitudeRef" in exif_flat:
        lat = _gps_dms_to_dd(exif_flat["gps_GPSLatitude"], exif_flat["gps_GPSLatitudeRef"])
    if "gps_GPSLongitude" in exif_flat and "gps_GPSLongitudeRef" in exif_flat:
        lon = _gps_dms_to_dd(exif_flat["gps_GPSLongitude"], exif_flat["gps_GPSLongitudeRef"])
    return lat, lon


def _parse_datetime_value(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.match(r"^\d{4}:\d{2}:\d{2} ", text):
        text = text.replace(":", "-", 2)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_gps_timestamp(date_value: Any, time_value: Any) -> datetime | None:
    date_text = str(date_value).strip() if date_value is not None else ""
    normalized_date = date_text.replace("-", ":")
    if not re.match(r"^\d{4}:\d{2}:\d{2}$", normalized_date):
        return None
    try:
        base = datetime.strptime(normalized_date, "%Y:%m:%d")
    except ValueError:
        return None
    if isinstance(time_value, (list, tuple)) and len(time_value) >= 3:
        try:
            hours = int(float(time_value[0]))
            minutes = int(float(time_value[1]))
            seconds = int(float(time_value[2]))
            return base.replace(hour=hours, minute=minutes, second=seconds)
        except Exception:
            return base
    return base


def _parse_datetime_from_filename(rel_path: str) -> datetime | None:
    stem = Path(rel_path).stem
    match = re.search(
        r"(?P<date>\d{4}-\d{2}-\d{2})(?:[ _-](?:at[ _-])?(?P<h>\d{2})[.\-:](?P<m>\d{2})[.\-:](?P<s>\d{2}))?",
        stem,
    )
    if not match:
        return None
    try:
        base = datetime.strptime(match.group("date"), "%Y-%m-%d")
    except ValueError:
        return None
    if match.group("h"):
        try:
            return base.replace(hour=int(match.group("h")), minute=int(match.group("m")), second=int(match.group("s")))
        except ValueError:
            return base
    return base


def _select_image_timestamp(rel_path: str, exif_flat: dict[str, Any], zip_modified_time: str | None) -> tuple[str | None, str]:
    """Pick the most trustworthy available date: GPS timestamp, then the
    date encoded in the filename, then plain EXIF date, then the zip
    archive's file-modified time."""
    gps_dt = _parse_gps_timestamp(exif_flat.get("gps_GPSDateStamp"), exif_flat.get("gps_GPSTimeStamp"))
    if gps_dt is not None:
        return gps_dt.isoformat(sep=" "), "gps"

    filename_dt = _parse_datetime_from_filename(rel_path)
    if filename_dt is not None:
        return filename_dt.isoformat(sep=" "), "filename"

    for source, value in (
        ("exif_datetime_original", exif_flat.get("DateTimeOriginal")),
        ("exif_datetime", exif_flat.get("DateTime")),
        ("zip_modified_time", zip_modified_time),
    ):
        if _parse_datetime_value(value) is not None:
            return str(value), source
    return None, "unknown"


def _load_lineage_sex_map(lineage_path: Path) -> dict[str, str]:
    if not lineage_path.exists():
        return {}
    lineage = pd.read_excel(lineage_path, sheet_name="Lineage")
    sex_map: dict[str, str] = {}
    for row in lineage.itertuples(index=False):
        key = normalize_name(getattr(row, "leopard_name", None))
        sex = str(getattr(row, "sex", "") or "").strip().title()
        if key and sex in {"Female", "Male"}:
            sex_map[key] = sex
    return sex_map


def normalize_name(value: Any) -> str | None:
    """Lower-case, de-punctuated leopard name key used to match records
    for the same individual across the photo archive, sightings log and
    lineage table."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower().replace("_", " ")
    text = re.sub(r"\b(male|female)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_image_metadata(zip_path: Path, lineage_path: Path) -> pd.DataFrame:
    """Walk the photo archive and build one row per image with its inferred
    leopard identity, sex, capture date/time and GPS coordinates."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Photo archive not found: {zip_path.resolve()}")

    lineage_sex_map = _load_lineage_sex_map(lineage_path)
    rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            rel_path = info.filename
            if Path(rel_path).suffix.lower() not in IMAGE_EXTS:
                continue

            leopard_name, sex = _infer_leopard_and_sex_from_path(rel_path)
            try:
                zip_modified_time = datetime(*info.date_time).isoformat(sep=" ")
            except Exception:
                zip_modified_time = None

            exif_flat: dict[str, Any] = {}
            lat = lon = None
            try:
                with Image.open(io.BytesIO(z.read(info))) as img:
                    exif_flat = _exif_to_dict(img)
                    lat, lon = _extract_lat_lon(exif_flat)
            except Exception:
                pass

            image_datetime, datetime_source = _select_image_timestamp(rel_path, exif_flat, zip_modified_time)
            dt = _parse_datetime_value(image_datetime)

            if sex is None and leopard_name is not None:
                sex = lineage_sex_map.get(normalize_name(leopard_name))

            rows.append(
                {
                    "file_name": Path(rel_path).name,
                    "leopard_name": leopard_name,
                    "sex": sex,
                    "image_datetime": image_datetime,
                    "image_date": dt.date().isoformat() if dt else None,
                    "image_time": dt.strftime("%H:%M:%S") if dt else None,
                    "image_datetime_source": datetime_source,
                    "gps_lat": lat,
                    "gps_lon": lon,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Extracted metadata for {len(df)} images ({df['gps_lat'].notna().sum()} with GPS coordinates)")
    return df


# ============================================================================
# 4. Stage: clean the sightings log
# ============================================================================

def clean_sightings(sightings_path: Path) -> pd.DataFrame:
    """Build a usable date column, fix lat/long swaps, and flag records
    that fall outside the study area."""
    if not sightings_path.exists():
        raise FileNotFoundError(f"Sightings log not found: {sightings_path.resolve()}")

    df = pd.read_excel(sightings_path, sheet_name="Sheet1")
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]), errors="coerce")
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df["has_gps"] = df["Lat"].notna() & df["Long"].notna()

    lat_min, lat_max = STUDY_AREA_LAT
    lon_min, lon_max = STUDY_AREA_LON
    swapped = df["has_gps"] & df["Lat"].between(lon_min, lon_max) & df["Long"].between(lat_min, lat_max)
    df.loc[swapped, ["Lat", "Long"]] = df.loc[swapped, ["Long", "Lat"]].to_numpy()

    df["in_bounds"] = df["has_gps"] & df["Lat"].between(lat_min, lat_max) & df["Long"].between(lon_min, lon_max)
    print(f"Cleaned {len(df)} sightings ({df['in_bounds'].sum()} with usable in-bounds GPS)")
    return df


# ============================================================================
# 5. Stage: match photo records to the sightings log
# ============================================================================

def match_images_to_sightings(images: pd.DataFrame, sightings: pd.DataFrame) -> pd.DataFrame:
    """For each image with GPS, find the closest same-individual sightings
    log entry. Close matches (<= HIGH_CONFIDENCE_MATCH_DISTANCE_M) lend
    their date/behaviour/guide fields to the image record, and are used to
    correct images stuck on the suspicious archive date."""
    images = images.copy()
    images["name_key"] = images["leopard_name"].map(normalize_name)

    sightings = sightings.copy()
    sightings["name_key"] = sightings["Leopard"].map(normalize_name)
    sightings["sighting_date"] = pd.to_datetime(
        dict(year=sightings["Year"], month=sightings["Month"], day=sightings["Day"]), errors="coerce"
    )
    grouped_sightings = {
        key: group.dropna(subset=["Lat", "Long"]).reset_index(drop=True)
        for key, group in sightings.groupby("name_key", dropna=True)
    }

    rows: list[dict[str, Any]] = []
    for row in images.itertuples(index=False):
        record = row._asdict()
        record["sighting_match_distance_m"] = None
        record["sighting_date"] = None
        record["use_sighting_date_correction"] = False

        candidates = grouped_sightings.get(record.get("name_key"))
        has_gps = pd.notna(record.get("gps_lat")) and pd.notna(record.get("gps_lon"))
        if candidates is not None and not candidates.empty and has_gps:
            distances = candidates.apply(
                lambda c: haversine_m(record["gps_lat"], record["gps_lon"], c["Lat"], c["Long"]), axis=1
            )
            best_idx = distances.idxmin()
            best_distance = float(distances.loc[best_idx])
            best = candidates.loc[best_idx]

            record["sighting_match_distance_m"] = round(best_distance, 2)
            if best_distance <= HIGH_CONFIDENCE_MATCH_DISTANCE_M:
                record["sighting_date"] = best["sighting_date"]

                suspicious = (
                    SUSPICIOUS_ARCHIVE_DATE is not None
                    and record.get("image_date") == SUSPICIOUS_ARCHIVE_DATE
                    and record.get("image_datetime_source") == "exif_datetime"
                    and pd.notna(best["sighting_date"])
                )
                if suspicious:
                    record["image_date"] = best["sighting_date"].date().isoformat()
                    record["image_datetime"] = best["sighting_date"].isoformat(sep=" ")
                    record["image_datetime_source"] = "sighting_log_date_match"
                    record["use_sighting_date_correction"] = True

        rows.append(record)

    matched = pd.DataFrame(rows).drop(columns=["name_key"], errors="ignore")
    n_corrected = matched["use_sighting_date_correction"].sum()
    print(f"Matched {matched['sighting_match_distance_m'].notna().sum()} images to sightings log rows "
          f"({n_corrected} suspicious dates corrected)")
    return matched


# ============================================================================
# 6. Stage: normalize repeat photos into discrete detection events
# ============================================================================

def _cluster_by_distance(points_ll: list[tuple[float, float]], threshold_m: float) -> list[list[int]]:
    if len(points_ll) <= 1:
        return [list(range(len(points_ll)))]
    lat0 = sum(lat for lat, _ in points_ll) / len(points_ll)
    pts_m = [meters_projection(lat, lon, lat0) for lat, lon in points_ll]

    uf = UnionFind(len(pts_m))
    for i in range(len(pts_m)):
        for j in range(i + 1, len(pts_m)):
            if math.hypot(pts_m[i].x - pts_m[j].x, pts_m[i].y - pts_m[j].y) <= threshold_m:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(len(pts_m)):
        clusters.setdefault(uf.find(idx), []).append(idx)
    return list(clusters.values())


def normalize_events(matched: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeat photos of the same leopard, on the same day, taken
    close together in space, into one representative detection event."""
    df = matched.copy()
    df["leopard_name"] = df["leopard_name"].fillna("Unknown")
    df["sex"] = df["sex"].fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df["gps_lat"], errors="coerce")
    df["gps_lon"] = pd.to_numeric(df["gps_lon"], errors="coerce")
    df["image_date_parsed"] = pd.to_datetime(df["image_date"], errors="coerce")
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", "image_date_parsed"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(resolve_sex)

    event_rows: list[dict[str, Any]] = []
    event_counter = 1
    for (_, image_date), group in df.groupby(["leopard_name", "image_date_parsed"], sort=True):
        group = group.reset_index(drop=True)
        points_ll = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
        for indices in _cluster_by_distance(points_ll, EVENT_CLUSTER_THRESHOLD_M):
            cluster = group.iloc[indices]
            rep = cluster.sort_values(["image_time", "file_name"], na_position="last").iloc[0].to_dict()
            rep["event_id"] = f"EVT{event_counter:05d}"
            rep["n_images_in_event"] = int(len(cluster))
            rep["gps_lat"] = float(cluster["gps_lat"].median())
            rep["gps_lon"] = float(cluster["gps_lon"].median())
            rep["image_date"] = pd.Timestamp(image_date).date().isoformat()
            event_rows.append(rep)
            event_counter += 1

    events = pd.DataFrame(event_rows).drop(columns=["image_date_parsed"], errors="ignore")
    print(f"Normalized {len(df)} raw photo records into {len(events)} detection events")
    return events


# ============================================================================
# 7. Stage: temporal analysis (monthly / seasonal / yearly patterns)
# ============================================================================

def temporal_analysis(events: pd.DataFrame) -> None:
    df = events.copy()
    df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")
    df = df.dropna(subset=["image_date"]).copy()
    df["year"] = df["image_date"].dt.year
    df["month_name"] = pd.Categorical(df["image_date"].dt.month_name(), categories=MONTH_ORDER, ordered=True)
    df["season"] = np.where(df["image_date"].dt.month.isin(WET_SEASON_MONTHS), "Wet season", "Dry season")

    timeline = (
        df.groupby(["leopard_name", "sex"], dropna=False)
        .agg(n_events=("event_id", "nunique"), first_seen=("image_date", "min"), last_seen=("image_date", "max"))
        .reset_index()
        .sort_values("n_events", ascending=False)
    )
    month_counts = df.groupby("month_name", observed=True).size().reindex(MONTH_ORDER, fill_value=0)
    season_counts = df.groupby("season").size().reindex(["Wet season", "Dry season"], fill_value=0)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(OUT_TABLES / "leopard_timeline_summary.csv", index=False)
    month_counts.rename("n_events").to_csv(OUT_TABLES / "detections_by_month.csv")
    season_counts.rename("n_events").to_csv(OUT_TABLES / "detections_by_season.csv")

    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    top = timeline.head(25).sort_values("first_seen")
    plt.figure(figsize=(8.5, 6.5))
    for i, row in enumerate(top.itertuples(index=False)):
        plt.plot([row.first_seen, row.last_seen], [i, i], linewidth=3)
    plt.yticks(range(len(top)), [f"{r.leopard_name} ({r.sex})" for r in top.itertuples(index=False)])
    plt.xlabel("Date")
    plt.title("Detection timelines: top 25 individuals by event count")
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "fig_timelines.png")
    plt.close()

    plt.figure(figsize=(9.0, 4.6))
    plt.bar(MONTH_ORDER, month_counts.to_numpy(), color="#7e8f68")
    plt.title("Detections by month")
    plt.ylabel("Detection events")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "fig_detections_by_month.png")
    plt.close()

    plt.figure(figsize=(6.0, 4.6))
    plt.bar(season_counts.index, season_counts.to_numpy(), color=["#7e8f68", "#b99254"])
    plt.title("Detections by Botswana season")
    plt.ylabel("Detection events")
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "fig_detections_by_season.png")
    plt.close()

    print(f"Saved -> {OUT_TABLES / 'leopard_timeline_summary.csv'}")
    print(f"Saved -> {OUT_TABLES / 'detections_by_month.csv'}")
    print(f"Saved -> {OUT_TABLES / 'detections_by_season.csv'}")


# ============================================================================
# 8. Stage: spatial subset (enough data for a robust home range)
# ============================================================================

def spatial_subset(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only individuals with >= MIN_UNIQUE_POINTS unique GPS points and
    >= MIN_UNIQUE_DATES unique dates, so sparse records don't produce an
    over-interpreted home range."""
    df = events.copy()
    df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")

    def _unique_points(g: pd.DataFrame) -> int:
        return len(set(zip(g["gps_lat"].astype(float), g["gps_lon"].astype(float))))

    individual_summary = (
        df.groupby(["leopard_name", "sex"], dropna=False)
        .agg(n_events=("event_id", "nunique"), n_unique_dates=("image_date", "nunique"))
        .reset_index()
    )
    unique_points = df.groupby("leopard_name").apply(_unique_points).rename("n_unique_points").reset_index()
    individual_summary = individual_summary.merge(unique_points, on="leopard_name", how="left")
    individual_summary["include_spatial_subset"] = (
        (individual_summary["n_unique_points"] >= MIN_UNIQUE_POINTS)
        & (individual_summary["n_unique_dates"] >= MIN_UNIQUE_DATES)
    )

    keep_names = set(individual_summary.loc[individual_summary["include_spatial_subset"], "leopard_name"])
    subset = df[df["leopard_name"].isin(keep_names)].copy()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    individual_summary.to_csv(OUT_TABLES / "individual_summary.csv", index=False)
    print(f"Spatial subset rule: >= {MIN_UNIQUE_POINTS} unique points and >= {MIN_UNIQUE_DATES} unique dates")
    print(f"Retained {len(keep_names)} of {individual_summary.shape[0]} individuals "
          f"({len(subset)} of {len(df)} detection events)")
    return subset, individual_summary


# ============================================================================
# 9. Stage: minimum convex polygon (MCP) home ranges
# ============================================================================

def compute_home_ranges(subset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[tuple[float, float]]]]:
    """MCP home range (km^2) per individual, plus its hull as a list of
    (lat, lon) ring vertices for the overlap and mapping stages."""
    lat0 = float(subset["gps_lat"].median())
    summary_rows: list[dict[str, Any]] = []
    hulls: dict[str, list[tuple[float, float]]] = {}

    for leopard, group in subset.groupby("leopard_name", dropna=False):
        sex = str(group["sex"].iloc[0])
        unique_pts_ll = sorted(set(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float))))

        row: dict[str, Any] = {
            "leopard_name": leopard,
            "sex": sex,
            "n_events": int(len(group)),
            "n_unique_points": len(unique_pts_ll),
            "home_range_km2_mcp": None,
        }

        if len(unique_pts_ll) >= 3:
            pts_m = [meters_projection(lat, lon, lat0) for lat, lon in unique_pts_ll]
            hull_m = convex_hull(pts_m)
            row["home_range_km2_mcp"] = polygon_area_sq_m(hull_m) / 1_000_000.0
            hulls[leopard] = [inv_meters_projection(p, lat0) for p in hull_m]

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("home_range_km2_mcp", ascending=False, na_position="last")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLES / "home_ranges_mcp.csv", index=False)
    print(f"Saved -> {OUT_TABLES / 'home_ranges_mcp.csv'}")
    return summary, hulls


# ============================================================================
# 10. Stage: pairwise home-range overlap
# ============================================================================

def compute_overlap(summary: pd.DataFrame, hulls: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    if not hulls:
        raise RuntimeError("No home ranges available to compute overlap from.")

    lat0 = next(iter(hulls.values()))[0][0]
    sex_by_name = dict(zip(summary["leopard_name"], summary["sex"]))

    polygons: dict[str, Polygon] = {}
    for name, ring_ll in hulls.items():
        ring_m = [(meters_projection(lat, lon, lat0).x, meters_projection(lat, lon, lat0).y) for lat, lon in ring_ll]
        poly = Polygon(ring_m)
        if poly.is_valid and not poly.is_empty:
            polygons[name] = poly

    names = sorted(polygons)
    pair_rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            poly_a, poly_b = polygons[a], polygons[b]
            inter_km2 = poly_a.intersection(poly_b).area / 1_000_000.0
            union_km2 = poly_a.union(poly_b).area / 1_000_000.0
            area_a, area_b = poly_a.area / 1_000_000.0, poly_b.area / 1_000_000.0
            pair_rows.append(
                {
                    "leopard_a": a,
                    "leopard_b": b,
                    "pair_sex": "-".join(sorted([sex_by_name.get(a, "Unknown"), sex_by_name.get(b, "Unknown")])),
                    "overlap_km2": inter_km2,
                    "jaccard_overlap": (inter_km2 / union_km2) if union_km2 else 0.0,
                    "pct_of_a_overlapped": (inter_km2 / area_a * 100.0) if area_a else 0.0,
                    "pct_of_b_overlapped": (inter_km2 / area_b * 100.0) if area_b else 0.0,
                }
            )

    overlap_pairs = pd.DataFrame(pair_rows).sort_values("overlap_km2", ascending=False).reset_index(drop=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    overlap_pairs.to_csv(OUT_TABLES / "home_range_overlap_pairs.csv", index=False)
    print(f"Saved -> {OUT_TABLES / 'home_range_overlap_pairs.csv'}")
    return overlap_pairs


# ============================================================================
# 11. Stage: statistical comparisons (sex, pair type)
# ============================================================================

def compute_sex_stats(summary: pd.DataFrame) -> pd.DataFrame:
    """Male vs female MCP home-range size: Mann-Whitney U, Cliff's delta,
    and a bootstrap median difference with a 95% CI."""
    base = summary.dropna(subset=["home_range_km2_mcp"])
    base = base[base["sex"].isin(["Female", "Male"])]

    female = base.loc[base["sex"] == "Female", "home_range_km2_mcp"].to_numpy(float)
    male = base.loc[base["sex"] == "Male", "home_range_km2_mcp"].to_numpy(float)
    if len(female) == 0 or len(male) == 0:
        raise RuntimeError("Need at least one female and one male home range to compare.")

    u = stats.mannwhitneyu(male, female, alternative="two-sided")
    delta = cliffs_delta(male, female)
    med_diff, ci_lo, ci_hi = bootstrap_median_diff(male, female)

    result = pd.DataFrame(
        [
            {
                "n_female": len(female),
                "n_male": len(male),
                "median_female_km2": float(np.median(female)),
                "median_male_km2": float(np.median(male)),
                "mann_whitney_u": float(u.statistic),
                "p_value": float(u.pvalue),
                "cliffs_delta_male_minus_female": delta,
                "median_diff_male_minus_female_km2": med_diff,
                "median_diff_ci95_low_km2": ci_lo,
                "median_diff_ci95_high_km2": ci_hi,
            }
        ]
    )
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_TABLES / "home_range_sex_stats.csv", index=False)
    print(f"Saved -> {OUT_TABLES / 'home_range_sex_stats.csv'}")
    return result


def compute_overlap_by_pair_type_stats(overlap_pairs: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis across Female-Female / Female-Male / Male-Male overlap
    (km^2, non-zero pairs only), followed by pairwise Mann-Whitney tests
    with a Holm correction for multiple comparisons."""
    labels = ["Female-Female", "Female-Male", "Male-Male"]
    sub = overlap_pairs[overlap_pairs["pair_sex"].isin(labels) & (overlap_pairs["overlap_km2"] > 0)]
    groups = {label: sub.loc[sub["pair_sex"] == label, "overlap_km2"].to_numpy(float) for label in labels}

    rows: list[dict[str, Any]] = []
    valid_groups = [arr for arr in groups.values() if len(arr) > 0]
    if len(valid_groups) >= 2:
        kw = stats.kruskal(*valid_groups)
        rows.append(
            {
                "comparison": "overall_kruskal_wallis", "group_a": None, "group_b": None,
                "statistic": float(kw.statistic), "p_value": float(kw.pvalue), "p_value_adjusted": float(kw.pvalue),
            }
        )

    pairwise = []
    pvals = []
    for a, b in [("Female-Female", "Female-Male"), ("Female-Female", "Male-Male"), ("Female-Male", "Male-Male")]:
        xa, xb = groups[a], groups[b]
        if len(xa) == 0 or len(xb) == 0:
            continue
        test = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        pvals.append(test.pvalue)
        pairwise.append(
            {
                "comparison": "pairwise_mann_whitney", "group_a": a, "group_b": b,
                "statistic": float(test.statistic), "p_value": float(test.pvalue),
            }
        )
    if pairwise:
        adjusted = multipletests(pvals, method="holm")[1]
        for row, p_adj in zip(pairwise, adjusted):
            row["p_value_adjusted"] = float(p_adj)
            rows.append(row)

    result = pd.DataFrame(rows)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_TABLES / "overlap_by_pair_type_stats.csv", index=False)
    print(f"Saved -> {OUT_TABLES / 'overlap_by_pair_type_stats.csv'}")
    return result


# ============================================================================
# 12. Stage: figures
# ============================================================================

def make_figures(
    summary: pd.DataFrame,
    hulls: dict[str, list[tuple[float, float]]],
    overlap_pairs: pd.DataFrame,
    sex_stats: pd.DataFrame,
) -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    base = summary.dropna(subset=["home_range_km2_mcp"])
    sexed = base[base["sex"].isin(["Female", "Male"])]

    # Top 15 largest home ranges.
    top = base.sort_values("home_range_km2_mcp", ascending=False).head(15).sort_values("home_range_km2_mcp")
    plt.figure(figsize=(8.0, 5.6))
    plt.barh(top["leopard_name"], top["home_range_km2_mcp"], color=[SEX_COLORS.get(s, "#888888") for s in top["sex"]])
    plt.xlabel("MCP home range (km$^2$)")
    plt.title("Top 15 home ranges")
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "fig_top15_home_ranges.png")
    plt.close()

    # Home range size by sex.
    female = sexed.loc[sexed["sex"] == "Female", "home_range_km2_mcp"].to_numpy(float)
    male = sexed.loc[sexed["sex"] == "Male", "home_range_km2_mcp"].to_numpy(float)
    stats_row = sex_stats.iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bp = ax.boxplot([female, male], tick_labels=[f"Female (n={len(female)})", f"Male (n={len(male)})"], patch_artist=True)
    for patch, sex in zip(bp["boxes"], ["Female", "Male"]):
        patch.set(facecolor="white", edgecolor=SEX_COLORS[sex])
    ax.set_ylabel("MCP home range (km$^2$)")
    ax.set_title("Home range size by sex")
    ax.text(
        0.03, 0.98,
        f"Mann-Whitney U = {stats_row['mann_whitney_u']:.0f}, p = {stats_row['p_value']:.3f}\n"
        f"Cliff's delta = {stats_row['cliffs_delta_male_minus_female']:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "fig_home_range_by_sex.png")
    plt.close(fig)

    # Overlap (km^2) by pair type.
    labels = ["Female-Female", "Female-Male", "Male-Male"]
    nonzero = overlap_pairs[overlap_pairs["overlap_km2"] > 0]
    data = [nonzero.loc[nonzero["pair_sex"] == label, "overlap_km2"].to_numpy(float) for label in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bp = ax.boxplot(data, tick_labels=[f"{l}\n(n={len(d)})" for l, d in zip(labels, data)], patch_artist=True)
    for patch, label in zip(bp["boxes"], labels):
        patch.set(facecolor="white", edgecolor=PAIR_COLORS[label])
    ax.set_ylabel("Pairwise overlap (km$^2$)")
    ax.set_title("Non-zero home-range overlap by pair type")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "fig_overlap_by_pair_type.png")
    plt.close(fig)

    # Simple spatial map: home-range hulls and points, coloured by sex.
    # (No basemap tiles are fetched here, to keep this script offline and
    # dependency-light - see README for adding a basemap if you need one.)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    for name, ring_ll in hulls.items():
        sex = sex_by_name = summary.loc[summary["leopard_name"] == name, "sex"].iloc[0]
        color = SEX_COLORS.get(sex_by_name, "#888888")
        lats = [p[0] for p in ring_ll] + [ring_ll[0][0]]
        lons = [p[1] for p in ring_ll] + [ring_ll[0][1]]
        ax.fill(lons, lats, color=color, alpha=0.15, edgecolor=color, linewidth=1.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_title("Home ranges (MCP), by sex")
    handles = [plt.Line2D([0], [0], color=c, lw=4, label=s) for s, c in SEX_COLORS.items()]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "fig_home_range_map.png")
    plt.close(fig)

    print(f"Saved figures -> {OUT_FIGURES}")


# ============================================================================
# 13. Stage: per-individual detection table
# ============================================================================

def export_detection_table(
    events: pd.DataFrame,
    individual_summary: pd.DataFrame,
    home_ranges: pd.DataFrame,
    lineage_path: Path,
) -> pd.DataFrame:
    """Final per-individual summary: identity, sex, detection counts,
    date range, MCP area and whether it met the spatial-subset criteria -
    the table referenced directly in the paper."""
    df = events.copy()
    df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")

    detections = (
        df.groupby("leopard_name", dropna=False)
        .agg(
            sex=("sex", resolve_sex),
            first_detection=("image_date", "min"),
            last_detection=("image_date", "max"),
            n_events=("event_id", "nunique"),
        )
        .reset_index()
    )

    if lineage_path.exists():
        lineage = pd.read_excel(lineage_path, sheet_name="Lineage")
        lineage = lineage.rename(columns={"leopard_name": "leopard_name", "Estimated age": "age_class"})
        lineage = lineage[["leopard_name", "age_class"]].drop_duplicates(subset=["leopard_name"])
        detections = detections.merge(lineage, on="leopard_name", how="left")
    else:
        detections["age_class"] = None

    detections = detections.merge(
        individual_summary[["leopard_name", "include_spatial_subset"]], on="leopard_name", how="left"
    )
    detections = detections.merge(
        home_ranges[["leopard_name", "home_range_km2_mcp"]], on="leopard_name", how="left"
    )
    detections["include_spatial_subset"] = detections["include_spatial_subset"].fillna(False)
    detections["first_detection"] = detections["first_detection"].dt.date.astype("string")
    detections["last_detection"] = detections["last_detection"].dt.date.astype("string")
    detections = detections.sort_values("n_events", ascending=False).reset_index(drop=True)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    out_path = OUT_TABLES / "individual_detection_table.csv"
    detections.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return detections


# ============================================================================
# 14. Pipeline entry point
# ============================================================================

def main() -> None:
    set_pub_style()

    print("\n[1/9] Extracting image metadata from the photo archive")
    images = extract_image_metadata(PHOTO_ARCHIVE_PATH, LINEAGE_PATH)

    print("\n[2/9] Cleaning the sightings log")
    sightings = clean_sightings(SIGHTINGS_LOG_PATH)

    print("\n[3/9] Matching photo records to the sightings log")
    matched = match_images_to_sightings(images, sightings)

    print("\n[4/9] Normalizing repeat photos into detection events")
    events = normalize_events(matched)

    print("\n[5/9] Temporal analysis")
    temporal_analysis(events)

    print("\n[6/9] Building the spatial subset")
    subset, individual_summary = spatial_subset(events)

    print("\n[7/9] Computing MCP home ranges")
    home_ranges, hulls = compute_home_ranges(subset)

    print("\n[8/9] Computing overlap and statistics")
    overlap_pairs = compute_overlap(home_ranges, hulls)
    sex_stats = compute_sex_stats(home_ranges)
    compute_overlap_by_pair_type_stats(overlap_pairs)
    make_figures(home_ranges, hulls, overlap_pairs, sex_stats)

    print("\n[9/9] Exporting the per-individual detection table")
    export_detection_table(events, individual_summary, home_ranges, LINEAGE_PATH)

    print("\nDone. Tables in ./outputs/tables, figures in ./outputs/figures.")


if __name__ == "__main__":
    main()
