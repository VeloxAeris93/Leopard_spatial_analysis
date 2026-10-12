from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from _paths import BATCH_ZIPS, LINEAGE_XLSX, PHOTO_ZIP, find_exiftool

try:
    from PIL import Image, ExifTags
except ImportError as e:
    raise SystemExit("Pillow not installed. Run: python -m pip install pillow") from e


OUT_XLSX = Path("outputs") / "tables" / "leopard_images_metadata.xlsx"
OUT_CSV = Path("outputs") / "tables" / "leopard_images_metadata.csv"
OUT_FILTERED_CSV = Path("outputs") / "tables" / "leopard_images_metadata.filtered.csv"
# Photos delivered as separate batches, not filed under a leopard folder (see 16c).
OUT_BATCH_CSV = Path("outputs") / "tables" / "leopard_images_metadata.batches.csv"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

GENERIC_NOTE_TEXTS = {
    "digital camera",
    "olympus digital camera",
}

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


# Input locations live in _paths.py (data/ first, then the project folder layout).
ZIP_PATH = PHOTO_ZIP
EXIFTOOL_PATH = find_exiftool()


def _safe_str(x: Any) -> str | None:
    if x is None:
        return None
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None


def _normalize_leopard_key(value: Any) -> str | None:
    text = _safe_str(value)
    if not text:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    return normalized or None


def _load_lineage_sex_map() -> dict[str, str]:
    if not LINEAGE_XLSX.exists():
        return {}

    lineage = pd.read_excel(LINEAGE_XLSX, sheet_name="Lineage")
    sex_map: dict[str, str] = {}
    for row in lineage.itertuples(index=False):
        key = _normalize_leopard_key(getattr(row, "leopard_name", None))
        sex = _safe_str(getattr(row, "sex", None))
        if not key or not sex:
            continue
        sex_title = sex.title()
        if sex_title in {"Female", "Male"}:
            sex_map[key] = sex_title
    return sex_map


LINEAGE_SEX_MAP = _load_lineage_sex_map()


def _apply_lineage_sex_backfill(df: pd.DataFrame) -> pd.DataFrame:
    if not LINEAGE_SEX_MAP:
        return df

    out = df.copy()
    name_keys = out.get("leopard_name", pd.Series(index=out.index, dtype="object")).map(_normalize_leopard_key)
    mapped_sex = name_keys.map(LINEAGE_SEX_MAP)
    current_sex = out.get("sex", pd.Series(index=out.index, dtype="object")).astype("string")
    missing_sex = current_sex.isna() | current_sex.str.strip().eq("")
    out.loc[missing_sex & mapped_sex.notna(), "sex"] = mapped_sex[missing_sex & mapped_sex.notna()]
    return out


def _parse_zip_dt(dt_tuple: tuple[int, int, int, int, int, int]) -> str | None:
    """Convert ZipInfo.date_time tuple to ISO string."""
    try:
        return datetime(*dt_tuple).isoformat(sep=" ")
    except Exception:
        return None


def _safe_note_str(value: Any) -> str | None:
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
        parts = _dedupe_preserve_order(_safe_note_str(item) for item in value)
        return "; ".join(parts) if parts else None

    text = _safe_str(value)
    if text is None:
        return None

    text = text.replace("\x00", " ")
    text = re.sub(r"^(ASCII|UNICODE|JIS)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ;|")
    return text or None


def _dedupe_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _is_meaningful_note_text(text: str | None) -> bool:
    if not text:
        return False
    normalized = text.casefold()
    if normalized in GENERIC_NOTE_TEXTS:
        return False
    if len(re.findall(r"\b[A-Z]{1,4}\[[^\]]+\]", text)) >= 3:
        return False
    return True


def _note_source_name(key: str) -> str | None:
    suffix = key.split(":")[-1].lower()
    if suffix == "usercomment":
        return "exif_user_comment"
    if suffix == "imagedescription":
        return "exif_image_description"
    if suffix in {"caption-abstract", "captionabstract"}:
        return "iptc_caption_abstract"
    if suffix == "description":
        return "xmp_description"
    if suffix == "subject":
        return "xmp_subject"
    if suffix == "title":
        return "xmp_title"
    if suffix in {"keywords", "xpkeywords"}:
        return "keywords"
    return None


def _filename_note_text(rel_path: str) -> str | None:
    stem = re.sub(r"[_\-.]+", " ", Path(rel_path).stem)
    stem = re.sub(r"\b(female|male|fem)\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return None

    matches = _dedupe_preserve_order(
        match.group(0)
        for _, pattern in NOTE_TAG_PATTERNS
        for match in [pattern.search(stem)]
        if match is not None
    )
    return "; ".join(matches) if matches else None


def _extract_note_tags(*texts: str | None) -> list[str]:
    haystack = " | ".join(text for text in texts if text)
    if not haystack:
        return []
    return [tag for tag, pattern in NOTE_TAG_PATTERNS if pattern.search(haystack)]


def _collect_note_fields(
    rel_path: str,
    exif_flat: dict[str, Any],
    exiftool_data: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    metadata_note_texts: list[str] = []
    note_sources: list[str] = []

    def add_note(source: str, value: Any) -> None:
        text = _safe_note_str(value)
        if not _is_meaningful_note_text(text):
            return
        metadata_note_texts.append(text)
        note_sources.append(source)

    add_note("exif_user_comment", exif_flat.get("exif_UserComment"))
    add_note("exif_image_description", exif_flat.get("exif_ImageDescription"))

    for key, value in exiftool_data.items():
        source = _note_source_name(key)
        if source is None:
            continue
        add_note(source, value)

    metadata_note_texts = _dedupe_preserve_order(metadata_note_texts)
    note_sources = _dedupe_preserve_order(note_sources)

    filename_note_text = _filename_note_text(rel_path)
    if filename_note_text:
        note_sources = _dedupe_preserve_order([*note_sources, "filename"])

    metadata_note_text = "; ".join(metadata_note_texts) if metadata_note_texts else None
    note_tags = _extract_note_tags(metadata_note_text, filename_note_text)
    note_tags_text = "; ".join(note_tags) if note_tags else None
    note_sources_text = "; ".join(note_sources) if note_sources else None

    return metadata_note_text, filename_note_text, note_tags_text, note_sources_text


def _parse_datetime_value(value: Any) -> datetime | None:
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

        for fmt in (
            "%Y:%m:%d %H:%M:%S",
            "%Y:%m:%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
        ):
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
                hours = int(float(time_value[0]))
                minutes = int(float(time_value[1]))
                seconds_float = float(time_value[2])
                seconds = int(seconds_float)
                micros = int(round((seconds_float - seconds) * 1_000_000))
                base = datetime.strptime(normalized_date, "%Y:%m:%d")
                return base.replace(hour=hours, minute=minutes, second=seconds, microsecond=micros)
            except Exception:
                pass
        if match and ":" in time_text:
            parsed = _parse_datetime_value(f"{normalized_date} {time_text}")
            if parsed is not None:
                return parsed

    try:
        return datetime.strptime(normalized_date, "%Y:%m:%d")
    except ValueError:
        return None


def _parse_datetime_from_filename(rel_path: str) -> datetime | None:
    filename = Path(rel_path).name
    stem = Path(filename).stem

    patterns = (
        r"(?P<date>\d{4}-\d{2}-\d{2})[ _-](?:at[ _-])?(?P<time>\d{2})[.\-:](?P<minute>\d{2})[.\-:](?P<second>\d{2})",
        r"(?P<date>\d{4}-\d{2}-\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if not match:
            continue

        date_text = match.group("date")
        try:
            base = datetime.strptime(date_text, "%Y-%m-%d")
        except ValueError:
            continue

        if {"time", "minute", "second"}.issubset(match.groupdict()):
            try:
                return base.replace(
                    hour=int(match.group("time")),
                    minute=int(match.group("minute")),
                    second=int(match.group("second")),
                )
            except ValueError:
                return base
        return base

    return None


def _split_datetime(value: Any) -> tuple[str | None, str | None]:
    dt = _parse_datetime_value(value)
    if dt is None:
        return None, None
    return dt.date().isoformat(), dt.strftime("%H:%M:%S")


def _infer_leopard_and_sex_from_path(rel_path: str) -> tuple[str | None, str | None]:
    """
    Infer leopard name + sex from folder names.

    Strategy:
      - Use the first non-empty directory segment as "folder label"
      - If it contains 'male' or 'female' -> sex
      - Leopard name = folder label with sex words removed
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    if len(parts) <= 1:
        # file at zip root
        label = None
    else:
        label = parts[0]  # top folder

    if not label:
        return None, None

    low = label.lower()

    sex = None
    if "female" in low:
        sex = "Female"
    elif "male" in low:
        sex = "Male"

    # remove sex words and separators from name
    name = re.sub(r"\b(male|female)\b", "", label, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name).strip()
    name = re.sub(r"\s+", " ", name).strip()

    return (name or None), sex


def _exif_to_dict(img: Image.Image) -> dict[str, Any]:
    """
    Extract EXIF into a flattened dict where possible.
    Includes basic tags and attempts to pull GPS.
    """
    out: dict[str, Any] = {}

    exif = None
    try:
        exif = img.getexif()
    except Exception:
        exif = None

    if not exif:
        return out

    # Map readable tag names back to numeric ids so nested IFDs can be loaded.
    tag_map = {v: k for k, v in ExifTags.TAGS.items()}
    gps_tag_id = tag_map.get("GPSInfo")

    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))

        if tag_name == "GPSInfo":
            gps = {}
            try:
                gps_ifd = exif.get_ifd(gps_tag_id) if gps_tag_id is not None else None
            except Exception:
                gps_ifd = None

            if isinstance(gps_ifd, dict):
                for gps_key, gps_val in gps_ifd.items():
                    gps_name = ExifTags.GPSTAGS.get(gps_key, str(gps_key))
                    gps[gps_name] = gps_val
            elif isinstance(value, dict):
                for gps_key, gps_val in value.items():
                    gps_name = ExifTags.GPSTAGS.get(gps_key, str(gps_key))
                    gps[gps_name] = gps_val

            # Preserve the raw pointer/placeholder value as well.
            out[f"exif_{tag_name}"] = value

            # Flatten GPS subfields
            for k, v in gps.items():
                out[f"exif_gps_{k}"] = v
        else:
            out[f"exif_{tag_name}"] = value

    return out


def _rational_to_float(x: Any) -> float | None:
    """
    Convert EXIF rational tuples / PIL IFDRational / (num,den) to float.
    """
    try:
        # PIL IFDRational behaves like float
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


def _gps_dms_to_dd(dms: Any, ref: Any) -> float | None:
    """
    Convert EXIF GPS DMS to decimal degrees.
    dms is usually (deg, min, sec) where each is rational.
    ref is 'N','S','E','W'
    """
    try:
        if not isinstance(dms, (list, tuple)) or len(dms) != 3:
            return None
        deg = _rational_to_float(dms[0])
        minutes = _rational_to_float(dms[1])
        seconds = _rational_to_float(dms[2])
        if deg is None or minutes is None or seconds is None:
            return None
        dd = deg + minutes / 60.0 + seconds / 3600.0
        ref_s = str(ref).upper()
        if ref_s in ("S", "W"):
            dd = -dd
        return dd
    except Exception:
        return None


def _extract_lat_lon(exif_flat: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = None
    lon = None
    if "exif_gps_GPSLatitude" in exif_flat and "exif_gps_GPSLatitudeRef" in exif_flat:
        lat = _gps_dms_to_dd(exif_flat["exif_gps_GPSLatitude"], exif_flat["exif_gps_GPSLatitudeRef"])
    if "exif_gps_GPSLongitude" in exif_flat and "exif_gps_GPSLongitudeRef" in exif_flat:
        lon = _gps_dms_to_dd(exif_flat["exif_gps_GPSLongitude"], exif_flat["exif_gps_GPSLongitudeRef"])
    return lat, lon


def _extract_with_exiftool(raw: bytes, suffix: str) -> dict[str, Any]:
    if EXIFTOOL_PATH is None:
        return {}

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        proc = subprocess.run(
            [
                EXIFTOOL_PATH,
                "-j",
                "-n",
                "-G1",
                "-EXIF:GPSLatitude",
                "-EXIF:GPSLongitude",
                "-XMP:GPSLatitude",
                "-XMP:GPSLongitude",
                "-Composite:GPSLatitude",
                "-Composite:GPSLongitude",
                "-EXIF:DateTimeOriginal",
                "-EXIF:DateTimeDigitized",
                "-EXIF:DateTime",
                "-EXIF:UserComment",
                "-EXIF:ImageDescription",
                "-IPTC:Caption-Abstract",
                "-XMP:CreateDate",
                "-XMP:Description",
                "-XMP:ModifyDate",
                "-XMP:Subject",
                "-XMP:Title",
                "-XMP:Keywords",
                "-File:FileModifyDate",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
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


def _extract_lat_lon_from_exiftool(exiftool_data: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = None
    lon = None

    for key in ("Composite:GPSLatitude", "EXIF:GPSLatitude", "XMP:GPSLatitude"):
        if key in exiftool_data:
            lat = _rational_to_float(exiftool_data.get(key))
            if lat is not None:
                break

    for key in ("Composite:GPSLongitude", "EXIF:GPSLongitude", "XMP:GPSLongitude"):
        if key in exiftool_data:
            lon = _rational_to_float(exiftool_data.get(key))
            if lon is not None:
                break

    return lat, lon


def _select_image_timestamp(
    rel_path: str,
    exif_flat: dict[str, Any],
    exiftool_data: dict[str, Any],
    zip_modified_time: str | None,
) -> tuple[str | None, str | None]:
    gps_dt = _parse_gps_timestamp(
        exif_flat.get("exif_gps_GPSDateStamp"),
        exif_flat.get("exif_gps_GPSTimeStamp"),
    )
    if gps_dt is not None:
        return gps_dt.isoformat(sep=" "), "exif_gps_datestamp"

    filename_dt = _parse_datetime_from_filename(rel_path)
    if filename_dt is not None:
        return filename_dt.isoformat(sep=" "), "filename_datetime"

    candidates = (
        ("exif_datetime_original", exif_flat.get("exif_DateTimeOriginal")),
        ("exif_datetime_digitized", exif_flat.get("exif_DateTimeDigitized")),
        ("exiftool_datetime_original", exiftool_data.get("EXIF:DateTimeOriginal")),
        ("exiftool_datetime_digitized", exiftool_data.get("EXIF:DateTimeDigitized")),
        ("exiftool_xmp_create_date", exiftool_data.get("XMP:CreateDate")),
        ("exif_datetime", exif_flat.get("exif_DateTime")),
        ("exiftool_datetime", exiftool_data.get("EXIF:DateTime")),
        ("exiftool_xmp_modify_date", exiftool_data.get("XMP:ModifyDate")),
        ("zip_modified_time", zip_modified_time),
        ("file_modify_time", exiftool_data.get("File:FileModifyDate")),
    )

    for source, value in candidates:
        if _parse_datetime_value(value) is not None:
            return _safe_str(value), source

    return None, None


def _gps_status_from_metadata(
    lat: float | None,
    lon: float | None,
    gps_source: str | None,
    exif_flat: dict[str, Any],
    exiftool_data: dict[str, Any],
) -> str:
    if lat is not None and lon is not None:
        return "gps_found" if gps_source else "gps_found"
    if any(k.startswith("exif_gps_") for k in exif_flat):
        return "incomplete_exif_gps"
    if any("GPS" in k for k in exiftool_data):
        return "incomplete_exiftool_gps"
    if exif_flat.get("exif_GPSInfo") is not None:
        return "gps_pointer_without_coordinates"
    return "no_gps_metadata"


def _fallback_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.updated{path.suffix}")


def _write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        return path
    except PermissionError:
        fallback = _fallback_output_path(path)
        df.to_csv(fallback, index=False, encoding="utf-8")
        return fallback


def _write_xlsx_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_excel(path, index=False)
        return path
    except PermissionError:
        fallback = _fallback_output_path(path)
        df.to_excel(fallback, index=False)
        return fallback


def _filter_complete_gps_rows(df: pd.DataFrame) -> pd.DataFrame:
    gps_lat = pd.to_numeric(df["gps_lat"], errors="coerce")
    gps_lon = pd.to_numeric(df["gps_lon"], errors="coerce")
    return df[gps_lat.notna() & gps_lon.notna()].copy()


def _extract_rows(zip_path: Path) -> list[dict[str, Any]]:
    """One metadata row per image in the archive."""
    rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue

            rel_path = info.filename
            ext = Path(rel_path).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue

            leopard, sex = _infer_leopard_and_sex_from_path(rel_path)

            raw = z.read(info)  # bytes
            file_size = info.file_size
            zip_modified_time = _parse_zip_dt(info.date_time)

            width = height = None
            exif_flat: dict[str, Any] = {}
            exiftool_data: dict[str, Any] = {}
            lat = lon = None
            gps_source = None

            try:
                with Image.open(io.BytesIO(raw)) as img:
                    width, height = img.size
                    exif_flat = _exif_to_dict(img)
                    lat, lon = _extract_lat_lon(exif_flat)
                    if lat is not None and lon is not None:
                        gps_source = "exif"
            except Exception:
                # Image may be unreadable or unsupported; keep going
                pass

            if lat is None or lon is None:
                exiftool_data = _extract_with_exiftool(raw, ext)
                exiftool_lat, exiftool_lon = _extract_lat_lon_from_exiftool(exiftool_data)
                if exiftool_lat is not None and exiftool_lon is not None:
                    lat, lon = exiftool_lat, exiftool_lon
                    gps_source = "exiftool"

            image_datetime, image_datetime_source = _select_image_timestamp(
                rel_path,
                exif_flat,
                exiftool_data,
                zip_modified_time,
            )
            image_date, image_time = _split_datetime(image_datetime)
            gps_status = _gps_status_from_metadata(lat, lon, gps_source, exif_flat, exiftool_data)
            metadata_note_text, filename_note_text, note_tags, note_sources = _collect_note_fields(
                rel_path=rel_path,
                exif_flat=exif_flat,
                exiftool_data=exiftool_data,
            )

            row: dict[str, Any] = {
                "zip_file": str(zip_path),
                "relative_path": rel_path,
                "folder_top": rel_path.replace("\\", "/").split("/")[0] if "/" in rel_path.replace("\\", "/") else "",
                "file_name": Path(rel_path).name,
                "file_ext": ext,
                "file_size_bytes": file_size,
                "zip_modified_time": zip_modified_time,
                "leopard_name": leopard,
                "sex": sex,
                "image_width_px": width,
                "image_height_px": height,
                "exif_datetime_original": _safe_str(exif_flat.get("exif_DateTimeOriginal")),
                "exif_datetime": _safe_str(exif_flat.get("exif_DateTime")),
                "image_datetime": image_datetime,
                "image_date": image_date,
                "image_time": image_time,
                "image_datetime_source": image_datetime_source,
                "exif_make": _safe_str(exif_flat.get("exif_Make")),
                "exif_model": _safe_str(exif_flat.get("exif_Model")),
                "exif_lens_model": _safe_str(exif_flat.get("exif_LensModel")),
                "exif_user_comment": _safe_str(exif_flat.get("exif_UserComment")),
                "metadata_note_text": metadata_note_text,
                "filename_note_text": filename_note_text,
                "note_tags": note_tags,
                "note_sources": note_sources,
                "gps_lat": lat,
                "gps_lon": lon,
                "gps_source": gps_source,
                "gps_status": gps_status,
            }

            # Add all EXIF fields (flattened) as extra columns
            # (This can create many columns; still useful for later filtering)
            existing_keys_lower = {k.lower() for k in row}
            for k, v in exif_flat.items():
                if k.lower() in existing_keys_lower:
                    continue
                # Keep it reasonably excel-friendly
                row[k] = _safe_str(v) if not isinstance(v, (int, float)) else v

            rows.append(row)
    return rows


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    df = pd.DataFrame(_extract_rows(ZIP_PATH))
    df = _apply_lineage_sex_backfill(df)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    filtered_df = _filter_complete_gps_rows(df)
    written_xlsx = _write_xlsx_with_fallback(df, OUT_XLSX)
    written_csv = _write_csv_with_fallback(df, OUT_CSV)
    written_filtered_csv = _write_csv_with_fallback(filtered_df, OUT_FILTERED_CSV)

    print(f"Extracted {len(df)} image rows")
    print(f"Saved -> {written_xlsx}")
    print(f"Saved -> {written_csv}")
    print(f"Saved -> {written_filtered_csv}")

    batch_rows = [row for batch_zip in BATCH_ZIPS if batch_zip.exists() for row in _extract_rows(batch_zip)]
    if batch_rows:
        batch_df = _filter_complete_gps_rows(pd.DataFrame(batch_rows))
        written_batch_csv = _write_csv_with_fallback(batch_df, OUT_BATCH_CSV)
        print(f"Extracted {len(batch_rows)} batch image rows ({len(batch_df)} with GPS)")
        print(f"Saved -> {written_batch_csv}")


if __name__ == "__main__":
    main()
