from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import re

import pandas as pd

from _paths import SIGHTINGS_XLSX


IMAGE_PATH = Path("outputs") / "tables" / "leopard_images_metadata.filtered.csv"
SIGHTINGS_PATH = SIGHTINGS_XLSX
OUT_DIR = Path("outputs") / "tables"
OUT_CORRECTED = OUT_DIR / "merged_leopard_images_metadata.filtered.corrected.csv"
OUT_COMBINED = OUT_DIR / "leopard_images_metadata_combined.csv"

HIGH_CONFIDENCE_DISTANCE_M = 50.0

NAME_REPLACEMENTS = {
    "maun rd": "maun road",
    "mojelemane": "mojeremane",
    "moretwla": "moretlwa",
}


def _normalize_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("_", " ")
    text = re.sub(r"\b(male|female)\b", "", text)
    for source, target in NAME_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _infer_sex(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "female" in text:
        return "Female"
    if "male" in text:
        return "Male"
    return None


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2
    return 2.0 * earth_radius_m * atan2(sqrt(a), sqrt(1.0 - a))


def _match_label(distance_m: float | None) -> str:
    if distance_m is None or pd.isna(distance_m):
        return "unmatched"
    if distance_m <= 5:
        return "exact"
    if distance_m <= 50:
        return "near"
    if distance_m <= 250:
        return "loose"
    return "far"


def _load_images() -> pd.DataFrame:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {IMAGE_PATH.resolve()}")
    images = pd.read_csv(IMAGE_PATH)
    images["gps_lat"] = pd.to_numeric(images.get("gps_lat"), errors="coerce")
    images["gps_lon"] = pd.to_numeric(images.get("gps_lon"), errors="coerce")
    images["name_norm"] = images.get("leopard_name").map(_normalize_name)
    return images


def _load_sightings() -> pd.DataFrame:
    if not SIGHTINGS_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {SIGHTINGS_PATH.resolve()}")
    sightings = pd.read_excel(SIGHTINGS_PATH, sheet_name="Sheet1")
    sightings["Lat"] = pd.to_numeric(sightings.get("Lat"), errors="coerce")
    sightings["Long"] = pd.to_numeric(sightings.get("Long"), errors="coerce")
    sightings["name_norm"] = sightings.get("Leopard").map(_normalize_name)
    sightings["sheet_sex"] = sightings.get("Leopard").map(_infer_sex)
    sightings["sheet_date"] = pd.to_datetime(
        dict(year=sightings["Year"], month=sightings["Month"], day=sightings["Day"]),
        errors="coerce",
    )
    return sightings


def _build_combined(images: pd.DataFrame, sightings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_sightings = {
        name: group.dropna(subset=["Lat", "Long"]).reset_index(drop=True)
        for name, group in sightings.groupby("name_norm", dropna=True)
    }

    for row in images.itertuples(index=False):
        output = row._asdict()
        output["sheet_match_distance_m"] = None
        output["sheet_match_quality"] = "unmatched"
        output["match_use_for_analysis"] = False
        output["sheet_date"] = None
        output["sheet_guide"] = None
        output["sheet_behaviour"] = None
        output["sheet_notes"] = None
        output["sheet_other_leopard"] = None
        output["sheet_leopard_2"] = None
        output["sheet_total_leopards"] = None

        candidates = grouped_sightings.get(output.get("name_norm"))
        if (
            candidates is not None
            and pd.notna(output.get("gps_lat"))
            and pd.notna(output.get("gps_lon"))
            and not candidates.empty
        ):
            distances = candidates.apply(
                lambda candidate: _distance_m(
                    float(output["gps_lat"]),
                    float(output["gps_lon"]),
                    float(candidate["Lat"]),
                    float(candidate["Long"]),
                ),
                axis=1,
            )
            best_idx = distances.idxmin()
            best = candidates.loc[best_idx]
            best_distance = float(distances.loc[best_idx])
            output["sheet_match_distance_m"] = round(best_distance, 2)
            output["sheet_match_quality"] = _match_label(best_distance)
            output["match_use_for_analysis"] = best_distance <= HIGH_CONFIDENCE_DISTANCE_M
            output["sheet_date"] = best["sheet_date"].date().isoformat() if pd.notna(best["sheet_date"]) else None
            output["sheet_guide"] = best.get("Guide")
            output["sheet_behaviour"] = best.get("Behaviour")
            output["sheet_notes"] = best.get("Notes")
            output["sheet_other_leopard"] = best.get("Other_Leopard")
            output["sheet_leopard_2"] = best.get("Leopard_2")
            output["sheet_total_leopards"] = best.get("Total_leopards")
            if (not output.get("sex")) and pd.notna(best.get("sheet_sex")):
                output["sex"] = best.get("sheet_sex")

        rows.append(output)

    combined = pd.DataFrame(rows)
    return combined.drop(columns=["name_norm"], errors="ignore")


def _build_corrected(combined: pd.DataFrame) -> pd.DataFrame:
    corrected = combined.copy()
    should_backfill = (
        corrected["match_use_for_analysis"].fillna(False)
        & corrected["sheet_date"].notna()
        & corrected["image_date"].astype("string").eq("2026-03-16")
        & corrected["image_datetime_source"].astype("string").eq("exif_datetime")
    )
    corrected.loc[should_backfill, "image_date"] = corrected.loc[should_backfill, "sheet_date"]
    corrected.loc[should_backfill, "image_time"] = None
    corrected.loc[should_backfill, "image_datetime"] = corrected.loc[should_backfill, "sheet_date"]
    corrected.loc[should_backfill, "image_datetime_source"] = "sheet_date_match"
    return corrected


def main() -> None:
    images = _load_images()
    sightings = _load_sightings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = _build_combined(images, sightings)
    corrected = _build_corrected(combined)

    combined.to_csv(OUT_COMBINED, index=False)
    corrected.to_csv(OUT_CORRECTED, index=False)

    matched = combined["sheet_match_distance_m"].notna().sum()
    high_conf = combined["match_use_for_analysis"].fillna(False).sum()
    corrected_dates = (corrected["image_datetime_source"] == "sheet_date_match").sum()

    print(f"Saved -> {OUT_COMBINED}")
    print(f"Saved -> {OUT_CORRECTED}")
    print()
    print(f"Matched rows to sightings sheet: {matched}")
    print(f"High-confidence matches (<= {HIGH_CONFIDENCE_DISTANCE_M:.0f} m): {high_conf}")
    print(f"Rows with backfilled sheet dates: {corrected_dates}")


if __name__ == "__main__":
    main()
