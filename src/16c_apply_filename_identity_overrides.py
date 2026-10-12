from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import re

import pandas as pd

from _paths import FILE_NAMES_LIST, LINEAGE_XLSX


ROOT = Path(".")
TABLES = ROOT / "outputs" / "tables"

RAW_CSV = TABLES / "leopard_images_metadata.csv"
RAW_FILTERED_CSV = TABLES / "leopard_images_metadata.filtered.csv"
RAW_XLSX = TABLES / "leopard_images_metadata.xlsx"
COMBINED_UPDATED_CSV = TABLES / "leopard_images_metadata_combined.updated.csv"
COMBINED_CSV = TABLES / "leopard_images_metadata_combined.csv"
CORRECTED_CSV = TABLES / "merged_leopard_images_metadata.filtered.corrected.csv"
# Photos from batch archives (no per-leopard folders), extracted by 0_extract.
BATCH_CSV = TABLES / "leopard_images_metadata.batches.csv"
IDENTITY_COLUMNS = [
    "identity_source",
    "identity_inference_confidence",
    "identity_reference_distance_m",
    "identity_reference_match_count",
    "identity_reference_second_choice",
    "identity_reference_second_distance_m",
]

TARGET_ZIP_NAME = "2026-05-27.zip"
MANUAL_IDENTITY_SOURCE = "file_names_list"

NAME_ALIASES = {
    "rose": "Rose",
    "montle": "Montle",
    "lesedi": "Lesedi",
    "mojelemane": "Mojeremane",
    "mojeremane": "Mojeremane",
    "puelo": "Puelo",
    "mooka": "Mooka",
    "maun rd femle": "Maun road",
    "maun rd female": "Maun road",
    "maun road": "Maun road",
    "maun road female": "Maun road",
    "mongololo": "Mongololo",
    "rradikolobe": "Rradikolobe",
    "river road": "River road",
    "mfana": "Mfana",
    "mfana male": "Mfana",
    "rradiphala": "Rradiphala",
    "kigelia ma e": "Kigelia",
    "kigelia male": "Kigelia",
    "kigelia": "Kigelia",
    "morula male": "Morula",
    "morula": "Morula",
    "tswaletswale": "Tswaletswale",
}

SEX_BY_NAME = {
    "Rose": "Female",
    "Mojeremane": "Male",
    "Mooka": "Female",
    "Maun road": "Female",
    "Mongololo": "Female",
    "Rradikolobe": "Male",
    "River road": "Female",
    "Mfana": "Male",
    "Rradiphala": "Male",
    "Kigelia": "Male",
    "Morula": "Male",
    "Tswaletswale": "Female",
}


def _load_lineage_sex_map() -> dict[str, str]:
    if not LINEAGE_XLSX.exists():
        return {}

    lineage = pd.read_excel(LINEAGE_XLSX, sheet_name="Lineage")
    if "leopard_name" not in lineage.columns:
        return {}

    sex_map: dict[str, str] = {}
    for row in lineage.itertuples(index=False):
        raw_name = str(getattr(row, "leopard_name", "") or "").strip()
        if not raw_name:
            continue
        lowered = raw_name.replace("?", " ").replace("_", " ")
        lowered = re.sub(r"\s+", " ", lowered).strip().lower()
        lowered = lowered.replace(" female", "").replace(" male", "").strip()
        canonical = NAME_ALIASES.get(lowered, lowered.title())
        raw_sex = str(getattr(row, "sex", "") or "").strip().title()
        if raw_sex in {"Female", "Male"}:
            sex_map[canonical] = raw_sex
    return sex_map


def _build_corrected_from_combined(combined: pd.DataFrame) -> pd.DataFrame:
    corrected = combined.copy()
    if not {"match_use_for_analysis", "sheet_date", "image_date", "image_datetime_source"}.issubset(corrected.columns):
        return corrected

    should_backfill = (
        corrected["match_use_for_analysis"].fillna(False)
        & corrected["sheet_date"].notna()
        & corrected["image_date"].astype("string").eq("2026-03-16")
        & corrected["image_datetime_source"].astype("string").eq("exif_datetime")
    )
    corrected.loc[should_backfill, "image_date"] = corrected.loc[should_backfill, "sheet_date"]
    corrected.loc[should_backfill, "image_time"] = pd.NA
    corrected.loc[should_backfill, "image_datetime"] = corrected.loc[should_backfill, "sheet_date"]
    corrected.loc[should_backfill, "image_datetime_source"] = "sheet_date_match"
    return corrected


def _normalize_match_key(text: object) -> str:
    stem = Path(str(text).strip()).stem
    folded = re.sub(r"[^A-Z0-9]+", "", stem.upper())
    return folded.translate(str.maketrans({"I": "1", "L": "1", "O": "0"}))


def _clean_token(text: str) -> str:
    text = text.replace("?", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _canonicalize_name(token: str) -> tuple[str, str | None]:
    lowered = _clean_token(token)
    if not lowered:
        raise ValueError("Cannot canonicalize an empty leopard name.")

    sex: str | None = None
    if "female" in lowered:
        sex = "Female"
    elif "male" in lowered:
        sex = "Male"

    lowered = lowered.replace(" female", "").replace(" male", "").strip()
    canonical = NAME_ALIASES.get(lowered, lowered.title())
    return canonical, LINEAGE_SEX_BY_NAME.get(canonical) or SEX_BY_NAME.get(canonical, sex)


LINEAGE_SEX_BY_NAME = _load_lineage_sex_map()


def _split_names(row: list[str]) -> list[str]:
    names: list[str] = []
    for cell in row[1:]:
        text = str(cell or "").strip()
        if not text:
            continue
        for token in re.split(r",|&", text):
            clean = token.strip()
            if clean:
                names.append(clean)
    return names


def _load_overrides() -> dict[str, dict[str, object]]:
    if not FILE_NAMES_LIST.exists():
        raise FileNotFoundError(f"Cannot find filename list: {FILE_NAMES_LIST.resolve()}")

    overrides: dict[str, dict[str, object]] = {}
    with FILE_NAMES_LIST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            file_label = str(row[0]).strip()
            if not file_label:
                continue

            parsed = [_canonicalize_name(name) for name in _split_names(row)]
            if not parsed:
                continue

            ordered_names: list[str] = []
            ordered_sexes: list[str | None] = []
            seen: set[str] = set()
            for name, sex in parsed:
                if name in seen:
                    continue
                seen.add(name)
                ordered_names.append(name)
                ordered_sexes.append(sex)

            primary_name = ordered_names[0]
            primary_sex = ordered_sexes[0] or "Unknown"
            overrides[_normalize_match_key(file_label)] = {
                "all_names": ordered_names,
                "primary_name": primary_name,
                "primary_sex": primary_sex,
            }
    return overrides


def _append_manual_note(series: pd.Series, notes: pd.Series) -> pd.Series:
    existing = series.fillna("").astype(str).str.strip()
    note_text = notes.fillna("").astype(str).map(lambda value: f"Filename identities: {value}" if value else "")

    merged = []
    for current, manual_note in zip(existing.tolist(), note_text.tolist()):
        if not manual_note:
            merged.append(current or pd.NA)
        elif not current:
            merged.append(manual_note)
        elif manual_note.lower() in current.lower():
            merged.append(current)
        else:
            merged.append(f"{current} | {manual_note}")
    return pd.Series(merged, index=series.index, dtype="object")


def _apply_identity_columns(df: pd.DataFrame, mask: pd.Series, overrides: pd.Series) -> None:
    df.loc[mask, "leopard_name"] = overrides.map(lambda item: item["primary_name"])
    df.loc[mask, "sex"] = overrides.map(lambda item: item["primary_sex"])

    if "metadata_note_text" in df.columns:
        df.loc[mask, "metadata_note_text"] = _append_manual_note(
            df.loc[mask, "metadata_note_text"],
            overrides.map(lambda item: ", ".join(item["all_names"])),
        )

    if "identity_source" in df.columns:
        df.loc[mask, "identity_source"] = MANUAL_IDENTITY_SOURCE
    if "identity_inference_confidence" in df.columns:
        df.loc[mask, "identity_inference_confidence"] = "manual"
    if "identity_reference_distance_m" in df.columns:
        df.loc[mask, "identity_reference_distance_m"] = pd.NA
    if "identity_reference_match_count" in df.columns:
        df.loc[mask, "identity_reference_match_count"] = pd.NA
    if "identity_reference_second_choice" in df.columns:
        df.loc[mask, "identity_reference_second_choice"] = pd.NA
    if "identity_reference_second_distance_m" in df.columns:
        df.loc[mask, "identity_reference_second_distance_m"] = pd.NA

    if "sheet_match_quality" in df.columns:
        df.loc[mask, "sheet_match_quality"] = "manual"
    if "match_use_for_analysis" in df.columns:
        df.loc[mask, "match_use_for_analysis"] = True
    if "sheet_other_leopard" in df.columns:
        df.loc[mask, "sheet_other_leopard"] = overrides.map(lambda item: "Yes" if len(item["all_names"]) > 1 else "No")
    if "sheet_leopard_2" in df.columns:
        df.loc[mask, "sheet_leopard_2"] = overrides.map(
            lambda item: ", ".join(item["all_names"][1:]) if len(item["all_names"]) > 1 else pd.NA
        )
    if "sheet_total_leopards" in df.columns:
        df.loc[mask, "sheet_total_leopards"] = overrides.map(lambda item: len(item["all_names"]))


def _apply_overrides(df: pd.DataFrame, *, require_target_zip: bool) -> tuple[pd.DataFrame, int]:
    work = df.copy()
    overrides = _load_overrides()
    if not overrides:
        return work, 0

    file_keys = work.get("file_name", pd.Series(index=work.index, dtype="object")).map(_normalize_match_key)
    mask = file_keys.isin(overrides)

    if require_target_zip and "zip_file" in work.columns:
        zip_mask = work["zip_file"].fillna("").astype(str).str.endswith(TARGET_ZIP_NAME)
        mask &= zip_mask

    matched_keys = file_keys[mask]
    if matched_keys.empty:
        return work, 0

    override_series = matched_keys.map(overrides)
    _apply_identity_columns(work, mask, override_series)
    return work, int(mask.sum())


def _expand_multi_identity_rows(df: pd.DataFrame, *, require_target_zip: bool) -> tuple[pd.DataFrame, int]:
    work = df.copy()
    overrides = _load_overrides()
    if not overrides:
        return work, 0

    file_keys = work.get("file_name", pd.Series(index=work.index, dtype="object")).map(_normalize_match_key)
    mask = file_keys.isin(overrides)

    if require_target_zip and "zip_file" in work.columns:
        zip_mask = work["zip_file"].fillna("").astype(str).str.endswith(TARGET_ZIP_NAME)
        mask &= zip_mask

    matched = work.loc[mask].copy()
    if matched.empty:
        return work, 0

    extra_rows: list[dict[str, object]] = []
    for idx, row in matched.iterrows():
        override = overrides.get(file_keys.at[idx])
        if not override:
            continue
        all_names = list(override["all_names"])
        if len(all_names) <= 1:
            continue

        for current_name in all_names[1:]:
            clone = row.copy()
            clone["leopard_name"] = current_name
            clone["sex"] = LINEAGE_SEX_BY_NAME.get(current_name) or SEX_BY_NAME.get(current_name, "Unknown")
            if "sheet_other_leopard" in clone.index:
                clone["sheet_other_leopard"] = "Yes"
            if "sheet_leopard_2" in clone.index:
                companions = [name for name in all_names if name != current_name]
                clone["sheet_leopard_2"] = ", ".join(companions) if companions else pd.NA
            if "sheet_total_leopards" in clone.index:
                clone["sheet_total_leopards"] = len(all_names)
            extra_rows.append(clone.to_dict())

    if not extra_rows:
        return work, 0

    expanded = pd.concat([work, pd.DataFrame(extra_rows)], ignore_index=True)
    return expanded, len(extra_rows)


def _match_batch_to_sightings(batch: pd.DataFrame) -> pd.DataFrame:
    """Name batch photos from file_names_list.csv, then give them 16a's sightings-log fields.

    16a matches each photo to the nearest sighting of the same leopard; batch photos only get
    a name here, so they are matched here, with 16a's own code. _apply_overrides later marks
    them manual and usable whatever the match distance.
    """
    spec = importlib.util.spec_from_file_location("match_16a", Path(__file__).with_name("16a_match_image_metadata_to_sightings.py"))
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)

    batch = batch.copy()
    for column in ("leopard_name", "sex", "metadata_note_text"):   # empty in the batch table, so read as numbers
        batch[column] = batch[column].astype("object")
    named, _ = _apply_overrides(batch, require_target_zip=True)
    named["gps_lat"] = pd.to_numeric(named["gps_lat"], errors="coerce")
    named["gps_lon"] = pd.to_numeric(named["gps_lon"], errors="coerce")
    named["name_norm"] = named["leopard_name"].map(matcher._normalize_name)
    return matcher._build_combined(named, matcher._load_sightings())


def _with_identity_and_batch_rows(combined: pd.DataFrame) -> pd.DataFrame:
    """Build the curated analysis table from 16a's matched table.

    Photos filed under a leopard's folder carry identity_source "path". Photos from the
    batch archive (TARGET_ZIP_NAME) have no leopard folder, so 16a never sees them: they
    are appended here from 0_extract's batch table, and _apply_overrides then names them
    from file_names_list.csv. Safe to re-run: columns and batch rows are only added once.
    """
    work = combined.copy()
    if "identity_source" not in work.columns:
        position = work.columns.get_loc("sheet_match_distance_m")
        for offset, column in enumerate(IDENTITY_COLUMNS):
            work.insert(position + offset, column, pd.NA)
        work["identity_source"] = "path"

    has_batch_rows = work["zip_file"].fillna("").astype(str).str.endswith(TARGET_ZIP_NAME).any()
    if not has_batch_rows and BATCH_CSV.exists():
        batch = pd.read_csv(BATCH_CSV, low_memory=False)
        batch = batch[batch["zip_file"].fillna("").astype(str).str.endswith(TARGET_ZIP_NAME)]
        batch = _match_batch_to_sightings(batch)
        work = pd.concat([work, batch.reindex(columns=work.columns)], ignore_index=True)
    return work


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    raw_updated, raw_matches = _apply_overrides(raw, require_target_zip=True)
    raw_updated.to_csv(RAW_CSV, index=False)
    raw_updated.to_excel(RAW_XLSX, index=False)

    filtered = pd.read_csv(RAW_FILTERED_CSV)
    filtered_updated, filtered_matches = _apply_overrides(filtered, require_target_zip=True)
    filtered_updated.to_csv(RAW_FILTERED_CSV, index=False)

    combined_updated = _with_identity_and_batch_rows(pd.read_csv(COMBINED_CSV, low_memory=False))
    combined_updated, combined_matches = _apply_overrides(combined_updated, require_target_zip=True)
    corrected_updated = _build_corrected_from_combined(combined_updated)
    corrected_updated, corrected_extra_rows = _expand_multi_identity_rows(corrected_updated, require_target_zip=True)
    combined_updated.to_csv(COMBINED_UPDATED_CSV, index=False)
    combined_updated.to_csv(COMBINED_CSV, index=False)
    corrected_updated.to_csv(CORRECTED_CSV, index=False)

    print(f"Saved -> {RAW_CSV}")
    print(f"Saved -> {RAW_XLSX}")
    print(f"Saved -> {RAW_FILTERED_CSV}")
    print(f"Saved -> {COMBINED_UPDATED_CSV}")
    print(f"Saved -> {COMBINED_CSV}")
    print(f"Saved -> {CORRECTED_CSV}")
    print()
    print(f"Updated {raw_matches} raw metadata rows.")
    print(f"Updated {filtered_matches} filtered metadata rows.")
    print(f"Updated {combined_matches} curated analysis rows.")
    print(f"Expanded {corrected_extra_rows} additional corrected rows for multi-leopard photos.")


if __name__ == "__main__":
    main()
