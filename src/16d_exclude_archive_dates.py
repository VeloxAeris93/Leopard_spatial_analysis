"""Drop encounter events whose only date is the archive export date.

Writes the dated events table that every downstream analysis reads, an audit of the
excluded events, and the date-source counts quoted in the Methods. See src/_dates.py
for why the export date is unrecoverable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _dates import ARCHIVE_EXPORT_DATE, is_archive_export_date

TABLES = Path("outputs") / "tables"
RAW_PATH = TABLES / "leopard_images_metadata.csv"
MERGED_PATH = TABLES / "merged_leopard_images_metadata.filtered.corrected.csv"
EVENTS_PATH = TABLES / "merged_leopard_images_metadata.normalized_events.csv"

OUT_EVENTS = TABLES / "merged_leopard_images_metadata.dated_events.csv"
OUT_AUDIT = TABLES / "date_audit_archive_export.csv"
OUT_SOURCES = TABLES / "date_source_counts.csv"


def _source_counts(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    bad = is_archive_export_date(df)
    counts = (
        df.assign(archive_export_date=bad)
        .groupby(["image_datetime_source", "archive_export_date"], dropna=False)
        .size()
        .rename("n_records")
        .reset_index()
    )
    counts.insert(0, "stage", stage)
    counts.insert(1, "n_stage_total", len(df))
    return counts


def main() -> None:
    for path in (RAW_PATH, MERGED_PATH, EVENTS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Cannot find: {path.resolve()}")

    raw = pd.read_csv(RAW_PATH, low_memory=False)
    merged = pd.read_csv(MERGED_PATH, low_memory=False)
    events = pd.read_csv(EVENTS_PATH, low_memory=False)

    sources = pd.concat(
        [
            _source_counts(raw, "extracted_images"),
            _source_counts(merged, "gps_images_after_sheet_matching"),
            _source_counts(events, "encounter_events"),
        ],
        ignore_index=True,
    )
    sources.to_csv(OUT_SOURCES, index=False)

    bad = is_archive_export_date(events)
    audit = events.loc[bad, ["event_id", "leopard_name", "sex", "image_date", "image_datetime",
                             "image_datetime_source", "n_images_in_event", "gps_lat", "gps_lon", "file_name"]].copy()
    audit["reason"] = f"EXIF date equals archive export date {ARCHIVE_EXPORT_DATE.date()}"
    audit.to_csv(OUT_AUDIT, index=False)

    dated = events.loc[~bad].copy()
    dated.to_csv(OUT_EVENTS, index=False)

    lost = sorted(set(events["leopard_name"].dropna()) - set(dated["leopard_name"].dropna()))
    print(f"Excluded {int(bad.sum())} of {len(events)} events ({int(merged.pipe(is_archive_export_date).sum())} "
          f"of {len(merged)} GPS image records) dated {ARCHIVE_EXPORT_DATE.date()} from EXIF")
    print(f"Individuals with no dated events left: {', '.join(lost) if lost else 'none'}")
    print(f"Saved -> {OUT_EVENTS}\nSaved -> {OUT_AUDIT}\nSaved -> {OUT_SOURCES}")


if __name__ == "__main__":
    main()
