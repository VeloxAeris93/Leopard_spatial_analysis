"""Date-validity rule for image records, defined once.

The original archive (Chitabe Leopards.zip) was exported from Adobe Lightroom Classic,
and the export dropped the EXIF capture date (DateTimeOriginal) from every image. The
only EXIF date left is DateTime, the file's last-saved time, which for all 1,098
affected images is the export session: 16 March 2026, 08:59-17:37. Images with a GPS
date stamp or a dated filename are unaffected; script 16a recovers some others from
the sightings workbook, but the remainder still carry the export date. Their locations are genuine, their dates are not, so they cannot be placed in a
time window and are excluded from every analysis that uses the events table.

The rule is deliberately narrow: only records whose date came from the EXIF
timestamp *and* falls on the export date are rejected. A GPS-stamped or
sheet-matched record from 16 March 2026 would be kept, although none exist.
"""

from __future__ import annotations

import pandas as pd

ARCHIVE_EXPORT_DATE = pd.Timestamp("2026-03-16")
UNRELIABLE_SOURCE = "exif_datetime"


def parse_image_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True, format="mixed").dt.normalize()


def is_archive_export_date(df: pd.DataFrame) -> pd.Series:
    source = df.get("image_datetime_source", pd.Series("", index=df.index)).astype("string")
    return parse_image_date(df).eq(ARCHIVE_EXPORT_DATE) & source.eq(UNRELIABLE_SOURCE)
