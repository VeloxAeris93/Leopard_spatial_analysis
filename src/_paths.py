"""Where the paper pipeline reads its raw inputs.

Every input is looked for first in ``data/`` (the layout of the public companion
repository, github.com/VeloxAeris93/Leopard_spatial_analysis) and then where the full
project folder keeps it, so the same scripts run unchanged in both. Paths are absolute
(anchored at the repository root), so scripts can be run from anywhere.

Set the ``EXIFTOOL`` environment variable if ExifTool is not on PATH.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# Photo archive: one top-level folder per leopard (e.g. "Mapula female/").
PHOTO_ZIP = _first_existing(
    DATA / "Chitabe Leopards.zip",
    ROOT / "Chitabe Leopards.zip",
    ROOT / "Raw_zipped_photos" / "Chitabe Leopards.zip",
)
# Later photo batches delivered without per-leopard folders; identities for these come
# from FILE_NAMES_LIST (applied in 16c_apply_filename_identity_overrides.py).
BATCH_ZIPS = [
    _first_existing(
        DATA / "2026-05-27.zip",
        ROOT / "2026-05-27.zip",
        ROOT / "Raw_zipped_photos" / "2026-05-27.zip",
    ),
]
# Guides' sightings log, sheet "Sheet1".
SIGHTINGS_XLSX = _first_existing(
    DATA / "Chitabe Leopards.xlsx",
    ROOT / "data_inputs" / "Chitabe Leopards.xlsx",
)
# Lineage register, sheet "Lineage" (sex, estimated age, relationships).
LINEAGE_XLSX = _first_existing(
    DATA / "leopard_lineage_master.xlsx",
    ROOT / "leopard_lineage_master.xlsx",
)
# Manual identities for photos whose folder holds several leopards.
FILE_NAMES_LIST = _first_existing(
    DATA / "file_names_list.csv",
    ROOT / "file_names_list.csv",
)
# 2013 game-drive road network (null-model road regions).
ROADS_GEOJSON = _first_existing(
    DATA / "chitabe_roads_2013.geojson",
    ROOT / "data_inputs" / "chitabe_roads_2013.geojson",
)
# NG31 concession boundary.
NG31_GEOJSON = _first_existing(
    DATA / "ng31.geojson",
    ROOT / "data_inputs" / "Concession Boundary" / "ng31.geojson",
)
# Edited NG31 outline drawn on the study-area map (Figure 1); shapefile with .shx/.dbf/.prj.
NG31_EDIT_SHP = _first_existing(
    DATA / "ng31_edit.shp",
    ROOT / "data_inputs" / "Concession Boundary" / "ng31_edit.shp",
    ROOT / "ng31_edit.shp",
)
# Natural Earth 1:110m country outlines for the Figure 1 inset (public domain), kept so the
# figure does not need a download.
NATURAL_EARTH_COUNTRIES = _first_existing(
    DATA / "ne_110m_admin_0_countries.geojson",
    ROOT / "reference" / "ne_110m_admin_0_countries.geojson",
)

# OpenStreetMap water/wetland snapshot behind the floodplain shading in Figure 3, taken on
# 25 Aug 2026 and kept so the figure does not depend on a live server or later OSM edits.
# (c) OpenStreetMap contributors, ODbL.
OSM_FLOODPLAIN_SNAPSHOT = _first_existing(
    DATA / "osm_floodplain_ng31_2026-08-25.json.gz",
    ROOT / "reference" / "osm_floodplain_ng31_2026-08-25.json.gz",
)


def find_exiftool() -> str | None:
    """ExifTool from $EXIFTOOL, the project's usual Windows install, or PATH."""
    candidates = [os.environ.get("EXIFTOOL", ""), r"C:\exiftool.exe\exiftool-13.32_64\exiftool.exe"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("exiftool")
