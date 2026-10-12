from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from _age import NON_TERRITORIAL_AGE_CLASSES
from _inclusion import MIN_UNIQUE_DATES, MIN_UNIQUE_POINTS as MIN_UNIQUE_LOCATIONS
from _paths import LINEAGE_XLSX


ROOT = Path(".")
EVENTS_CSV = ROOT / "outputs" / "tables" / "merged_leopard_images_metadata.dated_events.csv"
ALL_EVENTS_CSV = ROOT / "outputs" / "tables" / "merged_leopard_images_metadata.normalized_events.csv"
MCP_SUMMARY_CSV = ROOT / "outputs" / "tables" / "merged_image_home_ranges_mcp_summary.csv"
SPATIAL_SUBSET_CSV = ROOT / "outputs" / "tables" / "merged_leopard_images_metadata.spatial_subset_individuals.csv"
WINDOW_CSV = ROOT / "outputs" / "tables" / "merged_image_window_individuals__contemporary_24mo.csv"
OUT_CSV = ROOT / "outputs" / "tables" / "merged_leopard_individual_detection_table.csv"

# The primary analysis uses only the 2024-2026 window, so its individuals need their
# own columns: the pooled ones above do not identify which 17-18 animals it rests on.
WINDOW_COLUMNS = {
    "n_unique_points": "w24_n_unique_locations",
    "n_unique_dates": "w24_n_unique_dates",
    "mcp95_km2": "w24_mcp95_km2",
    "mcp100_km2": "w24_mcp100_km2",
    "include": "w24_included",
}


def _display_name(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text.replace("_", " "))


def _individual_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v).strip().title() for v in series.dropna() if str(v).strip() and str(v).strip().title() != "Unknown"]
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "Unknown"


def _fmt_date(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _clean_age_class(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def _build_inclusion_status(row: pd.Series) -> str:
    if bool(row.get("include_spatial_subset", False)):
        return "Included"

    n_dates = int(row.get("n_unique_dates", 0) or 0)
    n_locations = int(row.get("n_unique_locations", 0) or 0)
    n_events = int(row.get("n_encounter_events", 0) or 0)

    if n_events == 0:
        return "Excluded: no detections"
    age = str(row.get("age_class", "")).strip().lower()
    if age in NON_TERRITORIAL_AGE_CLASSES:
        # Reported ahead of the sampling thresholds: the reason this individual is out
        # is biological, and stays true however many locations it accumulates.
        return f"Excluded: {age} (territory not established)"
    if n_locations < MIN_UNIQUE_LOCATIONS and n_dates < MIN_UNIQUE_DATES:
        return f"Excluded: <{MIN_UNIQUE_LOCATIONS} unique locations and <{MIN_UNIQUE_DATES} unique dates"
    if n_locations < MIN_UNIQUE_LOCATIONS:
        return f"Excluded: <{MIN_UNIQUE_LOCATIONS} unique locations"
    return f"Excluded: <{MIN_UNIQUE_DATES} unique dates"


def _load_event_summary() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"Cannot find: {EVENTS_CSV.resolve()}")

    df = pd.read_csv(EVENTS_CSV)
    df["leopard_name"] = df.get("leopard_name", pd.Series(index=df.index, dtype="object")).map(_display_name)
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["image_date_parsed"] = pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True, format="mixed")
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df["event_id"] = df.get("event_id", pd.Series(index=df.index, dtype="object"))
    df = df[df["leopard_name"].ne("")].copy()

    summary = (
        df.groupby("leopard_name", dropna=False)
        .agg(
            sex_events=("sex", _resolve_sex),
            first_detection=("image_date_parsed", "min"),
            last_detection=("image_date_parsed", "max"),
            n_encounter_events=("event_id", "nunique"),
            n_unique_dates=("image_date_parsed", "nunique"),
        )
        .reset_index()
    )

    unique_locations = (
        df.dropna(subset=["gps_lat", "gps_lon"])
        .groupby("leopard_name")
        .apply(lambda g: int(len(set(zip(g["gps_lat"].astype(float), g["gps_lon"].astype(float))))))
        .rename("n_unique_locations")
        .reset_index()
    )

    unique_years = (
        df.dropna(subset=["image_date_parsed"])
        .assign(year=lambda d: d["image_date_parsed"].dt.year)
        .groupby("leopard_name")["year"]
        .nunique()
        .rename("n_years_detected")
        .reset_index()
    )

    summary = summary.merge(unique_locations, on="leopard_name", how="left")
    summary = summary.merge(unique_years, on="leopard_name", how="left")
    summary["n_unique_locations"] = summary["n_unique_locations"].fillna(0).astype(int)
    summary["n_years_detected"] = summary["n_years_detected"].fillna(0).astype(int)
    summary["individual_id"] = summary["leopard_name"].map(_individual_id)
    return summary


def _load_lineage_attributes() -> pd.DataFrame:
    if not LINEAGE_XLSX.exists():
        return pd.DataFrame(columns=["leopard_name", "sex_lineage", "age_class"])

    lineage = pd.read_excel(LINEAGE_XLSX, sheet_name="Lineage")
    lineage["leopard_name"] = lineage.get("leopard_name", pd.Series(index=lineage.index, dtype="object")).map(_display_name)
    lineage["sex_lineage"] = lineage.get("sex", pd.Series(index=lineage.index, dtype="object")).fillna("").astype(str).str.strip().str.title()
    lineage["age_class"] = lineage.get("Estimated age", pd.Series(index=lineage.index, dtype="object")).map(_clean_age_class)
    lineage = lineage[lineage["leopard_name"].ne("")].copy()
    lineage["individual_id"] = lineage["leopard_name"].map(_individual_id)
    lineage = lineage[["individual_id", "leopard_name", "sex_lineage", "age_class"]].drop_duplicates(subset=["individual_id"], keep="first")
    return lineage


def _load_mcp_summary() -> pd.DataFrame:
    if not MCP_SUMMARY_CSV.exists():
        return pd.DataFrame(columns=["leopard_name", "mcp_area_km2"])

    mcp = pd.read_csv(MCP_SUMMARY_CSV, usecols=["leopard_name", "mcp95_km2", "mcp100_km2", "n_points_dropped"])
    mcp["leopard_name"] = mcp["leopard_name"].map(_display_name)
    mcp["individual_id"] = mcp["leopard_name"].map(_individual_id)
    return mcp[["individual_id", "mcp95_km2", "mcp100_km2", "n_points_dropped"]].drop_duplicates(
        subset=["individual_id"], keep="first"
    )


def _load_spatial_subset() -> pd.DataFrame:
    if not SPATIAL_SUBSET_CSV.exists():
        return pd.DataFrame(columns=["leopard_name", "include_spatial_subset"])

    subset = pd.read_csv(SPATIAL_SUBSET_CSV, usecols=["leopard_name", "include_spatial_subset"])
    subset["leopard_name"] = subset["leopard_name"].map(_display_name)
    subset["individual_id"] = subset["leopard_name"].map(_individual_id)
    subset["include_spatial_subset"] = subset["include_spatial_subset"].fillna(False).astype(bool)
    return subset[["individual_id", "include_spatial_subset"]].drop_duplicates(subset=["individual_id"], keep="first")


def _load_window() -> pd.DataFrame:
    if not WINDOW_CSV.exists():
        return pd.DataFrame(columns=["individual_id", *WINDOW_COLUMNS.values()])
    window = pd.read_csv(WINDOW_CSV, usecols=["leopard_name", *WINDOW_COLUMNS])
    window["individual_id"] = window["leopard_name"].map(_display_name).map(_individual_id)
    return window.rename(columns=WINDOW_COLUMNS).drop(columns=["leopard_name"])


def export_individual_detection_table() -> Path:
    events = _load_event_summary()
    lineage = _load_lineage_attributes()
    mcp = _load_mcp_summary()
    subset = _load_spatial_subset()

    table = events.merge(lineage, on="individual_id", how="outer", suffixes=("_events", "_lineage"))
    table = table.merge(mcp, on="individual_id", how="left")
    table = table.merge(subset, on="individual_id", how="left")
    table = table.merge(_load_window(), on="individual_id", how="left")

    table["leopard_name"] = (
        table.get("leopard_name_events", pd.Series(index=table.index, dtype="object"))
        .fillna(table.get("leopard_name_lineage", pd.Series(index=table.index, dtype="object")))
        .fillna("")
    )

    table["sex"] = table["sex_lineage"].where(table["sex_lineage"].isin(["Female", "Male"]), table["sex_events"])
    table["sex"] = table["sex"].fillna("Unknown")
    table["age_class"] = table.get("age_class", pd.Series(index=table.index, dtype="object")).fillna("")
    table["include_spatial_subset"] = table["include_spatial_subset"].fillna(False).astype(bool)

    for column in ("n_encounter_events", "n_unique_dates", "n_unique_locations", "n_years_detected"):
        table[column] = table.get(column, pd.Series(index=table.index, dtype="float")).fillna(0).astype(int)

    table["first_detection"] = table["first_detection"].map(_fmt_date)
    table["last_detection"] = table["last_detection"].map(_fmt_date)
    for column in ("mcp95_km2", "mcp100_km2"):
        table[column] = pd.to_numeric(table[column], errors="coerce").round(6)
    table["n_points_dropped"] = pd.to_numeric(table["n_points_dropped"], errors="coerce").astype("Int64")
    table["inclusion_status"] = table.apply(_build_inclusion_status, axis=1)
    # Individuals photographed only on records carrying the archive export date are not
    # "no detections": they were seen, but none of their records can be dated.
    all_events = pd.read_csv(ALL_EVENTS_CSV, usecols=["leopard_name", "sex"])
    all_events["leopard_name"] = all_events["leopard_name"].map(_display_name)
    all_events = all_events[all_events["leopard_name"].ne("")]
    all_events["individual_id"] = all_events["leopard_name"].map(_individual_id)
    undated = all_events[~all_events["individual_id"].isin(table.loc[table["n_encounter_events"] > 0, "individual_id"])]
    undated = undated.groupby(["individual_id", "leopard_name"], as_index=False)["sex"].agg(_resolve_sex)
    table = table[~table["individual_id"].isin(undated["individual_id"])]
    extra = undated.assign(age_class="", first_detection="", last_detection="",
                           inclusion_status="Excluded: only undatable (archive-export) records")
    for column in ("n_encounter_events", "n_unique_dates", "n_unique_locations", "n_years_detected"):
        extra[column] = 0
    table = pd.concat([table, extra], ignore_index=True)
    for column in ("w24_n_unique_locations", "w24_n_unique_dates"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    for column in ("w24_mcp95_km2", "w24_mcp100_km2"):
        table[column] = pd.to_numeric(table[column], errors="coerce").round(6)
    table["w24_included"] = table["w24_included"].astype("boolean").fillna(False)

    table = table.sort_values(
        ["n_encounter_events", "n_unique_dates", "n_unique_locations", "leopard_name"],
        ascending=[False, False, False, True],
    ).drop_duplicates(subset=["individual_id"], keep="first")

    table = table[
        [
            "individual_id",
            "leopard_name",
            "sex",
            "age_class",
            "first_detection",
            "last_detection",
            "n_encounter_events",
            "n_unique_dates",
            "n_unique_locations",
            "n_years_detected",
            "mcp95_km2",
            "mcp100_km2",
            "n_points_dropped",
            "inclusion_status",
            *WINDOW_COLUMNS.values(),
        ]
    ].sort_values(
        ["n_encounter_events", "n_unique_dates", "leopard_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    return OUT_CSV


def main() -> None:
    out_path = export_individual_detection_table()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
