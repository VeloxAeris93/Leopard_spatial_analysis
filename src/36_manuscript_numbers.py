"""Emit every number the manuscript quotes, from one build, with provenance.

The submitted manuscript carried figures from three different builds of the
pipeline at once (Results text from one vintage, Figures 4-5 from a 9 June run,
their own input tables from an 11 June run). Nothing detects that by eye.

Revise the manuscript from this file: if a number appears in the text, it should
appear here, and it should carry the source table it came from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
from _inclusion import MIN_UNIQUE_DATES, MIN_UNIQUE_POINTS

OUT_TABLES = Path("outputs") / "tables"
OUT_JSON = OUT_TABLES / "manuscript_numbers.json"
OUT_MD = OUT_TABLES / "manuscript_numbers.md"

PRIMARY_WINDOW = "contemporary_24mo"


def _read(name: str) -> pd.DataFrame:
    path = OUT_TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Cannot find: {path.resolve()}")
    return pd.read_csv(path, low_memory=False)


def collect() -> dict[str, object]:
    raw = _read("leopard_images_metadata.csv")
    merged = _read("merged_leopard_images_metadata.filtered.corrected.csv")
    all_events = _read("merged_leopard_images_metadata.normalized_events.csv")
    events = _read("merged_leopard_images_metadata.dated_events.csv")
    date_sources = _read("date_source_counts.csv")
    null_model = _read(f"overlap_null_model_summary__{PRIMARY_WINDOW}.csv")
    null_regions = _read(f"overlap_null_model_regions__{PRIMARY_WINDOW}.csv")
    sex_power = _read("sex_test_power.csv")
    detection = _read("merged_leopard_individual_detection_table.csv")
    seasons = _read("merged_image_counts_by_botswana_season.csv")
    windows = _read("merged_image_window_summary.csv")
    sex_stats = _read("merged_image_window_sex_stats.csv")
    desc = _read("overlap_descriptives_by_pair_type.csv")
    models = _read("overlap_model_results.csv")
    decomposition = _read("reviewer_decomposition.csv")
    mcp_summary = _read("merged_image_home_ranges_mcp_summary.csv")

    dates = pd.to_datetime(events["image_date"], errors="coerce", dayfirst=True)
    # Sex comes from Table S2 so that the Results and the supplement cannot disagree.
    detected = detection[detection["n_encounter_events"] > 0]
    sex_by_individual = detected.set_index("leopard_name")["sex"]

    primary_window = windows[windows["window"] == PRIMARY_WINDOW].iloc[0]
    primary_sex = sex_stats[sex_stats["window"] == PRIMARY_WINDOW].iloc[0]

    out: dict[str, object] = {
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
        "date_sources": {
            "_source": "date_source_counts.csv",
            "rows": date_sources.to_dict(orient="records"),
        },
        "seasonality": {
            "_source": "merged_image_counts_by_botswana_season.csv",
            **{
                str(row.botswana_season).lower().replace(" ", "_"): {
                    "n_records": int(row.n_records),
                    "percent": round(float(row.percent_records), 1),
                }
                for row in seasons.itertuples()
            },
        },
        "spatial_subset": {
            "_source": "merged_image_window_summary.csv",
            "min_unique_points": MIN_UNIQUE_POINTS,
            "min_unique_dates": MIN_UNIQUE_DATES,
            "mcp_percent": 95,
            "windows": windows.to_dict(orient="records"),
            "primary_window": PRIMARY_WINDOW,
        },
        "range_size_by_sex": {
            "_source": "merged_image_window_sex_stats.csv",
            "primary_window": {
                key: (round(float(primary_sex[key]), 4) if isinstance(primary_sex[key], float) else primary_sex[key])
                for key in primary_sex.index
            },
            "all_windows": sex_stats.to_dict(orient="records"),
        },
        "sex_test_power": {
            "_source": "sex_test_power.csv",
            "rows": sex_power.to_dict(orient="records"),
        },
        "mcp_estimator_comparison": {
            "_source": "merged_image_home_ranges_mcp_summary.csv",
            "median_mcp95_km2": round(float(mcp_summary["mcp95_km2"].median()), 3),
            "median_mcp100_km2": round(float(mcp_summary["mcp100_km2"].median()), 3),
            "median_pct_area_reduction": round(
                float((100 * (1 - mcp_summary["mcp95_km2"] / mcp_summary["mcp100_km2"])).median()), 1
            ),
            "max_points_dropped": int(mcp_summary["n_points_dropped"].max()),
        },
        "overlap": {
            "_source": "overlap_descriptives_by_pair_type.csv, overlap_model_results.csv",
            "descriptives": desc.to_dict(orient="records"),
            "model_results": models.to_dict(orient="records"),
        },
        "overlap_null_model": {
            "_source": f"overlap_null_model_summary__{PRIMARY_WINDOW}.csv",
            "regions": null_regions.to_dict(orient="records"),
            "results": null_model.to_dict(orient="records"),
        },
        "reviewer_decomposition": {
            "_source": "reviewer_decomposition.csv",
            "steps": decomposition.to_dict(orient="records"),
        },
        "software": {
            "_source": "requirements.txt / venv",
            "python": "3.14.2",
            "pandas": "3.0.1",
            "numpy": "2.4.2",
            "scipy": "1.17.1",
            "statsmodels": "0.14.6",
            "shapely": "2.1.2",
            "pyproj": "3.7.2",
            "matplotlib": "3.10.8",
            "projection": "EPSG:32734 (UTM 34S)",
        },
    }
    return out


def to_markdown(data: dict[str, object]) -> str:
    lines = [
        "# Manuscript numbers",
        "",
        f"Build: {data['build_timestamp_utc']}",
        "",
        "Every number quoted in the manuscript should appear below. Regenerate with",
        "`python src/36_manuscript_numbers.py` after any pipeline change.",
        "",
    ]

    ds = data["dataset"]
    lines += [
        "## Dataset",
        "",
        f"- {ds['n_image_records_extracted']:,} image records extracted, "
        f"{ds['n_image_records_with_gps']:,} with valid GPS (original archive)",
        f"- {ds['n_gps_image_records_merged_archive']:,} GPS-referenced image records in the merged archive",
        f"- {ds['n_encounter_events_before_date_exclusion']:,} encounter events of "
        f"{ds['n_individuals_before_date_exclusion']} individuals before date exclusion",
        f"- {ds['n_encounter_events']:,} dated encounter events retained for analysis",
        f"- {ds['n_individuals_photographed']} individually identified leopards "
        f"({ds['n_female_individuals']} females, {ds['n_male_individuals']} males, "
        f"{ds['n_unknown_sex_individuals']} unknown sex)",
        f"- {ds['first_detection']} to {ds['last_detection']} "
        f"({ds['study_span_days']:,} days, {ds['n_unique_sampling_dates']} unique sampling dates)",
        "",
        "## Date sources",
        "",
        pd.DataFrame(data["date_sources"]["rows"]).to_markdown(index=False),
        "",
        "## Seasonality",
        "",
    ]
    for key, value in data["seasonality"].items():
        if key.startswith("_"):
            continue
        lines.append(f"- {key.replace('_', ' ').title()}: {value['n_records']} ({value['percent']}%)")

    lines += ["", "## Windows", ""]
    lines.append(pd.DataFrame(data["spatial_subset"]["windows"]).to_markdown(index=False))

    lines += ["", "## Range size by sex", ""]
    lines.append(pd.DataFrame(data["range_size_by_sex"]["all_windows"]).round(4).to_markdown(index=False))

    lines += ["", "## Sex-test power (80%, alpha 0.05)", ""]
    lines.append(pd.DataFrame(data["sex_test_power"]["rows"]).round(3).to_markdown(index=False))

    lines += ["", "## Overlap descriptives", ""]
    lines.append(pd.DataFrame(data["overlap"]["descriptives"]).round(4).to_markdown(index=False))

    lines += ["", "## Overlap models (permutation inference)", ""]
    lines.append(pd.DataFrame(data["overlap"]["model_results"]).round(4).to_markdown(index=False))

    lines += ["", "## Overlap null model (random placement)", ""]
    lines.append(pd.DataFrame(data["overlap_null_model"]["regions"]).round(2).to_markdown(index=False))
    lines.append("")
    lines.append(pd.DataFrame(data["overlap_null_model"]["results"]).round(4).to_markdown(index=False))

    lines += ["", "## Decomposition: submitted analysis to revised analysis", ""]
    lines.append(pd.DataFrame(data["reviewer_decomposition"]["steps"]).round(4).to_markdown(index=False))

    mcp = data["mcp_estimator_comparison"]
    lines += [
        "",
        "## MCP estimator",
        "",
        f"- Median 95% MCP: {mcp['median_mcp95_km2']} km2",
        f"- Median 100% MCP: {mcp['median_mcp100_km2']} km2",
        f"- Median area reduction from peripheral removal: {mcp['median_pct_area_reduction']}%",
        f"- Most locations dropped for any individual: {mcp['max_points_dropped']}",
        "",
        "## Software",
        "",
    ]
    for key, value in data["software"].items():
        if not key.startswith("_"):
            lines.append(f"- {key}: {value}")

    return "\n".join(lines) + "\n"


def _jsonable(value: object) -> object:
    """numpy scalars from pandas are not JSON serializable on their own."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def main() -> None:
    data = collect()
    OUT_JSON.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")
    OUT_MD.write_text(to_markdown(data), encoding="utf-8")
    print(f"Saved -> {OUT_JSON}")
    print(f"Saved -> {OUT_MD}")


if __name__ == "__main__":
    main()
