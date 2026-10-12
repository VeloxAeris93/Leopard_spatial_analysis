"""Build the supplementary tables document with the numbering the manuscript uses.

The submitted supplementary file numbered the individual-level table S1 and the
camera table S2, which is the inverse of the in-text citations (Methods cite Table S1
for cameras and Table S2 for individuals). A second standalone file contained another
copy of the individual table, also numbered S1. That is what the reviewer tripped over.

This script regenerates the whole supplementary set from the pipeline output with the
numbering the manuscript expects:

    S1  Camera makes and models
    S2  Individually identified leopards, detection history and inclusion status
    S3  Sensitivity of results to time window and MCP estimator
    S4  Effect of each analytical choice on the focal overlap comparison
    S5  Observed overlap against the random-placement null model

Captions and S3 headings follow the wording edited by hand in the August Word file.
Output goes to manuscript/ so that hand-edited copy in outputs/ is never overwritten.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import _docx

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
OUT_DOCX = ROOT / "manuscript" / "Supplementary Tables_V5.docx"
WINDOW_LABELS = {"contemporary_24mo": "24 months", "recent_12mo": "12 months", "pooled": "Pooled"}
OUT_MD = ROOT / "outputs" / "supplementary_tables.md"


def _read(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Cannot find: {path.resolve()}")
    return pd.read_csv(path)


def table_s1() -> pd.DataFrame:
    cameras = _read("merged_leopard_images_metadata_camera_summary.csv")
    cameras = cameras[cameras["exif_make"] != "Unknown"].copy()
    cameras = cameras.rename(
        columns={
            "exif_make": "Make",
            "exif_model": "Model",
            "n_images": "Images",
            "n_individuals": "Individuals",
        }
    )
    return cameras[["Make", "Model", "Images", "Individuals"]].sort_values(["Make", "Model"])


def table_s2() -> pd.DataFrame:
    detection = _read("merged_leopard_individual_detection_table.csv")
    out = detection.rename(
        columns={
            "leopard_name": "Leopard",
            "sex": "Sex",
            "age_class": "Age class",
            "first_detection": "First detection",
            "last_detection": "Last detection",
            "n_encounter_events": "Encounter events",
            "n_unique_dates": "Unique dates",
            "n_unique_locations": "Unique locations",
            "n_years_detected": "Years detected",
            "mcp95_km2": "95% MCP (km2)",
            "mcp100_km2": "100% MCP (km2)",
            "n_points_dropped": "Locations removed",
            "inclusion_status": "Status",
            "w24_n_unique_locations": "2024-26 unique locations",
            "w24_n_unique_dates": "2024-26 unique dates",
            "w24_mcp95_km2": "2024-26 95% MCP (km2)",
            "w24_mcp100_km2": "2024-26 100% MCP (km2)",
            "w24_included": "2024-26 included",
        }
    )
    out["2024-26 included"] = out["2024-26 included"].map({True: "Yes", False: ""})
    for column in ("95% MCP (km2)", "100% MCP (km2)", "2024-26 95% MCP (km2)", "2024-26 100% MCP (km2)"):
        out[column] = pd.to_numeric(out[column], errors="coerce").round(1)
    return out[
        [
            "Leopard",
            "Sex",
            "Age class",
            "First detection",
            "Last detection",
            "Encounter events",
            "Unique dates",
            "Unique locations",
            "Years detected",
            "95% MCP (km2)",
            "100% MCP (km2)",
            "Locations removed",
            "Status",
            "2024-26 unique locations",
            "2024-26 unique dates",
            "2024-26 95% MCP (km2)",
            "2024-26 100% MCP (km2)",
            "2024-26 included",
        ]
    ]


def table_s3() -> pd.DataFrame:
    windows = _read("merged_image_window_summary.csv")
    sex = _read("merged_image_window_sex_stats.csv")
    desc = _read("overlap_descriptives_by_pair_type.csv")
    models = _read("overlap_model_results.csv")

    rows: list[dict[str, object]] = []
    for window in windows["window"]:
        w = windows[windows["window"] == window].iloc[0]
        s = sex[sex["window"] == window].iloc[0]
        p1 = models[
            (models["window"] == window)
            & (models["part"] == "part1_any_overlap")
            & (models["test"] == "omnibus_pair_type_LR")
        ]
        p2 = models[
            (models["window"] == window)
            & (models["part"] == "part2_extent_given_overlap")
            & (models["test"] == "omnibus_pair_type_LR")
        ]
        d = desc[desc["window"] == window].set_index("pair_sex")

        rows.append(
            {
                "Timeframe": WINDOW_LABELS.get(window, window),
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
                "Part 1: chance of overlap permutation p-value": round(float(p1["p_permutation"].iloc[0]), 3) if len(p1) else None,
                "FF prop. overlap (%)": round(
                    100 * float(d.at["Female-Female", "median_prop_overlap_nonzero"]), 1
                ),
                "FM prop. overlap (%)": round(
                    100 * float(d.at["Female-Male", "median_prop_overlap_nonzero"]), 1
                ),
                "MM prop. overlap (%)": round(
                    100 * float(d.at["Male-Male", "median_prop_overlap_nonzero"]), 1
                ),
                "Part 2: extent of overlap permutation p-value": round(float(p2["p_permutation"].iloc[0]), 3) if len(p2) else None,
            }
        )
    return pd.DataFrame(rows)


def table_s4() -> pd.DataFrame:
    dec = _read("reviewer_decomposition.csv")
    return dec.rename(
        columns={
            "step": "Step",
            "cumulative_change": "Cumulative change",
            "estimator": "Estimator",
            "response": "Response",
            "dyads": "Dyads included",
            "inference": "Inference",
            "n_FF": "n FF",
            "n_FM": "n FM",
            "median_FF": "Median FF",
            "median_FM": "Median FM",
            "p_focal_FF_vs_FM": "p (FF vs FM)",
        }
    ).round(4)


def table_s5() -> pd.DataFrame:
    summary = _read("overlap_null_model_summary__contemporary_24mo.csv")
    regions = _read("overlap_null_model_regions__contemporary_24mo.csv").set_index("region")
    region_label = {
        "sampled_1000m": "Encounters + 1 km (primary)",
        "sampled_500m": "Encounters + 500 m",
        "roads2013_1000m": "2013 roads + 1 km",
        "roads2013_2000m": "2013 roads + 2 km",
    }
    metric_label = {"pct_overlapping": "Dyads overlapping (%)", "mean_prop_overlap": "Mean proportional overlap (%)"}
    pair_label = {"all": "All", "Female-Female": "FF", "Female-Male": "FM", "Male-Male": "MM"}
    rows = []
    for r in summary.itertuples():
        reg = regions.loc[r.region]
        rows.append({
            "Null area": region_label.get(r.region, r.region),
            "Area (km2)": round(float(reg["area_km2"]), 1),
            "Observed centroids inside": f"{int(reg['observed_centroids_inside'])}/{int(reg['n_polygons'])}",
            "Dyads": pair_label[r.group],
            "Metric": metric_label[r.metric],
            "Observed": round(r.observed, 1),
            "Null median": round(r.null_p50, 1),
            "Null 5th-95th percentile": f"{r.null_p05:.1f}-{r.null_p95:.1f}",
            "Observed percentile": round(r.observed_percentile, 1),
            "p (lower)": round(r.p_lower, 3),
            "p (higher)": round(r.p_upper, 3),
        })
    return pd.DataFrame(rows)


CAPTIONS = {
    "S1": "Supplementary Table S1. Camera makes and models used to photograph leopards in the Chitabe archive. Records without EXIF make/model information are omitted.",
    "S2": "Supplementary Table S2. Individually identified leopards: detection history, sampling metrics, MCP areas under both estimators, the number of locations removed by 95% peripheral removal, and inclusion status for the spatial analyses, from dated encounter events. The five columns headed 2024-26 describe each individual within the primary 2024-06-01 to 2026-05-25 window used for the range and overlap analyses. Individuals known from the lineage record but without georeferenced photographic detections are listed with the status \"Excluded: no detections\"; individuals photographed only on images whose sole date was the archive export date (16 March 2026) are listed as \"Excluded: only undatable (archive-export) records\".",
    "S3": "Supplementary Table S3. Home ranges and pairwise overlap statistics recalculated separately over three timeframes, to test whether the results are sensitive to the timeframe used. The 24 months row is the primary analysis, including individuals detected within the most recent, temporally overlapping 2-year period; the 12 months row restricts the analysis to the final year; the pooled row repeats the analysis using every dated record across the full study period as a sensitivity check, and is interpreted as an accumulated space-use footprint rather than evidence of simultaneous occupation. For each timeframe, the table reports the number of individuals and dyads (all possible pairs) included; median 95% minimum convex polygon (MCP) home-range size for females and males, with the Mann-Whitney U statistic and p-value testing whether home-range size differs between the sexes; and overlap statistics for female-female (FF), female-male (FM) and male-male (MM) pairs. Overlapping (%) gives the percentage of all dyads of that pair type whose home ranges share any area at all; prop. overlap (%) is the median percentage of combined home-range area shared, calculated only among the dyads that do overlap. Part 1 permutation p-values test whether the probability of overlapping at all differs among the three pair types; Part 2 permutation p-values test whether the extent of overlap, given that a pair already overlaps, differs among pair types. All permutation p-values are from 10,000 node-label permutations, which account for each individual contributing to multiple pairwise comparisons.",
    "S4": "Supplementary Table S4. Effect of each analytical choice on the focal female-female versus female-male overlap comparison, applied cumulatively to the pooled data used in the original submission. Step A reproduces the submitted analysis; step G is the revised analysis before the date correction, and step H additionally excludes encounter events whose only date was the archive export date.",
    "S5": "Supplementary Table S5. Observed pairwise overlap in the 2024-06-01 to 2026-05-25 window compared with overlap expected under random placement of ranges. Each individual's 95% MCP was rotated by a random angle and relocated so that its centroid fell at a random position within the null area, 2,000 times per area. Null areas are NG31 within the stated distance of any GPS-referenced leopard encounter (all individuals and years) or of the 2013 game-drive road map; the road map does not cover the full network driven in 2024-2026, so road-based areas are sensitivity analyses. Observed centroids inside gives how many of the observed range centroids lie within each area. p (lower) is the one-sided probability of overlap as low as observed under random placement (evidence of avoidance if below 0.05); p (higher) the probability of overlap as high as observed.",
}


def to_markdown() -> str:
    parts = ["# Supplementary Tables", ""]
    for key, frame in (("S1", table_s1()), ("S2", table_s2()), ("S3", table_s3()), ("S4", table_s4()), ("S5", table_s5())):
        parts += [CAPTIONS[key], "", frame.to_markdown(index=False), "", ""]
    return "\n".join(parts)


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(to_markdown(), encoding="utf-8")
    print(f"Saved -> {OUT_MD}")

    body = [_docx.paragraph("Supplementary Tables", bold=True, size_half_points=28)]
    for key, frame in (("S1", table_s1()), ("S2", table_s2()), ("S3", table_s3()), ("S4", table_s4()), ("S5", table_s5())):
        body.append(_docx.paragraph(CAPTIONS[key], bold=True))
        body.append(_docx.table(frame, total_width_dxa=15000))
        body.append(_docx.paragraph(""))
    _docx.write_docx(OUT_DOCX, "".join(body), landscape=True)
    print(f"Saved -> {OUT_DOCX}")


if __name__ == "__main__":
    main()
