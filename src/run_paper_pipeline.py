"""Ordered driver for the paper analysis, with a staleness guard.

The submitted manuscript carried figures built on 9 June from tables that were
rebuilt on 11 June: script 22 was simply never re-run after script 19. Nothing in
the pipeline noticed. This runner exists so that cannot happen again.

Each stage declares its inputs and outputs. A stage runs when any output is missing
or older than any input. After every stage has run, the manifest is checked: if any
output is older than any input anywhere in the graph, the run fails loudly rather
than leaving a mixed-vintage set of tables and figures on disk.

The first three stages start from the raw inputs (photo archives, sightings log,
lineage register, manual photo identities; locations in _paths.py), so a clean
checkout with those files in data/ rebuilds every number in the manuscript.

Usage:
    python src/run_paper_pipeline.py            # run what is stale
    python src/run_paper_pipeline.py --force    # rebuild everything
    python src/run_paper_pipeline.py --check    # report staleness, run nothing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys

from _paths import BATCH_ZIPS, FILE_NAMES_LIST, LINEAGE_XLSX, NG31_EDIT_SHP, NG31_GEOJSON, PHOTO_ZIP, ROADS_GEOJSON, SIGHTINGS_XLSX

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"


@dataclass(frozen=True)
class Stage:
    script: str
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    note: str = ""


STAGES: list[Stage] = [
    Stage(
        "0_extract_zip_image_metadata_to_excel.py",
        inputs=[PHOTO_ZIP, *BATCH_ZIPS, LINEAGE_XLSX],
        outputs=[
            TABLES / "leopard_images_metadata.csv",
            TABLES / "leopard_images_metadata.filtered.csv",
            TABLES / "leopard_images_metadata.batches.csv",
        ],
        note="EXIF/GPS/date metadata from the photo archives (slow, ~3 min)",
    ),
    Stage(
        "16a_match_image_metadata_to_sightings.py",
        inputs=[TABLES / "leopard_images_metadata.filtered.csv", SIGHTINGS_XLSX],
        outputs=[
            TABLES / "leopard_images_metadata_combined.csv",
            TABLES / "merged_leopard_images_metadata.filtered.corrected.csv",
        ],
        note="match photos to the guides' sightings log",
    ),
    Stage(
        "16c_apply_filename_identity_overrides.py",
        # It rewrites 0's and 16a's tables in place, ending with the corrected table, so it
        # depends on the raw sources behind those tables rather than on the tables themselves.
        inputs=[
            FILE_NAMES_LIST,
            LINEAGE_XLSX,
            SIGHTINGS_XLSX,
            TABLES / "leopard_images_metadata.batches.csv",
            TABLES / "leopard_images_metadata.filtered.csv",
        ],
        outputs=[TABLES / "merged_leopard_images_metadata.filtered.corrected.csv"],
        note="add the batch photos and their manual identities",
    ),
    Stage(
        "16b_normalize_merged_image_events.py",
        inputs=[
            TABLES / "merged_leopard_images_metadata.filtered.corrected.csv",
            SRC / "leopards" / "naming.py",
        ],
        outputs=[
            TABLES / "merged_leopard_images_metadata.normalized_events.csv",
            TABLES / "merged_leopard_images_metadata.event_normalization_rules.csv",
            TABLES / "merged_leopard_images_metadata.event_burst_examples.csv",
        ],
        note="group image bursts into events; applies MERGED_IDENTITIES",
    ),
    Stage(
        "01b_summarize_merged_image_metadata.py",
        inputs=[TABLES / "merged_leopard_images_metadata.normalized_events.csv", LINEAGE_XLSX],
        outputs=[
            TABLES / "merged_leopard_images_metadata_overview.csv",
            TABLES / "merged_leopard_images_metadata_by_individual.csv",
            TABLES / "merged_leopard_images_metadata_camera_summary.csv",
        ],
        note="dataset overview and camera summary (Supplementary Table S1)",
    ),
    Stage(
        "16d_exclude_archive_dates.py",
        inputs=[
            TABLES / "leopard_images_metadata.csv",
            TABLES / "merged_leopard_images_metadata.filtered.corrected.csv",
            TABLES / "merged_leopard_images_metadata.normalized_events.csv",
            SRC / "_dates.py",
        ],
        outputs=[
            TABLES / "merged_leopard_images_metadata.dated_events.csv",
            TABLES / "date_audit_archive_export.csv",
            TABLES / "date_source_counts.csv",
        ],
        note="drop events dated only by the archive export date",
    ),
    Stage(
        "25_merged_image_heatmap_and_seasons.py",
        inputs=[TABLES / "merged_leopard_images_metadata.dated_events.csv"],
        outputs=[
            TABLES / "merged_image_counts_by_botswana_season.csv",
            TABLES / "merged_image_counts_by_botswana_season_and_sex.csv",
        ],
        note="season counts (Figure 2, Results)",
    ),
    Stage(
        "21_prepare_merged_image_spatial_subset.py",
        inputs=[TABLES / "merged_leopard_images_metadata.dated_events.csv"],
        outputs=[
            TABLES / "merged_leopard_images_metadata.spatial_subset.csv",
            TABLES / "merged_leopard_images_metadata.spatial_subset_individuals.csv",
            TABLES / "date_audit_rejected.csv",
        ],
        note="spatial subset + date-validity audit",
    ),
    Stage(
        "18_merged_image_home_ranges.py",
        inputs=[TABLES / "merged_leopard_images_metadata.spatial_subset.csv"],
        outputs=[
            TABLES / "merged_image_home_ranges_mcp_summary.csv",
            FIGS / "merged_image_home_ranges_mcp95.geojson",
            FIGS / "merged_image_home_ranges_mcp.geojson",
        ],
        note="pooled 95% and 100% MCPs",
    ),
    Stage(
        "19_merged_image_overlap_and_stats.py",
        inputs=[
            TABLES / "merged_image_home_ranges_mcp_summary.csv",
            FIGS / "merged_image_home_ranges_mcp95.geojson",
        ],
        outputs=[
            TABLES / "merged_image_home_range_sex_stats.csv",
            TABLES / "merged_image_home_range_overlap_pairs.csv",
        ],
        note="legacy pooled tables (consumed by the dashboard build)",
    ),
    Stage(
        "33_temporal_windows.py",
        inputs=[TABLES / "merged_leopard_images_metadata.dated_events.csv"],
        outputs=[
            TABLES / "merged_image_window_summary.csv",
            TABLES / "merged_image_window_sex_stats.csv",
            TABLES / "merged_image_overlap_pairs__contemporary_24mo.csv",
            TABLES / "merged_image_overlap_pairs__pooled.csv",
            TABLES / "merged_leopard_images_metadata.window_subset__contemporary_24mo.csv",
            FIGS / "merged_image_home_ranges_mcp95__contemporary_24mo.geojson",
        ],
        note="window-restricted MCPs, dyads and point subsets",
    ),
    Stage(
        "34_overlap_models.py",
        inputs=[TABLES / "merged_image_overlap_pairs__contemporary_24mo.csv"],
        outputs=[
            TABLES / "overlap_model_results.csv",
            TABLES / "overlap_descriptives_by_pair_type.csv",
        ],
        note="two-part models, 10,000 permutations (slow, ~7 min)",
    ),
    Stage(
        "41_overlap_null_model.py",
        inputs=[
            FIGS / "merged_image_home_ranges_mcp95__contemporary_24mo.geojson",
            TABLES / "merged_leopard_images_metadata.normalized_events.csv",
            ROADS_GEOJSON,
            NG31_GEOJSON,
        ],
        outputs=[
            TABLES / "overlap_null_model_summary__contemporary_24mo.csv",
            TABLES / "overlap_null_model_regions__contemporary_24mo.csv",
            FIGS / "fig6_overlap_null_model__contemporary_24mo.png",
        ],
        note="random-placement null model for overlap, 2,000 iterations x 4 regions (~15 min)",
    ),
    Stage(
        "42_sex_test_power.py",
        inputs=[TABLES / "merged_image_window_sex_stats.csv"],
        outputs=[TABLES / "sex_test_power.csv"],
        note="detectable effect size for the sex comparison",
    ),
    Stage(
        "37_reviewer_decomposition.py",
        inputs=[
            TABLES / "merged_leopard_images_metadata.normalized_events.csv",
            TABLES / "merged_leopard_images_metadata.dated_events.csv",
        ],
        outputs=[TABLES / "reviewer_decomposition.csv"],
        note="submitted-to-revised decomposition for the response letter",
    ),
    Stage(
        "35_paper_figures_revision.py",
        inputs=[
            TABLES / "merged_image_overlap_pairs__contemporary_24mo.csv",
            TABLES / "overlap_model_results.csv",
            TABLES / "merged_image_window_sex_stats.csv",
            TABLES / "merged_image_home_ranges_mcp_summary.csv",
            TABLES / "merged_image_counts_by_botswana_season_and_sex.csv",
        ],
        outputs=[
            FIGS / "fig2_seasonal_detections_by_sex.png",
            FIGS / "fig4_mcp95_by_sex_contemporary.png",
            FIGS / "fig5_overlap_by_pair_type_revised.png",
            FIGS / "figS1_window_concordance.png",
            FIGS / "figS2_mcp95_vs_mcp100.png",
        ],
        note="revised Figures 2, 4, 5, S1, S2",
    ),
    Stage(
        "23c_paper_home_range_map_carto_positron.py",
        inputs=[
            FIGS / "merged_image_home_ranges_mcp95__contemporary_24mo.geojson",
            TABLES / "merged_leopard_images_metadata.window_subset__contemporary_24mo.csv",
        ],
        outputs=[FIGS / "new_fig_merged_subset_home_ranges_map_osm_vector_gray.png"],
        note="Figure 3 - contemporary-window 95% MCP map",
    ),
    Stage(
        "24c_paper_study_area_map_carto_positron.py",
        inputs=[TABLES / "merged_leopard_images_metadata.dated_events.csv", NG31_EDIT_SHP],
        outputs=[FIGS / "new_fig_study_area_ng31_osm_vector_gray.png"],
        note="Figure 1 - study area and encounter locations (basemap tiles cached on first run)",
    ),
    Stage(
        "32_export_individual_detection_table.py",
        inputs=[
            TABLES / "merged_leopard_images_metadata.dated_events.csv",
            TABLES / "merged_image_home_ranges_mcp_summary.csv",
            TABLES / "merged_image_window_individuals__contemporary_24mo.csv",
            TABLES / "merged_leopard_images_metadata.normalized_events.csv",
        ],
        outputs=[TABLES / "merged_leopard_individual_detection_table.csv"],
        note="Supplementary Table S2 (individuals)",
    ),
    Stage(
        "36_manuscript_numbers.py",
        inputs=[
            TABLES / "overlap_model_results.csv",
            TABLES / "merged_image_window_sex_stats.csv",
            TABLES / "reviewer_decomposition.csv",
            TABLES / "merged_leopard_individual_detection_table.csv",
            TABLES / "date_source_counts.csv",
            TABLES / "merged_image_counts_by_botswana_season.csv",
            TABLES / "overlap_null_model_summary__contemporary_24mo.csv",
            TABLES / "sex_test_power.csv",
        ],
        outputs=[TABLES / "manuscript_numbers.json", TABLES / "manuscript_numbers.md"],
        note="every number quoted in the manuscript",
    ),
    Stage(
        "39_build_supplementary_tables.py",
        inputs=[
            TABLES / "manuscript_numbers.json",
            TABLES / "merged_leopard_individual_detection_table.csv",
            TABLES / "merged_leopard_images_metadata_camera_summary.csv",
        ],
        outputs=[ROOT / "outputs" / "supplementary_tables.md"],
        note="Supplementary Tables S1-S5",
    ),
]

# Only the paper's analyses are staged here. Public-facing products built elsewhere in
# the project (dashboards, apps) are deliberately left out so a failure there cannot
# block the paper pipeline; rebuild them separately after a paper run.


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else -1.0


def is_stale(stage: Stage) -> tuple[bool, str]:
    missing = [p for p in stage.outputs if not p.exists()]
    if missing:
        return True, f"missing output: {missing[0].name}"

    newest_input = max((_mtime(p) for p in stage.inputs), default=-1.0)
    oldest_output = min(_mtime(p) for p in stage.outputs)
    if newest_input > oldest_output:
        stale_in = max(stage.inputs, key=_mtime)
        return True, f"input {stale_in.name} is newer than output"
    return False, "up to date"


def run(stage: Stage) -> None:
    print(f"\n=== {stage.script} :: {stage.note} ===", flush=True)
    result = subprocess.run([sys.executable, str(SRC / stage.script)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"FAILED: {stage.script} (exit {result.returncode})")


def verify_manifest() -> None:
    print("\n=== manifest ===")
    problems: list[str] = []
    for stage in STAGES:
        stale, reason = is_stale(stage)
        flag = "STALE" if stale else "ok"
        print(f"  [{flag:5s}] {stage.script:45s} {reason}")
        if stale:
            problems.append(f"{stage.script}: {reason}")
    if problems:
        raise SystemExit(
            "\nPipeline is internally inconsistent - some outputs are older than their inputs:\n  "
            + "\n  ".join(problems)
            + "\nDo not quote these numbers in the manuscript. Re-run with --force."
        )
    print("\nAll stages consistent. Numbers in outputs/tables/manuscript_numbers.md are safe to quote.")


def main() -> None:
    force = "--force" in sys.argv
    check_only = "--check" in sys.argv

    if check_only:
        verify_manifest()
        return

    for stage in STAGES:
        stale, reason = is_stale(stage)
        if force or stale:
            run(stage)
        else:
            print(f"skip {stage.script} ({reason})")

    verify_manifest()


if __name__ == "__main__":
    main()
