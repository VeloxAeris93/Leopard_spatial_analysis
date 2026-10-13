# Chitabe Leopards: Spatial Analysis

Companion code for *Patterns of spatial overlap of leopards in the Okavango Delta*
(Pretorius, Hamman, Keith & Somers). It rebuilds every table, figure and number in the
revised manuscript from the raw photographic and sightings records.

This is a **code companion, not a data repository**: no leopard location data is
included (see [Data](#data)).

The scripts here are the exact scripts that produced the revised manuscript, copied
from the project's working repository rather than re-implemented, so what you run is
what we ran. The version that accompanied the original submission (a single
`analysis.py`) is kept under the git tag
[`submitted-version`](../../tree/submitted-version).

## Methodology

Leopards are identified from an archive of guide and camera photographs, a manually
kept sightings log and the lineage register. In outline:

1. **Photo metadata** (`0`): GPS position and capture date from each photo's EXIF data
   (read with Pillow, and with ExifTool where Pillow finds no GPS), with the leopard's name
   taken from its folder.
   A later batch of photos delivered without per-leopard folders is read separately.
2. **Match to the sightings log** (`16a`): each photo is linked to the nearest sighting
   of the same leopard; matches within 50 m count as high confidence.
3. **Batch identities** (`16c`): the batch photos are named from a manual list
   (`file_names_list.csv`); photos showing several leopards become one record per animal.
4. **Encounter events** (`16b`): repeat photos of the same leopard on the same day within
   250 m become one encounter event. Names later confirmed to be one animal are merged
   first (`src/leopards/naming.py`: Morobe is Tswaletswale).
5. **Date validity** (`16d`): the archive was exported with its capture dates stripped, so
   photos whose only date is the export day (16 March 2026) keep their location but are
   excluded from every dated analysis (`src/_dates.py`).
6. **Seasonality** (`25`): detections by Botswana wet (Nov–Mar) and dry (Apr–Oct) season.
7. **Inclusion** (`21`, `src/_inclusion.py`, `src/_age.py`): range estimates need at least
   8 unique locations on at least 3 dates; cubs and subadults are excluded.
8. **Home ranges** (`18`, `33`): 95% minimum convex polygons (MCP), estimated both pooled
   and within time windows. The primary window is 1 June 2024 to 25 May 2026, so every
   polygon in a comparison comes from the same period.
9. **Overlap** (`33`, `34`): every pair of leopards in the window, including pairs that do
   not overlap. Two-part models (does a pair overlap; if so, by what proportion) by pair
   type (female–female, female–male, male–male), tested by node-label permutation
   (10,000 permutations).
10. **Null model** (`41`): observed overlap compared with overlap when the same polygons
    are placed at random within the sampled area (and, as sensitivity checks, within 1 and
    2 km of the 2013 road network).
11. **Sex difference in range size** (`33`, `42`): Mann–Whitney test, effect size, and the
    smallest difference the test could detect with 80% power.
12. **Reviewer decomposition** (`37`): the submitted analysis rebuilt step by step into the
    revised one, so each change's effect on the result can be seen.
13. **Outputs** (`01b`, `24c`, `35`, `23c`, `32`, `36`, `39`): Figures 1–6 and S1–S2, Supplementary Tables
    S1–S5, and `outputs/tables/manuscript_numbers.md`, which lists every number quoted in
    the manuscript with the table it comes from.

Home ranges summarise where each leopard was recorded, given uneven sampling effort
across individuals and years; they are not exact territorial boundaries.

## Data

No raw data is committed. To run the pipeline, place these files in `data/`:

| File | Contents |
|---|---|
| `Chitabe Leopards.zip` | Photo archive, one top-level folder per leopard (e.g. `Mapula female/`) |
| `2026-05-27.zip` | Later photo batch, without per-leopard folders |
| `file_names_list.csv` | Manual identities for the batch photos: `File Name`, `Leopard` (comma-separated when several leopards are in one photo) |
| `Chitabe Leopards.xlsx` | Sightings log, sheet `Sheet1`: `Year`, `Month`, `Day`, `Lat`, `Long`, `Leopard`, `Guide`, `Behaviour`, `Notes`, `Other_Leopard`, `Leopard_2`, `Total_leopards` |
| `leopard_lineage_master.xlsx` | Lineage register, sheet `Lineage`: `leopard_name`, `sex`, `Estimated age`, plus family links |
| `chitabe_roads_2013.geojson` | 2013 game-drive road network (null-model sensitivity regions) |
| `ng31.geojson` | NG31 concession boundary |
| `ng31_edit.shp` (with `.shx`, `.dbf`, `.prj`) | Edited NG31 outline drawn on the study-area map (Figure 1) |

Please contact the authors for access to the underlying data.

## Running

Tested with Python 3.14 on Windows.

```bash
pip install -r requirements.txt
python src/run_paper_pipeline.py            # run what is out of date
python src/run_paper_pipeline.py --force    # rebuild everything
python src/run_paper_pipeline.py --check    # report what is out of date, run nothing
```

- **ExifTool** must be installed (exiftool.org). Set the `EXIFTOOL` environment variable if
  it is not on your PATH.
- A full run takes about half an hour; the permutation models (`34`) and the null model
  (`41`) take most of it.
- Figure 3 (`23c`) draws its floodplain from a saved OpenStreetMap snapshot (`reference/`,
  © OpenStreetMap contributors, ODbL), so it matches the published figure exactly.
- Figure 1 (`24c`) downloads its greyscale background map tiles on the first run and
  caches them in `outputs/cache/`. The tiles change over time, so the background can
  differ slightly from the published figure; the plotted records, boundary and labels
  are identical. Its Africa inset uses Natural Earth outlines saved in `reference/`.

The runner records each stage's inputs and outputs. It reruns anything out of date and
finishes with a consistency check, so the tables, figures and numbers always come from
the same run. Tables are written to `outputs/tables/`, figures to `outputs/figures/`, and
Supplementary Tables to `manuscript/`.
