# Chitabe Leopards: Spatial Analysis

Companion code for *Patterns of spatial overlap of leopards in the Okavango Delta*
(Pretorius, Hamman, Keith & Somers). One script, `analysis.py`, rebuilds every table,
figure and number in the revised manuscript from the raw photographic and sightings
records.

This is a **code companion, not a data repository**: no leopard location data is
included (see [Data](#data)).

The version that accompanied the original submission is kept under the git tag
[`submitted-version`](../../tree/submitted-version).

## Running

Tested with Python 3.14 on Windows.

```bash
pip install -r requirements.txt
python analysis.py
```

- **ExifTool** must be installed (exiftool.org). Set the `EXIFTOOL` environment variable if
  it is not on your PATH.
- A full run takes about half an hour; the overlap models (step 11) and the null model
  (step 12) take most of it.
- Tables are written to `outputs/tables/`, figures to `outputs/figures/`, and the
  Supplementary Tables (Word and Markdown) to `outputs/`.
  `outputs/tables/manuscript_numbers.md` lists every number quoted in the manuscript.

## What the script does

`analysis.py` runs 18 steps, top to bottom. Each step saves its result in
`outputs/tables/`, so every stage can be inspected. All settings (file locations,
inclusion thresholds, time windows, random seeds) are at the top of the script.

| Step | | |
|---|---|---|
| 1 | Photo metadata | GPS position and capture date from each photo's EXIF data (Pillow, and ExifTool where Pillow finds no GPS). The leopard's name comes from its folder. |
| 2 | Sightings log | Each photo is linked to the nearest sighting of the same leopard; matches within 50 m count as reliable. |
| 3 | Batch identities | A later batch of photos, delivered without per-leopard folders, is named from a manual list (`file_names_list.csv`). A photo showing several leopards becomes one record per animal. |
| 4 | Encounter events | Repeat photos of one leopard on one day within 250 m become one encounter event. Names later confirmed to be one animal are merged first (Morobe is Tswaletswale). |
| 5 | Camera summary | Cameras used (Table S1). |
| 6 | Date validity | The archive was exported with its capture dates stripped. Photos whose only date is the export day (16 March 2026) keep their location but are left out of every dated analysis. |
| 7 | Seasonality | Detections by Botswana wet (Nov–Mar) and dry (Apr–Oct) season (Figure 2). |
| 8 | Inclusion | A range estimate needs at least 8 unique locations on at least 3 dates; cubs and subadults are excluded. |
| 9 | Pooled home ranges | 95% and 100% minimum convex polygons (MCP) over the whole study period (Figure S2). |
| 10 | Time windows | 95% MCPs and the overlap of every pair of leopards, including pairs that do not overlap, within each window. The primary window is 1 June 2024 to 25 May 2026, so every polygon in a comparison comes from the same period. |
| 11 | Overlap models | Two-part models (does a pair overlap; if so, by what proportion) by pair type (female–female, female–male, male–male), tested with 10,000 node-label permutations. |
| 12 | Null model | Observed overlap compared with overlap when the same polygons are placed at random within the sampled area, and, as sensitivity checks, near the 2013 road network (Figure 6, Table S5). |
| 13 | Power | The smallest sex difference in range size the Mann–Whitney test could detect with 80% power. |
| 14 | Reviewer decomposition | The submitted analysis rebuilt step by step into the revised one, so each change's effect can be seen (Table S4). |
| 15 | Detection table | One row per leopard: detection history, range sizes and inclusion status (Table S2). |
| 16 | Figures | Figures 1–5, S1 and S2 (Figure 6 is drawn in step 12). |
| 17 | Manuscript numbers | Every number quoted in the manuscript, with the table it comes from. |
| 18 | Supplementary Tables | Tables S1–S5. |

Home ranges summarise where each leopard was recorded, given uneven sampling effort
across individuals and years; they are not exact territorial boundaries.

## Data

No raw data is committed. To run the analysis, place these files in `data/`:

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

Two reference files are included in `reference/`, so the figures do not depend on live
servers:

- `osm_floodplain_ng31_2026-08-25.json.gz`: OpenStreetMap water and wetland around NG31,
  drawn as the seasonal floodplain on Figure 3 (© OpenStreetMap contributors, ODbL).
- `ne_110m_admin_0_countries.geojson`: Natural Earth country outlines for the Africa inset
  on Figure 1 (public domain).

Figure 1's greyscale background map tiles (© OpenStreetMap contributors, © CARTO) are
downloaded on the first run and cached in `outputs/cache/`. The tiles change over time,
so the background can differ slightly from the published figure; the plotted records,
boundary and labels are identical.
