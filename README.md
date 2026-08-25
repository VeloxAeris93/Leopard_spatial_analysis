# Chitabe Leopards: Spatial Analysis

Companion analysis code for the Chitabe leopard spatial-ecology paper
(Okavango Delta, Botswana). This repository reproduces the full analysis
pipeline — from raw photographic/sightings records through to the home-range
and statistical results reported in the paper — in a single, readable script.

This is a **code companion**, not a data repository: no raw leopard location
data is included here (see [Data](#data) below).

## Methodology

Leopards are identified from a photo archive of camera and guide images plus
a manually maintained sightings log. The pipeline:

1. **Extract image metadata** — reads GPS coordinates and capture dates from
   each photo's EXIF data (falling back to a date encoded in the filename,
   then the archive's file-modified time).
2. **Clean the sightings log** — validates coordinates and fixes obvious
   lat/long swaps.
3. **Match photos to the sightings log** — links each photo to the closest
   same-individual sightings-log entry within 50 m, borrowing its date to
   correct photos whose EXIF date was corrupted during archiving.
4. **Normalize detection events** — collapses repeat photos of the same
   leopard, on the same day, within 250 m of each other into a single
   detection event, so results reflect encounters rather than shutter count.
5. **Temporal analysis** — detection timelines and monthly/seasonal patterns
   (Botswana wet season: November–March, dry season: April–October).
6. **Spatial subset** — restricts the home-range comparison to individuals
   with at least 10 unique GPS points and 3 unique dates, to avoid
   over-interpreting sparse records.
7. **Home ranges** — minimum convex polygon (MCP) area per individual.
8. **Overlap** — pairwise home-range overlap (km², % of range shared).
9. **Statistics** — male vs. female home-range size (Mann-Whitney U, Cliff's
   delta, bootstrap median difference with 95% CI) and overlap by pair type
   (Female-Female / Female-Male / Male-Male; Kruskal-Wallis with Holm-corrected
   pairwise Mann-Whitney follow-ups).
10. **Detection table** — a final per-individual summary (identity, sex,
    detection counts, date range, home-range area, spatial-subset inclusion).

Home ranges are broad indicators of space use given the available sightings,
not exact territorial boundaries, and results should be read in light of
uneven sampling effort across individuals and years.

## Data

No raw data is committed to this repository. To run the pipeline, create a
`data/` folder containing:

| File | Description | Required columns |
|---|---|---|
| `data/leopard_photos.zip` | Photo archive, one top-level folder per leopard named e.g. `Chikara Female/` | JPEG/PNG files with EXIF GPS + date where available |
| `data/sightings_log.xlsx` | Manual sightings log, sheet `Sheet1` | `Year`, `Month`, `Day`, `Lat`, `Long`, `Leopard`, `Guide`, `Behaviour`, `Notes`, `Other_Leopard`, `Leopard_2`, `Total_leopards` |
| `data/leopard_lineage.xlsx` | Individual attributes, sheet `Lineage` | `leopard_name`, `sex`, `Estimated age` |

Get in touch with the project team for access to the underlying dataset.

## Running

```bash
pip install -r requirements.txt
python analysis.py
```

Tables are written to `outputs/tables/`, figures to `outputs/figures/`
(both gitignored — generated locally, not committed).


