"""Seasonal floodplain / wetland layer for the study-area figures, from OpenStreetMap.

The manuscript's Figure 1 and Figure 3 captions both state that shaded areas represent
seasonal floodplain, but no floodplain geometry exists anywhere in this project - only
the NG31 concession boundary. The original figures were almost certainly hand-drawn in
GIS software from an authoritative hydrological source (e.g. Wolski & Murray-Hudson
2006, or an Okavango Research Institute flood-extent product), which is not available
here.

This module is a stand-in, not that source: it renders OSM `natural=water` and
`natural=wetland` polygons for the Chitabe area as a visual approximation of the
floodplain extent. It is real, current public mapping data - Xini Lagoon, the Gomoti
River, and numerous named hippo pools all resolve correctly for this area - but it is
a snapshot of mapped-water extent, not a seasonal flood-pulse product, and figures
using it should be captioned accordingly (see FLOODPLAIN_CAPTION_NOTE).
"""

from __future__ import annotations

from pathlib import Path
import gzip
import json

import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union

from _paths import OSM_FLOODPLAIN_SNAPSHOT

CACHE_PATH = Path("outputs") / "cache" / "osm_floodplain_ng31.json"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Generous box around the Chitabe / NG31 concession (south, west, north, east) so one
# cached fetch covers any reasonable map extent without re-querying per figure.
BBOX = (-19.75, 23.15, -19.25, 23.65)

FLOODPLAIN_CAPTION_NOTE = (
    "Water and seasonal wetland extent from OpenStreetMap (natural=water, "
    "natural=wetland; © OpenStreetMap contributors, ODbL), shown as an "
    "approximation of the seasonal floodplain."
)


def _build_query() -> str:
    s, w, n, e = BBOX
    clauses = "".join(
        f'{kind}["natural"="{tag}"]({s},{w},{n},{e});'
        for kind in ("way", "relation")
        for tag in ("water", "wetland")
    )
    return f"[out:json][timeout:90];({clauses});out body;>;out skel qt;"


def _fetch_overpass() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    # The saved snapshot reproduces the published figure exactly; only fetch live without it.
    if OSM_FLOODPLAIN_SNAPSHOT.exists():
        return json.loads(gzip.decompress(OSM_FLOODPLAIN_SNAPSHOT.read_bytes()).decode("utf-8"))

    query = _build_query()
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            response = requests.post(
                url,
                data={"data": query},
                timeout=90,
                headers={"User-Agent": "chitabe-leopard-spatial-analysis/1.0 (research use)"},
            )
            response.raise_for_status()
            data = response.json()
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
    raise RuntimeError(f"Could not fetch OSM water/wetland data from any Overpass mirror: {last_error}")


def _way_ring(way_id: int, nodes: dict[int, tuple[float, float]], ways: dict[int, list[int]]) -> list[tuple[float, float]]:
    return [nodes[n] for n in ways.get(way_id, []) if n in nodes]


def _closed_polygon(coords: list[tuple[float, float]]) -> Polygon | None:
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    try:
        poly = Polygon(coords)
    except Exception:  # noqa: BLE001
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (not poly.is_empty and poly.area > 0) else None


def load_floodplain_polygons() -> list[Polygon]:
    """Water/wetland polygons (lon, lat) covering BBOX, merged into simple polygons."""
    data = _fetch_overpass()
    elements = data.get("elements", [])

    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}
    ways = {el["id"]: el.get("nodes", []) for el in elements if el["type"] == "way"}

    def is_tagged(el: dict) -> bool:
        return el.get("tags", {}).get("natural") in ("water", "wetland")

    polygons: list[Polygon] = []

    for way in elements:
        if way["type"] == "way" and is_tagged(way):
            poly = _closed_polygon(_way_ring(way["id"], nodes, ways))
            if poly is not None:
                polygons.append(poly)

    for rel in elements:
        if rel["type"] != "relation" or not is_tagged(rel):
            continue

        outer_polys: list[Polygon] = []
        inner_polys: list[Polygon] = []
        for member in rel.get("members", []):
            if member.get("type") != "way":
                continue
            coords = _way_ring(member["ref"], nodes, ways)
            if len(coords) < 3:
                continue

            # Only accept a member way as its own ring when it is already closed in
            # the source data. Stitching open arcs together across a relation (via
            # linemerge/polygonize) previously produced bogus polygons spanning the
            # whole query bbox whenever two unrelated arcs happened to cross in 2D
            # without sharing an endpoint - visually, a large wrong wedge cut into
            # the floodplain shading. Dropping genuinely open arcs loses a handful of
            # multi-way boundaries, which is an acceptable cost for a figure that is
            # explicitly a visual approximation, not a hydrological product.
            if coords[0] != coords[-1]:
                continue
            poly = _closed_polygon(coords)
            if poly is None:
                continue
            if member.get("role") == "inner":
                inner_polys.append(poly)
            else:
                outer_polys.append(poly)

        for outer in outer_polys:
            for inner in inner_polys:
                if outer.contains(inner.representative_point()):
                    outer = outer.difference(inner)
            polygons.append(outer)

    # Subtracting an inner ring can split an outer polygon into a MultiPolygon;
    # flatten so every returned geometry is a simple Polygon, as callers expect.
    flat: list[Polygon] = []
    for p in polygons:
        if p.is_empty:
            continue
        flat.extend(p.geoms if p.geom_type == "MultiPolygon" else [p])
    return flat


def load_floodplain_union(polygons: list[Polygon] | None = None):
    """Single (multi)polygon union, convenient for one add_patch-style call per ring."""
    polys = polygons if polygons is not None else load_floodplain_polygons()
    return unary_union(polys) if polys else None
