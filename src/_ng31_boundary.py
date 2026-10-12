from __future__ import annotations

import json
from pathlib import Path


NG31_PATH = Path("ng31.geojson")


def load_ng31_lonlat_lines(path: Path = NG31_PATH) -> list[list[tuple[float, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find concession boundary file: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    lines: list[list[tuple[float, float]]] = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])

        if geom_type == "LineString":
            lines.append([(float(lon), float(lat)) for lon, lat, *_ in coords])
        elif geom_type == "MultiLineString":
            for part in coords:
                lines.append([(float(lon), float(lat)) for lon, lat, *_ in part])
        elif geom_type == "Polygon":
            for ring in coords:
                lines.append([(float(lon), float(lat)) for lon, lat, *_ in ring])
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    lines.append([(float(lon), float(lat)) for lon, lat, *_ in ring])

    if not lines:
        raise ValueError(f"No usable boundary geometry found in: {path.resolve()}")
    return lines
