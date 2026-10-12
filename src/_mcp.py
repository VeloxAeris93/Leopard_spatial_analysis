"""Shared minimum convex polygon helpers for the Chitabe leopard spatial analysis.

The paper reports 95% MCPs. Earlier versions of this pipeline hulled every unique
location, which is a 100% MCP; both estimators are provided here so the two can be
reported side by side.

Coordinates are projected to UTM 34S (EPSG:32734), which covers the Chitabe
concession (~19.45 S, 23.40 E), so polygon areas are true metric areas rather than
the equirectangular approximation used previously.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from pyproj import Transformer
from shapely.geometry import Polygon


UTM_34S = "EPSG:32734"
WGS84 = "EPSG:4326"

_TO_UTM = Transformer.from_crs(WGS84, UTM_34S, always_xy=True)
_TO_WGS84 = Transformer.from_crs(UTM_34S, WGS84, always_xy=True)


@dataclass(frozen=True)
class Pt:
    x: float
    y: float


def to_utm(lat: float, lon: float) -> Pt:
    x, y = _TO_UTM.transform(lon, lat)
    return Pt(x=float(x), y=float(y))


def to_latlon(p: Pt) -> tuple[float, float]:
    lon, lat = _TO_WGS84.transform(p.x, p.y)
    return float(lat), float(lon)


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def convex_hull(points: list[Pt]) -> list[Pt]:
    pts = sorted(set((p.x, p.y) for p in points))
    pts = [Pt(x, y) for x, y in pts]

    if len(pts) <= 1:
        return pts

    lower: list[Pt] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Pt] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def polygon_area_sq_m(poly: list[Pt]) -> float:
    if len(poly) < 3:
        return 0.0

    area = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y
    return abs(area) / 2.0


def n_points_to_drop(n: int, percent: float) -> int:
    """Number of peripheral locations removed for a `percent` MCP.

    Uses ceil so the retained fraction never exceeds `percent`; this is the
    conventional, conservative reading of "95% MCP". For n = 10 this drops one
    location, i.e. 10% of that individual's data - reported per individual in the
    home-range summary so the cost is visible.
    """
    if percent >= 100:
        return 0
    # Integer arithmetic: float division makes e.g. 0.05 * 60 == 3.0000000000000004,
    # which would round up to 4 and silently drop an extra location.
    numerator = n * (100 - percent)
    if numerator != int(numerator):
        return int(math.ceil(numerator / 100.0))
    return -(-int(numerator) // 100)


def mcp(points_ll: list[tuple[float, float]], percent: float = 95.0) -> dict[str, object]:
    """Build a `percent` MCP from (lat, lon) pairs.

    Peripheral removal drops the points furthest from the arithmetic mean centre of
    the unique locations, recomputed once over the full set (not iteratively).

    Returns a dict with the hull ring in lat/lon, the area in km^2, and the counts
    needed to audit the peripheral-removal rule. Returns `area_km2 = None` when
    fewer than three points survive.
    """
    unique = sorted(set((float(lat), float(lon)) for lat, lon in points_ll))
    n_unique = len(unique)

    pts = [to_utm(lat, lon) for lat, lon in unique]
    n_drop = n_points_to_drop(n_unique, percent)
    n_retained = n_unique - n_drop

    if n_retained >= 3:
        cx = sum(p.x for p in pts) / n_unique
        cy = sum(p.y for p in pts) / n_unique
        ordered = sorted(pts, key=lambda p: math.hypot(p.x - cx, p.y - cy))
        kept = ordered[:n_retained]
    else:
        kept = []

    result: dict[str, object] = {
        "percent": float(percent),
        "n_unique_points": n_unique,
        "n_points_dropped": n_drop,
        "n_points_retained": n_retained,
        "pct_points_dropped": (100.0 * n_drop / n_unique) if n_unique else None,
        "area_km2": None,
        "ring_latlon": None,
    }

    if not kept:
        return result

    hull = convex_hull(kept)
    if len(hull) < 3:
        return result

    ring = [to_latlon(p) for p in hull]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    result["area_km2"] = polygon_area_sq_m(hull) / 1_000_000.0
    result["ring_latlon"] = ring
    return result


def polygon_from_ring(ring_latlon: list[tuple[float, float]]) -> Polygon:
    """Shapely polygon in UTM 34S metres, for intersection/union work."""
    return Polygon([(p.x, p.y) for p in (to_utm(lat, lon) for lat, lon in ring_latlon)])


def geojson_feature(ring_latlon: list[tuple[float, float]], props: dict[str, object]) -> dict[str, object]:
    coords = [[lon, lat] for lat, lon in ring_latlon]
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [coords]}}
