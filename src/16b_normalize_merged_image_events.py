from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import pandas as pd

from leopards.naming import MERGED_IDENTITIES


IN_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.filtered.corrected.csv"
OUT_TABLES = Path("outputs") / "tables"

OUT_EVENTS = OUT_TABLES / "merged_leopard_images_metadata.normalized_events.csv"
OUT_RULES = OUT_TABLES / "merged_leopard_images_metadata.event_normalization_rules.csv"
OUT_BURSTS = OUT_TABLES / "merged_leopard_images_metadata.event_burst_examples.csv"

SPATIAL_THRESHOLD_M = 250.0


@dataclass
class UnionFind:
    parent: list[int]

    @classmethod
    def create(cls, n: int) -> "UnionFind":
        return cls(parent=list(range(n)))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna() if str(v).strip() and str(v) != "Unknown"]
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "Unknown"


def _meters_projection(lat: float, lon: float, lat0: float) -> tuple[float, float]:
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    return lon * m_per_deg_lon, lat * m_per_deg_lat


def _cluster_indices(group: pd.DataFrame, threshold_m: float) -> list[list[int]]:
    coords = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
    if len(coords) <= 1:
        return [list(range(len(coords)))]

    lat0 = float(group["gps_lat"].mean())
    pts_m = [_meters_projection(lat, lon, lat0) for lat, lon in coords]
    uf = UnionFind.create(len(pts_m))

    for i in range(len(pts_m)):
        xi, yi = pts_m[i]
        for j in range(i + 1, len(pts_m)):
            xj, yj = pts_m[j]
            if math.hypot(xi - xj, yi - yj) <= threshold_m:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(len(pts_m)):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def _join_unique_text(series: pd.Series, sep: str = " | ") -> str | None:
    values = []
    seen: set[str] = set()
    for value in series.dropna():
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
    return sep.join(values) if values else None


def _join_unique_tags(series: pd.Series) -> str | None:
    tags: list[str] = []
    seen: set[str] = set()
    for value in series.dropna():
        for part in str(value).split(";"):
            tag = part.strip()
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            tags.append(tag)
    return "; ".join(tags) if tags else None


def _max_pair_distance_m(group: pd.DataFrame) -> float:
    coords = list(zip(group["gps_lat"].astype(float), group["gps_lon"].astype(float)))
    if len(coords) <= 1:
        return 0.0
    lat0 = float(group["gps_lat"].mean())
    pts_m = [_meters_projection(lat, lon, lat0) for lat, lon in coords]
    max_dist = 0.0
    for i in range(len(pts_m)):
        xi, yi = pts_m[i]
        for j in range(i + 1, len(pts_m)):
            xj, yj = pts_m[j]
            max_dist = max(max_dist, math.hypot(xi - xj, yi - yj))
    return float(max_dist)


def _pick_representative(cluster: pd.DataFrame) -> pd.Series:
    ordered = cluster.sort_values(
        ["image_time_parsed", "file_name"],
        ascending=[True, True],
        na_position="last",
    )
    return ordered.iloc[0]


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {IN_PATH.resolve()}")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH)
    df["leopard_name"] = df.get("leopard_name", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    # Names later confirmed to be one leopard are grouped under the current name before
    # events are formed, so one encounter can't count twice under two names.
    df["leopard_name"] = df["leopard_name"].replace(MERGED_IDENTITIES)
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df["image_date_parsed"] = pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True)
    df["image_time_parsed"] = pd.to_datetime(df.get("image_time"), errors="coerce", format="%H:%M:%S")
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", "image_date_parsed"]).copy()
    df["sex"] = df.groupby("leopard_name")["sex"].transform(_resolve_sex)

    burst_examples = (
        df.groupby(["leopard_name", "image_date_parsed"], dropna=False)
        .size()
        .reset_index(name="n_raw_images")
        .sort_values(["n_raw_images", "leopard_name", "image_date_parsed"], ascending=[False, True, True])
        .head(25)
        .reset_index(drop=True)
    )
    burst_examples["image_date"] = burst_examples["image_date_parsed"].dt.date.astype("string")
    burst_examples = burst_examples.drop(columns=["image_date_parsed"])

    event_rows: list[dict[str, object]] = []
    event_counter = 1

    for (_, image_date), group in df.groupby(["leopard_name", "image_date_parsed"], sort=True, dropna=False):
        group = group.reset_index(drop=True)
        for cluster_number, indices in enumerate(_cluster_indices(group, SPATIAL_THRESHOLD_M), start=1):
            cluster = group.iloc[indices].copy()
            rep = _pick_representative(cluster)
            row = rep.to_dict()

            row["event_id"] = f"EVT{event_counter:05d}"
            row["event_cluster_number"] = cluster_number
            row["n_images_in_event"] = int(len(cluster))
            row["n_unique_raw_points_in_event"] = int(
                len(set(zip(cluster["gps_lat"].astype(float), cluster["gps_lon"].astype(float))))
            )
            row["event_max_pair_distance_m"] = round(_max_pair_distance_m(cluster), 2)
            row["file_name"] = str(rep.get("file_name", "") or "")
            row["representative_file_name"] = str(rep.get("file_name", "") or "")
            row["source_file_names"] = "; ".join(cluster["file_name"].astype(str).tolist())
            row["gps_lat"] = float(cluster["gps_lat"].median())
            row["gps_lon"] = float(cluster["gps_lon"].median())
            row["image_date"] = pd.Timestamp(image_date).strftime("%d/%m/%Y")
            row["image_time"] = (
                rep["image_time_parsed"].strftime("%H:%M:%S")
                if pd.notna(rep.get("image_time_parsed"))
                else None
            )
            row["metadata_note_text"] = _join_unique_text(cluster.get("metadata_note_text", pd.Series(dtype="object")))
            row["note_tags"] = _join_unique_tags(cluster.get("note_tags", pd.Series(dtype="object")))
            if "exif_ImageDescription" in cluster.columns:
                row["exif_ImageDescription"] = _join_unique_text(cluster["exif_ImageDescription"])
            event_rows.append(row)
            event_counter += 1

    events = pd.DataFrame(event_rows)
    events = events.drop(columns=["image_date_parsed", "image_time_parsed"], errors="ignore")

    rules = pd.DataFrame(
        [
            {"rule": "grouping_unit", "value": "same_leopard_same_date"},
            {"rule": "spatial_threshold_m", "value": SPATIAL_THRESHOLD_M},
            {"rule": "n_raw_rows", "value": int(len(df))},
            {"rule": "n_normalized_events", "value": int(len(events))},
            {"rule": "n_rows_removed_by_collapse", "value": int(len(df) - len(events))},
            {
                "rule": "median_raw_images_per_leopard_day",
                "value": float(
                    df.groupby(["leopard_name", "image_date_parsed"], dropna=False).size().median()
                ),
            },
            {
                "rule": "max_raw_images_in_single_leopard_day",
                "value": int(
                    df.groupby(["leopard_name", "image_date_parsed"], dropna=False).size().max()
                ),
            },
        ]
    )

    events.to_csv(OUT_EVENTS, index=False)
    rules.to_csv(OUT_RULES, index=False)
    burst_examples.to_csv(OUT_BURSTS, index=False)

    print(f"Saved -> {OUT_EVENTS}")
    print(f"Saved -> {OUT_RULES}")
    print(f"Saved -> {OUT_BURSTS}")


if __name__ == "__main__":
    main()
