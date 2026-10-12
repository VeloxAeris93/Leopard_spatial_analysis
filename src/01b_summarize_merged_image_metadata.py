from __future__ import annotations

from pathlib import Path

import pandas as pd

from _paths import LINEAGE_XLSX


IN_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.normalized_events.csv"
OUT_DIR = Path("outputs") / "tables"
OUT_OVERVIEW = OUT_DIR / "merged_leopard_images_metadata_overview.csv"
OUT_BY_INDIVIDUAL = OUT_DIR / "merged_leopard_images_metadata_by_individual.csv"
OUT_BY_SEX = OUT_DIR / "merged_leopard_images_metadata_by_sex.csv"
OUT_NOTE_TAGS = OUT_DIR / "merged_leopard_images_metadata_note_tags.csv"
OUT_CAMERAS = OUT_DIR / "merged_leopard_images_metadata_camera_summary.csv"

IGNORED_LINEAGE_MARKERS = ("need names", "unknown cub", "seen with unknown cub")
IGNORED_LINEAGE_PREFIXES = ("cubs ", "new male cub", "unknown ")


def _load_data() -> pd.DataFrame:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {IN_PATH.resolve()}")

    df = pd.read_csv(IN_PATH)
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["leopard_name"] = df.get("leopard_name", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    image_date_raw = df.get("image_date", pd.Series(index=df.index, dtype="object"))
    df["image_date"] = pd.to_datetime(image_date_raw, errors="coerce", dayfirst=True, format="mixed")
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df["metadata_note_text"] = df.get("metadata_note_text", pd.Series(index=df.index, dtype="object"))
    df["note_tags"] = df.get("note_tags", pd.Series(index=df.index, dtype="object"))
    df["exif_make"] = df.get("exif_make", pd.Series(index=df.index, dtype="object"))
    df["exif_model"] = df.get("exif_model", pd.Series(index=df.index, dtype="object"))
    return df


def _display_name(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return " ".join(text.replace("_", " ").split())


def _is_ignored_lineage_name(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in IGNORED_LINEAGE_MARKERS) or any(
        lower.startswith(prefix) for prefix in IGNORED_LINEAGE_PREFIXES
    )


def _count_lineage_individuals() -> int | None:
    if not LINEAGE_XLSX.exists():
        return None

    lineage = pd.read_excel(LINEAGE_XLSX, sheet_name="Lineage")
    if "leopard_name" not in lineage.columns:
        return None

    row_names = {
        _display_name(value)
        for value in lineage["leopard_name"].dropna().astype(str)
        if _display_name(value)
    }
    if not row_names:
        return None

    relation_names: set[str] = set()
    for column in ("Mother", "Father", "Cub", "Sibling"):
        if column not in lineage.columns:
            continue
        for value in lineage[column].dropna().astype(str):
            for part in str(value).split(";"):
                for subpart in part.split(","):
                    clean = _display_name(subpart)
                    if clean:
                        relation_names.add(clean)

    extra_names = {
        name
        for name in relation_names
        if name not in row_names and not _is_ignored_lineage_name(name)
    }
    return len(row_names | extra_names)


def _build_note_tags(df: pd.DataFrame) -> pd.DataFrame:
    tags = (
        df["note_tags"]
        .dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )
    tags = tags[tags.ne("")]
    if tags.empty:
        return pd.DataFrame(columns=["note_tag", "n_images"])
    return (
        tags.value_counts()
        .rename_axis("note_tag")
        .reset_index(name="n_images")
        .sort_values(["n_images", "note_tag"], ascending=[False, True])
    )


def _build_by_individual(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["leopard_name", "sex"], dropna=False)
        .agg(
            n_images=("file_name", "size"),
            n_unique_dates=("image_date", "nunique"),
            first_date=("image_date", "min"),
            last_date=("image_date", "max"),
            n_images_with_note_text=("metadata_note_text", lambda s: int(s.notna().sum())),
            n_images_with_note_tags=("note_tags", lambda s: int(s.notna().sum())),
            min_gps_lat=("gps_lat", "min"),
            max_gps_lat=("gps_lat", "max"),
            min_gps_lon=("gps_lon", "min"),
            max_gps_lon=("gps_lon", "max"),
        )
        .reset_index()
        .sort_values(["n_images", "leopard_name"], ascending=[False, True])
    )
    for col in ("first_date", "last_date"):
        out[col] = out[col].dt.date.astype("string")
    return out


def _build_by_sex(df: pd.DataFrame, by_individual: pd.DataFrame) -> pd.DataFrame:
    image_counts = df.groupby("sex", dropna=False).size().rename("n_images")
    first_last = df.groupby("sex", dropna=False)["image_date"].agg(first_date="min", last_date="max")
    by_sex = (
        by_individual.groupby("sex", dropna=False)
        .agg(
            n_individuals=("leopard_name", "nunique"),
            mean_images_per_individual=("n_images", "mean"),
            median_images_per_individual=("n_images", "median"),
            min_images_per_individual=("n_images", "min"),
            max_images_per_individual=("n_images", "max"),
        )
        .join(image_counts)
        .join(first_last)
        .reset_index()
        .sort_values("sex")
    )
    by_sex["mean_images_per_individual"] = by_sex["mean_images_per_individual"].round(2)
    by_sex["median_images_per_individual"] = by_sex["median_images_per_individual"].round(2)
    by_sex["first_date"] = by_sex["first_date"].dt.date.astype("string")
    by_sex["last_date"] = by_sex["last_date"].dt.date.astype("string")
    return by_sex


def _build_camera_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["exif_make"] = out["exif_make"].fillna("Unknown").replace("", "Unknown")
    out["exif_model"] = out["exif_model"].fillna("Unknown").replace("", "Unknown")
    return (
        out.groupby(["exif_make", "exif_model"], dropna=False)
        .agg(n_images=("file_name", "size"), n_individuals=("leopard_name", "nunique"))
        .reset_index()
        .sort_values(["n_images", "exif_make", "exif_model"], ascending=[False, True, True])
    )


def _build_overview(df: pd.DataFrame, by_individual: pd.DataFrame, note_tags: pd.DataFrame) -> pd.DataFrame:
    lineage_individuals = _count_lineage_individuals()
    sex_counts = df["sex"].value_counts()
    camera_combo_count = (
        df[["exif_make", "exif_model"]]
        .fillna("Unknown")
        .replace("", "Unknown")
        .drop_duplicates()
        .shape[0]
    )
    overview = [
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_unique_file_names", "value": int(df.get("file_name", pd.Series(index=df.index)).nunique(dropna=True))},
        {"metric": "n_individuals", "value": int(lineage_individuals or by_individual["leopard_name"].nunique())},
        {"metric": "n_female_individuals", "value": int((by_individual["sex"] == "Female").sum())},
        {"metric": "n_male_individuals", "value": int((by_individual["sex"] == "Male").sum())},
        {"metric": "n_unknown_sex_individuals", "value": int((by_individual["sex"] == "Unknown").sum())},
        {"metric": "n_female_images", "value": int(sex_counts.get("Female", 0))},
        {"metric": "n_male_images", "value": int(sex_counts.get("Male", 0))},
        {"metric": "n_unknown_sex_images", "value": int(sex_counts.get("Unknown", 0))},
        {"metric": "n_camera_make_model_combinations", "value": int(camera_combo_count)},
        {"metric": "first_image_date", "value": df["image_date"].min().date().isoformat() if df["image_date"].notna().any() else None},
        {"metric": "last_image_date", "value": df["image_date"].max().date().isoformat() if df["image_date"].notna().any() else None},
        {
            "metric": "study_span_days",
            "value": int((df["image_date"].max() - df["image_date"].min()).days + 1) if df["image_date"].notna().any() else None,
        },
        {"metric": "n_unique_dates", "value": int(df["image_date"].nunique(dropna=True))},
        {"metric": "n_images_pre_2025", "value": int((df["image_date"] < pd.Timestamp("2025-01-01")).sum())},
        {"metric": "n_images_with_note_text", "value": int(df["metadata_note_text"].notna().sum())},
        {"metric": "n_images_with_note_tags", "value": int(df["note_tags"].notna().sum())},
        {"metric": "n_unique_note_tags", "value": int(note_tags["note_tag"].nunique()) if not note_tags.empty else 0},
        {"metric": "mean_images_per_individual", "value": round(float(by_individual["n_images"].mean()), 2) if not by_individual.empty else None},
        {"metric": "median_images_per_individual", "value": round(float(by_individual["n_images"].median()), 2) if not by_individual.empty else None},
        {"metric": "min_gps_lat", "value": float(df["gps_lat"].min()) if df["gps_lat"].notna().any() else None},
        {"metric": "max_gps_lat", "value": float(df["gps_lat"].max()) if df["gps_lat"].notna().any() else None},
        {"metric": "min_gps_lon", "value": float(df["gps_lon"].min()) if df["gps_lon"].notna().any() else None},
        {"metric": "max_gps_lon", "value": float(df["gps_lon"].max()) if df["gps_lon"].notna().any() else None},
    ]
    return pd.DataFrame(overview)


def main() -> None:
    df = _load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    note_tags = _build_note_tags(df)
    by_individual = _build_by_individual(df)
    by_sex = _build_by_sex(df, by_individual)
    overview = _build_overview(df, by_individual, note_tags)
    cameras = _build_camera_summary(df)

    overview.to_csv(OUT_OVERVIEW, index=False)
    by_individual.to_csv(OUT_BY_INDIVIDUAL, index=False)
    by_sex.to_csv(OUT_BY_SEX, index=False)
    note_tags.to_csv(OUT_NOTE_TAGS, index=False)
    cameras.to_csv(OUT_CAMERAS, index=False)

    print(f"Saved -> {OUT_OVERVIEW}")
    print(f"Saved -> {OUT_BY_INDIVIDUAL}")
    print(f"Saved -> {OUT_BY_SEX}")
    print(f"Saved -> {OUT_NOTE_TAGS}")
    print(f"Saved -> {OUT_CAMERAS}")
    print()
    print(f"Images: {len(df)}")
    print(f"Individuals: {by_individual['leopard_name'].nunique()}")
    if df["image_date"].notna().any():
        print(f"First image date: {df['image_date'].min().date().isoformat()}")
        print(f"Last image date: {df['image_date'].max().date().isoformat()}")
    print("Images by sex:")
    print(df['sex'].value_counts().to_string())


if __name__ == "__main__":
    main()
