from __future__ import annotations

from pathlib import Path

import pandas as pd
from _age import annotate
from _inclusion import MIN_UNIQUE_DATES, MIN_UNIQUE_POINTS


IN_PATH = Path("outputs") / "tables" / "merged_leopard_images_metadata.dated_events.csv"
OUT_TABLES = Path("outputs") / "tables"

OUT_SUBSET = OUT_TABLES / "merged_leopard_images_metadata.spatial_subset.csv"
OUT_SUMMARY = OUT_TABLES / "merged_leopard_images_metadata.spatial_subset_individuals.csv"
OUT_RULES = OUT_TABLES / "merged_leopard_images_metadata.spatial_subset_rules.csv"

OUT_DATE_AUDIT = OUT_TABLES / "date_audit_rejected.csv"


# First photograph in the archive; nothing can predate it.
STUDY_START = "2014-08-10"


def _resolve_sex(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna() if str(v) != "Unknown"]
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return "Unknown"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Cannot find: {IN_PATH.resolve()}")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_PATH)
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).fillna("Unknown")
    df["gps_lat"] = pd.to_numeric(df.get("gps_lat"), errors="coerce")
    df["gps_lon"] = pd.to_numeric(df.get("gps_lon"), errors="coerce")
    df["image_date_parsed"] = pd.to_datetime(df.get("image_date"), errors="coerce", dayfirst=True)
    df = df.dropna(subset=["leopard_name", "gps_lat", "gps_lon", "image_date_parsed"]).copy()

    # Detection dates outside the study period indicate a camera clock error or a
    # day/month transposition, not a real encounter. Reject them here rather than
    # letting them reach the supplementary tables.
    impossible = (df["image_date_parsed"] > pd.Timestamp.today().normalize()) | (
        df["image_date_parsed"] < pd.Timestamp(STUDY_START)
    )
    if impossible.any():
        rejected = df.loc[impossible, ["leopard_name", "image_date", "image_datetime_source", "file_name"]].copy()
        rejected["reason"] = "date outside study period"
        rejected.to_csv(OUT_DATE_AUDIT, index=False)
        print(f"WARNING: rejected {int(impossible.sum())} record(s) with impossible dates -> {OUT_DATE_AUDIT}")
    else:
        pd.DataFrame(columns=["leopard_name", "image_date", "image_datetime_source", "file_name", "reason"]).to_csv(
            OUT_DATE_AUDIT, index=False
        )
    df = df.loc[~impossible].copy()

    df["sex"] = df.groupby("leopard_name")["sex"].transform(_resolve_sex)

    individual_summary = (
        df.groupby(["leopard_name", "sex"], dropna=False)
        .agg(
            n_images=("file_name", "size"),
            n_unique_dates=("image_date_parsed", "nunique"),
            n_unique_points=("gps_lat", lambda s: None),
            first_seen=("image_date_parsed", "min"),
            last_seen=("image_date_parsed", "max"),
        )
        .reset_index()
    )

    unique_points = (
        df.groupby("leopard_name")
        .apply(lambda g: int(len(set(zip(g["gps_lat"].astype(float), g["gps_lon"].astype(float))))))
        .rename("n_unique_points")
        .reset_index()
    )
    individual_summary = individual_summary.drop(columns=["n_unique_points"]).merge(unique_points, on="leopard_name", how="left")

    # Cubs and subadults have not established territories, so they cannot inform a
    # question about territorial exclusivity.
    individual_summary = annotate(individual_summary)
    individual_summary["include_spatial_subset"] = (
        (individual_summary["n_unique_points"] >= MIN_UNIQUE_POINTS)
        & (individual_summary["n_unique_dates"] >= MIN_UNIQUE_DATES)
        & individual_summary["is_territorial_age"]
    )
    individual_summary["first_seen"] = individual_summary["first_seen"].dt.date.astype("string")
    individual_summary["last_seen"] = individual_summary["last_seen"].dt.date.astype("string")

    keep_names = set(individual_summary.loc[individual_summary["include_spatial_subset"], "leopard_name"])
    subset = df[df["leopard_name"].isin(keep_names)].copy()
    subset = subset.drop(columns=["image_date_parsed"])

    rules = pd.DataFrame(
        [
            {"rule": "min_unique_points", "value": MIN_UNIQUE_POINTS},
            {"rule": "min_unique_dates", "value": MIN_UNIQUE_DATES},
            {"rule": "excluded_age_classes", "value": "cub; subadult"},
            {
                "rule": "n_excluded_non_territorial_age",
                "value": int(
                    (
                        (individual_summary["n_unique_points"] >= MIN_UNIQUE_POINTS)
                        & (individual_summary["n_unique_dates"] >= MIN_UNIQUE_DATES)
                        & ~individual_summary["is_territorial_age"]
                    ).sum()
                ),
            },
            {"rule": "n_rows_retained", "value": int(len(subset))},
            {"rule": "n_individuals_retained", "value": int(len(keep_names))},
            {
                "rule": "n_female_individuals_retained",
                "value": int(individual_summary.query("include_spatial_subset and sex == 'Female'").shape[0]),
            },
            {
                "rule": "n_male_individuals_retained",
                "value": int(individual_summary.query("include_spatial_subset and sex == 'Male'").shape[0]),
            },
            {
                "rule": "n_unknown_individuals_retained",
                "value": int(individual_summary.query("include_spatial_subset and sex == 'Unknown'").shape[0]),
            },
        ]
    )

    subset.to_csv(OUT_SUBSET, index=False)
    individual_summary.to_csv(OUT_SUMMARY, index=False)
    rules.to_csv(OUT_RULES, index=False)

    print(f"Saved -> {OUT_SUBSET}")
    print(f"Saved -> {OUT_SUMMARY}")
    print(f"Saved -> {OUT_RULES}")


if __name__ == "__main__":
    main()
