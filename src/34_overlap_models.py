"""Two-part overlap models with node-label permutation inference.

Three problems with the submitted analysis are addressed here.

1. Zero-overlap dyads were excluded. They are the observations that speak most
   directly to territorial exclusivity, so Part 1 models whether a dyad overlaps at
   all, using every dyad.

2. Absolute overlap in km^2 is confounded with polygon size: larger polygons have
   more opportunity to intersect. Part 2 models `prop_overlap` (size-normalised) and
   both parts carry log mean polygon area as a covariate.

3. Dyadic observations are not independent - each leopard appears in up to n-1
   dyads, so Mann-Whitney and Kruskal-Wallis p-values are anticonservative.
   Inference here comes from permuting sex labels across *individuals* and
   rebuilding the dyad set, which preserves the dependence structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

OUT_TABLES = Path("outputs") / "tables"

PAIR_TYPES = ["Female-Female", "Female-Male", "Male-Male"]
CONTRASTS = [
    ("Female-Female", "Female-Male"),
    ("Female-Female", "Male-Male"),
    ("Female-Male", "Male-Male"),
]
N_PERMUTATIONS = 10_000
SEED = 20260821

WINDOWS = ["contemporary_24mo", "recent_12mo", "pooled"]


def _pair_label(sex_a: str, sex_b: str) -> str:
    return "-".join(sorted([sex_a, sex_b]))


def _design(pair_types: pd.Series, log_area: np.ndarray, include_pair_type: bool) -> np.ndarray:
    cols = [np.ones(len(pair_types)), log_area]
    if include_pair_type:
        # Female-Female is the reference level.
        for level in PAIR_TYPES[1:]:
            cols.append((pair_types == level).to_numpy(float))
    return np.column_stack(cols)


def _deviance(y: np.ndarray, X: np.ndarray, weights: np.ndarray | None = None) -> float:
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    try:
        return float(model.fit().deviance)
    except Exception:
        return float("nan")


def _lr_statistic(y: np.ndarray, pair_types: pd.Series, log_area: np.ndarray) -> float:
    """Likelihood-ratio statistic for the pair-type term, over and above polygon size."""
    dev_null = _deviance(y, _design(pair_types, log_area, include_pair_type=False))
    dev_full = _deviance(y, _design(pair_types, log_area, include_pair_type=True))
    return dev_null - dev_full


def _group_stat(values: np.ndarray, pair_types: pd.Series, label: str, how: str) -> float:
    sel = (pair_types == label).to_numpy()
    if not sel.any():
        return float("nan")
    return float(np.mean(values[sel]) if how == "mean" else np.median(values[sel]))


def _permuted_pair_types(
    individuals: list[str],
    sexes: np.ndarray,
    dyad_a: np.ndarray,
    dyad_b: np.ndarray,
    rng: np.random.Generator,
) -> pd.Series:
    """Relabel sex across individuals, then rebuild every dyad's pair type."""
    shuffled = rng.permutation(sexes)
    lookup = dict(zip(individuals, shuffled))
    return pd.Series([_pair_label(lookup[a], lookup[b]) for a, b in zip(dyad_a, dyad_b)])


def run_part(
    pairs: pd.DataFrame,
    response: str,
    how: str,
    part_label: str,
    window: str,
) -> list[dict[str, object]]:
    y = pairs[response].to_numpy(float)
    log_area = np.log(pairs["mean_polygon_area_km2"].to_numpy(float))
    pair_types = pairs["pair_sex"].reset_index(drop=True)

    dyad_a = pairs["leopard_a"].to_numpy()
    dyad_b = pairs["leopard_b"].to_numpy()
    individual_sex = (
        pd.concat(
            [
                pairs[["leopard_a", "sex_a"]].rename(columns={"leopard_a": "name", "sex_a": "sex"}),
                pairs[["leopard_b", "sex_b"]].rename(columns={"leopard_b": "name", "sex_b": "sex"}),
            ]
        )
        .drop_duplicates()
        .sort_values("name")
    )
    individuals = individual_sex["name"].tolist()
    sexes = individual_sex["sex"].to_numpy()

    obs_lr = _lr_statistic(y, pair_types, log_area)
    obs_contrasts = {
        (a, b): _group_stat(y, pair_types, a, how) - _group_stat(y, pair_types, b, how)
        for a, b in CONTRASTS
    }

    rng = np.random.default_rng(SEED)
    lr_ge = 0
    contrast_ge = {key: 0 for key in obs_contrasts}
    n_valid = 0

    for _ in range(N_PERMUTATIONS):
        perm_types = _permuted_pair_types(individuals, sexes, dyad_a, dyad_b, rng)
        lr = _lr_statistic(y, perm_types, log_area)
        if np.isnan(lr):
            continue
        n_valid += 1
        if lr >= obs_lr - 1e-12:
            lr_ge += 1
        for (a, b), observed in obs_contrasts.items():
            diff = _group_stat(y, perm_types, a, how) - _group_stat(y, perm_types, b, how)
            if np.isnan(diff):
                continue
            if abs(diff) >= abs(observed) - 1e-12:
                contrast_ge[(a, b)] += 1

    # Add-one correction: a permutation p-value is never exactly zero.
    rows: list[dict[str, object]] = [
        {
            "window": window,
            "part": part_label,
            "test": "omnibus_pair_type_LR",
            "group_a": None,
            "group_b": None,
            "n_dyads": int(len(pairs)),
            "statistic": obs_lr,
            "p_permutation": (lr_ge + 1) / (n_valid + 1),
            "p_permutation_adjusted": (lr_ge + 1) / (n_valid + 1),
            "n_permutations": n_valid,
        }
    ]

    raw_p = []
    contrast_rows: list[dict[str, object]] = []
    for (a, b), observed in obs_contrasts.items():
        p = (contrast_ge[(a, b)] + 1) / (n_valid + 1)
        raw_p.append(p)
        contrast_rows.append(
            {
                "window": window,
                "part": part_label,
                "test": f"contrast_{how}_difference",
                "group_a": a,
                "group_b": b,
                "n_a": int((pair_types == a).sum()),
                "n_b": int((pair_types == b).sum()),
                "stat_a": _group_stat(y, pair_types, a, how),
                "stat_b": _group_stat(y, pair_types, b, how),
                "statistic": observed,
                "p_permutation": p,
                "n_permutations": n_valid,
            }
        )
    adjusted = multipletests(raw_p, method="holm")[1]
    for row, p_adj in zip(contrast_rows, adjusted):
        row["p_permutation_adjusted"] = float(p_adj)
    rows.extend(contrast_rows)
    return rows


def fit_reported_glm(pairs: pd.DataFrame, response: str, window: str, part_label: str) -> pd.DataFrame:
    """Coefficient table for the full model, for reporting effect sizes."""
    y = pairs[response].to_numpy(float)
    log_area = np.log(pairs["mean_polygon_area_km2"].to_numpy(float))
    pair_types = pairs["pair_sex"].reset_index(drop=True)
    X = _design(pair_types, log_area, include_pair_type=True)
    names = ["intercept", "log_mean_polygon_area_km2"] + [f"pair_type[{lvl}]" for lvl in PAIR_TYPES[1:]]

    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return pd.DataFrame(
        {
            "window": window,
            "part": part_label,
            "term": names,
            "coefficient": fit.params,
            "std_error": fit.bse,
            "odds_ratio": np.exp(fit.params),
            # Model-based CIs are shown for scale only; inference uses permutation.
            "or_ci_low": np.exp(fit.conf_int()[:, 0]),
            "or_ci_high": np.exp(fit.conf_int()[:, 1]),
        }
    )


def descriptive_table(pairs: pd.DataFrame, window: str) -> pd.DataFrame:
    grouped = pairs.groupby("pair_sex")
    out = grouped.agg(
        n_dyads=("overlaps", "size"),
        n_overlapping=("overlaps", "sum"),
        median_overlap_km2_nonzero=("overlap_km2", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else np.nan),
        median_prop_overlap_nonzero=("prop_overlap", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else np.nan),
        mean_polygon_area_km2=("mean_polygon_area_km2", "mean"),
        median_cooccurrence_days=("temporal_cooccurrence_days", "median"),
    ).reset_index()
    out.insert(0, "window", window)
    out["pct_overlapping"] = 100.0 * out["n_overlapping"] / out["n_dyads"]
    # The manuscript quotes overlap extent as a percentage, so carry that form too:
    # manuscript_numbers.md must literally contain every number the text reports.
    out["median_pct_overlap_nonzero"] = 100.0 * out["median_prop_overlap_nonzero"]
    return out


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []
    all_coefs: list[pd.DataFrame] = []
    all_desc: list[pd.DataFrame] = []

    for window in WINDOWS:
        path = OUT_TABLES / f"merged_image_overlap_pairs__{window}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find: {path.resolve()}. Run src/33_temporal_windows.py first.")
        pairs = pd.read_csv(path)
        pairs = pairs[pairs["pair_sex"].isin(PAIR_TYPES)].reset_index(drop=True)

        all_desc.append(descriptive_table(pairs, window))

        # Part 1: does the dyad overlap at all? Every dyad, zeros included.
        all_results.extend(run_part(pairs, "overlaps", "mean", "part1_any_overlap", window))
        all_coefs.append(fit_reported_glm(pairs, "overlaps", window, "part1_any_overlap"))

        # Part 2: given overlap, how much? Size-normalised response.
        nonzero = pairs[pairs["overlaps"] == 1].reset_index(drop=True)
        if len(nonzero) >= 10:
            all_results.extend(run_part(nonzero, "prop_overlap", "median", "part2_extent_given_overlap", window))
            all_coefs.append(fit_reported_glm(nonzero, "prop_overlap", window, "part2_extent_given_overlap"))

        print(f"Modelled window: {window} ({len(pairs)} dyads, {int(pairs['overlaps'].sum())} overlapping)")

    results = pd.DataFrame(all_results)
    coefs = pd.concat(all_coefs, ignore_index=True)
    desc = pd.concat(all_desc, ignore_index=True)

    results_path = OUT_TABLES / "overlap_model_results.csv"
    coefs_path = OUT_TABLES / "overlap_model_coefficients.csv"
    desc_path = OUT_TABLES / "overlap_descriptives_by_pair_type.csv"

    results.to_csv(results_path, index=False)
    coefs.to_csv(coefs_path, index=False)
    desc.to_csv(desc_path, index=False)

    for path in (results_path, coefs_path, desc_path):
        print(f"Saved -> {path}")

    pd.set_option("display.width", 220)
    print()
    print(desc.to_string(index=False))
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
