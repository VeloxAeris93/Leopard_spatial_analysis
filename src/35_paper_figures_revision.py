"""Revised paper figures.

Supersedes src/22_paper_figures_spatial_subset.py. Three things changed materially:

* Figures are built from 95% MCPs estimated within a contemporary time window, so
  every polygon on a figure comes from the same period.
* Figure 5 no longer hides zero-overlap dyads. Panel A reports how often dyads
  overlap at all - the quantity that speaks to territorial exclusivity - and panel B
  reports how much they overlap when they do, as a proportion of polygon area rather
  than in km^2.
* No titles and no colour. Titles are set in the manuscript figure captions, not on
  the figure itself, so nothing here calls set_title/suptitle - descriptive text and
  statistics that would have lived in a title are placed as an axis label below the
  plot instead, which guarantees they cannot overlap the data (a title-region
  annotation did, over the female outlier point, in the previous version of Figure 4).
  Colour is replaced by a greyscale (src/_plot_style.py) plus marker shape, so
  sex/pair-type stays distinguishable in black-and-white print. Groups are encoded by
  shade and marker only, not hatching - the earlier crosshatched boxes/bars read as
  visual clutter next to jittered points and were dropped.

Figure 2 (seasonal detections by sex) is also built here even though it summarises the
full dataset rather than the spatial-analysis subset: no script previously generated
it, so the submitted manuscript's version was a static image quoting the pre-revision
dry-season count (404). The current archive (extended to 25 May 2026) gives 446.

Inference annotated on the figures is the node-label permutation p-value from
src/34_overlap_models.py, not the rank tests used in the submitted version.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from _plot_style import PAIR_GREY, PAIR_MARKER, PAIR_ORDER, SEX3_GREY, SEX3_ORDER, SEX_GREY, SEX_MARKER, set_pub_style

OUT_TABLES = Path("outputs") / "tables"
OUT_FIGS = Path("outputs") / "figures"

PRIMARY_WINDOW = "contemporary_24mo"


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        path = OUT_FIGS / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"Saved -> {path}")
    plt.close(fig)


def _read(name: str) -> pd.DataFrame:
    path = OUT_TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Cannot find: {path.resolve()}")
    return pd.read_csv(path)


def _perm_p(models: pd.DataFrame, part: str) -> float | None:
    row = models[
        (models["window"] == PRIMARY_WINDOW)
        & (models["part"] == part)
        & (models["test"] == "omnibus_pair_type_LR")
    ]
    return float(row["p_permutation"].iloc[0]) if len(row) else None


def figure2_seasonal_by_sex(season_sex: pd.DataFrame) -> None:
    season_order = ["Wet season", "Dry season"]
    pivot = (
        season_sex.pivot(index="botswana_season", columns="sex", values="n_records")
        .reindex(season_order)
        .reindex(columns=SEX3_ORDER)
        .fillna(0)
    )
    totals = pivot.sum(axis=1)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    bottom = np.zeros(len(pivot))
    for sex in SEX3_ORDER:
        ax.bar(
            pivot.index,
            pivot[sex].to_numpy(),
            bottom=bottom,
            color=SEX3_GREY[sex],
            edgecolor="black",
            linewidth=0.7,
            width=0.62,
            label=sex,
        )
        bottom += pivot[sex].to_numpy()

    for i, total in enumerate(totals):
        ax.text(i, total + totals.max() * 0.015, f"{int(total)}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Detections")
    ax.set_ylim(0, totals.max() * 1.1)
    # Legend sits below the axes, not inside the plot area, so it cannot collide with
    # the tallest bar regardless of how the season totals compare (see Figure 4).
    ax.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, handlelength=1.4, columnspacing=1.2
    )
    fig.tight_layout()
    _save(fig, "fig2_seasonal_detections_by_sex")


def figure4_mcp_by_sex(individuals: pd.DataFrame, sex_stats: pd.DataFrame) -> None:
    kept = individuals[individuals["include"]]
    row = sex_stats[sex_stats["window"] == PRIMARY_WINDOW].iloc[0]
    female = kept.loc[kept["sex"] == "Female", "mcp95_km2"].to_numpy(float)
    male = kept.loc[kept["sex"] == "Male", "mcp95_km2"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    bp = ax.boxplot(
        [female, male],
        tick_labels=[f"Female\n(n={len(female)})", f"Male\n(n={len(male)})"],
        showfliers=False,
        patch_artist=True,
    )
    for patch, sex in zip(bp["boxes"], ["Female", "Male"]):
        patch.set(facecolor="white", edgecolor=SEX_GREY[sex], linewidth=1.1)
    for median, sex in zip(bp["medians"], ["Female", "Male"]):
        median.set(color=SEX_GREY[sex], linewidth=1.3)

    rng = np.random.default_rng(42)
    for i, (arr, sex) in enumerate([(female, "Female"), (male, "Male")], start=1):
        jitter = rng.normal(0.0, 0.045, size=len(arr))
        ax.plot(
            np.full_like(arr, i, dtype=float) + jitter,
            arr,
            marker=SEX_MARKER[sex],
            linestyle="none",
            markersize=4.2,
            markerfacecolor=SEX_GREY[sex],
            markeredgecolor="black",
            markeredgewidth=0.4,
            alpha=0.9,
        )

    ax.set_ylabel("95% MCP area (km$^2$)")
    # Statistics live below the plot, not inside a title region, so they cannot
    # overlap a data point regardless of where the outliers fall.
    ax.set_xlabel(
        f"Mann-Whitney U = {row['mann_whitney_u']:.0f}, p = {row['p_value']:.3f}   |   "
        f"Cliff's delta = {row['cliffs_delta_male_minus_female']:.3f}\n"
        f"Median difference (male - female) = {row['median_diff_male_minus_female_km2']:.2f} km$^2$ "
        f"(95% CI [{row['median_diff_ci95_low_km2']:.2f}, {row['median_diff_ci95_high_km2']:.2f}])",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, "fig4_mcp95_by_sex_contemporary")


def figure5_overlap(pairs: pd.DataFrame, models: pd.DataFrame) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 5.2))

    # Panel A - propensity to overlap at all, every dyad included.
    counts = pairs.groupby("pair_sex")["overlaps"].agg(["sum", "size"]).reindex(PAIR_ORDER)
    props = counts["sum"] / counts["size"]
    lo, hi = proportion_confint(counts["sum"], counts["size"], method="wilson")
    err = np.vstack([props - lo, hi - props])

    bars = ax_a.bar(
        range(len(PAIR_ORDER)),
        props * 100.0,
        yerr=err * 100.0,
        color=[PAIR_GREY[p] for p in PAIR_ORDER],
        edgecolor="black",
        linewidth=0.8,
        width=0.62,
        capsize=4,
        error_kw={"linewidth": 0.9},
    )
    ax_a.set_xticks(range(len(PAIR_ORDER)))
    ax_a.set_xticklabels([p.replace("-", "–") for p in PAIR_ORDER])
    ax_a.set_ylabel("Dyads that overlap at all (%)")
    ax_a.set_ylim(0, 100)
    for i, pair in enumerate(PAIR_ORDER):
        ax_a.text(
            i,
            props.iloc[i] * 100.0 + (hi.iloc[i] - props.iloc[i]) * 100.0 + 3,
            f"{int(counts['sum'].iloc[i])}/{int(counts['size'].iloc[i])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    p1 = _perm_p(models, "part1_any_overlap")
    ax_a.set_xlabel(
        "A. Propensity to overlap" + (f"\npermutation p = {p1:.3f}" if p1 is not None else ""), fontsize=8.5
    )

    # Panel B - extent of overlap among dyads that do overlap.
    nonzero = pairs[pairs["overlaps"] == 1]
    data = [nonzero.loc[nonzero["pair_sex"] == p, "prop_overlap"].to_numpy(float) * 100.0 for p in PAIR_ORDER]
    bp = ax_b.boxplot(
        data,
        tick_labels=[f"{p.replace('-', chr(8211))}\n(n={len(d)})" for p, d in zip(PAIR_ORDER, data)],
        showfliers=False,
        patch_artist=True,
    )
    for patch, pair in zip(bp["boxes"], PAIR_ORDER):
        patch.set(facecolor="white", edgecolor=PAIR_GREY[pair], linewidth=1.1)
    for median, pair in zip(bp["medians"], PAIR_ORDER):
        median.set(color=PAIR_GREY[pair], linewidth=1.3)

    rng = np.random.default_rng(123)
    for i, (arr, pair) in enumerate(zip(data, PAIR_ORDER), start=1):
        jitter = rng.normal(0.0, 0.05, size=len(arr))
        ax_b.plot(
            np.full_like(arr, i, dtype=float) + jitter,
            arr,
            marker=PAIR_MARKER[pair],
            linestyle="none",
            markersize=3.8,
            markerfacecolor=PAIR_GREY[pair],
            markeredgecolor="black",
            markeredgewidth=0.3,
            alpha=0.85,
        )
    ax_b.set_ylabel("Shared area (% of mean polygon area)")
    p2 = _perm_p(models, "part2_extent_given_overlap")
    ax_b.set_xlabel(
        "B. Extent, given overlap" + (f"\npermutation p = {p2:.3f}" if p2 is not None else ""), fontsize=8.5
    )

    fig.tight_layout()
    _save(fig, "fig5_overlap_by_pair_type_revised")


def figure_s1_window_concordance(contemporary: pd.DataFrame, pooled: pd.DataFrame) -> None:
    def key(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["dyad"] = [
            "|".join(sorted([a, b])) for a, b in zip(out["leopard_a"], out["leopard_b"])
        ]
        return out[["dyad", "pair_sex", "prop_overlap"]]

    merged = key(contemporary).merge(key(pooled), on="dyad", suffixes=("_window", "_pooled"))
    rho = merged["prop_overlap_window"].corr(merged["prop_overlap_pooled"], method="spearman")

    fig, ax = plt.subplots(figsize=(5.8, 5.9))
    for pair in PAIR_ORDER:
        sub = merged[merged["pair_sex_window"] == pair]
        ax.plot(
            sub["prop_overlap_pooled"] * 100.0,
            sub["prop_overlap_window"] * 100.0,
            marker=PAIR_MARKER[pair],
            linestyle="none",
            markersize=4.4,
            markerfacecolor=PAIR_GREY[pair],
            markeredgecolor="black",
            markeredgewidth=0.3,
            alpha=0.85,
            label=pair.replace("-", "–"),
        )
    limit = max(merged[["prop_overlap_window", "prop_overlap_pooled"]].max()) * 100.0 * 1.08
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=0.8, color="black")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_ylabel("Proportional overlap, contemporary window (%)")
    ax.set_xlabel(
        f"Proportional overlap, pooled 2014-2026 (%)\n{len(merged)} shared dyads, Spearman rho = {rho:.2f}",
        fontsize=8.5,
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "figS1_window_concordance")


def figure_s2_mcp95_vs_mcp100(summary: pd.DataFrame) -> None:
    df = summary.dropna(subset=["mcp95_km2", "mcp100_km2"]).sort_values("mcp100_km2")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.hlines(y, df["mcp95_km2"], df["mcp100_km2"], color="#999999", linewidth=1.2, zorder=1)
    ax.scatter(
        df["mcp100_km2"], y, s=32, facecolor="white", edgecolor="black", linewidth=1.0,
        marker="o", label="100% MCP", zorder=2,
    )
    ax.scatter(
        df["mcp95_km2"], y, s=32, facecolor="black", edgecolor="black", linewidth=0.6,
        marker="o", label="95% MCP", zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{name}  (-{int(n)})" for name, n in zip(df["leopard_name"], df["n_points_dropped"])]
    )
    ax.set_xlabel(
        "MCP area (km$^2$)\nLabel shows locations removed by 95% peripheral removal; pooled 2014-2026 subset",
        fontsize=8.5,
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "figS2_mcp95_vs_mcp100")


def main() -> None:
    set_pub_style()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    individuals = _read(f"merged_image_window_individuals__{PRIMARY_WINDOW}.csv")
    sex_stats = _read("merged_image_window_sex_stats.csv")
    contemporary = _read(f"merged_image_overlap_pairs__{PRIMARY_WINDOW}.csv")
    pooled = _read("merged_image_overlap_pairs__pooled.csv")
    models = _read("overlap_model_results.csv")
    summary = _read("merged_image_home_ranges_mcp_summary.csv")
    season_sex = _read("merged_image_counts_by_botswana_season_and_sex.csv")

    figure2_seasonal_by_sex(season_sex)
    figure4_mcp_by_sex(individuals, sex_stats)
    figure5_overlap(contemporary, models)
    figure_s1_window_concordance(contemporary, pooled)
    figure_s2_mcp95_vs_mcp100(summary)


if __name__ == "__main__":
    main()
