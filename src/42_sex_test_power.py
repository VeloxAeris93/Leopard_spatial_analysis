"""Smallest sex difference in range size the Mann-Whitney test could detect.

Both headline results are null, so the Discussion has to say how large a difference
the sex comparison could have found. Power is estimated by simulation: female and
male values are drawn from unit normals separated by a location shift, and the shift
is increased until a two-sided Mann-Whitney test at alpha = 0.05 rejects in at least
80% of simulations. The shift is reported as the equivalent Cliff's delta,
2 * Phi(shift / sqrt(2)) - 1, so it is on the same scale as the observed effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

TABLES = Path("outputs") / "tables"
IN_PATH = TABLES / "merged_image_window_sex_stats.csv"
OUT_PATH = TABLES / "sex_test_power.csv"

ALPHA = 0.05
TARGET_POWER = 0.80
N_SIMULATIONS = 5_000
SHIFTS = np.round(np.arange(0.1, 3.01, 0.05), 2)
SEED = 20261005


def power(n_female: int, n_male: int, shift: float, rng: np.random.Generator) -> float:
    female = rng.normal(0.0, 1.0, (N_SIMULATIONS, n_female))
    male = rng.normal(shift, 1.0, (N_SIMULATIONS, n_male))
    p = mannwhitneyu(female, male, axis=1, alternative="two-sided").pvalue
    return float((p < ALPHA).mean())


def main() -> None:
    stats = pd.read_csv(IN_PATH)
    rng = np.random.default_rng(SEED)
    rows = []
    for row in stats.itertuples():
        n_f, n_m = int(row.n_female), int(row.n_male)
        detectable = next((s for s in SHIFTS if power(n_f, n_m, s, rng) >= TARGET_POWER), np.nan)
        rows.append(
            {
                "window": row.window,
                "n_female": n_f,
                "n_male": n_m,
                "alpha": ALPHA,
                "target_power": TARGET_POWER,
                "detectable_shift_sd": detectable,
                "detectable_cliffs_delta": 2 * norm.cdf(detectable / np.sqrt(2)) - 1,
                "observed_cliffs_delta": row.cliffs_delta_male_minus_female,
                "n_simulations": N_SIMULATIONS,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(out.round(3).to_string(index=False))
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
