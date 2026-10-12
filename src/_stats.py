"""Small statistical helpers shared across the spatial-analysis scripts."""

from __future__ import annotations

import numpy as np


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Non-parametric effect size: P(x > y) - P(x < y)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(1, -1)
    return float((x > y).mean() - (x < y).mean())


def bootstrap_median_diff(
    x: np.ndarray, y: np.ndarray, n: int = 20_000, seed: int = 42
) -> tuple[float, float, float]:
    """Median difference (x - y) with a percentile bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan"), float("nan")

    xb = rng.choice(x, size=(n, len(x)), replace=True)
    yb = rng.choice(y, size=(n, len(y)), replace=True)
    diffs = np.median(xb, axis=1) - np.median(yb, axis=1)
    return (
        float(np.median(x) - np.median(y)),
        float(np.quantile(diffs, 0.025)),
        float(np.quantile(diffs, 0.975)),
    )
