"""Inclusion criteria for the spatial analyses, defined once.

These thresholds were previously hardcoded in seven places (scripts 21, 22, 23, 23b,
32, 33 and 36). Changing one without the others produced supplementary tables that
disagreed with the figures - exactly the class of inconsistency this revision exists
to remove. Import from here instead.

A note on the interaction with the 95% MCP: peripheral removal drops
ceil(0.05 * n) locations, which is one location for any individual with 20 or fewer.
At the minimum of 8 unique locations that single removal is 12.5% of that
individual's data, so the threshold and the estimator have to be reported together.
"""

from __future__ import annotations

MIN_UNIQUE_POINTS = 8
MIN_UNIQUE_DATES = 3

# The criterion used in the original submission, retained so that
# src/37_reviewer_decomposition.py can reproduce the submitted analysis exactly.
SUBMITTED_MIN_UNIQUE_POINTS = 10
