from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd


def category_lag(
    frame: pd.DataFrame,
    *,
    target: str,
    grid_field: str = "grid_id",
    date_field: str = "date",
) -> pd.Series:
    """Return the strict within-grid lag of a category-specific count target."""

    required = [target, grid_field, date_field]
    if frame[required].isna().any().any():
        raise ValueError("Category-lag inputs must be complete.")
    ordered = frame.sort_values([grid_field, date_field], kind="mergesort")
    lagged = ordered.groupby(grid_field, observed=True)[target].shift(1)
    result = pd.Series(index=ordered.index, data=lagged)
    return result.reindex(frame.index)


def normal_interval(
    point: float,
    bootstrap: Sequence[float],
    *,
    confidence_level: float,
    alternative: str,
) -> dict[str, float]:
    """Return a point-centered normal interval and bootstrap-based p-value."""

    values = np.asarray(bootstrap, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("At least two finite bootstrap values are required.")
    standard_error = float(np.std(values, ddof=1))
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    statistic = float(point / standard_error) if standard_error > 0 else np.inf
    if alternative == "greater":
        p_value = (
            1.0 - NormalDist().cdf(statistic)
            if standard_error > 0
            else (0.0 if point > 0 else 1.0)
        )
    elif alternative == "two-sided":
        p_value = (
            2.0 * (1.0 - NormalDist().cdf(abs(statistic)))
            if standard_error > 0
            else (0.0 if point != 0 else 1.0)
        )
    else:
        raise ValueError("alternative must be 'greater' or 'two-sided'.")
    alpha = 1.0 - confidence_level
    return {
        "bootstrap_standard_error": standard_error,
        "normal_ci_low": float(point - critical * standard_error),
        "normal_ci_high": float(point + critical * standard_error),
        "percentile_ci_low": float(np.quantile(values, alpha / 2.0)),
        "percentile_ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
        "z_statistic": statistic,
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
    }


def paired_category_difference(
    point_a: float,
    point_b: float,
    bootstrap_a: Sequence[float],
    bootstrap_b: Sequence[float],
    *,
    confidence_level: float,
) -> dict[str, float]:
    """Compare two category estimands using aligned bootstrap replicates."""

    first = np.asarray(bootstrap_a, dtype=float)
    second = np.asarray(bootstrap_b, dtype=float)
    if first.shape != second.shape:
        raise ValueError("Paired category bootstrap arrays must match.")
    difference = first - second
    point = float(point_a - point_b)
    return {
        "point_difference_a_minus_b": point,
        **normal_interval(
            point,
            difference,
            confidence_level=confidence_level,
            alternative="two-sided",
        ),
    }
