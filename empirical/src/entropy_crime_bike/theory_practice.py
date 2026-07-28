from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def calculate_gap_metrics(
    mse_baseline: float | np.ndarray,
    mse_bicycle: float | np.ndarray,
    bound_baseline: float | np.ndarray,
    bound_bicycle: float | np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Calculate the matched theory--practice quantities used by E07."""

    mse0 = np.asarray(mse_baseline, dtype=float)
    mseb = np.asarray(mse_bicycle, dtype=float)
    l0 = np.asarray(bound_baseline, dtype=float)
    lb = np.asarray(bound_bicycle, dtype=float)
    if np.any(mse0 <= 0) or np.any(mseb <= 0):
        raise ValueError("Empirical MSE must be strictly positive.")
    if np.any(l0 < 0) or np.any(lb < 0):
        raise ValueError("DELB values must be nonnegative.")
    delta_mse = mse0 - mseb
    delta_l = l0 - lb
    gap0 = mse0 - l0
    gapb = mseb - lb
    result: dict[str, float | np.ndarray] = {
        "gap_baseline": gap0,
        "gap_bicycle": gapb,
        "information_use_efficiency_baseline": l0 / mse0,
        "information_use_efficiency_bicycle": lb / mseb,
        "delta_mse_integer": delta_mse,
        "delta_l_exact_mse": delta_l,
        "gap_narrowing": gap0 - gapb,
        "realization_ratio_diagnostic": np.divide(
            delta_mse,
            delta_l,
            out=np.full(np.broadcast(delta_mse, delta_l).shape, np.nan),
            where=np.abs(delta_l) > np.finfo(float).eps,
        ),
    }
    if all(np.ndim(value) == 0 for value in result.values()):
        return {key: float(value) for key, value in result.items()}
    return result


def paired_grid_classification(
    delta_l: Sequence[float],
    delta_mse: Sequence[float],
) -> tuple[np.ndarray, float, float]:
    """Classify theory and practice using medians with ties assigned low."""

    theory = np.asarray(delta_l, dtype=float)
    empirical = np.asarray(delta_mse, dtype=float)
    if theory.shape != empirical.shape or theory.ndim != 1:
        raise ValueError("Classification arrays must be matching vectors.")
    if not np.isfinite(theory).all() or not np.isfinite(empirical).all():
        raise ValueError("Classification inputs must be finite.")
    theory_threshold = float(np.median(theory))
    empirical_threshold = float(np.median(empirical))
    high_theory = theory > theory_threshold
    high_empirical = empirical > empirical_threshold
    labels = np.select(
        [
            high_theory & high_empirical,
            high_theory & ~high_empirical,
            ~high_theory & high_empirical,
        ],
        [
            "high_theory__high_empirical",
            "high_theory__low_empirical",
            "low_theory__high_empirical",
        ],
        default="low_theory__low_empirical",
    )
    return labels.astype(object), theory_threshold, empirical_threshold


def spearman_with_spatial_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float | int]:
    """Spearman association with paired-grid percentile bootstrap limits."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape or x_values.ndim != 1:
        raise ValueError("Spearman inputs must be matching vectors.")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) < 3:
        raise ValueError("At least three finite paired grids are required.")
    observed = stats.spearmanr(x_values, y_values)
    rng = np.random.default_rng(seed)
    bootstrap = np.full(repetitions, np.nan, dtype=float)
    for replicate in range(repetitions):
        indices = rng.integers(0, len(x_values), size=len(x_values))
        if (
            np.unique(x_values[indices]).size > 1
            and np.unique(y_values[indices]).size > 1
        ):
            bootstrap[replicate] = stats.spearmanr(
                x_values[indices], y_values[indices]
            ).statistic
    valid = bootstrap[np.isfinite(bootstrap)]
    if len(valid) < max(100, int(0.9 * repetitions)):
        raise ValueError("Too few finite spatial bootstrap correlations.")
    alpha = 1.0 - confidence_level
    return {
        "grids": len(x_values),
        "spearman_rho": float(observed.statistic),
        "asymptotic_two_sided_p_value": float(observed.pvalue),
        "bootstrap_ci_low": float(np.quantile(valid, alpha / 2.0)),
        "bootstrap_ci_high": float(np.quantile(valid, 1.0 - alpha / 2.0)),
        "bootstrap_standard_error": float(np.std(valid, ddof=1)),
        "finite_bootstrap_replicates": len(valid),
    }


def ols_city_fixed_effect_hc3(
    frame: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    city_field: str = "city",
    confidence_level: float = 0.95,
) -> dict[str, float | int | str]:
    """Estimate an OLS slope with city fixed effects and HC3 covariance."""

    required = [outcome, exposure, city_field]
    if frame[required].isna().any().any():
        raise ValueError("Regression inputs must be complete.")
    cities = sorted(frame[city_field].astype(str).unique())
    if len(cities) < 2:
        raise ValueError("At least two cities are required for fixed effects.")
    y = frame[outcome].to_numpy(dtype=float)
    columns = [
        np.ones(len(frame), dtype=float),
        frame[exposure].to_numpy(dtype=float),
    ]
    column_names = ["intercept", exposure]
    for city in cities[1:]:
        columns.append(frame[city_field].astype(str).eq(city).to_numpy(dtype=float))
        column_names.append(f"city_{city}")
    x = np.column_stack(columns)
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inverse, x)
    adjusted = residual / np.maximum(1.0 - leverage, np.finfo(float).eps)
    meat = x.T @ ((adjusted * adjusted)[:, None] * x)
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    slope_index = column_names.index(exposure)
    slope = float(beta[slope_index])
    standard_error = float(standard_errors[slope_index])
    degrees_freedom = len(y) - np.linalg.matrix_rank(x)
    statistic = slope / standard_error if standard_error > 0 else np.nan
    p_value = (
        2.0 * stats.t.sf(abs(statistic), degrees_freedom)
        if np.isfinite(statistic) and degrees_freedom > 0
        else np.nan
    )
    critical = (
        stats.t.ppf(0.5 + confidence_level / 2.0, degrees_freedom)
        if degrees_freedom > 0
        else NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    )
    return {
        "observations": len(frame),
        "cities": len(cities),
        "reference_city": cities[0],
        "slope_delta_l": slope,
        "hc3_standard_error": standard_error,
        "t_statistic": float(statistic),
        "two_sided_p_value": float(p_value),
        "ci_low": slope - critical * standard_error,
        "ci_high": slope + critical * standard_error,
        "r_squared": float(
            1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)
        ),
        "residual_degrees_freedom": int(degrees_freedom),
    }
