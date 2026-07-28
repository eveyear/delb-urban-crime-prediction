from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq


LOG_TWO = math.log(2.0)
PI_SQUARED = math.pi**2


@dataclass(frozen=True)
class DiscreteGaussianStats:
    """Numerical quantities for the centered integer-lattice Gaussian."""

    lambda_value: float
    partition: float
    second_moment: float
    entropy_bits: float
    summation_method: str
    support_radius: int
    omitted_sum_bound: float


def _tail_bound(rate: float, radius: int) -> float:
    """Bound the two-sided unnormalized tail outside ``[-radius, radius]``."""

    first_index = radius + 1
    first_term = math.exp(-rate * first_index * first_index)
    ratio = math.exp(-rate * (2 * first_index + 1))
    return 2.0 * first_term / max(1.0 - ratio, np.finfo(float).tiny)


def _lattice_sums(
    rate: float, tolerance: float
) -> tuple[float, float, int, float]:
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("The lattice rate must be finite and strictly positive.")
    if not 0 < tolerance < 1:
        raise ValueError("The summation tolerance must lie in (0, 1).")

    radius = max(
        1,
        int(
            math.ceil(
                math.sqrt(max(0.0, -math.log(tolerance / 4.0)) / rate)
            )
        ),
    )
    while _tail_bound(rate, radius) > tolerance:
        radius += 1

    positive = np.arange(1, radius + 1, dtype=np.float64)
    weights = np.exp(-rate * positive * positive)
    partition = 1.0 + 2.0 * float(weights.sum())
    second_sum = 2.0 * float(np.dot(positive * positive, weights))
    return partition, second_sum, radius, _tail_bound(rate, radius)


def discrete_gaussian_stats(
    lambda_value: float, tail_tolerance: float = 1e-15
) -> DiscreteGaussianStats:
    r"""Evaluate \(\Theta(\lambda)\), \(D(\lambda)\), and \(H_\lambda\).

    Direct integer-lattice summation is used for moderate and large
    ``lambda``. For small ``lambda``, Poisson summation evaluates the dual
    theta series, preventing an unnecessarily wide real-lattice truncation.
    """

    if not math.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("lambda_value must be finite and strictly positive.")

    if lambda_value >= 1.0:
        partition, second_sum, radius, tail_bound = _lattice_sums(
            lambda_value, tail_tolerance
        )
        second_moment = second_sum / partition
        method = "direct_lattice"
    else:
        dual_rate = PI_SQUARED / lambda_value
        dual_partition, dual_second, radius, tail_bound = _lattice_sums(
            dual_rate, tail_tolerance
        )
        partition = math.sqrt(math.pi / lambda_value) * dual_partition
        second_moment = (
            1.0 / (2.0 * lambda_value)
            - PI_SQUARED
            * dual_second
            / (lambda_value * lambda_value * dual_partition)
        )
        # Roundoff can produce a tiny negative value only at an extreme limit.
        second_moment = max(0.0, second_moment)
        method = "poisson_dual"

    entropy_bits = (
        lambda_value * second_moment + math.log(partition)
    ) / LOG_TWO
    return DiscreteGaussianStats(
        lambda_value=lambda_value,
        partition=partition,
        second_moment=second_moment,
        entropy_bits=max(0.0, entropy_bits),
        summation_method=method,
        support_radius=radius,
        omitted_sum_bound=tail_bound,
    )


def inverse_entropy_envelope(
    entropy_bits: float,
    *,
    tail_tolerance: float = 1e-15,
    root_tolerance: float = 1e-12,
) -> float:
    r"""Return \(\mathcal H_{\mathbb Z}^{-1}(h)\) for ``h`` in bits."""

    if not math.isfinite(entropy_bits):
        raise ValueError("entropy_bits must be finite.")
    if entropy_bits <= 0:
        return 0.0

    def residual(lambda_value: float) -> float:
        return (
            discrete_gaussian_stats(lambda_value, tail_tolerance).entropy_bits
            - entropy_bits
        )

    unit_residual = residual(1.0)
    if abs(unit_residual) <= root_tolerance:
        return discrete_gaussian_stats(
            1.0, tail_tolerance
        ).second_moment
    lower = 1.0
    while residual(lower) < 0:
        lower /= 2.0
        if lower < np.finfo(float).tiny:
            raise RuntimeError("Unable to bracket the entropy inverse below.")

    upper = 1.0
    while residual(upper) > 0:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Unable to bracket the entropy inverse above.")

    lambda_solution = brentq(
        residual,
        lower,
        upper,
        xtol=root_tolerance,
        rtol=max(root_tolerance, 4 * np.finfo(float).eps),
        maxiter=200,
    )
    return discrete_gaussian_stats(
        lambda_solution, tail_tolerance
    ).second_moment


def entropy_envelope(
    second_moment: float,
    *,
    tail_tolerance: float = 1e-15,
    root_tolerance: float = 1e-12,
) -> float:
    r"""Return \(\mathcal H_{\mathbb Z}(D)\) for a raw second moment."""

    if not math.isfinite(second_moment):
        raise ValueError("second_moment must be finite.")
    if second_moment <= 0:
        return 0.0

    def residual(lambda_value: float) -> float:
        return (
            discrete_gaussian_stats(lambda_value, tail_tolerance).second_moment
            - second_moment
        )

    unit_residual = residual(1.0)
    if abs(unit_residual) <= root_tolerance:
        return discrete_gaussian_stats(1.0, tail_tolerance).entropy_bits
    lower = 1.0
    while residual(lower) < 0:
        lower /= 2.0
        if lower < np.finfo(float).tiny:
            raise RuntimeError("Unable to bracket the moment inverse below.")

    upper = 1.0
    while residual(upper) > 0:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Unable to bracket the moment inverse above.")

    lambda_solution = brentq(
        residual,
        lower,
        upper,
        xtol=root_tolerance,
        rtol=max(root_tolerance, 4 * np.finfo(float).eps),
        maxiter=200,
    )
    return discrete_gaussian_stats(
        lambda_solution, tail_tolerance
    ).entropy_bits


def lambda_for_second_moment(
    second_moment: float,
    *,
    tail_tolerance: float = 1e-15,
    root_tolerance: float = 1e-12,
) -> float:
    r"""Return the discrete-Gaussian rate having moment ``second_moment``."""

    if not math.isfinite(second_moment) or second_moment <= 0:
        raise ValueError(
            "second_moment must be finite and strictly positive."
        )

    def residual(lambda_value: float) -> float:
        return (
            discrete_gaussian_stats(lambda_value, tail_tolerance).second_moment
            - second_moment
        )

    if abs(residual(1.0)) <= root_tolerance:
        return 1.0
    lower = 1.0
    while residual(lower) < 0:
        lower /= 2.0
        if lower < np.finfo(float).tiny:
            raise RuntimeError("Unable to bracket the moment inverse below.")
    upper = 1.0
    while residual(upper) > 0:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Unable to bracket the moment inverse above.")
    return float(
        brentq(
            residual,
            lower,
            upper,
            xtol=root_tolerance,
            rtol=max(root_tolerance, 4 * np.finfo(float).eps),
            maxiter=200,
        )
    )


def closed_form_delb(entropy_bits: float) -> float:
    """Massey-type conservative lattice bound with the 1/12 correction."""

    if not math.isfinite(entropy_bits):
        raise ValueError("entropy_bits must be finite.")
    if entropy_bits <= 0:
        return 0.0
    return max(
        0.0,
        math.exp(2.0 * LOG_TWO * entropy_bits) / (2.0 * math.pi * math.e)
        - 1.0 / 12.0,
    )


def discrete_gaussian_pmf(
    lambda_value: float, tail_tolerance: float = 1e-15
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a normalized, symmetric finite representation of ``q_lambda``."""

    if not math.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("lambda_value must be finite and strictly positive.")
    radius = max(
        1,
        int(
            math.ceil(
                math.sqrt(
                    max(0.0, -math.log(tail_tolerance / 4.0))
                    / lambda_value
                )
            )
        ),
    )
    while _tail_bound(lambda_value, radius) > tail_tolerance:
        radius += 1
    support = np.arange(-radius, radius + 1, dtype=np.int64)
    weights = np.exp(-lambda_value * support.astype(float) ** 2)
    probabilities = weights / weights.sum()
    return support, probabilities, _tail_bound(lambda_value, radius)


def lambda_for_entropy(
    entropy_bits: float,
    *,
    tail_tolerance: float = 1e-15,
    root_tolerance: float = 1e-12,
) -> float:
    """Return the discrete-Gaussian rate associated with a positive entropy."""

    if not math.isfinite(entropy_bits) or entropy_bits <= 0:
        raise ValueError("entropy_bits must be finite and strictly positive.")

    def residual(lambda_value: float) -> float:
        return (
            discrete_gaussian_stats(lambda_value, tail_tolerance).entropy_bits
            - entropy_bits
        )

    if abs(residual(1.0)) <= root_tolerance:
        return 1.0
    lower = 1.0
    while residual(lower) < 0:
        lower /= 2.0
    upper = 1.0
    while residual(upper) > 0:
        upper *= 2.0
    return float(
        brentq(
            residual,
            lower,
            upper,
            xtol=root_tolerance,
            rtol=max(root_tolerance, 4 * np.finfo(float).eps),
            maxiter=200,
        )
    )


def shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in bits for a one-dimensional probability vector."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("probabilities must be finite and nonnegative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass.")
    normalized = values / total
    positive = normalized > 0
    return float(-np.dot(normalized[positive], np.log2(normalized[positive])))
