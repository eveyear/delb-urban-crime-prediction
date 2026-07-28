from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from entropy_crime_bike.discrete_bound import inverse_entropy_envelope


def maximum_true_run(values: Sequence[bool]) -> int:
    """Return the longest consecutive run of true values."""

    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini--Hochberg adjusted p-values, preserving NaNs."""

    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    selected = np.clip(values[finite], 0.0, 1.0)
    order = np.argsort(selected, kind="mergesort")
    ranked = selected[order]
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(count, dtype=float)
    restored[order] = adjusted
    result[finite] = restored
    return result


@dataclass(frozen=True)
class BoundInterpolator:
    """Monotone interpolation of log(1 + exact DELB) for bootstrap arrays."""

    entropy_grid: np.ndarray
    exact_bound_grid: np.ndarray
    interpolator: PchipInterpolator

    @classmethod
    def build(
        cls,
        maximum_entropy_bits: float,
        *,
        tail_tolerance: float,
        root_tolerance: float,
    ) -> "BoundInterpolator":
        if maximum_entropy_bits < 4.0:
            raise ValueError("The DELB interpolation domain must reach 4 bits.")
        entropy_grid = np.unique(
            np.concatenate(
                [
                    np.arange(0.0, 0.1000001, 0.001),
                    np.arange(0.105, 4.0001, 0.005),
                    np.arange(4.01, maximum_entropy_bits + 0.0001, 0.01),
                ]
            )
        )
        exact = np.asarray(
            [
                inverse_entropy_envelope(
                    float(value),
                    tail_tolerance=tail_tolerance,
                    root_tolerance=root_tolerance,
                )
                for value in entropy_grid
            ],
            dtype=float,
        )
        interpolator = PchipInterpolator(
            entropy_grid, np.log1p(exact), extrapolate=False
        )
        return cls(entropy_grid, exact, interpolator)

    def transform(self, entropy_bits: Sequence[float]) -> np.ndarray:
        values = np.asarray(entropy_bits, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Entropy values for DELB interpolation must be finite.")
        clipped = np.maximum(values, 0.0)
        if np.any(clipped > self.entropy_grid[-1]):
            raise ValueError("Entropy exceeds the validated DELB interpolation domain.")
        return np.expm1(self.interpolator(clipped))

    def midpoint_validation(
        self,
        *,
        tail_tolerance: float,
        root_tolerance: float,
    ) -> pd.DataFrame:
        midpoints = (
            self.entropy_grid[:-1] + self.entropy_grid[1:]
        ) / 2.0
        exact = np.asarray(
            [
                inverse_entropy_envelope(
                    float(value),
                    tail_tolerance=tail_tolerance,
                    root_tolerance=root_tolerance,
                )
                for value in midpoints
            ]
        )
        interpolated = self.transform(midpoints)
        absolute_error = np.abs(exact - interpolated)
        return pd.DataFrame(
            {
                "entropy_bits": midpoints,
                "exact_delb_mse": exact,
                "interpolated_delb_mse": interpolated,
                "absolute_error_mse": absolute_error,
                "relative_error": absolute_error
                / np.maximum(exact, np.finfo(float).tiny),
            }
        )


def queen_neighbor_indices(
    x_index: Sequence[int], y_index: Sequence[int]
) -> list[np.ndarray]:
    """Return queen-contiguous neighbor indices for integer grid coordinates."""

    x_values = np.asarray(x_index, dtype=int)
    y_values = np.asarray(y_index, dtype=int)
    if x_values.shape != y_values.shape:
        raise ValueError("x_index and y_index must have matching shapes.")
    coordinate_to_index = {
        (int(x), int(y)): index
        for index, (x, y) in enumerate(zip(x_values, y_values, strict=True))
    }
    if len(coordinate_to_index) != len(x_values):
        raise ValueError("Grid coordinates must be unique.")
    offsets = [
        (dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    ]
    neighbors: list[np.ndarray] = []
    for x, y in zip(x_values, y_values, strict=True):
        adjacent = [
            coordinate_to_index[(int(x) + dx, int(y) + dy)]
            for dx, dy in offsets
            if (int(x) + dx, int(y) + dy) in coordinate_to_index
        ]
        neighbors.append(np.asarray(sorted(adjacent), dtype=np.int64))
    return neighbors


def _standardize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Moran inputs must be finite.")
    centered = array - array.mean()
    scale = math.sqrt(float(np.mean(centered * centered)))
    if scale <= 0:
        raise ValueError("Moran inputs must have positive variance.")
    return centered / scale


def global_moran(
    values: Sequence[float],
    neighbors: Sequence[np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    """Calculate row-standardized global Moran's I and randomization p-value."""

    z_values = _standardize(values)
    if len(z_values) != len(neighbors):
        raise ValueError("Moran values and neighbor rows must have equal length.")
    non_islands = np.asarray([len(row) > 0 for row in neighbors])
    s0 = float(non_islands.sum())
    if s0 <= 0:
        raise ValueError("At least one non-island spatial unit is required.")

    def statistic(z: np.ndarray) -> float:
        spatial_lag = np.asarray(
            [
                float(z[row].mean()) if len(row) else 0.0
                for row in neighbors
            ]
        )
        return float(
            len(z) / s0 * np.dot(z, spatial_lag) / np.dot(z, z)
        )

    observed = statistic(z_values)
    rng = np.random.default_rng(seed)
    simulated = np.asarray(
        [statistic(rng.permutation(z_values)) for _ in range(permutations)]
    )
    expected = -1.0 / (len(z_values) - 1.0)
    p_value = (
        1.0
        + np.count_nonzero(
            np.abs(simulated - expected) >= abs(observed - expected)
        )
    ) / (permutations + 1.0)
    return {
        "moran_i": observed,
        "analytical_expected_i": expected,
        "permutation_mean_i": float(simulated.mean()),
        "permutation_standard_deviation_i": float(simulated.std(ddof=1)),
        "permutation_p_value_two_sided": float(p_value),
        "spatial_units": len(z_values),
        "non_island_units": int(non_islands.sum()),
        "island_units": int((~non_islands).sum()),
        "permutations": permutations,
    }


def local_moran(
    values: Sequence[float],
    neighbors: Sequence[np.ndarray],
    *,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate local Moran statistics using shared random permutations."""

    z_values = _standardize(values)
    if len(z_values) != len(neighbors):
        raise ValueError("Moran values and neighbor rows must have equal length.")
    spatial_lag = np.asarray(
        [
            float(z_values[row].mean()) if len(row) else np.nan
            for row in neighbors
        ]
    )
    observed = z_values * spatial_lag
    rng = np.random.default_rng(seed)
    permutation_values = np.empty((permutations, len(z_values)), dtype=float)
    for repetition in range(permutations):
        permuted = rng.permutation(z_values)
        permutation_values[repetition] = permuted * np.asarray(
            [
                float(permuted[row].mean()) if len(row) else np.nan
                for row in neighbors
            ]
        )
    permutation_mean = np.full(len(z_values), np.nan, dtype=float)
    non_islands = np.asarray([len(row) > 0 for row in neighbors])
    permutation_mean[non_islands] = np.mean(
        permutation_values[:, non_islands], axis=0
    )
    centered_observed = np.abs(observed - permutation_mean)
    centered_simulated = np.abs(permutation_values - permutation_mean)
    p_values = (
        1.0
        + np.sum(centered_simulated >= centered_observed, axis=0)
    ) / (permutations + 1.0)
    islands = ~non_islands
    p_values[islands] = np.nan
    clusters = np.select(
        [
            islands,
            (z_values >= 0) & (spatial_lag >= 0),
            (z_values < 0) & (spatial_lag < 0),
            (z_values >= 0) & (spatial_lag < 0),
            (z_values < 0) & (spatial_lag >= 0),
        ],
        ["island", "high_high", "low_low", "high_low", "low_high"],
        default="undefined",
    )
    return pd.DataFrame(
        {
            "standardized_value": z_values,
            "spatial_lag_standardized": spatial_lag,
            "local_moran_i": observed,
            "permutation_mean_i": permutation_mean,
            "permutation_p_value_two_sided": p_values,
            "cluster": clusters,
            "neighbor_count": [len(row) for row in neighbors],
        }
    )
