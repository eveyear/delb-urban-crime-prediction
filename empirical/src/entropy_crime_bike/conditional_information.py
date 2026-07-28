from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse


LOG_TWO = math.log(2.0)


def crime_lag_bins(values: pd.Series) -> pd.Categorical:
    """Map nonnegative lagged counts to the prespecified four-state alphabet."""

    numeric = pd.to_numeric(values, errors="coerce")
    labels = np.select(
        [
            numeric.eq(0),
            numeric.eq(1),
            numeric.eq(2),
            numeric.ge(3),
        ],
        ["zero", "one", "two", "three_plus"],
        default=None,
    )
    return pd.Categorical(
        labels,
        categories=["zero", "one", "two", "three_plus"],
        ordered=True,
    )


def fit_positive_bicycle_quantiles(
    values: pd.Series, quantiles: Sequence[float]
) -> tuple[float, ...]:
    """Fit strictly positive mobility cut points using training values only."""

    numeric = pd.to_numeric(values, errors="coerce")
    positive = numeric.loc[numeric.gt(0) & numeric.notna()]
    if positive.empty:
        raise ValueError("At least one strictly positive training flow is required.")
    probabilities = np.asarray(tuple(quantiles), dtype=float)
    if (
        probabilities.ndim != 1
        or np.any(probabilities <= 0)
        or np.any(probabilities >= 1)
        or np.any(np.diff(probabilities) <= 0)
    ):
        raise ValueError("Quantiles must be strictly increasing within (0, 1).")
    return tuple(float(value) for value in positive.quantile(probabilities))


def apply_bicycle_bins(
    values: pd.Series, cut_points: Sequence[float]
) -> pd.Categorical:
    """Apply zero plus positive training-quantile mobility states."""

    if len(cut_points) != 2:
        raise ValueError("The primary bicycle mapping requires two cut points.")
    lower, upper = (float(value) for value in cut_points)
    if lower > upper:
        raise ValueError("Bicycle cut points must be nondecreasing.")
    numeric = pd.to_numeric(values, errors="coerce")
    labels = np.select(
        [
            numeric.eq(0),
            numeric.gt(0) & numeric.le(lower),
            numeric.gt(lower) & numeric.le(upper),
            numeric.gt(upper),
        ],
        ["zero", "low", "medium", "high"],
        default=None,
    )
    return pd.Categorical(
        labels,
        categories=["zero", "low", "medium", "high"],
        ordered=True,
    )


def _entropy_from_count_matrix(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Every count-matrix row must have positive mass.")
    with np.errstate(divide="ignore", invalid="ignore"):
        logarithms = np.where(counts > 0, np.log(counts), 0.0)
    sum_count_log_count = (counts * logarithms).sum(axis=1)
    return (np.log(totals) - sum_count_log_count / totals) / LOG_TWO


@dataclass(frozen=True)
class ConditionalCodebook:
    """Sparse block-by-atom representation for repeated entropy estimates."""

    block_count: int
    observation_count: int
    baseline_joint_by_block: sparse.csr_matrix
    bicycle_joint_by_block: sparse.csr_matrix
    baseline_joint_to_state: sparse.csr_matrix
    bicycle_joint_to_state: sparse.csr_matrix
    baseline_state_count: int
    bicycle_state_count: int
    baseline_joint_count: int
    bicycle_joint_count: int
    baseline_state_frequencies: np.ndarray
    bicycle_state_frequencies: np.ndarray

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        target: str,
        baseline_state: Sequence[str],
        bicycle_state_field: str,
        block_field: str,
    ) -> "ConditionalCodebook":
        required = [
            target,
            *baseline_state,
            bicycle_state_field,
            block_field,
        ]
        missing = [column for column in required if column not in frame]
        if missing:
            raise KeyError(f"Missing columns for conditional codebook: {missing}")
        if frame[required].isna().any().any():
            raise ValueError("Conditional codebook inputs must not contain missing values.")

        baseline_index = pd.MultiIndex.from_frame(
            frame[list(baseline_state)].astype(object)
        )
        baseline_codes, baseline_uniques = pd.factorize(
            baseline_index, sort=False
        )
        bicycle_state_index = pd.MultiIndex.from_arrays(
            [
                baseline_codes,
                frame[bicycle_state_field].astype(object).to_numpy(),
            ]
        )
        bicycle_state_codes, bicycle_state_uniques = pd.factorize(
            bicycle_state_index, sort=False
        )
        target_values = pd.to_numeric(
            frame[target], errors="raise"
        ).to_numpy(dtype=np.int64)
        if np.any(target_values < 0):
            raise ValueError("The E04 crime target must be nonnegative.")

        baseline_joint_index = pd.MultiIndex.from_arrays(
            [baseline_codes, target_values]
        )
        baseline_joint_codes, baseline_joint_uniques = pd.factorize(
            baseline_joint_index, sort=False
        )
        bicycle_joint_index = pd.MultiIndex.from_arrays(
            [bicycle_state_codes, target_values]
        )
        bicycle_joint_codes, bicycle_joint_uniques = pd.factorize(
            bicycle_joint_index, sort=False
        )
        block_codes, block_uniques = pd.factorize(
            frame[block_field], sort=True
        )

        baseline_joint_count = len(baseline_joint_uniques)
        bicycle_joint_count = len(bicycle_joint_uniques)
        baseline_state_count = len(baseline_uniques)
        bicycle_state_count = len(bicycle_state_uniques)
        block_count = len(block_uniques)
        ones = np.ones(len(frame), dtype=np.float64)
        baseline_joint_by_block = sparse.coo_matrix(
            (
                ones,
                (block_codes, baseline_joint_codes),
            ),
            shape=(block_count, baseline_joint_count),
        ).tocsr()
        bicycle_joint_by_block = sparse.coo_matrix(
            (
                ones,
                (block_codes, bicycle_joint_codes),
            ),
            shape=(block_count, bicycle_joint_count),
        ).tocsr()
        baseline_joint_state_codes = (
            baseline_joint_uniques.get_level_values(0)
            .to_numpy(dtype=np.int64)
        )
        bicycle_joint_state_codes = (
            bicycle_joint_uniques.get_level_values(0)
            .to_numpy(dtype=np.int64)
        )
        baseline_joint_to_state = sparse.coo_matrix(
            (
                np.ones(baseline_joint_count, dtype=np.float64),
                (
                    np.arange(baseline_joint_count),
                    baseline_joint_state_codes,
                ),
            ),
            shape=(baseline_joint_count, baseline_state_count),
        ).tocsr()
        bicycle_joint_to_state = sparse.coo_matrix(
            (
                np.ones(bicycle_joint_count, dtype=np.float64),
                (
                    np.arange(bicycle_joint_count),
                    bicycle_joint_state_codes,
                ),
            ),
            shape=(bicycle_joint_count, bicycle_state_count),
        ).tocsr()
        baseline_state_frequencies = np.bincount(
            baseline_codes, minlength=baseline_state_count
        )
        bicycle_state_frequencies = np.bincount(
            bicycle_state_codes, minlength=bicycle_state_count
        )
        return cls(
            block_count=block_count,
            observation_count=len(frame),
            baseline_joint_by_block=baseline_joint_by_block,
            bicycle_joint_by_block=bicycle_joint_by_block,
            baseline_joint_to_state=baseline_joint_to_state,
            bicycle_joint_to_state=bicycle_joint_to_state,
            baseline_state_count=baseline_state_count,
            bicycle_state_count=bicycle_state_count,
            baseline_joint_count=baseline_joint_count,
            bicycle_joint_count=bicycle_joint_count,
            baseline_state_frequencies=baseline_state_frequencies,
            bicycle_state_frequencies=bicycle_state_frequencies,
        )

    def estimate(self, block_weights: np.ndarray) -> pd.DataFrame:
        """Estimate plugin/MM conditional entropies for weight rows."""

        weights = np.asarray(block_weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if weights.shape[1] != self.block_count:
            raise ValueError("Block weights do not match the codebook.")
        if np.any(weights < 0):
            raise ValueError("Block weights must be nonnegative.")

        baseline_joint = np.asarray(
            weights @ self.baseline_joint_by_block
        )
        bicycle_joint = np.asarray(
            weights @ self.bicycle_joint_by_block
        )
        baseline_state = np.asarray(
            baseline_joint @ self.baseline_joint_to_state
        )
        bicycle_state = np.asarray(
            bicycle_joint @ self.bicycle_joint_to_state
        )
        baseline_plugin = _entropy_from_count_matrix(
            baseline_joint
        ) - _entropy_from_count_matrix(baseline_state)
        bicycle_plugin = _entropy_from_count_matrix(
            bicycle_joint
        ) - _entropy_from_count_matrix(bicycle_state)
        sample_sizes = baseline_joint.sum(axis=1)
        baseline_joint_atoms = (baseline_joint > 0).sum(axis=1)
        baseline_state_atoms = (baseline_state > 0).sum(axis=1)
        bicycle_joint_atoms = (bicycle_joint > 0).sum(axis=1)
        bicycle_state_atoms = (bicycle_state > 0).sum(axis=1)
        baseline_mm = baseline_plugin + (
            baseline_joint_atoms - baseline_state_atoms
        ) / (2.0 * sample_sizes * LOG_TWO)
        bicycle_mm = bicycle_plugin + (
            bicycle_joint_atoms - bicycle_state_atoms
        ) / (2.0 * sample_sizes * LOG_TWO)
        return pd.DataFrame(
            {
                "sample_size": sample_sizes,
                "h0_plugin_bits": baseline_plugin,
                "hb_plugin_bits": bicycle_plugin,
                "delta_h_plugin_raw_bits": (
                    baseline_plugin - bicycle_plugin
                ),
                "h0_miller_madow_bits": baseline_mm,
                "hb_miller_madow_bits": bicycle_mm,
                "delta_h_miller_madow_raw_bits": (
                    baseline_mm - bicycle_mm
                ),
                "baseline_joint_atoms": baseline_joint_atoms,
                "baseline_state_atoms": baseline_state_atoms,
                "bicycle_joint_atoms": bicycle_joint_atoms,
                "bicycle_state_atoms": bicycle_state_atoms,
            }
        )

    def sparsity_diagnostics(self) -> dict[str, float | int]:
        baseline = self.baseline_state_frequencies
        bicycle = self.bicycle_state_frequencies
        return {
            "baseline_states": self.baseline_state_count,
            "bicycle_states": self.bicycle_state_count,
            "baseline_joint_atoms": self.baseline_joint_count,
            "bicycle_joint_atoms": self.bicycle_joint_count,
            "baseline_median_observations_per_state": float(
                np.median(baseline)
            ),
            "bicycle_median_observations_per_state": float(
                np.median(bicycle)
            ),
            "baseline_singleton_state_share": float(
                (baseline == 1).mean()
            ),
            "bicycle_singleton_state_share": float(
                (bicycle == 1).mean()
            ),
            "baseline_observation_share_in_singleton_states": float(
                baseline[baseline == 1].sum() / baseline.sum()
            ),
            "bicycle_observation_share_in_singleton_states": float(
                bicycle[bicycle == 1].sum() / bicycle.sum()
            ),
        }
