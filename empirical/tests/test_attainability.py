from __future__ import annotations

import numpy as np
import pytest

from entropy_crime_bike.attainability import attainability_decomposition
from entropy_crime_bike.discrete_bound import discrete_gaussian_pmf


def test_zero_error_boundary() -> None:
    result = attainability_decomposition(
        np.array([0], dtype=int),
        np.array([[0.25, 0.75]], dtype=float),
    )
    assert result.second_moment == 0.0
    assert result.history_dependence_bits == 0.0
    assert result.shape_mismatch_bits == 0.0
    assert result.universal_slack_bits == 0.0
    assert result.identity_residual_bits == 0.0


def test_independent_nearly_discrete_gaussian_error() -> None:
    support, probability, _ = discrete_gaussian_pmf(
        0.8, tail_tolerance=1e-15
    )
    history_probability = np.array([0.2, 0.3, 0.5])
    joint = np.outer(probability, history_probability)
    result = attainability_decomposition(support, joint)
    assert result.history_dependence_bits == pytest.approx(0.0, abs=2e-14)
    assert result.shape_mismatch_bits == pytest.approx(0.0, abs=2e-12)
    assert result.universal_slack_bits == pytest.approx(0.0, abs=2e-12)
    assert result.identity_residual_bits == pytest.approx(0.0, abs=2e-12)


def test_history_dependence_and_shape_mismatch_add_exactly() -> None:
    support = np.array([-2, -1, 0, 1, 2], dtype=int)
    joint = np.array(
        [
            [0.04, 0.01],
            [0.18, 0.02],
            [0.20, 0.20],
            [0.02, 0.18],
            [0.01, 0.14],
        ]
    )
    result = attainability_decomposition(support, joint)
    assert result.history_dependence_bits > 0
    assert result.shape_mismatch_bits > 0
    assert result.predictor_specific_slack_bits == pytest.approx(
        result.shape_mismatch_bits, abs=2e-12
    )
    assert result.universal_slack_bits == pytest.approx(
        result.history_dependence_bits + result.shape_mismatch_bits,
        abs=2e-12,
    )
    assert result.identity_residual_bits == pytest.approx(0.0, abs=2e-12)


def test_non_gaussian_independent_error_has_only_shape_slack() -> None:
    support = np.array([-1, 0, 1], dtype=int)
    error_probability = np.array([0.1, 0.8, 0.1])
    joint = np.outer(error_probability, np.array([0.4, 0.6]))
    result = attainability_decomposition(support, joint)
    assert result.history_dependence_bits == pytest.approx(0.0, abs=2e-14)
    assert result.shape_mismatch_bits > 0
    assert result.universal_slack_bits == pytest.approx(
        result.shape_mismatch_bits, abs=2e-12
    )


def test_invalid_support_or_joint_raises() -> None:
    with pytest.raises(ValueError):
        attainability_decomposition(
            np.array([0.0, 1.0]),
            np.array([[0.5], [0.5]]),
        )
    with pytest.raises(ValueError):
        attainability_decomposition(
            np.array([0, 1]),
            np.array([[0.5, -0.1], [0.5, 0.1]]),
        )
