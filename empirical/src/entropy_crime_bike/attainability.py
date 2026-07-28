from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .discrete_bound import (
    LOG_TWO,
    discrete_gaussian_stats,
    lambda_for_second_moment,
    shannon_entropy,
)


@dataclass(frozen=True)
class AttainabilityDecomposition:
    """Numerical terms in the exact DELB entropy-domain decomposition."""

    second_moment: float
    conditional_entropy_bits: float
    error_entropy_bits: float
    envelope_entropy_bits: float
    history_dependence_bits: float
    shape_mismatch_bits: float
    predictor_specific_slack_bits: float
    universal_slack_bits: float
    identity_residual_bits: float
    lambda_value: float

    def as_record(self) -> dict[str, float]:
        return asdict(self)


def _normalized_joint(joint_probability: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint_probability, dtype=float)
    if joint.ndim != 2:
        raise ValueError("joint_probability must be a two-dimensional array.")
    if np.any(joint < 0) or not np.isfinite(joint).all():
        raise ValueError("joint_probability must be finite and nonnegative.")
    total = float(joint.sum())
    if total <= 0:
        raise ValueError("joint_probability must contain positive mass.")
    return joint / total


def attainability_decomposition(
    error_values: np.ndarray,
    joint_error_history: np.ndarray,
) -> AttainabilityDecomposition:
    r"""Evaluate the exact entropy-domain DELB slack identity.

    Rows of ``joint_error_history`` correspond to the integer values in
    ``error_values`` and columns correspond to arbitrary finite history
    states. The conditional entropy returned here equals ``H(X | F)`` for
    every integer predictor because conditioning turns ``X -> E`` into an
    integer translation.
    """

    errors = np.asarray(error_values)
    if errors.ndim != 1:
        raise ValueError("error_values must be one-dimensional.")
    if not np.issubdtype(errors.dtype, np.integer):
        raise ValueError("error_values must be integer valued.")
    if len(np.unique(errors)) != len(errors):
        raise ValueError("error_values must not contain duplicates.")

    joint = _normalized_joint(joint_error_history)
    if joint.shape[0] != errors.size:
        raise ValueError(
            "The number of joint-probability rows must match error_values."
        )

    error_probability = joint.sum(axis=1)
    history_probability = joint.sum(axis=0)
    error_entropy = shannon_entropy(error_probability)
    history_entropy = shannon_entropy(history_probability)
    joint_entropy = shannon_entropy(joint.ravel())
    history_dependence = max(
        0.0, error_entropy + history_entropy - joint_entropy
    )
    conditional_entropy = max(0.0, error_entropy - history_dependence)
    second_moment = float(
        np.dot(errors.astype(float) ** 2, error_probability)
    )

    if second_moment <= np.finfo(float).eps:
        if np.any(error_probability[errors != 0] > 1e-14):
            raise ValueError(
                "A zero second moment requires all mass at error zero."
            )
        lambda_value = math.inf
        envelope_entropy = 0.0
        shape_mismatch = 0.0
    else:
        lambda_value = lambda_for_second_moment(second_moment)
        statistics = discrete_gaussian_stats(lambda_value)
        envelope_entropy = statistics.entropy_bits
        positive = error_probability > 0
        log_q_bits = (
            -lambda_value * errors[positive].astype(float) ** 2
            - math.log(statistics.partition)
        ) / LOG_TWO
        shape_mismatch = float(
            np.dot(
                error_probability[positive],
                np.log2(error_probability[positive]) - log_q_bits,
            )
        )
        shape_mismatch = max(0.0, shape_mismatch)

    predictor_specific_slack = envelope_entropy - error_entropy
    universal_slack = envelope_entropy - conditional_entropy
    identity_residual = universal_slack - (
        history_dependence + shape_mismatch
    )

    return AttainabilityDecomposition(
        second_moment=second_moment,
        conditional_entropy_bits=conditional_entropy,
        error_entropy_bits=error_entropy,
        envelope_entropy_bits=envelope_entropy,
        history_dependence_bits=history_dependence,
        shape_mismatch_bits=shape_mismatch,
        predictor_specific_slack_bits=predictor_specific_slack,
        universal_slack_bits=universal_slack,
        identity_residual_bits=identity_residual,
        lambda_value=lambda_value,
    )
