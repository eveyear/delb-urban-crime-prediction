from __future__ import annotations

import math

import numpy as np
import pytest

from entropy_crime_bike.discrete_bound import (
    closed_form_delb,
    discrete_gaussian_pmf,
    discrete_gaussian_stats,
    entropy_envelope,
    inverse_entropy_envelope,
    lambda_for_entropy,
    lambda_for_second_moment,
    shannon_entropy,
)


def test_zero_entropy_boundary() -> None:
    assert inverse_entropy_envelope(0.0) == 0.0
    assert inverse_entropy_envelope(-1.0) == 0.0
    assert entropy_envelope(0.0) == 0.0
    assert closed_form_delb(0.0) == 0.0


@pytest.mark.parametrize("entropy_bits", [0.01, 0.25, 1.0, 3.0, 8.0])
def test_entropy_inverse_round_trip(entropy_bits: float) -> None:
    moment = inverse_entropy_envelope(entropy_bits)
    recovered = entropy_envelope(moment)
    assert recovered == pytest.approx(entropy_bits, abs=2e-9)


@pytest.mark.parametrize("lambda_value", [1e-4, 0.1, 1.0, 3.0, 20.0])
def test_pmf_matches_partition_statistics(lambda_value: float) -> None:
    statistics = discrete_gaussian_stats(lambda_value)
    support, probabilities, tail = discrete_gaussian_pmf(lambda_value)
    entropy = shannon_entropy(probabilities)
    moment = float(np.dot(support.astype(float) ** 2, probabilities))
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-14)
    assert tail <= 1e-15
    assert moment == pytest.approx(
        statistics.second_moment, rel=1e-11, abs=1e-12
    )
    assert entropy == pytest.approx(
        statistics.entropy_bits, rel=1e-11, abs=1e-12
    )


def test_envelope_and_closed_form_are_ordered() -> None:
    entropy_values = np.linspace(0.0, 8.0, 81)
    exact = np.array(
        [inverse_entropy_envelope(value) for value in entropy_values]
    )
    closed = np.array([closed_form_delb(value) for value in entropy_values])
    assert np.all(np.diff(exact) >= 0)
    assert np.all(exact + 1e-11 >= closed)


def test_lambda_parameterization_is_strictly_decreasing() -> None:
    lambdas = np.geomspace(1e-4, 20.0, 100)
    statistics = [discrete_gaussian_stats(value) for value in lambdas]
    moments = np.array([value.second_moment for value in statistics])
    entropies = np.array([value.entropy_bits for value in statistics])
    assert np.all(np.diff(moments) < 0)
    assert np.all(np.diff(entropies) < 0)


def test_lambda_for_entropy() -> None:
    target = 2.0
    lambda_value = lambda_for_entropy(target)
    statistics = discrete_gaussian_stats(lambda_value)
    assert statistics.entropy_bits == pytest.approx(target, abs=1e-10)


@pytest.mark.parametrize("target", [0.001, 0.2, 1.0, 10.0, 1000.0])
def test_lambda_for_second_moment(target: float) -> None:
    lambda_value = lambda_for_second_moment(target)
    statistics = discrete_gaussian_stats(lambda_value)
    assert statistics.second_moment == pytest.approx(
        target, rel=2e-10, abs=2e-12
    )


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        discrete_gaussian_stats(0.0)
    with pytest.raises(ValueError):
        inverse_entropy_envelope(math.inf)
    with pytest.raises(ValueError):
        shannon_entropy(np.array([0.5, -0.5, 1.0]))
    with pytest.raises(ValueError):
        lambda_for_second_moment(0.0)
