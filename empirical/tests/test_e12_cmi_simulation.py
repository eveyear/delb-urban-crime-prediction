from __future__ import annotations

import numpy as np
import pytest

from entropy_crime_bike.e12_cmi_simulation import (
    build_population_model,
    estimate_cmi,
    simulate_series,
    support_diagnostics,
)
from entropy_crime_bike.e10_placebo import conditional_information


DGP = {
    "dispersion": 2.5,
    "minimum_mean": 0.35,
    "maximum_mean": 2.0,
    "tail_probability_tolerance": 1.0e-12,
    "markov_persistence": 0.85,
}


def test_population_null_and_alternative() -> None:
    null = build_population_model(4, 3, "skewed", 0.0, DGP)
    alternative = build_population_model(4, 3, "skewed", 0.6, DGP)
    assert null.population_cmi_bits == pytest.approx(0.0, abs=1e-12)
    assert alternative.population_cmi_bits > 0
    assert null.maximum_tail_probability <= 1e-12
    assert alternative.maximum_tail_probability <= 1e-12


def test_oracle_delb_and_binary_fano_hold() -> None:
    model = build_population_model(8, 4, "balanced", 0.5, DGP)
    assert model.rounded_predictor_mse >= model.exact_delb_mse
    assert model.binary_bayes_error >= model.binary_fano_floor


def test_estimator_mm_reuses_e10_definition() -> None:
    target = np.array([0, 1, 0, 2, 1, 0, 3, 0] * 20)
    baseline = np.array([0, 0, 1, 1, 2, 2, 3, 3] * 20)
    bicycle = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 20)
    estimate = estimate_cmi(target, baseline, bicycle)
    h0, hb, cmi = conditional_information(target, baseline, bicycle)
    assert estimate["h0_miller_madow_bits"] == pytest.approx(h0)
    assert estimate["hb_miller_madow_bits"] == pytest.approx(hb)
    assert estimate["cmi_miller_madow_bits"] == pytest.approx(cmi)


def test_simulation_is_reproducible_and_markov_supported() -> None:
    model = build_population_model(4, 2, "balanced", 0.4, DGP)
    first = simulate_series(
        model, 200, "markov", DGP, np.random.default_rng(77)
    )
    second = simulate_series(
        model, 200, "markov", DGP, np.random.default_rng(77)
    )
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_support_diagnostics_have_expected_ranges() -> None:
    result = support_diagnostics(
        np.array([0, 0, 1, 2, 2]),
        np.array([0, 0, 0, 1, 1]),
        np.array([0, 0, 1, 0, 1]),
    )
    assert result["enhanced_joint_atoms"] == 4
    assert result["singleton_atom_count"] == 3
    assert result["singleton_observation_share"] == pytest.approx(0.6)
    assert 0 <= result["effective_df_per_observation"]
