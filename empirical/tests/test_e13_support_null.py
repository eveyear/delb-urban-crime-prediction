from __future__ import annotations

import numpy as np
import pytest

from entropy_crime_bike.e10_placebo import conditional_information
from entropy_crime_bike.e12_cmi_simulation import support_diagnostics
from entropy_crime_bike.e13_support_null import panel_cmi_support


def test_panel_support_matches_scalar_definition() -> None:
    target = np.array(
        [[0, 0, 1, 2, 2], [0, 1, 0, 1, 2]], dtype=int
    )
    baseline = np.array(
        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=int
    )
    bicycle = np.array(
        [[0, 0, 1, 0, 1], [1, 0, 1, 0, 1]], dtype=int
    )
    result = panel_cmi_support(target, baseline, bicycle)
    for index in range(2):
        scalar = support_diagnostics(
            target[index], baseline[index], bicycle[index]
        )
        for metric in [
            "support_atoms_per_observation",
            "singleton_observation_share",
            "effective_df_per_observation",
        ]:
            assert result.loc[index, metric] == pytest.approx(scalar[metric])


def test_panel_cmi_matches_e10_definition() -> None:
    target = np.array(
        [[0, 1, 0, 2, 1, 0, 3, 0], [1, 0, 1, 0, 2, 0, 1, 0]],
        dtype=int,
    )
    baseline = np.array(
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3]],
        dtype=int,
    )
    bicycle = np.array(
        [[0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]],
        dtype=int,
    )
    result = panel_cmi_support(target, baseline, bicycle)
    for index in range(2):
        h0, hb, cmi = conditional_information(
            target[index], baseline[index], bicycle[index]
        )
        assert result.loc[index, "h0_miller_madow_bits"] == pytest.approx(h0)
        assert result.loc[index, "hb_miller_madow_bits"] == pytest.approx(hb)
        assert result.loc[index, "delta_h_raw_bits"] == pytest.approx(cmi)


def test_effective_df_uses_only_observed_baseline_states() -> None:
    target = np.array([[0, 1, 0, 1]], dtype=int)
    baseline = np.array([[0, 0, 2, 2]], dtype=int)
    bicycle = np.array([[0, 1, 0, 1]], dtype=int)
    result = panel_cmi_support(target, baseline, bicycle).iloc[0]
    assert result["effective_df"] == 2
    assert result["effective_df_per_observation"] == pytest.approx(0.5)


def test_support_ratios_are_bounded() -> None:
    rng = np.random.default_rng(12)
    target = rng.integers(0, 5, size=(4, 100))
    baseline = rng.integers(0, 8, size=(4, 100))
    bicycle = rng.integers(0, 4, size=(4, 100))
    result = panel_cmi_support(target, baseline, bicycle)
    assert result["support_atoms_per_observation"].between(0, 1).all()
    assert result["singleton_observation_share"].between(0, 1).all()
    assert result["effective_df_per_observation"].ge(0).all()


def test_misaligned_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        panel_cmi_support(
            np.zeros((2, 3), dtype=int),
            np.zeros((2, 3), dtype=int),
            np.zeros((2, 4), dtype=int),
        )
