from pathlib import Path

import numpy as np
import pytest

from entropy_crime_bike.e19_scale_sample_identifiability import (
    add_delb_truth,
    calibrate_beta,
    baseline_state_matched_permutation,
    estimate_all,
    load_config,
    minimum_detectable_cmi,
    population_truth,
    run,
    summarize_stage3,
    synthetic_fixed_design,
    _matched_groups,
)


CONFIG = Path(__file__).parents[1] / "config" / "e19.yaml"


def test_frozen_design_configuration() -> None:
    config = load_config(CONFIG)
    formal = config["formal_design"]
    assert len(formal["scales"]) * len(formal["sample_lengths"]) * len(
        formal["target_cmi_bits"]
    ) == 48
    assert formal["monte_carlo_replicates"] == 200
    assert formal["randomization"]["routine_surrogates"] == 199
    assert formal["randomization"]["confirmation_surrogates"] == 999


def test_population_identity_and_zero_target() -> None:
    baseline, bicycle, means = synthetic_fixed_design("1km_day", 90)
    beta, truth = calibrate_beta(0.0, baseline, bicycle, means, 2.0)
    assert beta == 0.0
    assert truth["cmi_true_bits"] == pytest.approx(0.0, abs=1e-10)
    assert truth["h0_true_bits"] - truth["hb_true_bits"] == pytest.approx(
        truth["cmi_true_bits"]
    )
    bounded = add_delb_truth(
        truth, tail_tolerance=1e-12, root_tolerance=1e-10
    )
    assert bounded["l0_true_mse"] == pytest.approx(bounded["lb_true_mse"])
    assert bounded["delta_l_true_mse"] == pytest.approx(0.0, abs=1e-9)


def test_positive_target_calibration() -> None:
    baseline, bicycle, means = synthetic_fixed_design("500m_day", 90)
    beta, truth = calibrate_beta(0.06, baseline, bicycle, means, 2.0)
    assert beta > 0
    assert truth["cmi_true_bits"] == pytest.approx(0.06, abs=1e-6)
    assert truth["maximum_tail_probability"] < 1e-8
    bounded = add_delb_truth(
        truth, tail_tolerance=1e-12, root_tolerance=1e-10
    )
    assert bounded["delta_l_true_mse"] > 0


def test_three_estimators_are_finite() -> None:
    baseline, bicycle, means = synthetic_fixed_design("2km_week", 90)
    rng = np.random.default_rng(17)
    y = rng.poisson(means).astype(int)
    rows = estimate_all(y, baseline, bicycle, jeffreys_alpha=0.5)
    assert {row["estimator"] for row in rows} == {
        "plugin",
        "miller_madow",
        "jeffreys_dirichlet",
    }
    assert all(np.isfinite(row["cmi_hat_bits"]) for row in rows)


def test_stage_one_blocks_formal_execution() -> None:
    with pytest.raises(RuntimeError, match="blocks formal execution"):
        run(CONFIG, development=False)


def test_stage_three_summary_uses_population_truth() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "scale": ["500m_day"] * 2,
            "sample_length_periods": [90] * 2,
            "target_cmi_bits": [0.03] * 2,
            "estimator": ["plugin"] * 2,
            "cmi_true_bits": [0.03] * 2,
            "cmi_hat_bits": [0.02, 0.04],
            "delta_l_true_mse": [0.2] * 2,
            "delta_l_hat_mse": [0.1, 0.3],
            "projection_applied": [False, True],
        }
    )
    row = summarize_stage3(frame, relative_error_floor=1e-12).iloc[0]
    assert row["cmi_bias_bits"] == pytest.approx(0.0)
    assert row["cmi_rmse_bits"] == pytest.approx(0.01)
    assert row["delb_bias_mse"] == pytest.approx(0.0)
    assert row["projection_rate"] == pytest.approx(0.5)


def test_baseline_matched_permutation_preserves_stratum_margins() -> None:
    baseline = np.array([0, 0, 0, 1, 1, 2])
    bicycle = np.array([0, 1, 2, 1, 1, 3])
    groups = _matched_groups(baseline, bicycle)
    result = baseline_state_matched_permutation(
        bicycle, groups, np.random.default_rng(8)
    )
    for state in np.unique(baseline):
        mask = baseline == state
        assert np.array_equal(np.sort(result[mask]), np.sort(bicycle[mask]))


def test_minimum_detectable_cmi_interpolates_target_power() -> None:
    import pandas as pd

    curve = pd.DataFrame(
        {
            "target_cmi_bits": [0.0, 0.03, 0.06, 0.12],
            "power_or_type1_error": [0.05, 0.50, 0.90, 1.0],
        }
    )
    result = minimum_detectable_cmi(curve, target_power=0.80)
    assert result["mdcmi80_discrete_bits"] == pytest.approx(0.06)
    assert result["mdcmi80_interpolated_bits"] == pytest.approx(0.0525)
    assert result["monotonicity_adjusted"] is False


def test_minimum_detectable_cmi_reports_unreached_grid() -> None:
    import pandas as pd

    curve = pd.DataFrame(
        {
            "target_cmi_bits": [0.0, 0.03, 0.06, 0.12],
            "power_or_type1_error": [0.05, 0.20, 0.50, 0.70],
        }
    )
    result = minimum_detectable_cmi(curve, target_power=0.80)
    assert np.isnan(result["mdcmi80_discrete_bits"])
    assert result["threshold_status"] == "greater_than_0.12_bits"
