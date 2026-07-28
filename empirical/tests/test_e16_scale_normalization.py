from __future__ import annotations

import numpy as np
import pandas as pd

from entropy_crime_bike.e16_scale_normalization import (
    add_scale_normalization,
    add_stability,
    bootstrap_intervals,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": ["DC", "DC"],
            "resolution_m": [2000, 2000],
            "temporal_resolution": ["week", "week"],
            "estimator": ["miller_madow", "miller_madow"],
            "replicate": [1, 0],
            "fraction": [0.5, 1.0],
            "h0_raw_bits": [0.4, 0.5],
            "hb_raw_bits": [0.3, 0.4],
            "delta_h_raw_bits": [0.1, 0.1],
            "delta_h_projected_bits": [0.1, 0.1],
            "l0_exact_mse": [8.0, 10.0],
            "lb_exact_mse": [4.0, 5.0],
            "delta_l_exact_mse": [4.0, 5.0],
            "l0_closed_mse": [6.0, 8.0],
            "lb_closed_mse": [3.0, 4.0],
            "delta_l_closed_mse": [3.0, 4.0],
            "observations": [50, 100],
            "singleton_observation_share": [0.2, 0.1],
            "effective_df_per_observation": [0.2, 0.1],
        }
    )


def test_mse_uses_squared_area_time_and_rmse_uses_linear_scale():
    result = add_scale_normalization(_rows())
    scale = 4 * 7
    assert np.isclose(result.iloc[1]["l0_exact_intensity_mse"], 10 / scale**2)
    assert np.isclose(result.iloc[1]["l0_exact_intensity_rmse"], np.sqrt(10) / scale)
    assert np.isclose(result.iloc[1]["relative_reduction_exact"], 0.5)
    assert np.isclose(result.iloc[1]["relative_reduction_exact_from_intensity"], 0.5)


def test_stability_is_zero_at_reference_and_scale_invariant_for_mse():
    result = add_stability(add_scale_normalization(_rows()), epsilon=1e-12)
    assert result.loc[result["fraction"].eq(1), "stability_l0_exact_mse"].iloc[0] == 0
    assert np.allclose(
        result["stability_l0_exact_mse"], result["stability_l0_exact_intensity_mse"]
    )


def test_bootstrap_interval_retains_paired_group():
    point = add_scale_normalization(_rows().loc[_rows()["fraction"].eq(1)])
    bootstrap = pd.concat([point.assign(replicate=i) for i in range(1, 6)], ignore_index=True)
    interval = bootstrap_intervals(bootstrap, point)
    assert len(interval) == 1
    assert interval.iloc[0]["bootstrap_repetitions"] == 5
    assert interval.iloc[0]["delta_l_exact_mse_low"] <= interval.iloc[0]["delta_l_exact_mse_high"]

