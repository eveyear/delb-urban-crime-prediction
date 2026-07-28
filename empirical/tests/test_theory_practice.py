import numpy as np
import pandas as pd

from entropy_crime_bike.theory_practice import (
    calculate_gap_metrics,
    ols_city_fixed_effect_hc3,
    paired_grid_classification,
    spearman_with_spatial_bootstrap,
)


def test_gap_identities_hold():
    result = calculate_gap_metrics(4.0, 3.0, 1.0, 0.8)
    assert np.isclose(result["gap_narrowing"], 0.8)
    assert np.isclose(
        result["gap_narrowing"],
        result["delta_mse_integer"] - result["delta_l_exact_mse"],
    )
    assert np.isclose(result["information_use_efficiency_baseline"], 0.25)


def test_classification_assigns_median_ties_low():
    labels, theory_threshold, empirical_threshold = paired_grid_classification(
        [0.0, 1.0, 1.0, 2.0], [-1.0, 0.0, 0.0, 1.0]
    )
    assert theory_threshold == 1.0
    assert empirical_threshold == 0.0
    assert labels.tolist() == [
        "low_theory__low_empirical",
        "low_theory__low_empirical",
        "low_theory__low_empirical",
        "high_theory__high_empirical",
    ]


def test_spatial_bootstrap_recovers_monotone_association():
    x = np.arange(20, dtype=float)
    result = spearman_with_spatial_bootstrap(
        x, x * 2.0, repetitions=200, seed=11, confidence_level=0.95
    )
    assert np.isclose(result["spearman_rho"], 1.0)
    assert result["bootstrap_ci_low"] > 0.99


def test_hc3_fixed_effect_slope():
    frame = pd.DataFrame(
        {
            "city": np.repeat(["A", "B", "C"], 10),
            "delta_l": np.tile(np.arange(10, dtype=float), 3),
        }
    )
    frame["delta_mse"] = (
        2.0 * frame["delta_l"]
        + frame["city"].map({"A": 0.0, "B": 5.0, "C": -3.0})
    )
    result = ols_city_fixed_effect_hc3(
        frame, outcome="delta_mse", exposure="delta_l"
    )
    assert np.isclose(result["slope_delta_l"], 2.0)
    assert result["r_squared"] > 0.999999
