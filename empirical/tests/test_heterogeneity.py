import numpy as np
import pandas as pd

from entropy_crime_bike.heterogeneity import (
    category_lag,
    normal_interval,
    paired_category_difference,
)


def test_category_lag_is_strictly_within_grid():
    frame = pd.DataFrame(
        {
            "grid_id": ["b", "a", "a", "b"],
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-01", "2020-01-02", "2020-01-01"]
            ),
            "target": [4, 1, 3, 2],
        }
    )
    lag = category_lag(frame, target="target")
    assert np.isnan(lag.iloc[1])
    assert lag.iloc[2] == 1
    assert lag.iloc[0] == 2
    assert np.isnan(lag.iloc[3])


def test_normal_interval_positive_test():
    bootstrap = np.linspace(0.5, 1.5, 1000)
    result = normal_interval(
        1.0, bootstrap, confidence_level=0.95, alternative="greater"
    )
    assert result["normal_ci_low"] > 0
    assert result["p_value"] < 0.01


def test_paired_category_difference_uses_aligned_replicates():
    first = np.array([2.0, 4.0, 6.0])
    second = np.array([1.0, 2.0, 3.0])
    result = paired_category_difference(
        4.0,
        2.0,
        first,
        second,
        confidence_level=0.95,
    )
    assert result["point_difference_a_minus_b"] == 2.0
    assert result["bootstrap_standard_error"] == 1.0
