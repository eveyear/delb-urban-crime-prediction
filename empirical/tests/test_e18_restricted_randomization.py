import numpy as np
import pandas as pd

from entropy_crime_bike.e18_restricted_randomization import (
    _calibrate_beta,
    restricted_matched_blocks,
)


def test_restricted_blocks_preserve_grid_marginals() -> None:
    values = np.tile(np.arange(112), (2, 1))
    dates = pd.Series(pd.date_range("2020-01-01", periods=112))
    crime = np.zeros_like(values)
    holiday = np.zeros_like(values)
    result, share = restricted_matched_blocks(
        values, dates, crime, holiday, np.random.default_rng(4)
    )
    assert share > 0.8
    for index in range(2):
        assert np.array_equal(np.sort(values[index]), np.sort(result[index]))


def test_zero_target_cmi_has_zero_beta() -> None:
    baseline = np.repeat(np.arange(4), 20)
    bicycle = np.tile(np.arange(4), 20)
    means = np.ones(80)
    beta, achieved = _calibrate_beta(0.0, baseline, bicycle, means, 2.0)
    assert beta == 0.0
    assert achieved == 0.0
