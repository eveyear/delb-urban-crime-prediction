import numpy as np

from entropy_crime_bike.e10_placebo import (
    circular_shift_surrogate,
    conditional_information,
    randomization_p_value,
    spatial_series_permutation,
    temporal_block_permutation,
)


def test_conditional_information_detects_deterministic_side_information() -> None:
    target = np.tile([0, 1], 100)
    baseline = np.zeros(len(target), dtype=int)
    bicycle = target.copy()
    _, _, cmi = conditional_information(target, baseline, bicycle)
    assert cmi > 0.9


def test_temporal_blocks_preserve_each_grid_marginal() -> None:
    values = np.arange(2 * 14).reshape(2, 14)
    result = temporal_block_permutation(
        values, np.random.default_rng(3), block_days=7
    )
    assert result.shape == values.shape
    for row in range(2):
        assert np.array_equal(np.sort(result[row]), np.sort(values[row]))


def test_spatial_permutation_preserves_complete_series() -> None:
    values = np.arange(12).reshape(3, 4)
    result = spatial_series_permutation(values, np.random.default_rng(4))
    assert {tuple(row) for row in result} == {tuple(row) for row in values}


def test_circular_surrogate_preserves_grid_marginal() -> None:
    values = np.arange(3 * 100).reshape(3, 100)
    result = circular_shift_surrogate(
        values, np.random.default_rng(5), minimum_shift=10
    )
    for row in range(3):
        assert np.array_equal(np.sort(result[row]), np.sort(values[row]))


def test_randomization_p_value_uses_plus_one_rule() -> None:
    null = np.array([0.0, 0.1, 0.2, 0.3])
    assert randomization_p_value(0.4, null) == 0.2
    assert randomization_p_value(0.2, null) == 0.6
