from __future__ import annotations

import numpy as np

from entropy_crime_bike.e16_randomization_calibration import (
    circular_shift_randomization,
    conditional_entropy_and_support,
    spatial_series_randomization,
    temporal_block_randomization,
)


def test_randomizations_preserve_global_bicycle_marginal():
    matrix = np.arange(48).reshape(4, 12) % 4
    for transformed in [
        temporal_block_randomization(matrix, np.random.default_rng(1), 3),
        spatial_series_randomization(matrix, np.random.default_rng(2)),
        circular_shift_randomization(matrix, np.random.default_rng(3), 2),
    ]:
        assert np.array_equal(np.sort(transformed.ravel()), np.sort(matrix.ravel()))


def test_temporal_blocks_preserve_each_grid_marginal():
    matrix = np.arange(60).reshape(5, 12) % 4
    transformed = temporal_block_randomization(matrix, np.random.default_rng(4), 3)
    for observed, randomized in zip(matrix, transformed):
        assert np.array_equal(np.sort(observed), np.sort(randomized))


def test_circular_shift_preserves_each_grid_marginal():
    matrix = np.arange(80).reshape(5, 16) % 4
    transformed = circular_shift_randomization(matrix, np.random.default_rng(5), 3)
    for observed, randomized in zip(matrix, transformed):
        assert np.array_equal(np.sort(observed), np.sort(randomized))


def test_conditional_entropy_support_is_finite_and_valid():
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    pairs = list(dict.fromkeys(zip(z.tolist(), y.tolist())))
    mapping = {pair: index for index, pair in enumerate(pairs)}
    zy = np.array([mapping[pair] for pair in zip(z.tolist(), y.tolist())])
    zy_to_z = np.array([pair[0] for pair in pairs], dtype=np.int64)
    bicycle = np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64)
    result = conditional_entropy_and_support(
        z, zy, zy_to_z, bicycle, np.ones(len(z), dtype=bool), z_count=2
    )
    assert np.isfinite(result["hb_miller_madow_bits"])
    assert 0 <= result["singleton_observation_share"] <= 1
    assert result["effective_df"] >= 0
