from __future__ import annotations

import numpy as np
import pytest

from entropy_crime_bike.spatial_information import (
    benjamini_hochberg,
    global_moran,
    maximum_true_run,
    queen_neighbor_indices,
)


def test_maximum_true_run() -> None:
    assert maximum_true_run([False, True, True, False, True]) == 2
    assert maximum_true_run([False, False]) == 0
    assert maximum_true_run([True, True, True]) == 3


def test_benjamini_hochberg_known_values() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert adjusted[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_queen_neighbors_on_three_cell_line() -> None:
    neighbors = queen_neighbor_indices([0, 1, 2], [0, 0, 0])
    assert [row.tolist() for row in neighbors] == [[1], [0, 2], [1]]


def test_global_moran_detects_ordering() -> None:
    neighbors = queen_neighbor_indices([0, 1, 2, 3], [0, 0, 0, 0])
    result = global_moran(
        [0.0, 1.0, 2.0, 3.0],
        neighbors,
        permutations=99,
        seed=7,
    )
    assert np.isfinite(result["moran_i"])
    assert result["spatial_units"] == 4
    assert 0 < result["permutation_p_value_two_sided"] <= 1
