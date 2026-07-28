from __future__ import annotations

import numpy as np
import pandas as pd

from entropy_crime_bike.e16_delb_estimation import (
    SupportCodebook,
    baseline_state_fields,
    block_weight_matrix,
    bound_metrics,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 2, 2],
            "z": ["a", "a", "a", "b", "b", "b"],
            "b": ["x", "x", "y", "x", "y", "y"],
            "block": ["D7_001", "D7_001", "D7_002", "D7_002", "D7_003", "D7_003"],
        }
    )


def test_support_codebook_reports_singletons_and_effective_df():
    codebook = SupportCodebook.from_frame(
        _frame(), target="y", baseline_state=["z"], bicycle_state="b", block_field="block"
    )
    result = codebook.diagnose(np.ones(3)).iloc[0]
    assert result["support_atoms"] == 5
    assert result["singleton_atom_count"] == 4
    assert np.isclose(result["singleton_observation_share"], 4 / 6)
    assert result["effective_df"] == 2


def test_block_weight_matrix_uses_frozen_membership():
    summary = pd.DataFrame(
        {"replicate": [1, 1, 0], "fraction": [0.2, 0.4, 1.0]}
    )
    membership = pd.DataFrame(
        {
            "replicate": [1, 1, 1, 0, 0, 0],
            "fraction": [0.2, 0.4, 0.4, 1.0, 1.0, 1.0],
            "block_id": ["a", "a", "b", "a", "b", "c"],
        }
    )
    weights, metadata = block_weight_matrix(["a", "b", "c"], summary, membership)
    assert metadata[["replicate", "fraction"]].values.tolist() == [[0, 1.0], [1, 0.2], [1, 0.4]]
    assert weights.tolist() == [[1, 1, 1], [1, 0, 0], [1, 1, 0]]


def test_bound_metrics_projects_only_for_bound_and_preserves_order():
    result = bound_metrics(0.5, 0.7, tail=1e-15, root=1e-12)
    assert result["projection_applied"]
    assert result["hb_for_bound_bits"] == result["h0_for_bound_bits"]
    assert result["delta_l_exact_mse"] == 0
    assert result["lb_exact_mse"] <= result["l0_exact_mse"]


def test_temporal_information_states_are_scale_specific():
    assert "day_of_week" in baseline_state_fields("day")
    assert "contains_holiday" in baseline_state_fields("week")
    assert "day_of_week" not in baseline_state_fields("week")
