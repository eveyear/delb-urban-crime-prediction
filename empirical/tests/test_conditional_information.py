from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entropy_crime_bike.conditional_information import (
    ConditionalCodebook,
    apply_bicycle_bins,
    crime_lag_bins,
    fit_positive_bicycle_quantiles,
)


def _estimate(frame: pd.DataFrame) -> pd.Series:
    codebook = ConditionalCodebook.from_frame(
        frame,
        target="x",
        baseline_state=["grid", "lag"],
        bicycle_state_field="bike",
        block_field="block",
    )
    return codebook.estimate(np.ones(codebook.block_count)).iloc[0]


def test_crime_lag_bins() -> None:
    result = crime_lag_bins(pd.Series([0, 1, 2, 3, 9]))
    assert list(result.astype(str)) == [
        "zero",
        "one",
        "two",
        "three_plus",
        "three_plus",
    ]


def test_training_bicycle_bins() -> None:
    values = pd.Series([0, 1, 2, 3, 4, 5, 6])
    cuts = fit_positive_bicycle_quantiles(values, [1 / 3, 2 / 3])
    assert cuts == pytest.approx((2.6666666667, 4.3333333333))
    bins = apply_bicycle_bins(pd.Series([0, 1, 3, 5]), cuts)
    assert list(bins.astype(str)) == ["zero", "low", "medium", "high"]


def test_perfect_bicycle_information_reduces_entropy_to_zero() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1] * 100,
            "grid": ["g"] * 200,
            "lag": ["l"] * 200,
            "bike": ["b0", "b1"] * 100,
            "block": np.repeat(np.arange(20), 10),
        }
    )
    estimate = _estimate(frame)
    assert estimate["h0_plugin_bits"] == pytest.approx(1.0)
    assert estimate["hb_plugin_bits"] == pytest.approx(0.0)
    assert estimate["delta_h_plugin_raw_bits"] == pytest.approx(1.0)


def test_irrelevant_bicycle_state_leaves_entropy_unchanged() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1] * 100,
            "grid": ["g"] * 200,
            "lag": ["l"] * 200,
            "bike": ["constant"] * 200,
            "block": np.repeat(np.arange(20), 10),
        }
    )
    estimate = _estimate(frame)
    assert estimate["h0_plugin_bits"] == pytest.approx(
        estimate["hb_plugin_bits"]
    )
    assert estimate["delta_h_plugin_raw_bits"] == pytest.approx(0.0)


def test_missing_state_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1],
            "grid": ["g", "g"],
            "lag": ["l", None],
            "bike": ["b0", "b1"],
            "block": [0, 0],
        }
    )
    with pytest.raises(ValueError):
        _estimate(frame)
