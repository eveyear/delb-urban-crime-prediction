import numpy as np
import pandas as pd

from entropy_crime_bike.e09_robustness import (
    apply_flow_bins,
    crime_bins,
    fit_flow_cut_points,
    jeffreys_conditional_pair,
)


def test_coarse_crime_bins_are_prespecified() -> None:
    result = crime_bins(pd.Series([0, 1, 2, 8]), "zero_one_two_plus")
    assert list(result.astype(str)) == ["zero", "one", "two_plus", "two_plus"]


def test_generic_flow_bins_support_median_and_quintiles() -> None:
    values = pd.Series([0, 1, 2, 3, 4, 5, 6])
    median = fit_flow_cut_points(values, "zero_plus_positive_median")
    mapped = apply_flow_bins(values, median)
    assert len(mapped.categories) == 3
    quintiles = fit_flow_cut_points(values, "zero_plus_positive_quintiles")
    mapped_quintiles = apply_flow_bins(values, quintiles)
    assert len(mapped_quintiles.categories) == 6
    assert str(mapped_quintiles[0]) == "zero"


def test_jeffreys_conditional_pair_is_coherent() -> None:
    frame = pd.DataFrame(
        {
            "y": [0, 0, 1, 1, 2, 2],
            "s": ["a", "a", "a", "b", "b", "b"],
            "b": ["x", "x", "y", "x", "y", "y"],
        }
    )
    h0, hb = jeffreys_conditional_pair(
        frame,
        target="y",
        baseline_state=["s"],
        bicycle_state_field="b",
        alpha=0.5,
    )
    assert np.isfinite([h0, hb]).all()
    assert h0 >= hb - 1.0e-12


def test_jeffreys_entropy_vanishes_toward_deterministic_large_sample() -> None:
    frame = pd.DataFrame(
        {"y": [0] * 10000, "s": ["a"] * 10000, "b": ["x"] * 10000}
    )
    h0, hb = jeffreys_conditional_pair(
        frame,
        target="y",
        baseline_state=["s"],
        bicycle_state_field="b",
        alpha=0.5,
    )
    assert h0 == 0.0
    assert hb == 0.0
