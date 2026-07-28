from __future__ import annotations

import pandas as pd

from entropy_crime_bike.e16_sample_indices import (
    _nested_check,
    balanced_stratified_order,
    expanding_windows,
    make_block_universe,
    sample_prefixes,
)


def test_daily_and_weekly_block_universes_have_frozen_sizes():
    daily, daily_map = make_block_universe(
        pd.date_range("2020-01-01", "2022-12-31", freq="D"), temporal="day"
    )
    weekly, weekly_map = make_block_universe(
        pd.date_range("2020-01-06", periods=155, freq="7D"), temporal="week"
    )
    assert len(daily) == 156
    assert int((~daily_map["included_in_primary_universe"]).sum()) == 4
    assert len(weekly) == 38
    assert int((~weekly_map["included_in_primary_universe"]).sum()) == 3


def test_balanced_order_is_seed_reproducible():
    blocks, _ = make_block_universe(
        pd.date_range("2020-01-01", periods=70, freq="D"), temporal="day"
    )
    first = balanced_stratified_order(blocks, seed=42)
    second = balanced_stratified_order(blocks, seed=42)
    assert first["block_id"].tolist() == second["block_id"].tolist()
    assert sorted(first["selection_rank"].tolist()) == list(range(1, 11))


def test_prefix_samples_are_exact_and_nested():
    blocks, _ = make_block_universe(
        pd.date_range("2020-01-01", periods=140, freq="D"), temporal="day"
    )
    membership, summary = sample_prefixes(
        blocks, fractions=[0.2, 0.4, 0.6, 0.8], seed=8
    )
    assert summary["selected_blocks"].tolist() == [4, 8, 12, 16]
    assert _nested_check(membership, [0.2, 0.4, 0.6, 0.8])


def test_expanding_windows_are_chronological_and_nested():
    result = expanding_windows(
        pd.date_range("2020-01-01", periods=100, freq="D"),
        [0.2, 0.4, 0.6, 0.8, 1.0],
        temporal="day",
    )
    assert result["selected_periods"].tolist() == [20, 40, 60, 80, 100]
    assert result["window_start"].nunique() == 1
    assert result["window_end"].is_monotonic_increasing

