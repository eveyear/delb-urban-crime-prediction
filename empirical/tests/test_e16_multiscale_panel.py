from __future__ import annotations

import pandas as pd

from entropy_crime_bike.e16_multiscale_panel import (
    _map_scale,
    complete_nested_core,
    full_iso_week_bounds,
    weekly_panel,
)


def test_complete_nested_core_keeps_only_full_two_km_parents():
    reference = pd.DataFrame(
        {
            "city": ["DC"] * 5,
            "x_index": [0, 0, 1, 1, 2],
            "y_index": [0, 1, 0, 1, 0],
        }
    )
    core = complete_nested_core(reference)
    assert len(core) == 16
    assert core["parent_2km_grid_id"].nunique() == 1
    assert core["parent_1km_grid_id"].nunique() == 4
    assert core["grid_500m_id"].nunique() == 16


def test_full_iso_week_bounds_exclude_partial_boundary_weeks():
    first, last = full_iso_week_bounds("2020-01-01", "2022-12-31")
    assert first == pd.Timestamp("2020-01-06")
    assert last == pd.Timestamp("2022-12-19")
    assert ((last - first).days // 7) + 1 == 155


def test_scale_mapping_conserves_counts():
    frame = pd.DataFrame(
        {
            "city": ["DC"] * 4,
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "x_index": [0, 1, 0, 1],
            "y_index": [0, 0, 1, 1],
            "crime_count_all": [1, 2, 3, 4],
        }
    )
    result = _map_scale(frame, 1000, ["crime_count_all"])
    assert len(result) == 1
    assert int(result["crime_count_all"].sum()) == 10


def test_weekly_panel_requires_and_sums_complete_weeks():
    dates = pd.date_range("2020-01-01", "2020-01-19", freq="D")
    daily = pd.DataFrame(
        {
            "city": "DC",
            "grid_id": "DC_R500_E0_N0",
            "x_index": 0,
            "y_index": 0,
            "date": dates,
            "centroid_longitude": -77.0,
            "centroid_latitude": 38.9,
            "bike_coverage_training": True,
            "bike_coverage_ever": True,
            "is_holiday": False,
            **{column: 1 for column in [
                "crime_count_property_theft", "crime_count_vehicle_theft",
                "crime_count_burglary", "crime_count_all", "bike_outflow",
                "bike_inflow", "bike_total_flow", "bike_net_flow"
            ]},
        }
    )
    e02 = {
        "date_range": {"start": "2020-01-01", "end": "2020-01-19"},
        "validation": {"train_end": "2020-01-31", "validation_end": "2020-02-29"},
    }
    result = weekly_panel(daily, e02)
    assert len(result) == 2
    assert result["days_in_period"].eq(7).all()
    assert result["crime_count_all"].tolist() == [7, 7]
    assert pd.isna(result.iloc[0]["crime_count_lag1"])
    assert result.iloc[1]["crime_count_lag1"] == 7

