from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from entropy_crime_bike.panel import (
    _build_city_panel,
    _grid_id,
    _grid_key,
    assign_projected_grid,
    holiday_dates,
)


class IdentityTransformer:
    def transform(self, longitude, latitude):
        return np.asarray(longitude), np.asarray(latitude)


def _minimal_e02() -> dict[str, object]:
    return {
        "date_range": {"start": "2020-01-01", "end": "2020-01-03"},
        "validation": {
            "train_end": "2020-01-01",
            "validation_end": "2020-01-02",
        },
        "parquet": {
            "compression": "zstd",
            "compression_level": 3,
            "row_group_size": 100,
        },
    }


def test_grid_assignment_has_deterministic_half_open_cells():
    x_index, y_index, valid = assign_projected_grid(
        pd.Series([0.10, 0.99, 1.00]),
        pd.Series([0.10, 0.99, 1.00]),
        IdentityTransformer(),
        1,
    )
    assert valid.tolist() == [True, True, True]
    assert x_index.tolist() == [0, 0, 1]
    assert y_index.tolist() == [0, 0, 1]
    assert _grid_key(x_index, y_index).tolist() == [0, 0, 10_000_001]
    assert _grid_id("DC", 1000, 323, 4307) == "DC_R1000_E323_N4307"


def test_grid_assignment_rejects_missing_and_out_of_range_coordinates():
    _, _, valid = assign_projected_grid(
        pd.Series([np.nan, 181.0, 0.0]),
        pd.Series([0.0, 0.0, 91.0]),
        IdentityTransformer(),
        1,
    )
    assert valid.tolist() == [False, False, False]


def test_holiday_calendars_cover_known_us_and_vancouver_dates():
    us = holiday_dates("DC", date(2020, 1, 1), date(2022, 12, 31))
    van = holiday_dates("VAN", date(2020, 1, 1), date(2022, 12, 31))
    assert date(2021, 7, 4) in us
    assert date(2021, 7, 5) in us
    assert date(2022, 2, 21) in van
    assert date(2022, 7, 1) in van


def test_panel_zero_fill_and_lags_are_strictly_previous_day(tmp_path):
    reference = pd.DataFrame(
        {
            "city": ["DC", "DC"],
            "grid_id": ["DC_R1_E0_N0", "DC_R1_E1_N1"],
            "x_index": [0, 1],
            "y_index": [0, 1],
            "centroid_longitude": [0.5, 1.5],
            "centroid_latitude": [0.5, 1.5],
            "bike_coverage_training": [True, False],
            "bike_coverage_ever": [True, False],
        }
    )
    crime = pd.DataFrame(
        {
            "city": ["DC"],
            "date": [pd.Timestamp("2020-01-01")],
            "x_index": [0],
            "y_index": [0],
            "grid_id": ["DC_R1_E0_N0"],
            "crime_count_property_theft": [2],
            "crime_count_vehicle_theft": [1],
            "crime_count_burglary": [0],
            "crime_count_all": [3],
        }
    )
    flow = pd.DataFrame(
        {
            "city": ["DC"],
            "date": [pd.Timestamp("2020-01-02")],
            "x_index": [0],
            "y_index": [0],
            "grid_id": ["DC_R1_E0_N0"],
            "bike_outflow": [2],
            "bike_inflow": [1],
            "bike_total_flow": [3],
            "bike_net_flow": [-1],
        }
    )
    trips = pd.DataFrame(
        {
            "city": ["DC"],
            "date": [pd.Timestamp("2020-01-02")],
            "bike_trips_any_endpoint": [2],
        }
    )
    panel, monthly, _, _, _ = _build_city_panel(
        "DC",
        reference,
        crime,
        flow,
        trips,
        _minimal_e02(),
        tmp_path,
        type("Logger", (), {"info": lambda *args, **kwargs: None})(),
    )
    assert len(panel) == 6
    assert int(panel["crime_count_all"].sum()) == 3
    assert int(panel["bike_total_flow"].sum()) == 3
    first_grid = panel.loc[panel["grid_id"].eq("DC_R1_E0_N0")].sort_values(
        "date"
    )
    assert pd.isna(first_grid.iloc[0]["crime_count_lag1"])
    assert first_grid.iloc[1]["crime_count_lag1"] == 3
    assert first_grid.iloc[2]["bike_total_flow_lag1"] == 3
    assert int(monthly["bike_trips_any_endpoint"].sum()) == 2
