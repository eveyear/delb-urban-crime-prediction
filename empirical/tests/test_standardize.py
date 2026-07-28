from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from entropy_crime_bike.standardize import (
    StationResolver,
    _combine_flags,
    _extract_station_fields,
    _haversine_m,
    _normal_column_map,
    _source_period,
    _station_key_series,
)


def test_source_period_supports_all_source_names() -> None:
    assert _source_period(Path("2020_DC.csv")) == "2020"
    assert _source_period(Path("202001-capitalbikeshare-tripdata.zip")) == "2020-01"
    assert _source_period(Path("Mobi_System_Data_2022-12.csv")) == "2022-12"


def test_vancouver_station_parser_preserves_four_digit_id() -> None:
    frame = pd.DataFrame(
        {
            "Departure station": [
                "0026 Beatty & Robson",
                "0981 Workshop - Service Complete",
                None,
            ]
        }
    )
    station_id, station_name, latitude, longitude = _extract_station_fields(
        frame,
        _normal_column_map(frame.columns),
        "start",
        "VAN",
    )
    assert station_id.tolist()[:2] == ["0026", "0981"]
    assert station_name.tolist()[:2] == [
        "Beatty & Robson",
        "Workshop - Service Complete",
    ]
    assert latitude.isna().all()
    assert longitude.isna().all()


def test_station_key_prefers_identifier() -> None:
    station_ids = pd.Series(["31623", None, ""])
    station_names = pd.Series(
        ["Columbus Circle / Union Station", "Unique Name", None]
    )
    result = _station_key_series(station_ids, station_names)
    assert result.iloc[0] == "I:31623"
    assert result.iloc[1] == "N:unique name"
    assert pd.isna(result.iloc[2])


def test_flag_combination_is_ordered_and_pipe_delimited() -> None:
    index = pd.RangeIndex(3)
    flags = _combine_flags(
        index,
        [
            ("first", pd.Series([True, False, True])),
            ("second", pd.Series([True, True, False])),
        ],
    )
    assert flags.tolist() == ["first|second", "second", "first"]


def test_haversine_detects_station_movement() -> None:
    assert _haversine_m(49.28, -123.12, 49.28, -123.12) == 0
    assert _haversine_m(49.28, -123.12, 49.281, -123.12) > 100


def test_station_resolver_priority_and_non_public_classification(
    tmp_path: Path,
) -> None:
    current = tmp_path / "station_information_current.csv"
    current.write_text(
        "station_id,name,lat,lon\n"
        "0026,Beatty & Robson,49.27744,-123.11432\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    historical = pd.DataFrame(
        {
            "city": ["DC"],
            "source_month": ["2020-04"],
            "station_key": ["I:31623"],
            "station_id": ["31623"],
            "station_name": ["Columbus Circle / Union Station"],
            "normalized_name": ["columbus circle / union station"],
            "latitude": [38.89696],
            "longitude": [-77.00493],
            "observations": [10],
            "episode_id": [1],
            "movement_from_previous_m": [np.nan],
        }
    )
    resolver = StationResolver(
        historical=historical,
        vancouver_current_path=current,
        vancouver_historical_path=None,
        expected_current_hash=digest,
        non_public_patterns=["workshop", "yard", "temporary", "valet"],
    )

    station_ids = pd.Series(["0026", "0981", "9998"])
    station_names = pd.Series(
        ["Beatty & Robson", "Workshop - Service Complete", "Unknown"]
    )
    latitude = pd.Series([np.nan, np.nan, np.nan])
    longitude = pd.Series([np.nan, np.nan, np.nan])
    lat, lon, source, eligible = resolver.resolve(
        "VAN",
        "2020-01",
        station_ids,
        station_names,
        latitude,
        longitude,
        "start",
    )
    assert lat.iloc[0] == 49.27744
    assert lon.iloc[0] == -123.11432
    assert source.tolist() == [
        "current_snapshot_retrofit",
        "non_public_node",
        "unresolved",
    ]
    assert eligible.tolist() == [True, False, False]

    dc_lat, dc_lon, dc_source, dc_eligible = resolver.resolve(
        "DC",
        "2020-01",
        pd.Series(["31623"]),
        pd.Series(["Columbus Circle / Union Station"]),
        pd.Series([np.nan]),
        pd.Series([np.nan]),
        "start",
    )
    assert dc_lat.iloc[0] == 38.89696
    assert dc_lon.iloc[0] == -77.00493
    assert dc_source.iloc[0] == "historical_trip_station_nearest_month"
    assert dc_eligible.iloc[0]

    _, _, ny_source, ny_eligible = resolver.resolve(
        "NY",
        "2020-01",
        pd.Series(["5351.03", "SYS030"]),
        pd.Series(["Bayard St & Baxter St", "Morgan Tech Shop parts testing"]),
        pd.Series([40.71, np.nan]),
        pd.Series([-74.0, np.nan]),
        "start",
    )
    assert ny_source.tolist() == ["source", "non_public_node"]
    assert ny_eligible.tolist() == [True, False]

    _, _, dc_source_name, dc_eligible_name = resolver.resolve(
        "DC",
        "2022-01",
        pd.Series(["32218"]),
        pd.Series(["Temporary Rd & Old Reston Ave"]),
        pd.Series([38.95]),
        pd.Series([-77.35]),
        "start",
    )
    assert dc_source_name.iloc[0] == "source"
    assert dc_eligible_name.iloc[0]
