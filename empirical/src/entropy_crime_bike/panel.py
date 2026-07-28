from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pandas.tseries.holiday import USFederalHolidayCalendar
from pyproj import Transformer

from entropy_crime_bike.standardize import _load_yaml, _write_environment


CITY_ORDER = ["DC", "NY", "VAN"]
GRID_KEY_MULTIPLIER = 10_000_000


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e02_panel")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e02_panel.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _atomic_parquet(
    frame: pd.DataFrame,
    path: Path,
    compression: str,
    compression_level: int,
    row_group_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(
        temporary,
        engine="pyarrow",
        compression=compression,
        compression_level=compression_level,
        index=False,
        row_group_size=row_group_size,
    )
    temporary.replace(path)


def _city_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("city="):
            return part.split("=", 1)[1]
    raise ValueError(f"Missing city partition in {path}")


def _year_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("year="):
            return int(part.split("=", 1)[1])
    raise ValueError(f"Missing year partition in {path}")


def _grid_key(x_index: np.ndarray, y_index: np.ndarray) -> np.ndarray:
    return (
        x_index.astype(np.int64) * GRID_KEY_MULTIPLIER
        + y_index.astype(np.int64)
    )


def _grid_id(
    city: str, resolution: int, x_index: int, y_index: int
) -> str:
    return f"{city}_R{resolution}_E{x_index}_N{y_index}"


def assign_projected_grid(
    longitude: pd.Series | np.ndarray,
    latitude: pd.Series | np.ndarray,
    transformer: Transformer,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude_values = pd.to_numeric(
        pd.Series(longitude), errors="coerce"
    ).to_numpy(dtype=float)
    latitude_values = pd.to_numeric(
        pd.Series(latitude), errors="coerce"
    ).to_numpy(dtype=float)
    valid = (
        np.isfinite(longitude_values)
        & np.isfinite(latitude_values)
        & (longitude_values >= -180)
        & (longitude_values <= 180)
        & (latitude_values >= -90)
        & (latitude_values <= 90)
    )
    x_coordinate = np.full(len(longitude_values), np.nan)
    y_coordinate = np.full(len(latitude_values), np.nan)
    if valid.any():
        x_valid, y_valid = transformer.transform(
            longitude_values[valid], latitude_values[valid]
        )
        x_coordinate[valid] = x_valid
        y_coordinate[valid] = y_valid
    valid &= np.isfinite(x_coordinate) & np.isfinite(y_coordinate)
    x_index = np.full(len(longitude_values), np.iinfo(np.int32).min)
    y_index = np.full(len(latitude_values), np.iinfo(np.int32).min)
    x_index[valid] = np.floor(
        x_coordinate[valid] / resolution
    ).astype(np.int32)
    y_index[valid] = np.floor(
        y_coordinate[valid] / resolution
    ).astype(np.int32)
    return x_index, y_index, valid


def _crime_column_name(crime_type: str) -> str:
    return {
        "PROPERTY_THEFT": "crime_count_property_theft",
        "VEHICLE_THEFT": "crime_count_vehicle_theft",
        "BURGLARY": "crime_count_burglary",
    }[crime_type]


def _aggregate_crime(
    e01_data: Path,
    selected_cities: list[str],
    cities_config: dict[str, object],
    e02: dict[str, object],
    run_dir: Path,
    data_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[int]]]:
    resolution = int(e02["spatial"]["resolution_m"])
    training_start = pd.Timestamp(e02["domain"]["training_start"])
    training_end = pd.Timestamp(e02["domain"]["training_end"])
    compression = str(e02["parquet"]["compression"])
    compression_level = int(e02["parquet"]["compression_level"])
    row_group_size = int(e02["parquet"]["row_group_size"])
    records: list[pd.DataFrame] = []
    files = [
        path
        for path in sorted((e01_data / "crime_events").rglob("*.parquet"))
        if _city_from_path(path) in selected_cities
    ]
    for index, path in enumerate(files, start=1):
        city = _city_from_path(path)
        projected_crs = cities_config["cities"][city]["projected_crs"]
        transformer = Transformer.from_crs(
            "EPSG:4326", projected_crs, always_xy=True
        )
        frame = pq.read_table(
            path,
            columns=[
                "event_id",
                "city",
                "event_date",
                "crime_type_unified",
                "longitude",
                "latitude",
            ],
        ).to_pandas()
        x_index, y_index, valid = assign_projected_grid(
            frame["longitude"],
            frame["latitude"],
            transformer,
            resolution,
        )
        if not valid.all():
            raise AssertionError(
                f"Crime coordinates failed projection in {path}: "
                f"{int((~valid).sum())}"
            )
        frame["event_date"] = pd.to_datetime(frame["event_date"])
        frame["x_index"] = x_index
        frame["y_index"] = y_index
        frame["grid_key"] = _grid_key(x_index, y_index)
        frame["grid_id"] = [
            _grid_id(city, resolution, int(x_value), int(y_value))
            for x_value, y_value in zip(x_index, y_index)
        ]
        records.append(
            frame[
                [
                    "event_id",
                    "city",
                    "event_date",
                    "crime_type_unified",
                    "longitude",
                    "latitude",
                    "x_index",
                    "y_index",
                    "grid_key",
                    "grid_id",
                ]
            ]
        )
        logger.info(
            "Crime grid assignment %d/%d: %s (%d rows)",
            index,
            len(files),
            path.name,
            len(frame),
        )

    assignments = pd.concat(records, ignore_index=True)
    active_keys: dict[str, set[int]] = {}
    assignments["in_domain_training_window"] = assignments[
        "event_date"
    ].between(training_start, training_end)
    assignments["in_primary_domain"] = False
    for city in selected_cities:
        city_training = assignments.loc[
            assignments["city"].eq(city)
            & assignments["in_domain_training_window"],
            "grid_key",
        ]
        active_keys[city] = set(city_training.astype(np.int64).unique())
        city_mask = assignments["city"].eq(city)
        assignments.loc[city_mask, "in_primary_domain"] = (
            assignments.loc[city_mask, "grid_key"]
            .astype(np.int64)
            .isin(active_keys[city])
        )
        logger.info(
            "%s primary domain: %d training crime-occupied 1 km grids",
            city,
            len(active_keys[city]),
        )

    for (city, year), partition in assignments.groupby(
        ["city", assignments["event_date"].dt.year], sort=True
    ):
        output = (
            data_dir
            / "crime_grid_assignments"
            / f"city={city}"
            / f"year={int(year)}"
            / "part-crime-grid-assignments.parquet"
        )
        _atomic_parquet(
            partition.reset_index(drop=True),
            output,
            compression,
            compression_level,
            row_group_size,
        )

    inside = assignments.loc[assignments["in_primary_domain"]].copy()
    grouped = (
        inside.groupby(
            [
                "city",
                "event_date",
                "x_index",
                "y_index",
                "grid_id",
                "crime_type_unified",
            ],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["city", "event_date", "x_index", "y_index", "grid_id"],
        columns="crime_type_unified",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={"event_date": "date"}, inplace=True)
    for crime_type in e02["panel"]["crime_categories"]:
        if crime_type not in pivot:
            pivot[crime_type] = 0
        pivot.rename(
            columns={crime_type: _crime_column_name(crime_type)},
            inplace=True,
        )
    crime_columns = [
        _crime_column_name(value)
        for value in e02["panel"]["crime_categories"]
    ]
    pivot["crime_count_all"] = pivot[crime_columns].sum(axis=1)
    for column in [*crime_columns, "crime_count_all"]:
        pivot[column] = pivot[column].astype(np.int32)

    outside = assignments.loc[~assignments["in_primary_domain"]]
    outside_summary = (
        outside.groupby(
            [
                "city",
                "event_date",
                "grid_id",
                "crime_type_unified",
            ],
            observed=True,
        )
        .size()
        .rename("event_rows")
        .reset_index()
    )
    outside_summary.to_csv(
        run_dir / "tables" / "crime_outside_primary_domain.csv",
        index=False,
    )
    return assignments, pivot, active_keys


def _coordinate_counter_key(
    endpoint: str, source: object, disposition: str
) -> tuple[str, str, str]:
    return endpoint, str(source or "missing"), disposition


def _aggregate_bicycle_city(
    city: str,
    e01_data: Path,
    active_keys: set[int],
    cities_config: dict[str, object],
    e02: dict[str, object],
    run_dir: Path,
    data_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolution = int(e02["spatial"]["resolution_m"])
    date_start = pd.Timestamp(e02["date_range"]["start"])
    date_end = pd.Timestamp(e02["date_range"]["end"])
    projected_crs = cities_config["cities"][city]["projected_crs"]
    transformer = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )
    active_array = np.array(sorted(active_keys), dtype=np.int64)
    compression = str(e02["parquet"]["compression"])
    compression_level = int(e02["parquet"]["compression_level"])
    row_group_size = int(e02["parquet"]["row_group_size"])
    state_file = run_dir / "state" / f"{city}_bike_complete.json"
    checkpoint_dir = data_dir / "checkpoints"
    checkpoint_paths = {
        "flow": checkpoint_dir / f"{city}_bike_flow_grid_day.parquet",
        "trips": checkpoint_dir / f"{city}_bike_trip_daily.parquet",
        "reconciliation": checkpoint_dir
        / f"{city}_bike_reconciliation.parquet",
        "coordinate": checkpoint_dir
        / f"{city}_coordinate_domain_summary.parquet",
    }
    if state_file.exists() and all(path.exists() for path in checkpoint_paths.values()):
        logger.info("Loading completed bicycle checkpoint for %s", city)
        return tuple(
            pd.read_parquet(checkpoint_paths[name])
            for name in ["flow", "trips", "reconciliation", "coordinate"]
        )

    files = sorted(
        (e01_data / "bicycle_trips" / f"city={city}").rglob("*.parquet")
    )
    flow_parts: list[pd.DataFrame] = []
    trip_parts: list[pd.DataFrame] = []
    reconciliation_records: list[dict[str, object]] = []
    coordinate_counts: Counter[tuple[str, str, str]] = Counter()
    columns = [
        "start_date",
        "end_date",
        "start_longitude",
        "start_latitude",
        "start_coordinate_source",
        "start_flow_eligible",
        "end_longitude",
        "end_latitude",
        "end_coordinate_source",
        "end_flow_eligible",
    ]
    for file_index, path in enumerate(files, start=1):
        file_counts: Counter[str] = Counter()
        file_flow_parts: list[pd.DataFrame] = []
        file_trip_parts: list[pd.DataFrame] = []
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=row_group_size, columns=columns
        ):
            frame = batch.to_pandas()
            rows = len(frame)
            file_counts["trip_rows"] += rows
            start_date = pd.to_datetime(frame["start_date"], errors="coerce")
            end_date = pd.to_datetime(frame["end_date"], errors="coerce")
            start_e01 = (
                frame["start_flow_eligible"].fillna(False).to_numpy(bool)
            )
            end_e01 = (
                frame["end_flow_eligible"].fillna(False).to_numpy(bool)
            )
            start_period = (
                start_date.between(date_start, date_end).to_numpy()
            )
            end_period = end_date.between(date_start, date_end).to_numpy()
            start_candidate = start_e01 & start_period
            end_candidate = end_e01 & end_period

            start_x = np.full(rows, np.iinfo(np.int32).min, dtype=np.int32)
            start_y = np.full(rows, np.iinfo(np.int32).min, dtype=np.int32)
            start_valid = np.zeros(rows, dtype=bool)
            if start_candidate.any():
                candidate_index = np.flatnonzero(start_candidate)
                x_values, y_values, valid_values = assign_projected_grid(
                    frame.loc[
                        start_candidate, "start_longitude"
                    ].reset_index(drop=True),
                    frame.loc[
                        start_candidate, "start_latitude"
                    ].reset_index(drop=True),
                    transformer,
                    resolution,
                )
                start_x[candidate_index] = x_values
                start_y[candidate_index] = y_values
                start_valid[candidate_index] = valid_values
            start_keys = _grid_key(start_x, start_y)
            start_inside = (
                start_candidate
                & start_valid
                & np.isin(start_keys, active_array)
            )

            end_x = np.full(rows, np.iinfo(np.int32).min, dtype=np.int32)
            end_y = np.full(rows, np.iinfo(np.int32).min, dtype=np.int32)
            end_valid = np.zeros(rows, dtype=bool)
            if end_candidate.any():
                candidate_index = np.flatnonzero(end_candidate)
                x_values, y_values, valid_values = assign_projected_grid(
                    frame.loc[
                        end_candidate, "end_longitude"
                    ].reset_index(drop=True),
                    frame.loc[
                        end_candidate, "end_latitude"
                    ].reset_index(drop=True),
                    transformer,
                    resolution,
                )
                end_x[candidate_index] = x_values
                end_y[candidate_index] = y_values
                end_valid[candidate_index] = valid_values
            end_keys = _grid_key(end_x, end_y)
            end_inside = (
                end_candidate
                & end_valid
                & np.isin(end_keys, active_array)
            )

            file_counts["start_e01_flow_eligible"] += int(start_e01.sum())
            file_counts["end_e01_flow_eligible"] += int(end_e01.sum())
            file_counts["start_outside_panel_period"] += int(
                (start_e01 & ~start_period).sum()
            )
            file_counts["end_outside_panel_period"] += int(
                (end_e01 & ~end_period).sum()
            )
            file_counts["start_in_primary_domain"] += int(
                start_inside.sum()
            )
            file_counts["end_in_primary_domain"] += int(end_inside.sum())
            file_counts["start_outside_primary_domain"] += int(
                (start_candidate & ~start_inside).sum()
            )
            file_counts["end_outside_primary_domain"] += int(
                (end_candidate & ~end_inside).sum()
            )
            any_inside = start_inside | end_inside
            file_counts["trips_any_endpoint_in_primary_domain"] += int(
                any_inside.sum()
            )

            for endpoint, source_values, e01_mask, inside_mask, period_mask in [
                (
                    "start",
                    frame["start_coordinate_source"],
                    start_e01,
                    start_inside,
                    start_period,
                ),
                (
                    "end",
                    frame["end_coordinate_source"],
                    end_e01,
                    end_inside,
                    end_period,
                ),
            ]:
                source_array = source_values.fillna("missing").astype(str)
                dispositions = np.full(rows, "not_flow_eligible", dtype=object)
                dispositions[e01_mask & ~period_mask] = "outside_panel_period"
                dispositions[e01_mask & period_mask] = "outside_primary_domain"
                dispositions[inside_mask] = "in_primary_domain"
                coordinate_frame = pd.DataFrame(
                    {
                        "source": source_array,
                        "disposition": dispositions,
                    }
                )
                counts = coordinate_frame.value_counts()
                for (source, disposition), count in counts.items():
                    coordinate_counts[
                        _coordinate_counter_key(
                            endpoint, source, disposition
                        )
                    ] += int(count)

            if start_inside.any():
                start_flow = pd.DataFrame(
                    {
                        "city": city,
                        "date": start_date[start_inside].to_numpy(),
                        "x_index": start_x[start_inside],
                        "y_index": start_y[start_inside],
                        "bike_outflow": 1,
                        "bike_inflow": 0,
                    }
                )
                file_flow_parts.append(
                    start_flow.groupby(
                        ["city", "date", "x_index", "y_index"],
                        as_index=False,
                    )[["bike_outflow", "bike_inflow"]].sum()
                )
            if end_inside.any():
                end_flow = pd.DataFrame(
                    {
                        "city": city,
                        "date": end_date[end_inside].to_numpy(),
                        "x_index": end_x[end_inside],
                        "y_index": end_y[end_inside],
                        "bike_outflow": 0,
                        "bike_inflow": 1,
                    }
                )
                file_flow_parts.append(
                    end_flow.groupby(
                        ["city", "date", "x_index", "y_index"],
                        as_index=False,
                    )[["bike_outflow", "bike_inflow"]].sum()
                )
            if any_inside.any():
                trip_frame = pd.DataFrame(
                    {
                        "city": city,
                        "date": start_date[any_inside].to_numpy(),
                        "bike_trips_any_endpoint": 1,
                    }
                )
                file_trip_parts.append(
                    trip_frame.groupby(
                        ["city", "date"], as_index=False
                    )["bike_trips_any_endpoint"].sum()
                )

        if file_flow_parts:
            flow_parts.append(
                pd.concat(file_flow_parts, ignore_index=True)
                .groupby(
                    ["city", "date", "x_index", "y_index"],
                    as_index=False,
                )[["bike_outflow", "bike_inflow"]]
                .sum()
            )
        if file_trip_parts:
            trip_parts.append(
                pd.concat(file_trip_parts, ignore_index=True)
                .groupby(["city", "date"], as_index=False)[
                    "bike_trips_any_endpoint"
                ]
                .sum()
            )
        reconciliation_records.append(
            {
                "city": city,
                "file": str(path),
                "source_year": _year_from_path(path),
                **file_counts,
            }
        )
        if (
            file_index % 6 == 0
            or file_index == len(files)
            or file_index == 1
        ):
            logger.info(
                "%s bicycle spatial aggregation %d/%d files",
                city,
                file_index,
                len(files),
            )

    flow = (
        pd.concat(flow_parts, ignore_index=True)
        .groupby(
            ["city", "date", "x_index", "y_index"], as_index=False
        )[["bike_outflow", "bike_inflow"]]
        .sum()
    )
    flow["date"] = pd.to_datetime(flow["date"])
    flow["bike_outflow"] = flow["bike_outflow"].astype(np.int32)
    flow["bike_inflow"] = flow["bike_inflow"].astype(np.int32)
    flow["bike_total_flow"] = (
        flow["bike_outflow"] + flow["bike_inflow"]
    ).astype(np.int32)
    flow["bike_net_flow"] = (
        flow["bike_inflow"] - flow["bike_outflow"]
    ).astype(np.int32)
    flow["grid_id"] = [
        _grid_id(city, resolution, int(x_value), int(y_value))
        for x_value, y_value in zip(flow["x_index"], flow["y_index"])
    ]
    trips = (
        pd.concat(trip_parts, ignore_index=True)
        .groupby(["city", "date"], as_index=False)[
            "bike_trips_any_endpoint"
        ]
        .sum()
    )
    trips["date"] = pd.to_datetime(trips["date"])
    trips["bike_trips_any_endpoint"] = trips[
        "bike_trips_any_endpoint"
    ].astype(np.int32)
    reconciliation = pd.DataFrame(reconciliation_records)
    coordinate_summary = pd.DataFrame(
        [
            {
                "city": city,
                "endpoint": endpoint,
                "coordinate_source": source,
                "disposition": disposition,
                "rows": count,
            }
            for (endpoint, source, disposition), count in sorted(
                coordinate_counts.items()
            )
        ]
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("flow", flow),
        ("trips", trips),
        ("reconciliation", reconciliation),
        ("coordinate", coordinate_summary),
    ]:
        _atomic_parquet(
            frame,
            checkpoint_paths[name],
            compression,
            compression_level,
            row_group_size,
        )
    state_file.write_text(
        json.dumps(
            {
                "city": city,
                "status": "complete",
                "files": len(files),
                "trip_rows": int(reconciliation["trip_rows"].sum()),
                "completed_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return flow, trips, reconciliation, coordinate_summary


def _observed_fixed_holidays(day_values: set[date]) -> set[date]:
    observed = set(day_values)
    occupied = set(day_values)
    for day_value in sorted(day_values):
        if day_value.weekday() == 5:
            candidate = day_value + timedelta(days=2)
        elif day_value.weekday() == 6:
            candidate = day_value + timedelta(days=1)
        else:
            continue
        while candidate in occupied:
            candidate += timedelta(days=1)
        observed.add(candidate)
        occupied.add(candidate)
    return observed


def _nth_weekday(
    year: int, month: int, weekday: int, occurrence: int
) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _monday_before(year: int, month: int, day: int) -> date:
    target = date(year, month, day)
    offset = (target.weekday() - 0) % 7
    if offset == 0:
        offset = 7
    return target - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def holiday_dates(city: str, start: date, end: date) -> set[date]:
    if city in {"DC", "NY"}:
        observed = USFederalHolidayCalendar().holidays(
            pd.Timestamp(start), pd.Timestamp(end)
        )
        result = {value.date() for value in observed}
        for year in range(start.year, end.year + 1):
            fixed = {
                date(year, 1, 1),
                date(year, 7, 4),
                date(year, 11, 11),
                date(year, 12, 25),
            }
            if year >= 2021:
                fixed.add(date(year, 6, 19))
            result.update(fixed)
        return {value for value in result if start <= value <= end}

    result: set[date] = set()
    for year in range(start.year, end.year + 1):
        fixed = {
            date(year, 1, 1),
            date(year, 7, 1),
            date(year, 11, 11),
            date(year, 12, 25),
            date(year, 12, 26),
        }
        if year >= 2021:
            fixed.add(date(year, 9, 30))
        result.update(_observed_fixed_holidays(fixed))
        result.update(
            {
                _nth_weekday(year, 2, 0, 3),
                _easter_sunday(year) - timedelta(days=2),
                _monday_before(year, 5, 25),
                _nth_weekday(year, 8, 0, 1),
                _nth_weekday(year, 9, 0, 1),
                _nth_weekday(year, 10, 0, 2),
            }
        )
    return {value for value in result if start <= value <= end}


def _calendar_frame(
    city: str, start: pd.Timestamp, end: pd.Timestamp, e02: dict[str, object]
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame({"date": dates})
    frame["year"] = frame["date"].dt.year.astype(np.int16)
    frame["month"] = frame["date"].dt.month.astype(np.int8)
    frame["day_of_week"] = frame["date"].dt.dayofweek.astype(np.int8)
    frame["day_of_year"] = frame["date"].dt.dayofyear.astype(np.int16)
    frame["iso_week"] = frame["date"].dt.isocalendar().week.astype(np.int8)
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6])
    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
    frame["season"] = frame["month"].map(season_map)
    holidays = holiday_dates(city, start.date(), end.date())
    frame["is_holiday"] = frame["date"].dt.date.isin(holidays)
    validation = e02["validation"]
    train_end = pd.Timestamp(validation["train_end"])
    validation_end = pd.Timestamp(validation["validation_end"])
    frame["split"] = np.select(
        [
            frame["date"].le(train_end),
            frame["date"].le(validation_end),
        ],
        ["train", "validation"],
        default="test",
    )
    return frame


def _grid_reference(
    city: str,
    active_keys: set[int],
    assignments: pd.DataFrame,
    flow: pd.DataFrame,
    cities_config: dict[str, object],
    e02: dict[str, object],
) -> pd.DataFrame:
    resolution = int(e02["spatial"]["resolution_m"])
    projected_crs = cities_config["cities"][city]["projected_crs"]
    inverse = Transformer.from_crs(
        projected_crs, "EPSG:4326", always_xy=True
    )
    keys = np.array(sorted(active_keys), dtype=np.int64)
    x_index = (keys // GRID_KEY_MULTIPLIER).astype(np.int32)
    y_index = (keys % GRID_KEY_MULTIPLIER).astype(np.int32)
    x_min = x_index.astype(np.int64) * resolution
    y_min = y_index.astype(np.int64) * resolution
    x_max = x_min + resolution
    y_max = y_min + resolution
    centroid_x = x_min + resolution / 2
    centroid_y = y_min + resolution / 2
    centroid_lon, centroid_lat = inverse.transform(centroid_x, centroid_y)
    corners = []
    for left, bottom, right, top in zip(x_min, y_min, x_max, y_max):
        x_values = [left, right, right, left, left]
        y_values = [bottom, bottom, top, top, bottom]
        lon_values, lat_values = inverse.transform(x_values, y_values)
        points = ", ".join(
            f"{lon:.8f} {lat:.8f}"
            for lon, lat in zip(lon_values, lat_values)
        )
        corners.append(f"POLYGON (({points}))")
    reference = pd.DataFrame(
        {
            "city": city,
            "grid_id": [
                _grid_id(city, resolution, int(x_value), int(y_value))
                for x_value, y_value in zip(x_index, y_index)
            ],
            "resolution_m": resolution,
            "projected_crs": projected_crs,
            "x_index": x_index,
            "y_index": y_index,
            "easting_min": x_min,
            "northing_min": y_min,
            "easting_max": x_max,
            "northing_max": y_max,
            "centroid_longitude": centroid_lon,
            "centroid_latitude": centroid_lat,
            "polygon_wkt_wgs84": corners,
        }
    )
    city_assignments = assignments.loc[assignments["city"].eq(city)]
    training_end = pd.Timestamp(e02["domain"]["training_end"])
    training_counts = (
        city_assignments.loc[
            city_assignments["event_date"].le(training_end)
        ]
        .groupby("grid_id")
        .size()
    )
    all_counts = city_assignments.groupby("grid_id").size()
    bike_training = (
        flow.loc[flow["date"].le(training_end)]
        .groupby("grid_id")["bike_total_flow"]
        .sum()
    )
    bike_all = flow.groupby("grid_id")["bike_total_flow"].sum()
    reference["training_crime_events"] = (
        reference["grid_id"].map(training_counts).fillna(0).astype(np.int32)
    )
    reference["all_crime_events"] = (
        reference["grid_id"].map(all_counts).fillna(0).astype(np.int32)
    )
    reference["training_bike_total_flow"] = (
        reference["grid_id"].map(bike_training).fillna(0).astype(np.int64)
    )
    reference["all_bike_total_flow"] = (
        reference["grid_id"].map(bike_all).fillna(0).astype(np.int64)
    )
    reference["bike_coverage_training"] = reference[
        "training_bike_total_flow"
    ].gt(0)
    reference["bike_coverage_ever"] = reference[
        "all_bike_total_flow"
    ].gt(0)
    return reference


def _build_city_panel(
    city: str,
    grid_reference: pd.DataFrame,
    crime: pd.DataFrame,
    flow: pd.DataFrame,
    trips: pd.DataFrame,
    e02: dict[str, object],
    data_dir: Path,
    logger: logging.Logger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    start = pd.Timestamp(e02["date_range"]["start"])
    end = pd.Timestamp(e02["date_range"]["end"])
    calendar = _calendar_frame(city, start, end, e02)
    grid_columns = [
        "city",
        "grid_id",
        "x_index",
        "y_index",
        "centroid_longitude",
        "centroid_latitude",
        "bike_coverage_training",
        "bike_coverage_ever",
    ]
    grids = grid_reference[grid_columns].copy()
    grids["_join"] = 1
    calendar["_join"] = 1
    panel = grids.merge(calendar, on="_join", how="inner").drop(
        columns="_join"
    )
    city_crime = crime.loc[crime["city"].eq(city)].copy()
    city_flow = flow.loc[flow["city"].eq(city)].copy()
    panel = panel.merge(
        city_crime.drop(columns=["city"]),
        on=["date", "x_index", "y_index", "grid_id"],
        how="left",
    )
    panel = panel.merge(
        city_flow.drop(columns=["city", "grid_id"]),
        on=["date", "x_index", "y_index"],
        how="left",
    )
    count_columns = [
        "crime_count_property_theft",
        "crime_count_vehicle_theft",
        "crime_count_burglary",
        "crime_count_all",
        "bike_outflow",
        "bike_inflow",
        "bike_total_flow",
        "bike_net_flow",
    ]
    for column in count_columns:
        panel[column] = panel[column].fillna(0).astype(np.int32)
    panel["bike_any_flow"] = panel["bike_total_flow"].gt(0)
    panel.sort_values(["grid_id", "date"], inplace=True)
    panel["crime_count_lag1"] = (
        panel.groupby("grid_id", sort=False)["crime_count_all"]
        .shift(1)
        .astype("Int32")
    )
    panel["bike_total_flow_lag1"] = (
        panel.groupby("grid_id", sort=False)["bike_total_flow"]
        .shift(1)
        .astype("Int32")
    )
    panel.sort_values(["date", "grid_id"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    compression = str(e02["parquet"]["compression"])
    compression_level = int(e02["parquet"]["compression_level"])
    row_group_size = int(e02["parquet"]["row_group_size"])
    for year, partition in panel.groupby("year", sort=True):
        output = (
            data_dir
            / "grid_day_panel"
            / f"city={city}"
            / f"year={int(year)}"
            / "part-grid-day-panel.parquet"
        )
        _atomic_parquet(
            partition.reset_index(drop=True),
            output,
            compression,
            compression_level,
            row_group_size,
        )

    monthly = (
        panel.assign(month=panel["date"].dt.to_period("M").astype(str))
        .groupby(["city", "month"], as_index=False)[
            [
                "crime_count_all",
                "crime_count_property_theft",
                "crime_count_vehicle_theft",
                "crime_count_burglary",
                "bike_outflow",
                "bike_inflow",
                "bike_total_flow",
            ]
        ]
        .sum()
    )
    monthly = monthly.merge(
        trips.assign(month=trips["date"].dt.to_period("M").astype(str))
        .groupby(["city", "month"], as_index=False)[
            "bike_trips_any_endpoint"
        ]
        .sum(),
        on=["city", "month"],
        how="left",
    )
    monthly["bike_trips_any_endpoint"] = monthly[
        "bike_trips_any_endpoint"
    ].fillna(0).astype(np.int32)
    weekday = (
        panel.groupby(["city", "day_of_week"], as_index=False)[
            ["crime_count_all", "bike_total_flow"]
        ]
        .mean()
        .rename(
            columns={
                "crime_count_all": "mean_crime_count_per_grid_day",
                "bike_total_flow": "mean_bike_total_flow_per_grid_day",
            }
        )
    )
    spatial = (
        panel.groupby(
            [
                "city",
                "grid_id",
                "x_index",
                "y_index",
                "centroid_longitude",
                "centroid_latitude",
            ],
            as_index=False,
        )[
            [
                "crime_count_all",
                "crime_count_property_theft",
                "crime_count_vehicle_theft",
                "crime_count_burglary",
                "bike_outflow",
                "bike_inflow",
                "bike_total_flow",
            ]
        ]
        .sum()
    )
    variable_records: list[dict[str, object]] = []
    for variable in [
        "crime_count_all",
        "bike_outflow",
        "bike_inflow",
        "bike_total_flow",
        "bike_net_flow",
    ]:
        values = panel[variable].astype(float)
        variable_records.append(
            {
                "city": city,
                "variable": variable,
                "observations": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "minimum": values.min(),
                "p50": values.quantile(0.5),
                "p90": values.quantile(0.9),
                "p99": values.quantile(0.99),
                "maximum": values.max(),
                "zero_share": values.eq(0).mean(),
            }
        )
    logger.info(
        "%s panel complete: %d grids x %d days = %d rows",
        city,
        len(grid_reference),
        len(calendar),
        len(panel),
    )
    return panel, monthly, weekday, pd.DataFrame(variable_records), spatial


def _write_interpretation(
    run_dir: Path,
    city_summary: pd.DataFrame,
    resolution: int,
) -> None:
    lines = [
        "# E02 Interpretation and E03 Gate",
        "",
        f"The primary panel uses a deterministic {resolution:,} m projected grid "
        "and local calendar days from 2020-01-01 through 2022-12-31.",
        "",
        "The primary domain is defined only from crime observations available "
        "in the 2020-2021 training window. This prevents validation/test "
        "locations from determining the analysis grid. Bicycle endpoints "
        "outside that domain remain in reconciliation tables.",
        "",
        "## City-level gate",
        "",
    ]
    for row in city_summary.to_dict("records"):
        lines.append(
            f"- **{row['city']}**: {int(row['active_grids']):,} grids, "
            f"{int(row['panel_rows']):,} grid-days, "
            f"{int(row['crime_events_in_primary_domain']):,} in-domain "
            f"crime events, and {int(row['bike_total_flow_in_panel']):,} "
            "eligible bicycle endpoint observations."
        )
    lines.extend(
        [
            "",
            "E02 creates descriptive and modeling inputs only. It does not "
            "estimate entropy, the discrete lower bound, causal effects, or "
            "predictive-model performance. Those claims require E03-E10.",
            "",
            "Weather is not included because no versioned, forecast-available "
            "weather source has yet passed the data-provenance gate.",
        ]
    )
    (run_dir / "interpretation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    paths = _load_yaml(Path(args.paths))
    e02 = _load_yaml(Path(args.e02))
    cities_config = _load_yaml(Path(args.cities))
    timezone = ZoneInfo(str(paths.get("timezone", "Asia/Shanghai")))
    run_id = args.resume_run or (
        datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        + "_E02_spatial_temporal_panel"
    )
    run_dir = Path(str(paths["runs"])) / run_id
    data_dir = Path(str(paths["processed_data"])) / "e02" / run_id
    if args.resume_run and (not run_dir.exists() or not data_dir.exists()):
        raise FileNotFoundError(f"Cannot resume missing E02 run: {run_id}")
    for directory in [
        run_dir / "logs",
        run_dir / "tables",
        run_dir / "figures",
        run_dir / "state",
        data_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)

    e01_run = (
        Path(args.e01_run).resolve()
        if args.e01_run
        else Path(
            (
                Path(str(paths["empirical_root"]))
                / str(e02["source_e01_pointer"])
            )
            .read_text(encoding="utf-8")
            .strip()
        )
    )
    e01_status = json.loads(
        (e01_run / "run_status.json").read_text(encoding="utf-8")
    )
    if e01_status.get("status") != "complete_targeted_patch":
        raise ValueError("E02 requires the accepted complete E01 run")
    e01_data = Path(e01_status["data_directory"])
    selected_cities = (
        [args.city] if args.city else list(CITY_ORDER)
    )
    if not set(selected_cities).issubset(cities_config["cities"]):
        raise ValueError(f"Unknown city selection: {selected_cities}")

    if not args.resume_run:
        (run_dir / "command.txt").write_text(
            " ".join([sys.executable, *sys.argv]) + "\n",
            encoding="utf-8",
        )
        _write_environment(run_dir / "environment.txt")
        (run_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": paths,
                    "e02": e02,
                    "cities": cities_config,
                    "source_e01_run": str(e01_run),
                    "selected_cities": selected_cities,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    else:
        (run_dir / "resume_command.txt").write_text(
            " ".join([sys.executable, *sys.argv]) + "\n",
            encoding="utf-8",
        )

    logger.info(
        "Starting E02 from %s for %s",
        e01_run.name,
        ", ".join(selected_cities),
    )
    assignments_path = data_dir / "checkpoints" / "crime_assignments.parquet"
    crime_path = data_dir / "checkpoints" / "crime_grid_day.parquet"
    active_path = run_dir / "tables" / "active_grid_keys.csv"
    if (
        args.resume_run
        and assignments_path.exists()
        and crime_path.exists()
        and active_path.exists()
    ):
        assignments = pd.read_parquet(assignments_path)
        crime = pd.read_parquet(crime_path)
        active_table = pd.read_csv(active_path)
        active_keys = {
            city: set(
                active_table.loc[
                    active_table["city"].eq(city), "grid_key"
                ].astype(np.int64)
            )
            for city in selected_cities
        }
        logger.info("Loaded completed crime-grid checkpoint")
    else:
        assignments, crime, active_keys = _aggregate_crime(
            e01_data,
            selected_cities,
            cities_config,
            e02,
            run_dir,
            data_dir,
            logger,
        )
        assignments_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(
            assignments,
            assignments_path,
            str(e02["parquet"]["compression"]),
            int(e02["parquet"]["compression_level"]),
            int(e02["parquet"]["row_group_size"]),
        )
        _atomic_parquet(
            crime,
            crime_path,
            str(e02["parquet"]["compression"]),
            int(e02["parquet"]["compression_level"]),
            int(e02["parquet"]["row_group_size"]),
        )
        pd.DataFrame(
            [
                {"city": city, "grid_key": grid_key}
                for city, keys in active_keys.items()
                for grid_key in sorted(keys)
            ]
        ).to_csv(active_path, index=False)

    flow_frames: list[pd.DataFrame] = []
    trip_frames: list[pd.DataFrame] = []
    reconciliation_frames: list[pd.DataFrame] = []
    coordinate_frames: list[pd.DataFrame] = []
    for city in selected_cities:
        flow, trips, reconciliation, coordinate = _aggregate_bicycle_city(
            city,
            e01_data,
            active_keys[city],
            cities_config,
            e02,
            run_dir,
            data_dir,
            logger,
        )
        flow_frames.append(flow)
        trip_frames.append(trips)
        reconciliation_frames.append(reconciliation)
        coordinate_frames.append(coordinate)
    flow = pd.concat(flow_frames, ignore_index=True)
    trips = pd.concat(trip_frames, ignore_index=True)
    bicycle_reconciliation = pd.concat(
        reconciliation_frames, ignore_index=True
    )
    coordinate_summary = pd.concat(
        coordinate_frames, ignore_index=True
    )

    compression = str(e02["parquet"]["compression"])
    compression_level = int(e02["parquet"]["compression_level"])
    row_group_size = int(e02["parquet"]["row_group_size"])
    for city, partition in flow.groupby("city"):
        for year, year_partition in partition.groupby(
            partition["date"].dt.year
        ):
            _atomic_parquet(
                year_partition.reset_index(drop=True),
                data_dir
                / "bike_flow_grid_day"
                / f"city={city}"
                / f"year={int(year)}"
                / "part-bike-flow-grid-day.parquet",
                compression,
                compression_level,
                row_group_size,
            )
    bicycle_reconciliation.to_csv(
        run_dir / "tables" / "bicycle_spatial_reconciliation.csv",
        index=False,
    )
    coordinate_summary.to_csv(
        run_dir / "tables" / "coordinate_domain_summary.csv", index=False
    )

    monthly_frames: list[pd.DataFrame] = []
    weekday_frames: list[pd.DataFrame] = []
    variable_frames: list[pd.DataFrame] = []
    spatial_frames: list[pd.DataFrame] = []
    city_summary_records: list[dict[str, object]] = []
    for city in selected_cities:
        reference = _grid_reference(
            city,
            active_keys[city],
            assignments,
            flow.loc[flow["city"].eq(city)],
            cities_config,
            e02,
        )
        _atomic_parquet(
            reference,
            data_dir
            / "grid_reference"
            / f"city={city}"
            / "grid_reference.parquet",
            compression,
            compression_level,
            row_group_size,
        )
        panel, monthly, weekday, variable, spatial = _build_city_panel(
            city,
            reference,
            crime,
            flow,
            trips,
            e02,
            data_dir,
            logger,
        )
        monthly_frames.append(monthly)
        weekday_frames.append(weekday)
        variable_frames.append(variable)
        spatial_frames.append(spatial)
        city_assignment = assignments.loc[assignments["city"].eq(city)]
        city_reconciliation = bicycle_reconciliation.loc[
            bicycle_reconciliation["city"].eq(city)
        ]
        city_summary_records.append(
            {
                "city": city,
                "active_grids": len(reference),
                "panel_days": panel["date"].nunique(),
                "panel_rows": len(panel),
                "crime_events_e01": len(city_assignment),
                "crime_events_in_primary_domain": int(
                    city_assignment["in_primary_domain"].sum()
                ),
                "crime_events_outside_primary_domain": int(
                    (~city_assignment["in_primary_domain"]).sum()
                ),
                "bicycle_trip_rows_e01": int(
                    city_reconciliation["trip_rows"].sum()
                ),
                "bike_start_endpoints_e01_eligible": int(
                    city_reconciliation["start_e01_flow_eligible"].sum()
                ),
                "bike_end_endpoints_e01_eligible": int(
                    city_reconciliation["end_e01_flow_eligible"].sum()
                ),
                "bike_start_endpoints_in_primary_domain": int(
                    city_reconciliation["start_in_primary_domain"].sum()
                ),
                "bike_end_endpoints_in_primary_domain": int(
                    city_reconciliation["end_in_primary_domain"].sum()
                ),
                "bike_start_endpoints_outside_primary_domain": int(
                    city_reconciliation[
                        "start_outside_primary_domain"
                    ].sum()
                ),
                "bike_end_endpoints_outside_primary_domain": int(
                    city_reconciliation[
                        "end_outside_primary_domain"
                    ].sum()
                ),
                "bike_total_flow_in_panel": int(
                    panel["bike_total_flow"].sum()
                ),
                "bike_trips_any_endpoint_in_primary_domain": int(
                    city_reconciliation[
                        "trips_any_endpoint_in_primary_domain"
                    ].sum()
                ),
                "zero_crime_grid_day_share": float(
                    panel["crime_count_all"].eq(0).mean()
                ),
                "zero_bike_flow_grid_day_share": float(
                    panel["bike_total_flow"].eq(0).mean()
                ),
                "bike_covered_grid_share": float(
                    reference["bike_coverage_ever"].mean()
                ),
                "bike_covered_training_grid_share": float(
                    reference["bike_coverage_training"].mean()
                ),
            }
        )
        del panel

    city_summary = pd.DataFrame(city_summary_records)
    monthly_summary = pd.concat(monthly_frames, ignore_index=True)
    weekday_summary = pd.concat(weekday_frames, ignore_index=True)
    variable_summary = pd.concat(variable_frames, ignore_index=True)
    spatial_summary = pd.concat(spatial_frames, ignore_index=True)
    city_summary.to_csv(
        run_dir / "tables" / "city_panel_summary.csv", index=False
    )
    monthly_summary.to_csv(
        run_dir / "tables" / "monthly_patterns.csv", index=False
    )
    weekday_summary.to_csv(
        run_dir / "tables" / "weekday_patterns.csv", index=False
    )
    variable_summary.to_csv(
        run_dir / "tables" / "variable_summary.csv", index=False
    )
    spatial_summary.to_csv(
        run_dir / "tables" / "spatial_grid_totals.csv", index=False
    )

    e01_city_summary = pd.read_csv(
        e01_run / "tables" / "city_summary.csv"
    )
    e01_bicycle = e01_city_summary.loc[
        e01_city_summary["dataset"].eq("bicycle")
        & e01_city_summary["city"].isin(selected_cities)
    ]
    expected_crime = int(
        e01_city_summary.loc[
            e01_city_summary["dataset"].eq("crime")
            & e01_city_summary["city"].isin(selected_cities),
            "retained_rows",
        ].sum()
    )
    expected_bike = int(e01_bicycle["retained_rows"].sum())
    checks = [
        (
            "Crime assignment rows equal E01 retained rows",
            expected_crime,
            len(assignments),
        ),
        (
            "Bicycle rows processed equal E01 retained rows",
            expected_bike,
            int(bicycle_reconciliation["trip_rows"].sum()),
        ),
        (
            "Start eligible endpoints equal E01",
            int(e01_bicycle["start_flow_eligible_rows"].sum()),
            int(
                bicycle_reconciliation["start_e01_flow_eligible"].sum()
            ),
        ),
        (
            "End eligible endpoints equal E01",
            int(e01_bicycle["end_flow_eligible_rows"].sum()),
            int(bicycle_reconciliation["end_e01_flow_eligible"].sum()),
        ),
        (
            "Panel crime equals in-domain assignments",
            int(assignments["in_primary_domain"].sum()),
            int(city_summary["crime_events_in_primary_domain"].sum()),
        ),
        (
            "Panel bicycle total equals in-domain endpoints",
            int(
                bicycle_reconciliation["start_in_primary_domain"].sum()
                + bicycle_reconciliation["end_in_primary_domain"].sum()
            ),
            int(city_summary["bike_total_flow_in_panel"].sum()),
        ),
    ]
    acceptance = pd.DataFrame(
        checks, columns=["check", "expected", "observed"]
    )
    acceptance["difference"] = (
        acceptance["observed"] - acceptance["expected"]
    )
    acceptance["status"] = np.where(
        acceptance["difference"].eq(0), "PASS", "FAIL"
    )
    acceptance.to_csv(
        run_dir / "tables" / "acceptance_checklist.csv", index=False
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(
            "E02 acceptance gate failed: "
            + ", ".join(
                acceptance.loc[
                    acceptance["status"].ne("PASS"), "check"
                ]
            )
        )

    _write_interpretation(
        run_dir, city_summary, int(e02["spatial"]["resolution_m"])
    )
    elapsed = time.monotonic() - started
    is_full_run = set(selected_cities) == set(CITY_ORDER)
    completion_status = "complete" if is_full_run else "partial_complete"
    status = {
        "run_id": run_id,
        "status": completion_status,
        "source_e01_run": e01_run.name,
        "selected_cities": selected_cities,
        "spatial_resolution_m": int(e02["spatial"]["resolution_m"]),
        "panel_rows": int(city_summary["panel_rows"].sum()),
        "active_grids": int(city_summary["active_grids"].sum()),
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now().isoformat(),
        "data_directory": str(data_dir),
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    if is_full_run:
        Path(str(paths["runs"])).joinpath("latest_e02_run.txt").write_text(
            str(run_dir) + "\n", encoding="utf-8"
        )
        latest_data = Path(str(paths["processed_data"])) / "e02"
        latest_data.mkdir(parents=True, exist_ok=True)
        (latest_data / "latest_e02_data.txt").write_text(
            str(data_dir) + "\n", encoding="utf-8"
        )
    registry = (
        Path(str(paths["empirical_root"]))
        / "docs"
        / "experiment_registry.csv"
    )
    registry_frame = pd.read_csv(registry)
    if run_id not in set(registry_frame["run_id"].astype(str)):
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "experiment": "E02",
                    "status": completion_status,
                    "started_at": datetime.fromtimestamp(
                        (run_dir / "command.txt").stat().st_mtime
                    ).isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "elapsed_seconds": round(elapsed, 3),
                    "run_directory": str(run_dir),
                    "notes": (
                        "1 km daily grid panel with crime counts, bicycle "
                        "flows, calendar controls, lags, and full QC."
                    ),
                }
            ]
        ).to_csv(registry, mode="a", header=False, index=False)
    logger.info("E02 %s in %.1f seconds", completion_status, elapsed)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the E02 spatial-temporal panel"
    )
    parser.add_argument("--paths", required=True)
    parser.add_argument("--cities", required=True)
    parser.add_argument("--e02", required=True)
    parser.add_argument("--e01-run")
    parser.add_argument("--city", choices=CITY_ORDER)
    parser.add_argument("--resume-run")
    return parser


def main() -> None:
    run_dir = run(build_parser().parse_args())
    print(run_dir)


if __name__ == "__main__":
    main()
