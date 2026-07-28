from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml
from pyproj import Transformer

from entropy_crime_bike.panel import (
    CITY_ORDER,
    _aggregate_bicycle_city,
    _atomic_parquet,
    _build_city_panel,
    _calendar_frame,
    _crime_column_name,
    _grid_id,
    _grid_key,
    _grid_reference,
    assign_projected_grid,
)


COUNT_COLUMNS = [
    "crime_count_property_theft",
    "crime_count_vehicle_theft",
    "crime_count_burglary",
    "crime_count_all",
    "bike_outflow",
    "bike_inflow",
    "bike_total_flow",
    "bike_net_flow",
]
FLOW_COLUMNS = ["bike_outflow", "bike_inflow", "bike_total_flow", "bike_net_flow"]
CRIME_COLUMNS = [
    "crime_count_property_theft",
    "crime_count_vehicle_theft",
    "crime_count_burglary",
    "crime_count_all",
]


def _load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_pointer(root: Path, relative: str) -> Path:
    value = Path((root / relative).read_text(encoding="utf-8").strip()).resolve()
    if not value.exists():
        raise FileNotFoundError(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e16_multiscale_panel")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in [
        logging.FileHandler(run_dir / "logs" / "e16_s16_2.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _environment(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"timestamp={datetime.now().isoformat()}",
                f"python={sys.version.replace(os.linesep, ' ')}",
                f"executable={sys.executable}",
                f"platform={platform.platform()}",
                f"pandas={pd.__version__}",
                f"numpy={np.__version__}",
                f"pyarrow={pyarrow.__version__}",
                f"matplotlib={plt.matplotlib.__version__}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def complete_nested_core(reference_1km: pd.DataFrame) -> pd.DataFrame:
    """Return complete 2 km parents and their exact 1 km/500 m descendants."""
    required = {"city", "x_index", "y_index"}
    if not required.issubset(reference_1km.columns):
        raise ValueError(f"Missing reference fields: {required-reference_1km.columns.to_list()}")
    records: list[dict[str, object]] = []
    for city, city_frame in reference_1km.groupby("city", sort=True):
        keys = {
            (int(row.x_index), int(row.y_index))
            for row in city_frame.itertuples(index=False)
        }
        parents = sorted({(x // 2, y // 2) for x, y in keys})
        for parent_x, parent_y in parents:
            children_1km = {
                (2 * parent_x + dx, 2 * parent_y + dy)
                for dx in (0, 1)
                for dy in (0, 1)
            }
            if not children_1km.issubset(keys):
                continue
            parent_id = _grid_id(str(city), 2000, parent_x, parent_y)
            for x_1km, y_1km in sorted(children_1km):
                id_1km = _grid_id(str(city), 1000, x_1km, y_1km)
                for dx in (0, 1):
                    for dy in (0, 1):
                        x_500 = 2 * x_1km + dx
                        y_500 = 2 * y_1km + dy
                        records.append(
                            {
                                "city": city,
                                "parent_2km_grid_id": parent_id,
                                "parent_2km_x_index": parent_x,
                                "parent_2km_y_index": parent_y,
                                "parent_1km_grid_id": id_1km,
                                "parent_1km_x_index": x_1km,
                                "parent_1km_y_index": y_1km,
                                "grid_500m_id": _grid_id(str(city), 500, x_500, y_500),
                                "x_500m_index": x_500,
                                "y_500m_index": y_500,
                            }
                        )
    result = pd.DataFrame(records)
    if result.empty or result.duplicated(["city", "x_500m_index", "y_500m_index"]).any():
        raise AssertionError("Invalid or duplicated nested common core.")
    return result.sort_values(
        ["city", "parent_2km_x_index", "parent_2km_y_index", "x_500m_index", "y_500m_index"]
    ).reset_index(drop=True)


def full_iso_week_bounds(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    first = start + pd.Timedelta(days=(7 - start.dayofweek) % 7)
    last = end - pd.Timedelta(days=end.dayofweek)
    if last + pd.Timedelta(days=6) > end:
        last -= pd.Timedelta(days=7)
    if first > last:
        raise ValueError("No complete ISO week in range.")
    return first, last


def _load_reference_1km(e02_data: Path) -> pd.DataFrame:
    parts = []
    for city in CITY_ORDER:
        path = e02_data / "grid_reference" / f"city={city}" / "grid_reference.parquet"
        frame = pd.read_parquet(path)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _aggregate_crime_to_master(
    e01_data: Path,
    hierarchy: pd.DataFrame,
    cities: dict[str, object],
    e02: dict[str, object],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments: list[pd.DataFrame] = []
    reconciliation: list[dict[str, object]] = []
    categories = list(e02["panel"]["crime_categories"])
    for city in CITY_ORDER:
        allowed = hierarchy.loc[hierarchy["city"].eq(city)]
        allowed_keys = set(
            _grid_key(
                allowed["x_500m_index"].to_numpy(np.int32),
                allowed["y_500m_index"].to_numpy(np.int32),
            ).astype(np.int64)
        )
        transformer = Transformer.from_crs(
            "EPSG:4326", cities["cities"][city]["projected_crs"], always_xy=True
        )
        files = sorted((e01_data / "crime_events" / f"city={city}").rglob("*.parquet"))
        city_rows = 0
        city_inside = 0
        for path in files:
            frame = pq.read_table(
                path,
                columns=["event_id", "city", "event_date", "crime_type_unified", "longitude", "latitude"],
            ).to_pandas()
            x_index, y_index, valid = assign_projected_grid(
                frame["longitude"], frame["latitude"], transformer, 500
            )
            if not valid.all():
                raise AssertionError(f"Invalid crime projection: {path}")
            keys = _grid_key(x_index, y_index).astype(np.int64)
            inside = np.fromiter((int(value) in allowed_keys for value in keys), dtype=bool, count=len(keys))
            frame["event_date"] = pd.to_datetime(frame["event_date"])
            frame["x_index"] = x_index
            frame["y_index"] = y_index
            frame["grid_key"] = keys
            frame["grid_id"] = [
                _grid_id(city, 500, int(x), int(y)) for x, y in zip(x_index, y_index)
            ]
            frame["in_primary_domain"] = inside
            assignments.append(
                frame[["event_id", "city", "event_date", "crime_type_unified", "x_index", "y_index", "grid_key", "grid_id", "in_primary_domain"]]
            )
            city_rows += len(frame)
            city_inside += int(inside.sum())
        reconciliation.append(
            {
                "city": city,
                "source_crime_events": city_rows,
                "inside_common_core": city_inside,
                "outside_common_core": city_rows - city_inside,
            }
        )
        logger.info("%s crime assigned to common core: %d/%d", city, city_inside, city_rows)
    assignment = pd.concat(assignments, ignore_index=True)
    inside = assignment.loc[assignment["in_primary_domain"]].copy()
    grouped = (
        inside.groupby(
            ["city", "event_date", "x_index", "y_index", "grid_id", "crime_type_unified"],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    crime = grouped.pivot_table(
        index=["city", "event_date", "x_index", "y_index", "grid_id"],
        columns="crime_type_unified",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    crime.columns.name = None
    crime.rename(columns={"event_date": "date"}, inplace=True)
    for category in categories:
        if category not in crime:
            crime[category] = 0
        crime.rename(columns={category: _crime_column_name(category)}, inplace=True)
    crime_cols = [_crime_column_name(value) for value in categories]
    crime["crime_count_all"] = crime[crime_cols].sum(axis=1)
    for column in [*crime_cols, "crime_count_all"]:
        crime[column] = crime[column].astype(np.int32)
    return assignment, crime, pd.DataFrame(reconciliation)


def _map_scale(frame: pd.DataFrame, resolution: int, value_columns: list[str]) -> pd.DataFrame:
    if resolution not in {500, 1000, 2000}:
        raise ValueError(resolution)
    factor = resolution // 500
    mapped = frame.copy()
    mapped["x_index"] = np.floor_divide(mapped["x_index"].astype(np.int64), factor).astype(np.int32)
    mapped["y_index"] = np.floor_divide(mapped["y_index"].astype(np.int64), factor).astype(np.int32)
    result = mapped.groupby(["city", "date", "x_index", "y_index"], as_index=False)[value_columns].sum()
    result["grid_id"] = [
        _grid_id(str(city), resolution, int(x), int(y))
        for city, x, y in zip(result["city"], result["x_index"], result["y_index"])
    ]
    for column in value_columns:
        result[column] = result[column].astype(np.int32 if column != "bike_total_flow" else np.int64)
    return result


def _scale_assignments(assignments_500: pd.DataFrame, resolution: int) -> pd.DataFrame:
    factor = resolution // 500
    result = assignments_500.loc[assignments_500["in_primary_domain"]].copy()
    result["x_index"] = np.floor_divide(result["x_index"].astype(np.int64), factor).astype(np.int32)
    result["y_index"] = np.floor_divide(result["y_index"].astype(np.int64), factor).astype(np.int32)
    result["grid_id"] = [
        _grid_id(str(city), resolution, int(x), int(y))
        for city, x, y in zip(result["city"], result["x_index"], result["y_index"])
    ]
    return result


def _active_keys(hierarchy: pd.DataFrame, city: str, resolution: int) -> set[int]:
    part = hierarchy.loc[hierarchy["city"].eq(city)]
    columns = {
        500: ("x_500m_index", "y_500m_index"),
        1000: ("parent_1km_x_index", "parent_1km_y_index"),
        2000: ("parent_2km_x_index", "parent_2km_y_index"),
    }[resolution]
    unique = part[list(columns)].drop_duplicates()
    return set(_grid_key(unique[columns[0]].to_numpy(np.int32), unique[columns[1]].to_numpy(np.int32)).astype(np.int64))


def weekly_panel(daily: pd.DataFrame, e02: dict[str, object]) -> pd.DataFrame:
    first, last = full_iso_week_bounds(e02["date_range"]["start"], e02["date_range"]["end"])
    working = daily.copy()
    working["week_start"] = working["date"] - pd.to_timedelta(working["date"].dt.dayofweek, unit="D")
    working = working.loc[working["week_start"].between(first, last)].copy()
    keys = ["city", "grid_id", "x_index", "y_index", "week_start"]
    grouped = working.groupby(keys, observed=True, as_index=False).agg(
        **{column: (column, "sum") for column in COUNT_COLUMNS},
        days_in_period=("date", "nunique"),
        contains_holiday=("is_holiday", "max"),
        centroid_longitude=("centroid_longitude", "first"),
        centroid_latitude=("centroid_latitude", "first"),
        bike_coverage_training=("bike_coverage_training", "first"),
        bike_coverage_ever=("bike_coverage_ever", "first"),
    ).rename(columns={"week_start": "date"})
    if not grouped["days_in_period"].eq(7).all():
        raise AssertionError("Incomplete ISO week survived aggregation.")
    grouped["period_end"] = grouped["date"] + pd.Timedelta(days=6)
    grouped["year"] = grouped["date"].dt.year.astype(np.int16)
    grouped["month"] = grouped["date"].dt.month.astype(np.int8)
    grouped["day_of_week"] = np.int8(0)
    grouped["iso_week"] = grouped["date"].dt.isocalendar().week.astype(np.int8)
    grouped["season"] = grouped["month"].map(
        {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    )
    train_end = pd.Timestamp(e02["validation"]["train_end"])
    validation_end = pd.Timestamp(e02["validation"]["validation_end"])
    grouped["split"] = np.select(
        [grouped["period_end"].le(train_end), grouped["period_end"].le(validation_end)],
        ["train", "validation"],
        default="test",
    )
    grouped.sort_values(["grid_id", "date"], inplace=True)
    grouped["crime_count_lag1"] = grouped.groupby("grid_id", sort=False)["crime_count_all"].shift(1).astype("Int32")
    grouped["bike_total_flow_lag1"] = grouped.groupby("grid_id", sort=False)["bike_total_flow"].shift(1).astype("Int64")
    grouped["bike_any_flow"] = grouped["bike_total_flow"].gt(0)
    grouped.sort_values(["date", "grid_id"], inplace=True)
    return grouped.reset_index(drop=True)


def _write_weekly(panel: pd.DataFrame, data_dir: Path, city: str, resolution: int, e02: dict[str, object]) -> None:
    for year, partition in panel.groupby("year", sort=True):
        _atomic_parquet(
            partition.reset_index(drop=True),
            data_dir / "common_core" / f"resolution={resolution}m" / "temporal=week" / f"city={city}" / f"year={int(year)}" / "part-grid-week-panel.parquet",
            str(e02["parquet"]["compression"]),
            int(e02["parquet"]["compression_level"]),
            int(e02["parquet"]["row_group_size"]),
        )


def _copy_daily_layout(source_dir: Path, destination: Path, city: str, resolution: int) -> None:
    for path in sorted((source_dir / "grid_day_panel" / f"city={city}").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["resolution_m"] = np.int16(resolution)
        frame["nominal_area_km2"] = float((resolution / 1000) ** 2)
        frame["temporal_resolution"] = "day"
        frame["duration_days"] = np.int8(1)
        year = int(frame["year"].iloc[0])
        _atomic_parquet(
            frame,
            destination / "common_core" / f"resolution={resolution}m" / "temporal=day" / f"city={city}" / f"year={year}" / "part-grid-day-panel.parquet",
            "zstd", 3, 250000,
        )


def _read_daily(data_dir: Path, city: str, resolution: int) -> pd.DataFrame:
    files = sorted((data_dir / "common_core" / f"resolution={resolution}m" / "temporal=day" / f"city={city}").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError((data_dir, city, resolution))
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["grid_id", "date"]).reset_index(drop=True)


def _accepted_e02_panel(e02_data: Path, city: str) -> pd.DataFrame:
    files = sorted((e02_data / "grid_day_panel" / f"city={city}").rglob("*.parquet"))
    columns = ["grid_id", "date", *COUNT_COLUMNS]
    return pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)


def _qc_plot(summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    daily = summary.loc[summary["temporal_resolution"].eq("day")]
    weekly = summary.loc[summary["temporal_resolution"].eq("week")]
    for city, marker in zip(CITY_ORDER, ["o", "s", "^"]):
        part = daily.loc[daily["city"].eq(city)].sort_values("resolution_m")
        axes[0].plot(part["resolution_m"], part["grids"], marker=marker, label=city)
        part = weekly.loc[weekly["city"].eq(city)].sort_values("resolution_m")
        axes[1].plot(part["resolution_m"], part["panel_rows"], marker=marker, label=city)
    axes[0].set(title="Nested common-core grids", xlabel="Spatial resolution (m)", ylabel="Grid cells")
    axes[1].set(title="Complete-week panel rows", xlabel="Spatial resolution (m)", ylabel="Rows")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "e16_s16_2_panel_inventory.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "e16_s16_2_panel_inventory.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    config_path = Path(args.config).resolve()
    empirical_root = config_path.parent.parent
    project_root = empirical_root.parent
    config = _load_yaml(config_path)
    e02 = _load_yaml(Path(args.e02).resolve())
    cities = _load_yaml(Path(args.cities).resolve())
    timezone = ZoneInfo("Asia/Shanghai")
    run_id = args.resume_run or datetime.now(timezone).strftime("%Y%m%d_%H%M%S") + "_E16_S16_2_multiscale_panel"
    run_dir = empirical_root / "runs" / run_id
    data_dir = empirical_root / "processed_data" / "e16" / run_id
    for directory in [run_dir / "logs", run_dir / "tables", run_dir / "figures", run_dir / "state", data_dir / "checkpoints"]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    if not args.resume_run:
        (run_dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
        _environment(run_dir / "environment.txt")
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    e01_data = _read_pointer(empirical_root, str(config["source_policy"]["source_e01_data_pointer"]))
    e02_data = _read_pointer(empirical_root, str(config["source_policy"]["source_e02_data_pointer"]))
    e09_run = _read_pointer(empirical_root, "runs/latest_e09_run.txt")
    e09_status = json.loads((e09_run / "run_status.json").read_text(encoding="utf-8"))
    e09_data = e09_run.parents[1] / "processed_data" / "e09" / e09_run.name
    if not e09_data.exists():
        e09_data = Path(str(e09_status["source_e02_data"])).parents[1] / "e09" / e09_run.name
    logger.info("S16.2 sources: E01=%s E02=%s E09=%s", e01_data, e02_data, e09_data)

    reference_1km = _load_reference_1km(e02_data)
    hierarchy = complete_nested_core(reference_1km)
    hierarchy.to_csv(run_dir / "tables" / "nested_grid_hierarchy.csv", index=False)
    _atomic_parquet(hierarchy, data_dir / "nested_grid_hierarchy.parquet", "zstd", 3, 250000)

    local_e02 = copy.deepcopy(e02)
    local_e02["spatial"]["resolution_m"] = 500
    assignments_path = data_dir / "checkpoints" / "crime_assignments_500m.parquet"
    crime_path = data_dir / "checkpoints" / "crime_grid_day_500m.parquet"
    crime_recon_path = run_dir / "tables" / "crime_common_core_reconciliation.csv"
    if assignments_path.exists() and crime_path.exists() and crime_recon_path.exists():
        assignments_500 = pd.read_parquet(assignments_path)
        crime_500 = pd.read_parquet(crime_path)
        crime_recon = pd.read_csv(crime_recon_path)
        logger.info("Loaded crime common-core checkpoint")
    else:
        assignments_500, crime_500, crime_recon = _aggregate_crime_to_master(e01_data, hierarchy, cities, local_e02, logger)
        _atomic_parquet(assignments_500, assignments_path, "zstd", 3, 250000)
        _atomic_parquet(crime_500, crime_path, "zstd", 3, 250000)
        crime_recon.to_csv(crime_recon_path, index=False)

    flow_parts = []
    trip_parts = []
    bike_recon_parts = []
    coordinate_parts = []
    for city in CITY_ORDER:
        flow, trips, bike_recon, coordinate = _aggregate_bicycle_city(
            city, e01_data, _active_keys(hierarchy, city, 500), cities, local_e02, run_dir, data_dir, logger
        )
        flow_parts.append(flow)
        trip_parts.append(trips)
        bike_recon_parts.append(bike_recon)
        coordinate_parts.append(coordinate)
    flow_500 = pd.concat(flow_parts, ignore_index=True)
    trips = pd.concat(trip_parts, ignore_index=True)
    bicycle_recon = pd.concat(bike_recon_parts, ignore_index=True)
    coordinate = pd.concat(coordinate_parts, ignore_index=True)
    bicycle_recon.to_csv(run_dir / "tables" / "bicycle_common_core_reconciliation.csv", index=False)
    coordinate.to_csv(run_dir / "tables" / "coordinate_common_core_summary.csv", index=False)

    inventory_records = []
    conservation_records = []
    reconciliation_records = []
    scale_assignments = {r: _scale_assignments(assignments_500, r) for r in [500, 1000, 2000]}
    scale_crime = {r: _map_scale(crime_500, r, CRIME_COLUMNS) for r in [500, 1000, 2000]}
    scale_flow = {r: _map_scale(flow_500, r, FLOW_COLUMNS) for r in [500, 1000, 2000]}

    for resolution in [500, 1000, 2000]:
        scale_dir = data_dir / "build" / f"resolution_{resolution}m"
        scale_dir.mkdir(parents=True, exist_ok=True)
        for city in CITY_ORDER:
            reference = _grid_reference(
                city,
                _active_keys(hierarchy, city, resolution),
                scale_assignments[resolution],
                scale_flow[resolution].loc[scale_flow[resolution]["city"].eq(city)],
                cities,
                {**local_e02, "spatial": {**local_e02["spatial"], "resolution_m": resolution}},
            )
            reference["nominal_area_km2"] = float((resolution / 1000) ** 2)
            _atomic_parquet(reference, data_dir / "common_core" / f"resolution={resolution}m" / "grid_reference" / f"city={city}" / "grid_reference.parquet", "zstd", 3, 250000)
            panel, _, _, _, _ = _build_city_panel(
                city,
                reference,
                scale_crime[resolution],
                scale_flow[resolution],
                trips,
                {**local_e02, "spatial": {**local_e02["spatial"], "resolution_m": resolution}},
                scale_dir,
                logger,
            )
            _copy_daily_layout(scale_dir, data_dir, city, resolution)
            daily = _read_daily(data_dir, city, resolution)
            weekly = weekly_panel(daily, e02)
            weekly["resolution_m"] = np.int16(resolution)
            weekly["nominal_area_km2"] = float((resolution / 1000) ** 2)
            weekly["temporal_resolution"] = "week"
            weekly["duration_days"] = np.int8(7)
            _write_weekly(weekly, data_dir, city, resolution, e02)
            for temporal, frame in [("day", daily), ("week", weekly)]:
                inventory_records.append(
                    {
                        "city": city,
                        "resolution_m": resolution,
                        "temporal_resolution": temporal,
                        "grids": int(frame["grid_id"].nunique()),
                        "periods": int(frame["date"].nunique()),
                        "panel_rows": len(frame),
                        "crime_count_all": int(frame["crime_count_all"].sum()),
                        "bike_total_flow": int(frame["bike_total_flow"].sum()),
                        "bike_covered_training_grids": int(frame.groupby("grid_id")["bike_coverage_training"].first().sum()),
                    }
                )
            accepted = _accepted_e02_panel(e02_data, city)
            if resolution == 1000:
                merged = daily[["grid_id", "date", *COUNT_COLUMNS]].merge(
                    accepted, on=["grid_id", "date"], suffixes=("_e16", "_e02"), how="left", validate="one_to_one"
                )
                if merged.filter(like="_e02").isna().any().any():
                    raise AssertionError(f"Missing E02 reconciliation rows: {city}")
                for column in COUNT_COLUMNS:
                    difference = pd.to_numeric(merged[f"{column}_e16"]) - pd.to_numeric(merged[f"{column}_e02"])
                    reconciliation_records.append(
                        {"city": city, "variable": column, "rows": len(merged), "max_abs_difference": float(difference.abs().max()), "sum_difference": float(difference.sum())}
                    )
            del panel, daily, weekly

    inventory = pd.DataFrame(inventory_records)
    inventory.to_csv(run_dir / "tables" / "multiscale_panel_inventory.csv", index=False)
    reconciliation = pd.DataFrame(reconciliation_records)
    reconciliation.to_csv(run_dir / "tables" / "e02_1km_daily_reconciliation.csv", index=False)

    for city in CITY_ORDER:
        for temporal in ["day", "week"]:
            part = inventory.loc[(inventory["city"].eq(city)) & (inventory["temporal_resolution"].eq(temporal))]
            for variable in ["crime_count_all", "bike_total_flow"]:
                values = part.set_index("resolution_m")[variable]
                conservation_records.append(
                    {
                        "city": city,
                        "temporal_resolution": temporal,
                        "variable": variable,
                        "value_500m": int(values.loc[500]),
                        "value_1000m": int(values.loc[1000]),
                        "value_2000m": int(values.loc[2000]),
                        "max_difference": int(max(values) - min(values)),
                    }
                )
    conservation = pd.DataFrame(conservation_records)
    conservation.to_csv(run_dir / "tables" / "scale_conservation.csv", index=False)

    native_records = []
    native_sources = {1000: e02_data, 500: e09_data / "resolution_500m", 2000: e09_data / "resolution_2000m"}
    for resolution, source in native_sources.items():
        for city in CITY_ORDER:
            files = sorted((source / "grid_day_panel" / f"city={city}").rglob("*.parquet"))
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
            native_records.append({"city": city, "resolution_m": resolution, "source_directory": str(source), "parquet_files": len(files), "panel_rows": rows, "source_status": "accepted_E02" if resolution == 1000 else "accepted_E09"})
    pd.DataFrame(native_records).to_csv(run_dir / "tables" / "native_domain_source_inventory.csv", index=False)

    checks = [
        ("six scale combinations per city", len(inventory), 18),
        ("daily periods equal 1096", int(inventory.loc[inventory["temporal_resolution"].eq("day"), "periods"].eq(1096).sum()), 9),
        ("weekly periods are complete", int(inventory.loc[inventory["temporal_resolution"].eq("week"), "periods"].eq(155).sum()), 9),
        ("row count equals grids times periods", int((inventory["panel_rows"] == inventory["grids"] * inventory["periods"]).sum()), 18),
        ("cross-spatial totals conserved", int(conservation["max_difference"].eq(0).sum()), len(conservation)),
        ("E02 1 km daily values reconcile", int(reconciliation["max_abs_difference"].eq(0).sum()), len(reconciliation)),
        ("nested hierarchy has 16 children per 2 km parent", int((hierarchy.groupby(["city", "parent_2km_grid_id"]).size() == 16).sum()), int(hierarchy.groupby(["city", "parent_2km_grid_id"]).ngroups)),
    ]
    acceptance = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    acceptance["status"] = np.where(acceptance["observed"].eq(acceptance["expected"]), "PASS", "FAIL")
    acceptance.to_csv(run_dir / "tables" / "acceptance_checklist.csv", index=False)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("S16.2 acceptance failed: " + "; ".join(acceptance.loc[acceptance["status"].ne("PASS"), "check"]))

    _qc_plot(inventory, run_dir / "figures", int(config["reporting"]["figure_dpi"]))
    figure_manifest = []
    for path in sorted((run_dir / "figures").iterdir()):
        figure_manifest.append({"file": path.name, "sha256": _sha256(path), "generator": "empirical/run_e16_multiscale_panel.py"})
    pd.DataFrame(figure_manifest).to_csv(run_dir / "tables" / "figure_manifest.csv", index=False)

    elapsed = time.monotonic() - started
    status = {
        "experiment": "E16",
        "stage": "S16.2",
        "status": "complete",
        "run_id": run_id,
        "source_e01_data": str(e01_data),
        "source_e02_data": str(e02_data),
        "source_e09_run": str(e09_run),
        "data_directory": str(data_dir),
        "panel_combinations": 18,
        "panel_rows": int(inventory["panel_rows"].sum()),
        "acceptance_passed": int(acceptance["status"].eq("PASS").sum()),
        "acceptance_total": len(acceptance),
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(timezone).isoformat(),
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (run_dir / "interpretation.md").write_text(
        "# S16.2 multiscale panel\n\n"
        "The accepted 1 km E02 domain defined complete nested 2 km parents. Their exact 500 m descendants were rebuilt once from accepted E01 standardized crime and bicycle endpoints; 1 km and 2 km panels were then obtained by exact summation. Complete ISO weeks contain seven days. All cross-resolution crime and bicycle totals reconcile exactly, and the common-core 1 km daily panel reconciles to accepted E02. No entropy, DELB, CMI, randomization, or prediction model was run.\n",
        encoding="utf-8",
    )
    (empirical_root / "runs" / "latest_e16_s16_2_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    latest_data = empirical_root / "processed_data" / "e16"
    latest_data.mkdir(parents=True, exist_ok=True)
    (latest_data / "latest_s16_2_data.txt").write_text(str(data_dir) + "\n", encoding="utf-8")
    logger.info("S16.2 complete in %.1f seconds", elapsed)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build E16 S16.2 common-core multiscale panels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--e02", required=True)
    parser.add_argument("--cities", required=True)
    parser.add_argument("--resume-run")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
