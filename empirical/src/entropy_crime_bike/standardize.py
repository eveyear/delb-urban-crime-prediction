from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


CRIME_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("source_event_id", pa.string()),
        ("city", pa.string()),
        ("source_year", pa.int16()),
        ("event_datetime_local", pa.timestamp("ns")),
        ("event_date", pa.date32()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("crime_type_raw", pa.string()),
        ("crime_type_unified", pa.string()),
        ("source_file", pa.string()),
        ("source_file_sha256", pa.string()),
        ("source_row_number", pa.int64()),
        ("qc_flags", pa.string()),
    ]
)

BICYCLE_SCHEMA = pa.schema(
    [
        ("trip_record_id", pa.string()),
        ("source_ride_id", pa.string()),
        ("city", pa.string()),
        ("source_month", pa.string()),
        ("started_at_local", pa.timestamp("ns")),
        ("ended_at_local", pa.timestamp("ns")),
        ("start_date", pa.date32()),
        ("end_date", pa.date32()),
        ("start_station_id", pa.string()),
        ("start_station_name", pa.string()),
        ("start_longitude", pa.float64()),
        ("start_latitude", pa.float64()),
        ("start_coordinate_source", pa.string()),
        ("start_flow_eligible", pa.bool_()),
        ("end_station_id", pa.string()),
        ("end_station_name", pa.string()),
        ("end_longitude", pa.float64()),
        ("end_latitude", pa.float64()),
        ("end_coordinate_source", pa.string()),
        ("end_flow_eligible", pa.bool_()),
        ("rideable_type", pa.string()),
        ("member_type", pa.string()),
        ("duration_seconds", pa.float64()),
        ("source_file", pa.string()),
        ("archive_member", pa.string()),
        ("source_file_sha256", pa.string()),
        ("source_row_number", pa.int64()),
        ("qc_flags", pa.string()),
    ]
)

EXCLUSION_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("dataset", pa.string()),
        ("city", pa.string()),
        ("source_period", pa.string()),
        ("source_file", pa.string()),
        ("archive_member", pa.string()),
        ("source_file_sha256", pa.string()),
        ("source_row_number", pa.int64()),
        ("primary_reason", pa.string()),
        ("all_reasons", pa.string()),
        ("raw_datetime", pa.string()),
        ("raw_start_station", pa.string()),
        ("raw_end_station", pa.string()),
    ]
)

CHUNK_DEFAULT = 250_000


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    data_dir: Path
    paths: dict[str, object]
    e01: dict[str, object]
    cities: dict[str, object]
    logger: logging.Logger
    sample_rows: int | None
    selected_cities: set[str]
    station_dictionary_override: Path | None


class AtomicParquetSink:
    def __init__(
        self,
        final_path: Path,
        schema: pa.Schema,
        compression: str,
        compression_level: int | None,
    ) -> None:
        self.final_path = final_path
        self.temp_path = final_path.with_name(f".{final_path.name}.partial")
        self.schema = schema
        self.compression = compression
        self.compression_level = compression_level
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.temp_path.exists():
            self.temp_path.unlink()

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(
            frame,
            schema=self.schema,
            preserve_index=False,
            safe=False,
        )
        if self.writer is None:
            unique_identifier_columns = {
                "event_id",
                "source_event_id",
                "trip_record_id",
                "source_ride_id",
                "record_id",
            }
            dictionary_columns = [
                field.name
                for field in self.schema
                if pa.types.is_string(field.type)
                and field.name not in unique_identifier_columns
            ]
            self.writer = pq.ParquetWriter(
                self.temp_path,
                self.schema,
                compression=self.compression,
                compression_level=self.compression_level,
                use_dictionary=dictionary_columns,
                write_statistics=True,
            )
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.temp_path.replace(self.final_path)
        self.writer = None

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.temp_path.exists():
            self.temp_path.unlink()


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return loaded


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e01")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e01_standardize.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _write_environment(path: Path) -> None:
    lines = [
        f"timestamp={datetime.now().isoformat()}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
        f"pyarrow={pa.__version__}",
    ]
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        lines.extend(["", "[pip-freeze]", freeze])
    except subprocess.SubprocessError as exc:
        lines.append(f"pip_freeze_error={exc}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _normal_column_map(columns: Iterable[object]) -> dict[str, str]:
    return {
        re.sub(r"\s+", " ", str(column).replace("\ufeff", "").strip()).lower(): str(
            column
        )
        for column in columns
    }


def _first_column(column_map: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        if name.lower() in column_map:
            return column_map[name.lower()]
    return None


def _clean_string(series: pd.Series, index: pd.Index | None = None) -> pd.Series:
    if index is not None and series.empty:
        series = pd.Series(pd.NA, index=index, dtype="string")
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def _empty_string(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="string")


def _get_string(
    frame: pd.DataFrame, column_map: dict[str, str], names: Iterable[str]
) -> pd.Series:
    column = _first_column(column_map, names)
    return _clean_string(frame[column]) if column else _empty_string(frame.index)


def _get_numeric(
    frame: pd.DataFrame, column_map: dict[str, str], names: Iterable[str]
) -> pd.Series:
    column = _first_column(column_map, names)
    if not column:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _get_datetime(
    frame: pd.DataFrame, column_map: dict[str, str], names: Iterable[str]
) -> pd.Series:
    column = _first_column(column_map, names)
    if not column:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce", format="mixed")


def _normalize_name_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def _normalize_names(series: pd.Series) -> pd.Series:
    return series.astype("string").map(_normalize_name_value).astype("string")


def _normalize_station_ids(series: pd.Series) -> pd.Series:
    value = series.astype("string").str.strip()
    value = value.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    return value.mask(value.eq(""))


def _station_key_series(ids: pd.Series, names: pd.Series) -> pd.Series:
    ids = _normalize_station_ids(ids)
    normalized_names = _normalize_names(names)
    result = pd.Series(pd.NA, index=ids.index, dtype="string")
    id_mask = ids.notna()
    result.loc[id_mask] = "I:" + ids.loc[id_mask]
    name_mask = ~id_mask & normalized_names.ne("")
    result.loc[name_mask] = "N:" + normalized_names.loc[name_mask]
    return result


def _valid_coordinates(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    return (
        latitude.notna()
        & longitude.notna()
        & latitude.between(-90, 90)
        & longitude.between(-180, 180)
    )


def _combine_flags(
    index: pd.Index, rules: list[tuple[str, pd.Series | np.ndarray]]
) -> pd.Series:
    result = np.full(len(index), "", dtype=object)
    for name, mask in rules:
        boolean = np.asarray(pd.Series(mask, index=index).fillna(False), dtype=bool)
        current = result[boolean]
        result[boolean] = np.where(current == "", name, current + "|" + name)
    return pd.Series(result, index=index, dtype="string")


def _source_period(path: Path) -> str:
    match = re.search(r"(20\d{2})[-_]?([01]\d)", path.name)
    if match and 1 <= int(match.group(2)) <= 12:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.search(r"(20\d{2})", path.name)
    return match.group(1) if match else ""


def _crime_city(path: Path) -> str:
    match = re.match(r"20\d{2}_(DC|NY|VAN)\.csv$", path.name, flags=re.I)
    return match.group(1).upper() if match else "UNKNOWN"


def _bike_city(path: Path) -> str:
    for part in path.parts:
        if part.upper() in {"DC", "NY", "VAN"}:
            return part.upper()
    return "UNKNOWN"


def _csv_members(path: Path) -> list[str]:
    if path.suffix.lower() != ".zip":
        return [path.name]
    with zipfile.ZipFile(path) as archive:
        return [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and "/._" not in name
        ]


def _iter_member_chunks(
    path: Path,
    member_name: str,
    chunk_size: int,
    sample_rows: int | None,
) -> Iterator[pd.DataFrame]:
    kwargs = {
        "chunksize": chunk_size,
        "nrows": sample_rows,
        "dtype": str,
        "encoding": "utf-8-sig",
        "encoding_errors": "replace",
        "low_memory": False,
        "on_bad_lines": "error",
        "skip_blank_lines": False,
    }
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open(member_name) as stream:
                yield from pd.read_csv(stream, **kwargs)
    else:
        yield from pd.read_csv(path, **kwargs)


def _input_files(paths: dict[str, object]) -> tuple[list[Path], list[Path], Path]:
    crime_root = Path(str(paths["crime_raw"]))
    bicycle_root = Path(str(paths["bicycle_raw"]))
    crime = sorted(crime_root.glob("20??_*.csv"))
    bicycle = sorted((bicycle_root / "DC").rglob("*.zip"))
    bicycle.extend(sorted((bicycle_root / "NY").rglob("*.zip")))
    bicycle.extend(
        sorted((bicycle_root / "VAN").glob("Mobi_System_Data_20??-??.csv"))
    )
    station = bicycle_root / "VAN" / "station_information_current.csv"
    return crime, bicycle, station


def _e00_hashes(paths: dict[str, object]) -> dict[str, str]:
    runs = Path(str(paths["runs"]))
    pointer = runs / "latest_e00_run.txt"
    if not pointer.exists():
        return {}
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    manifest = run_dir / "tables" / "input_manifest.csv"
    if not manifest.exists():
        return {}
    frame = pd.read_csv(manifest, dtype=str)
    return dict(zip(frame["absolute_path"], frame["sha256"], strict=False))


def _build_input_manifest(
    crime: list[Path],
    bicycle: list[Path],
    station_current: Path,
    historical_station: Path | None,
    known_hashes: dict[str, str],
    logger: logging.Logger,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    inputs: list[tuple[str, Path]] = [
        *(("crime", path) for path in crime),
        *(("bicycle", path) for path in bicycle),
        ("station_reference_current", station_current),
    ]
    if historical_station and historical_station.exists():
        inputs.append(("station_reference_historical", historical_station))
    for dataset, path in inputs:
        digest = known_hashes.get(str(path))
        if not digest:
            logger.info("Hashing new E01 input: %s", path)
            digest = _sha256(path)
        stat = path.stat()
        records.append(
            {
                "dataset": dataset,
                "city": (
                    _crime_city(path)
                    if dataset == "crime"
                    else _bike_city(path)
                    if dataset == "bicycle"
                    else "VAN"
                ),
                "source_period": _source_period(path),
                "absolute_path": str(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "sha256": digest,
                "e01_role": (
                    "canonical_event_input"
                    if dataset in {"crime", "bicycle"}
                    else "coordinate_reference"
                ),
            }
        )
    return pd.DataFrame(records)


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return float("nan")
    value_array = values.loc[valid].to_numpy(dtype=float)
    weight_array = weights.loc[valid].to_numpy(dtype=float)
    order = np.argsort(value_array)
    value_array = value_array[order]
    weight_array = weight_array[order]
    threshold = weight_array.sum() / 2
    return float(value_array[np.searchsorted(np.cumsum(weight_array), threshold)])


def _haversine_m(
    latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float
) -> float:
    radius = 6_371_008.8
    lat1, lat2 = np.radians([latitude_1, latitude_2])
    dlat = lat2 - lat1
    dlon = np.radians(longitude_2 - longitude_1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def _extract_station_fields(
    frame: pd.DataFrame, column_map: dict[str, str], side: str, city: str
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if side == "start":
        station_id_names = ["start_station_id", "start station number"]
        station_name_names = ["start_station_name", "start station"]
        combined_names = ["departure station"]
        latitude_names = ["start_lat"]
        longitude_names = ["start_lng"]
    else:
        station_id_names = ["end_station_id", "end station number"]
        station_name_names = ["end_station_name", "end station"]
        combined_names = ["return station"]
        latitude_names = ["end_lat"]
        longitude_names = ["end_lng"]

    station_id = _get_string(frame, column_map, station_id_names)
    station_name = _get_string(frame, column_map, station_name_names)
    if city == "VAN":
        combined = _get_string(frame, column_map, combined_names)
        extracted = combined.str.extract(r"^\s*(\d{4})\b", expand=False)
        station_id = _normalize_station_ids(extracted)
        station_name = combined.str.replace(r"^\s*\d{4}\s*", "", regex=True).str.strip()
        station_name = station_name.mask(station_name.eq(""))
    latitude = _get_numeric(frame, column_map, latitude_names)
    longitude = _get_numeric(frame, column_map, longitude_names)
    return station_id, station_name, latitude, longitude


def build_historical_station_dictionary(
    bicycle_files: list[Path],
    manifest_hashes: dict[str, str],
    ctx: RunContext,
) -> pd.DataFrame:
    output_csv = ctx.run_dir / "tables" / "historical_station_dictionary.csv"
    output_parquet = ctx.data_dir / "reference" / "historical_station_dictionary.parquet"
    if output_csv.exists() and output_parquet.exists():
        ctx.logger.info("Reusing completed historical station dictionary")
        return pd.read_csv(output_csv, dtype={"station_id": str})
    if ctx.station_dictionary_override is not None:
        source = ctx.station_dictionary_override
        ctx.logger.info(
            "Reusing verified station dictionary %s (sha256=%s)",
            source,
            _sha256(source),
        )
        result = pd.read_csv(source, dtype={"station_id": str})
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        result.to_parquet(output_parquet, index=False, compression="zstd")
        return result

    partials: list[pd.DataFrame] = []
    chunk_size = int(ctx.e01.get("chunk_size", CHUNK_DEFAULT))
    candidates = [
        path
        for path in bicycle_files
        if _bike_city(path) in {"DC", "NY"}
        and _bike_city(path) in ctx.selected_cities
    ]
    for file_index, path in enumerate(candidates, start=1):
        city = _bike_city(path)
        source_month = _source_period(path)
        ctx.logger.info(
            "Station prepass %d/%d: %s", file_index, len(candidates), path.name
        )
        for member_name in _csv_members(path):
            for chunk in _iter_member_chunks(
                path, member_name, chunk_size, ctx.sample_rows
            ):
                chunk = chunk.loc[~chunk.isna().all(axis=1)]
                if chunk.empty:
                    continue
                column_map = _normal_column_map(chunk.columns)
                for side in ("start", "end"):
                    station_id, station_name, latitude, longitude = (
                        _extract_station_fields(chunk, column_map, side, city)
                    )
                    valid = _valid_coordinates(latitude, longitude)
                    if not valid.any():
                        continue
                    keys = _station_key_series(station_id, station_name)
                    normalized_names = _normalize_names(station_name)
                    usable = valid & keys.notna()
                    small = pd.DataFrame(
                        {
                            "station_key": keys.loc[usable],
                            "station_id": _normalize_station_ids(
                                station_id.loc[usable]
                            ),
                            "station_name": station_name.loc[usable],
                            "normalized_name": normalized_names.loc[usable],
                            "latitude": latitude.loc[usable],
                            "longitude": longitude.loc[usable],
                        }
                    )
                    if small.empty:
                        continue
                    grouped = (
                        small.groupby("station_key", sort=False, dropna=False)
                        .agg(
                            station_id=("station_id", "first"),
                            station_name=("station_name", "first"),
                            normalized_name=("normalized_name", "first"),
                            latitude=("latitude", "median"),
                            longitude=("longitude", "median"),
                            observations=("station_key", "size"),
                        )
                        .reset_index()
                    )
                    grouped.insert(0, "source_month", source_month)
                    grouped.insert(0, "city", city)
                    partials.append(grouped)

    if not partials:
        columns = [
            "city",
            "source_month",
            "station_key",
            "station_id",
            "station_name",
            "normalized_name",
            "latitude",
            "longitude",
            "observations",
            "episode_id",
            "movement_from_previous_m",
        ]
        result = pd.DataFrame(columns=columns)
    else:
        partial = pd.concat(partials, ignore_index=True)
        records: list[dict[str, object]] = []
        group_columns = ["city", "source_month", "station_key"]
        for (city, month, key), group in partial.groupby(
            group_columns, sort=True, dropna=False
        ):
            station_ids = group["station_id"].dropna()
            station_names = group["station_name"].dropna()
            normalized_names = group["normalized_name"].dropna()
            records.append(
                {
                    "city": city,
                    "source_month": month,
                    "station_key": key,
                    "station_id": (
                        station_ids.mode().iloc[0] if not station_ids.empty else pd.NA
                    ),
                    "station_name": (
                        station_names.mode().iloc[0]
                        if not station_names.empty
                        else pd.NA
                    ),
                    "normalized_name": (
                        normalized_names.mode().iloc[0]
                        if not normalized_names.empty
                        else ""
                    ),
                    "latitude": _weighted_median(
                        group["latitude"], group["observations"]
                    ),
                    "longitude": _weighted_median(
                        group["longitude"], group["observations"]
                    ),
                    "observations": int(group["observations"].sum()),
                }
            )
        result = pd.DataFrame(records)
        threshold = float(
            ctx.e01["station_resolution"].get("movement_threshold_m", 100)
        )
        result["episode_id"] = 0
        result["movement_from_previous_m"] = np.nan
        for (_, _), positions in result.groupby(
            ["city", "station_key"], sort=False
        ).groups.items():
            ordered = result.loc[list(positions)].sort_values("source_month")
            episode = 1
            previous: pd.Series | None = None
            for row_index, row in ordered.iterrows():
                movement = np.nan
                if previous is not None:
                    movement = _haversine_m(
                        float(previous["latitude"]),
                        float(previous["longitude"]),
                        float(row["latitude"]),
                        float(row["longitude"]),
                    )
                    if movement > threshold:
                        episode += 1
                result.loc[row_index, "episode_id"] = episode
                result.loc[row_index, "movement_from_previous_m"] = movement
                previous = row

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    result.to_parquet(output_parquet, index=False, compression="zstd")
    ctx.logger.info("Historical station dictionary rows: %d", len(result))
    return result


class StationResolver:
    def __init__(
        self,
        historical: pd.DataFrame,
        vancouver_current_path: Path,
        vancouver_historical_path: Path | None,
        expected_current_hash: str,
        non_public_patterns: list[str],
    ) -> None:
        self.exact: dict[tuple[str, str, str], tuple[float, float]] = {}
        self.by_key: dict[tuple[str, str], list[tuple[str, float, float]]] = (
            defaultdict(list)
        )
        self.name_alias: dict[tuple[str, str], str] = {}
        self.non_public_regex = re.compile(
            "|".join(re.escape(pattern) for pattern in non_public_patterns), re.I
        )
        self.unresolved_counts: Counter[tuple[str, str, str, str]] = Counter()
        self.non_public_counts: Counter[tuple[str, str, str, str]] = Counter()

        name_keys: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        for row in historical.itertuples(index=False):
            key = str(row.station_key)
            month = str(row.source_month)
            city = str(row.city)
            latitude = float(row.latitude)
            longitude = float(row.longitude)
            self.exact[(city, month, key)] = (latitude, longitude)
            self.by_key[(city, key)].append((month, latitude, longitude))
            normalized_name = str(row.normalized_name or "")
            if normalized_name:
                name_keys[(city, normalized_name)].add(key)
        for name_key, keys in name_keys.items():
            if len(keys) == 1:
                self.name_alias[name_key] = next(iter(keys))
        for key in self.by_key:
            self.by_key[key].sort(key=lambda value: value[0])

        actual_hash = _sha256(vancouver_current_path)
        if actual_hash != expected_current_hash:
            raise ValueError(
                "Vancouver station snapshot hash mismatch: "
                f"expected {expected_current_hash}, found {actual_hash}"
            )
        current = pd.read_csv(
            vancouver_current_path, dtype=str, encoding="utf-8-sig"
        )
        current["station_id"] = _normalize_station_ids(current["station_id"])
        current["normalized_name"] = _normalize_names(current["name"])
        current["lat"] = pd.to_numeric(current["lat"], errors="coerce")
        current["lon"] = pd.to_numeric(current["lon"], errors="coerce")
        if current["station_id"].duplicated().any():
            raise ValueError("Duplicate station_id in Vancouver current station file")
        self.van_current_id = {
            str(row.station_id): (float(row.lat), float(row.lon))
            for row in current.itertuples(index=False)
        }
        name_counts = current["normalized_name"].value_counts()
        self.van_current_name = {
            str(row.normalized_name): (float(row.lat), float(row.lon))
            for row in current.itertuples(index=False)
            if name_counts.get(row.normalized_name, 0) == 1
        }

        self.van_historical_id: dict[str, tuple[float, float]] = {}
        self.van_historical_name: dict[str, tuple[float, float]] = {}
        if vancouver_historical_path and vancouver_historical_path.exists():
            archived = pd.read_csv(
                vancouver_historical_path, dtype=str, encoding="utf-8-sig"
            )
            required = {"station_id", "name", "lat", "lon", "source_url"}
            missing = required.difference(archived.columns)
            if missing:
                raise ValueError(
                    "Historical Vancouver station file is missing: "
                    + ", ".join(sorted(missing))
                )
            archived["station_id"] = _normalize_station_ids(archived["station_id"])
            archived["normalized_name"] = _normalize_names(archived["name"])
            archived["lat"] = pd.to_numeric(archived["lat"], errors="coerce")
            archived["lon"] = pd.to_numeric(archived["lon"], errors="coerce")
            archived = archived.loc[
                _valid_coordinates(archived["lat"], archived["lon"])
            ]
            self.van_historical_id = {
                str(row.station_id): (float(row.lat), float(row.lon))
                for row in archived.itertuples(index=False)
                if pd.notna(row.station_id)
            }
            self.van_historical_name = {
                str(row.normalized_name): (float(row.lat), float(row.lon))
                for row in archived.itertuples(index=False)
                if row.normalized_name
            }

    @staticmethod
    def _month_distance(month_1: str, month_2: str) -> int:
        return abs(pd.Period(month_1, freq="M").ordinal - pd.Period(month_2, freq="M").ordinal)

    def _historical_lookup(
        self, city: str, month: str, station_id: str, normalized_name: str
    ) -> tuple[float, float, str] | None:
        key = f"I:{station_id}" if station_id else ""
        if not key and normalized_name:
            key = self.name_alias.get((city, normalized_name), f"N:{normalized_name}")
        if key:
            exact = self.exact.get((city, month, key))
            if exact:
                return exact[0], exact[1], "historical_trip_station_month"
            candidates = self.by_key.get((city, key), [])
            if candidates:
                nearest = min(
                    candidates, key=lambda item: self._month_distance(item[0], month)
                )
                return (
                    nearest[1],
                    nearest[2],
                    "historical_trip_station_nearest_month",
                )
        if normalized_name:
            alias_key = self.name_alias.get((city, normalized_name))
            if alias_key:
                exact = self.exact.get((city, month, alias_key))
                if exact:
                    return exact[0], exact[1], "historical_trip_station_month"
                candidates = self.by_key.get((city, alias_key), [])
                if candidates:
                    nearest = min(
                        candidates,
                        key=lambda item: self._month_distance(item[0], month),
                    )
                    return (
                        nearest[1],
                        nearest[2],
                        "historical_trip_station_nearest_month",
                    )
        return None

    def resolve(
        self,
        city: str,
        month: str,
        station_ids: pd.Series,
        station_names: pd.Series,
        source_latitudes: pd.Series,
        source_longitudes: pd.Series,
        side: str,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        station_ids = _normalize_station_ids(station_ids)
        normalized_names = _normalize_names(station_names)
        valid_source = _valid_coordinates(source_latitudes, source_longitudes)
        latitude = source_latitudes.astype(float).copy()
        longitude = source_longitudes.astype(float).copy()
        source = pd.Series("unresolved", index=station_ids.index, dtype="string")
        source.loc[valid_source] = "source"
        if city == "VAN":
            non_public = station_names.fillna("").astype(str).str.contains(
                self.non_public_regex, na=False
            )
        elif city == "NY":
            # Citi Bike uses SYS-prefixed identifiers for test/shop nodes.
            # City-specific logic prevents false matches such as Bayard Street.
            non_public = station_ids.fillna("").str.upper().str.startswith(
                "SYS"
            ) | station_names.fillna("").astype(str).str.contains(
                r"\btech shop\b|\bstation demo\b", case=False, regex=True, na=False
            )
        else:
            # Capital Bikeshare street names can legitimately contain words such
            # as "Temporary"; no non-public source identifiers were found in E00.
            non_public = pd.Series(False, index=station_ids.index)

        unresolved = ~valid_source & ~non_public
        combinations = pd.DataFrame(
            {
                "station_id": station_ids.loc[unresolved].fillna(""),
                "normalized_name": normalized_names.loc[unresolved].fillna(""),
            }
        ).drop_duplicates()
        lookup: dict[tuple[str, str], tuple[float, float, str] | None] = {}
        for row in combinations.itertuples(index=False):
            station_id = str(row.station_id)
            normalized_name = str(row.normalized_name)
            resolved: tuple[float, float, str] | None = None
            if city in {"DC", "NY"}:
                resolved = self._historical_lookup(
                    city, month, station_id, normalized_name
                )
            elif city == "VAN":
                if station_id in self.van_current_id:
                    lat, lon = self.van_current_id[station_id]
                    resolved = lat, lon, "current_snapshot_retrofit"
                elif normalized_name in self.van_current_name:
                    lat, lon = self.van_current_name[normalized_name]
                    resolved = lat, lon, "current_snapshot_retrofit"
                elif station_id in self.van_historical_id:
                    lat, lon = self.van_historical_id[station_id]
                    resolved = lat, lon, "official_historical_snapshot"
                elif normalized_name in self.van_historical_name:
                    lat, lon = self.van_historical_name[normalized_name]
                    resolved = lat, lon, "official_historical_snapshot"
            lookup[(station_id, normalized_name)] = resolved

        for (station_id, normalized_name), resolved in lookup.items():
            mask = (
                unresolved
                & station_ids.fillna("").eq(station_id)
                & normalized_names.fillna("").eq(normalized_name)
            )
            count = int(mask.sum())
            if resolved is None:
                display = (
                    station_names.loc[mask].dropna().iloc[0]
                    if station_names.loc[mask].notna().any()
                    else ""
                )
                self.unresolved_counts[(city, side, station_id, str(display))] += count
            else:
                latitude.loc[mask] = resolved[0]
                longitude.loc[mask] = resolved[1]
                source.loc[mask] = resolved[2]

        source.loc[non_public] = "non_public_node"
        for station_id, name, count in (
            pd.DataFrame(
                {
                    "station_id": station_ids.loc[non_public].fillna(""),
                    "station_name": station_names.loc[non_public].fillna(""),
                }
            )
            .value_counts()
            .rename("count")
            .reset_index()
            .itertuples(index=False)
        ):
            self.non_public_counts[
                (city, side, str(station_id), str(name))
            ] += int(count)

        resolved_valid = _valid_coordinates(latitude, longitude)
        latitude = latitude.where(resolved_valid)
        longitude = longitude.where(resolved_valid)
        eligible = resolved_valid & ~non_public
        return latitude, longitude, source, eligible

    def unresolved_table(self) -> pd.DataFrame:
        records = [
            {
                "city": city,
                "endpoint": endpoint,
                "station_id": station_id,
                "station_name": station_name,
                "row_count": count,
                "classification": "unresolved_public_or_unknown",
            }
            for (city, endpoint, station_id, station_name), count in self.unresolved_counts.items()
        ]
        records.extend(
            {
                "city": city,
                "endpoint": endpoint,
                "station_id": station_id,
                "station_name": station_name,
                "row_count": count,
                "classification": "non_public_node",
            }
            for (city, endpoint, station_id, station_name), count in self.non_public_counts.items()
        )
        if not records:
            return pd.DataFrame(
                columns=[
                    "city",
                    "endpoint",
                    "station_id",
                    "station_name",
                    "row_count",
                    "classification",
                ]
            )
        return pd.DataFrame(records).sort_values(
            ["city", "classification", "endpoint", "row_count"],
            ascending=[True, True, True, False],
        )


def _make_exclusion_frame(
    *,
    record_ids: pd.Series,
    dataset: str,
    city: str,
    source_period: str,
    source_file: Path,
    archive_member: str,
    source_hash: str,
    row_numbers: pd.Series,
    primary_reason: pd.Series,
    all_reasons: pd.Series,
    raw_datetime: pd.Series,
    raw_start_station: pd.Series | None = None,
    raw_end_station: pd.Series | None = None,
) -> pd.DataFrame:
    index = record_ids.index
    return pd.DataFrame(
        {
            "record_id": record_ids.astype("string"),
            "dataset": dataset,
            "city": city,
            "source_period": source_period,
            "source_file": str(source_file),
            "archive_member": archive_member,
            "source_file_sha256": source_hash,
            "source_row_number": row_numbers.astype("int64"),
            "primary_reason": primary_reason.astype("string"),
            "all_reasons": all_reasons.astype("string"),
            "raw_datetime": raw_datetime.astype("string"),
            "raw_start_station": (
                raw_start_station.astype("string")
                if raw_start_station is not None
                else _empty_string(index)
            ),
            "raw_end_station": (
                raw_end_station.astype("string")
                if raw_end_station is not None
                else _empty_string(index)
            ),
        },
        index=index,
    ).reset_index(drop=True)


def standardize_crime(
    crime_files: list[Path],
    manifest_hashes: dict[str, str],
    ctx: RunContext,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    allowed = set(ctx.e01["crime"]["allowed_types"])
    source_id_fields = ctx.e01["crime"].get("source_id_fields", {})
    chunk_size = int(ctx.e01.get("chunk_size", CHUNK_DEFAULT))
    compression = str(ctx.e01["parquet"].get("compression", "zstd"))
    compression_level = int(ctx.e01["parquet"].get("compression_level", 6))

    selected = [
        path for path in crime_files if _crime_city(path) in ctx.selected_cities
    ]
    for file_index, path in enumerate(selected, start=1):
        city = _crime_city(path)
        year = int(_source_period(path))
        source_hash = manifest_hashes[str(path)]
        part = f"part-{source_hash[:12]}"
        final_path = (
            ctx.data_dir
            / "crime_events"
            / f"city={city}"
            / f"year={year}"
            / f"{part}.parquet"
        )
        exclusion_path = (
            ctx.data_dir
            / "exclusions"
            / "crime"
            / f"city={city}"
            / f"year={year}"
            / f"{part}.parquet"
        )
        retained_sink = AtomicParquetSink(
            final_path, CRIME_SCHEMA, compression, compression_level
        )
        exclusion_sink = AtomicParquetSink(
            exclusion_path, EXCLUSION_SCHEMA, compression, compression_level
        )
        totals = Counter()
        row_offset = 0
        ctx.logger.info(
            "Crime %d/%d: %s", file_index, len(selected), path.name
        )
        try:
            for chunk in pd.read_csv(
                path,
                chunksize=chunk_size,
                nrows=ctx.sample_rows,
                dtype=str,
                encoding="utf-8-sig",
                encoding_errors="replace",
                low_memory=False,
                on_bad_lines="error",
                skip_blank_lines=False,
            ):
                rows = len(chunk)
                row_numbers = pd.Series(
                    np.arange(row_offset + 1, row_offset + rows + 1),
                    index=chunk.index,
                    dtype="int64",
                )
                row_offset += rows
                totals["physical_rows"] += rows
                column_map = _normal_column_map(chunk.columns)
                raw_datetime = _get_string(chunk, column_map, ["datetime"])
                datetimes = _get_datetime(chunk, column_map, ["datetime"])
                latitude = _get_numeric(chunk, column_map, ["latitude"])
                longitude = _get_numeric(chunk, column_map, ["longitude"])
                crime_raw = _get_string(chunk, column_map, ["crime_type_raw"])
                crime_unified = _get_string(
                    chunk, column_map, ["crime_type_unified"]
                )
                source_id_field = source_id_fields.get(city)
                source_event_id = (
                    _get_string(chunk, column_map, [str(source_id_field)])
                    if source_id_field
                    else _empty_string(chunk.index)
                )
                record_ids = pd.Series(
                    [
                        f"CR-{city}-{source_hash[:12]}-{number:09d}"
                        for number in row_numbers
                    ],
                    index=chunk.index,
                    dtype="string",
                )
                invalid_datetime = datetimes.isna()
                outside_year = (
                    datetimes.notna()
                    & (
                        (datetimes < pd.Timestamp(f"{year}-01-01"))
                        | (datetimes >= pd.Timestamp(f"{year + 1}-01-01"))
                    )
                )
                invalid_coordinate = ~_valid_coordinates(latitude, longitude)
                missing_type = crime_unified.isna()
                unexpected_type = crime_unified.notna() & ~crime_unified.isin(allowed)
                all_reasons = _combine_flags(
                    chunk.index,
                    [
                        ("invalid_datetime", invalid_datetime),
                        ("outside_source_year", outside_year),
                        ("invalid_or_missing_coordinate", invalid_coordinate),
                        ("missing_unified_crime_type", missing_type),
                        ("unexpected_unified_crime_type", unexpected_type),
                    ],
                )
                excluded = all_reasons.ne("")
                primary_reason = pd.Series("", index=chunk.index, dtype="string")
                for reason, mask in [
                    ("invalid_datetime", invalid_datetime),
                    ("outside_source_year", outside_year),
                    ("invalid_or_missing_coordinate", invalid_coordinate),
                    ("missing_unified_crime_type", missing_type),
                    ("unexpected_unified_crime_type", unexpected_type),
                ]:
                    primary_reason = primary_reason.mask(
                        primary_reason.eq("") & mask, reason
                    )

                keep = ~excluded
                retained = pd.DataFrame(
                    {
                        "event_id": record_ids.loc[keep],
                        "source_event_id": source_event_id.loc[keep],
                        "city": city,
                        "source_year": year,
                        "event_datetime_local": datetimes.loc[keep],
                        "event_date": datetimes.loc[keep].dt.date,
                        "latitude": latitude.loc[keep],
                        "longitude": longitude.loc[keep],
                        "crime_type_raw": crime_raw.loc[keep],
                        "crime_type_unified": crime_unified.loc[keep],
                        "source_file": str(path),
                        "source_file_sha256": source_hash,
                        "source_row_number": row_numbers.loc[keep],
                        "qc_flags": "",
                    }
                ).reset_index(drop=True)
                retained_sink.write(retained)
                if excluded.any():
                    exclusions = _make_exclusion_frame(
                        record_ids=record_ids.loc[excluded],
                        dataset="crime",
                        city=city,
                        source_period=str(year),
                        source_file=path,
                        archive_member=path.name,
                        source_hash=source_hash,
                        row_numbers=row_numbers.loc[excluded],
                        primary_reason=primary_reason.loc[excluded],
                        all_reasons=all_reasons.loc[excluded],
                        raw_datetime=raw_datetime.loc[excluded],
                    )
                    exclusion_sink.write(exclusions)

                totals["retained_rows"] += int(keep.sum())
                totals["excluded_rows"] += int(excluded.sum())
                totals["outside_source_period_rows"] += int(outside_year.sum())
                totals["invalid_datetime_rows"] += int(invalid_datetime.sum())
                totals["invalid_coordinate_rows"] += int(
                    invalid_coordinate.sum()
                )
                totals["missing_or_unexpected_type_rows"] += int(
                    (missing_type | unexpected_type).sum()
                )
        except Exception:
            retained_sink.abort()
            exclusion_sink.abort()
            raise
        retained_sink.close()
        exclusion_sink.close()
        records.append(
            {
                "dataset": "crime",
                "city": city,
                "source_period": str(year),
                "source_file": str(path),
                **totals,
            }
        )
    return pd.DataFrame(records)


def _bike_rideable_type(
    frame: pd.DataFrame, column_map: dict[str, str]
) -> pd.Series:
    rideable = _get_string(frame, column_map, ["rideable_type"])
    electric_column = _first_column(column_map, ["electric bike"])
    if electric_column:
        electric = (
            frame[electric_column].astype("string").str.strip().str.casefold()
        )
        mapped = electric.map(
            {
                "true": "electric_bike",
                "false": "classic_bike",
                "1": "electric_bike",
                "0": "classic_bike",
            }
        )
        rideable = rideable.fillna(mapped)
    return rideable


def standardize_bicycle(
    bicycle_files: list[Path],
    manifest_hashes: dict[str, str],
    resolver: StationResolver,
    ctx: RunContext,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    chunk_size = int(ctx.e01.get("chunk_size", CHUNK_DEFAULT))
    compression = str(ctx.e01["parquet"].get("compression", "zstd"))
    compression_level = int(ctx.e01["parquet"].get("compression_level", 6))
    selected = [
        path for path in bicycle_files if _bike_city(path) in ctx.selected_cities
    ]
    for file_index, path in enumerate(selected, start=1):
        city = _bike_city(path)
        source_month = _source_period(path)
        year, month = source_month.split("-")
        source_hash = manifest_hashes[str(path)]
        ctx.logger.info(
            "Bicycle %d/%d: %s", file_index, len(selected), path.name
        )
        for member_index, member_name in enumerate(_csv_members(path), start=1):
            member_token = hashlib.sha256(member_name.encode("utf-8")).hexdigest()[:8]
            part = (
                f"part-{source_hash[:12]}-{member_index:03d}-{member_token}"
            )
            final_path = (
                ctx.data_dir
                / "bicycle_trips"
                / f"city={city}"
                / f"year={year}"
                / f"month={month}"
                / f"{part}.parquet"
            )
            exclusion_path = (
                ctx.data_dir
                / "exclusions"
                / "bicycle"
                / f"city={city}"
                / f"year={year}"
                / f"month={month}"
                / f"{part}.parquet"
            )
            retained_sink = AtomicParquetSink(
                final_path, BICYCLE_SCHEMA, compression, compression_level
            )
            exclusion_sink = AtomicParquetSink(
                exclusion_path, EXCLUSION_SCHEMA, compression, compression_level
            )
            totals = Counter()
            row_offset = 0
            try:
                for chunk in _iter_member_chunks(
                    path, member_name, chunk_size, ctx.sample_rows
                ):
                    rows = len(chunk)
                    row_numbers = pd.Series(
                        np.arange(row_offset + 1, row_offset + rows + 1),
                        index=chunk.index,
                        dtype="int64",
                    )
                    row_offset += rows
                    totals["physical_rows"] += rows
                    blank = chunk.isna().all(axis=1)
                    totals["blank_rows"] += int(blank.sum())
                    column_map = _normal_column_map(chunk.columns)
                    started_raw = _get_string(
                        chunk, column_map, ["started_at", "start date", "departure"]
                    )
                    started = _get_datetime(
                        chunk, column_map, ["started_at", "start date", "departure"]
                    )
                    ended = _get_datetime(
                        chunk, column_map, ["ended_at", "end date", "return"]
                    )
                    start_id, start_name, start_lat_source, start_lon_source = (
                        _extract_station_fields(chunk, column_map, "start", city)
                    )
                    end_id, end_name, end_lat_source, end_lon_source = (
                        _extract_station_fields(chunk, column_map, "end", city)
                    )
                    source_ride_id = _get_string(
                        chunk, column_map, ["ride_id"]
                    )
                    rideable_type = _bike_rideable_type(chunk, column_map)
                    member_type = _get_string(
                        chunk,
                        column_map,
                        ["member_casual", "member type", "membership type", "formula"],
                    )
                    duration = _get_numeric(
                        chunk, column_map, ["duration", "duration (sec.)"]
                    )
                    duration = duration.where(
                        duration.notna(), (ended - started).dt.total_seconds()
                    )

                    record_ids = pd.Series(
                        [
                            (
                                f"BI-{city}-{source_hash[:12]}-"
                                f"{member_index:03d}-{number:09d}"
                            )
                            for number in row_numbers
                        ],
                        index=chunk.index,
                        dtype="string",
                    )
                    invalid_start = ~blank & started.isna()
                    period = pd.Period(source_month, freq="M")
                    outside_month = (
                        ~blank
                        & started.notna()
                        & (
                            (started < period.start_time)
                            | (started >= (period + 1).start_time)
                        )
                    )
                    all_exclusion_reasons = _combine_flags(
                        chunk.index,
                        [
                            ("blank_source_row", blank),
                            ("invalid_start_datetime", invalid_start),
                            ("outside_source_month", outside_month),
                        ],
                    )
                    excluded = all_exclusion_reasons.ne("")
                    primary_reason = pd.Series(
                        "", index=chunk.index, dtype="string"
                    )
                    for reason, mask in [
                        ("blank_source_row", blank),
                        ("invalid_start_datetime", invalid_start),
                        ("outside_source_month", outside_month),
                    ]:
                        primary_reason = primary_reason.mask(
                            primary_reason.eq("") & mask, reason
                        )

                    keep = ~excluded
                    keep_index = chunk.index[keep]
                    (
                        start_lat,
                        start_lon,
                        start_coordinate_source,
                        start_flow_eligible,
                    ) = resolver.resolve(
                        city,
                        source_month,
                        start_id.loc[keep],
                        start_name.loc[keep],
                        start_lat_source.loc[keep],
                        start_lon_source.loc[keep],
                        "start",
                    )
                    (
                        end_lat,
                        end_lon,
                        end_coordinate_source,
                        end_flow_eligible,
                    ) = resolver.resolve(
                        city,
                        source_month,
                        end_id.loc[keep],
                        end_name.loc[keep],
                        end_lat_source.loc[keep],
                        end_lon_source.loc[keep],
                        "end",
                    )
                    trip_has_non_public = start_coordinate_source.eq(
                        "non_public_node"
                    ) | end_coordinate_source.eq("non_public_node")
                    start_flow_eligible = (
                        start_flow_eligible & ~trip_has_non_public
                    )
                    end_flow_eligible = (
                        end_flow_eligible
                        & ended.loc[keep].notna()
                        & ~trip_has_non_public
                    )
                    qc_flags = _combine_flags(
                        keep_index,
                        [
                            (
                                "missing_start_station",
                                start_id.loc[keep].isna()
                                & start_name.loc[keep].isna(),
                            ),
                            (
                                "missing_end_station",
                                end_id.loc[keep].isna()
                                & end_name.loc[keep].isna(),
                            ),
                            ("invalid_end_datetime", ended.loc[keep].isna()),
                            (
                                "nonpositive_duration",
                                duration.loc[keep].le(0),
                            ),
                            (
                                "start_coordinate_imputed",
                                ~start_coordinate_source.isin(
                                    ["source", "unresolved", "non_public_node"]
                                ),
                            ),
                            (
                                "end_coordinate_imputed",
                                ~end_coordinate_source.isin(
                                    ["source", "unresolved", "non_public_node"]
                                ),
                            ),
                            (
                                "start_coordinate_unresolved",
                                start_coordinate_source.eq("unresolved"),
                            ),
                            (
                                "end_coordinate_unresolved",
                                end_coordinate_source.eq("unresolved"),
                            ),
                            ("non_public_endpoint", trip_has_non_public),
                            (
                                "current_snapshot_retrofit",
                                start_coordinate_source.eq(
                                    "current_snapshot_retrofit"
                                )
                                | end_coordinate_source.eq(
                                    "current_snapshot_retrofit"
                                ),
                            ),
                        ],
                    )
                    retained = pd.DataFrame(
                        {
                            "trip_record_id": record_ids.loc[keep],
                            "source_ride_id": source_ride_id.loc[keep],
                            "city": city,
                            "source_month": source_month,
                            "started_at_local": started.loc[keep],
                            "ended_at_local": ended.loc[keep],
                            "start_date": started.loc[keep].dt.date,
                            "end_date": ended.loc[keep].dt.date,
                            "start_station_id": start_id.loc[keep],
                            "start_station_name": start_name.loc[keep],
                            "start_longitude": start_lon,
                            "start_latitude": start_lat,
                            "start_coordinate_source": start_coordinate_source,
                            "start_flow_eligible": start_flow_eligible,
                            "end_station_id": end_id.loc[keep],
                            "end_station_name": end_name.loc[keep],
                            "end_longitude": end_lon,
                            "end_latitude": end_lat,
                            "end_coordinate_source": end_coordinate_source,
                            "end_flow_eligible": end_flow_eligible,
                            "rideable_type": rideable_type.loc[keep],
                            "member_type": member_type.loc[keep],
                            "duration_seconds": duration.loc[keep],
                            "source_file": str(path),
                            "archive_member": member_name,
                            "source_file_sha256": source_hash,
                            "source_row_number": row_numbers.loc[keep],
                            "qc_flags": qc_flags,
                        }
                    ).reset_index(drop=True)
                    retained_sink.write(retained)

                    if excluded.any():
                        exclusions = _make_exclusion_frame(
                            record_ids=record_ids.loc[excluded],
                            dataset="bicycle",
                            city=city,
                            source_period=source_month,
                            source_file=path,
                            archive_member=member_name,
                            source_hash=source_hash,
                            row_numbers=row_numbers.loc[excluded],
                            primary_reason=primary_reason.loc[excluded],
                            all_reasons=all_exclusion_reasons.loc[excluded],
                            raw_datetime=started_raw.loc[excluded],
                            raw_start_station=start_name.loc[excluded],
                            raw_end_station=end_name.loc[excluded],
                        )
                        exclusion_sink.write(exclusions)

                    totals["retained_rows"] += int(keep.sum())
                    totals["excluded_rows"] += int(excluded.sum())
                    totals["outside_source_period_rows"] += int(
                        outside_month.sum()
                    )
                    totals["invalid_datetime_rows"] += int(invalid_start.sum())
                    totals["nonpositive_duration_rows"] += int(
                        (keep & duration.le(0)).sum()
                    )
                    totals["start_flow_eligible_rows"] += int(
                        start_flow_eligible.sum()
                    )
                    totals["end_flow_eligible_rows"] += int(
                        end_flow_eligible.sum()
                    )
                    totals["start_coordinate_unresolved_rows"] += int(
                        start_coordinate_source.eq("unresolved").sum()
                    )
                    totals["end_coordinate_unresolved_rows"] += int(
                        end_coordinate_source.eq("unresolved").sum()
                    )
                    totals["current_snapshot_retrofit_rows"] += int(
                        (
                            start_coordinate_source.eq(
                                "current_snapshot_retrofit"
                            )
                            | end_coordinate_source.eq(
                                "current_snapshot_retrofit"
                            )
                        ).sum()
                    )
                    for source_label, count in (
                        start_coordinate_source.value_counts().items()
                    ):
                        totals[f"start_coordinate_source__{source_label}"] += int(
                            count
                        )
                    for source_label, count in (
                        end_coordinate_source.value_counts().items()
                    ):
                        totals[f"end_coordinate_source__{source_label}"] += int(
                            count
                        )
            except Exception:
                retained_sink.abort()
                exclusion_sink.abort()
                raise
            retained_sink.close()
            exclusion_sink.close()
            records.append(
                {
                    "dataset": "bicycle",
                    "city": city,
                    "source_period": source_month,
                    "source_file": str(path),
                    "archive_member": member_name,
                    **totals,
                }
            )
    return pd.DataFrame(records)


def _aggregate_summary(reconciliation: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in reconciliation.columns
        if column
        not in {
            "dataset",
            "city",
            "source_period",
            "source_file",
            "archive_member",
        }
    ]
    for column in numeric:
        reconciliation[column] = pd.to_numeric(
            reconciliation[column], errors="coerce"
        ).fillna(0)
    summary = (
        reconciliation.groupby(["dataset", "city"], as_index=False)[numeric]
        .sum()
        .sort_values(["dataset", "city"])
    )
    return summary


def _assert_expected(
    reconciliation: pd.DataFrame, ctx: RunContext
) -> dict[str, int]:
    totals = (
        reconciliation.groupby("dataset")[
            ["physical_rows", "retained_rows", "excluded_rows", "blank_rows", "outside_source_period_rows"]
        ]
        .sum()
        .fillna(0)
        .astype("int64")
    )
    observed = {
        "crime_physical_rows": int(totals.loc["crime", "physical_rows"]),
        "crime_retained_rows": int(totals.loc["crime", "retained_rows"]),
        "crime_excluded_rows": int(totals.loc["crime", "excluded_rows"]),
        "bicycle_physical_rows": int(totals.loc["bicycle", "physical_rows"]),
        "bicycle_retained_rows": int(totals.loc["bicycle", "retained_rows"]),
        "bicycle_outside_source_month_rows": int(
            totals.loc["bicycle", "outside_source_period_rows"]
        ),
        "bicycle_blank_rows": int(totals.loc["bicycle", "blank_rows"]),
    }
    expected = ctx.e01.get("expected_counts", {})
    mismatches = {
        key: (int(expected[key]), observed[key])
        for key in observed
        if key in expected and int(expected[key]) != observed[key]
    }
    if mismatches:
        raise AssertionError(f"E01 count reconciliation failed: {mismatches}")
    if (
        observed["crime_retained_rows"] + observed["crime_excluded_rows"]
        != observed["crime_physical_rows"]
    ):
        raise AssertionError("Crime retained + excluded does not equal physical")
    if (
        observed["bicycle_retained_rows"]
        + observed["bicycle_outside_source_month_rows"]
        + observed["bicycle_blank_rows"]
        != observed["bicycle_physical_rows"]
    ):
        raise AssertionError("Bicycle retained + outside-month + blank does not equal physical")
    return observed


def _schema_inventory(data_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset in ["crime_events", "bicycle_trips", "exclusions"]:
        root = data_dir / dataset
        for path in sorted(root.rglob("*.parquet")) if root.exists() else []:
            parquet = pq.ParquetFile(path)
            records.append(
                {
                    "dataset": dataset,
                    "file": str(path),
                    "rows": parquet.metadata.num_rows,
                    "row_groups": parquet.metadata.num_row_groups,
                    "schema": str(parquet.schema_arrow),
                }
            )
    return pd.DataFrame(records)


def _write_interpretation(
    ctx: RunContext,
    observed: dict[str, int],
    summary: pd.DataFrame,
    unresolved: pd.DataFrame,
) -> None:
    unresolved_public = unresolved.loc[
        unresolved["classification"].eq("unresolved_public_or_unknown"),
        "row_count",
    ].sum() if not unresolved.empty else 0
    non_public = unresolved.loc[
        unresolved["classification"].eq("non_public_node"), "row_count"
    ].sum() if not unresolved.empty else 0
    lines = [
        "# E01 Interpretation and E02 Gate",
        "",
        "## Decision",
        "",
        "E01 completed only if all formal row reconciliations and Parquet checks pass.",
        "The standardized events are suitable for E02 panel construction subject to",
        "the endpoint-coverage flags documented below.",
        "",
        "## Reconciled event counts",
        "",
        f"- Crime physical rows: {observed['crime_physical_rows']:,}.",
        f"- Crime retained rows: {observed['crime_retained_rows']:,}.",
        f"- Crime excluded rows: {observed['crime_excluded_rows']:,}.",
        f"- Bicycle physical rows: {observed['bicycle_physical_rows']:,}.",
        f"- Bicycle retained rows: {observed['bicycle_retained_rows']:,}.",
        f"- Bicycle outside-source-month rows: {observed['bicycle_outside_source_month_rows']:,}.",
        f"- Bicycle blank rows: {observed['bicycle_blank_rows']:,}.",
        "",
        "## Coordinate policy",
        "",
        "- Valid trip coordinates were retained as the first-priority source.",
        "- DC and New York missing coordinates were reconstructed only by exact",
        "  station identifiers or unique normalized station names from historical",
        "  trip-coordinate dictionaries.",
        "- Vancouver current station coordinates were applied with the explicit",
        "  `current_snapshot_retrofit` flag.",
        f"- Unresolved endpoint observations: {int(unresolved_public):,}.",
        f"- Non-public workshop/yard/temporary endpoint observations: {int(non_public):,}.",
        "- No fuzzy station matching was used.",
        "",
        "## E02 requirements",
        "",
        "1. Use only endpoint rows whose corresponding `*_flow_eligible` field is true.",
        "2. Preserve coordinate-source and coverage flags in every city-month panel.",
        "3. Report a robustness specification excluding `current_snapshot_retrofit`",
        "   endpoints if their share is material.",
        "4. Keep zero-count grid-days explicit rather than treating them as missing.",
        "5. Do not calculate entropy or model results from excluded records.",
        "",
        "## City-level summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (ctx.run_dir / "interpretation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _append_registry(ctx: RunContext, elapsed: float, status: str) -> None:
    registry = Path(str(ctx.paths["empirical_root"])) / "docs" / "experiment_registry.csv"
    row = pd.DataFrame(
        [
            {
                "run_id": ctx.run_id,
                "experiment": "E01",
                "status": status,
                "started_at": datetime.fromtimestamp(
                    (ctx.run_dir / "command.txt").stat().st_mtime
                ).isoformat(),
                "completed_at": datetime.now().isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "run_directory": str(ctx.run_dir),
                "notes": "Standardized crime and bicycle events with station reconciliation; raw data unchanged.",
            }
        ]
    )
    row.to_csv(registry, mode="a", header=not registry.exists(), index=False)


def _initialize_context(args: argparse.Namespace) -> RunContext:
    paths = _load_yaml(Path(args.paths))
    e01 = _load_yaml(Path(args.e01))
    cities = _load_yaml(Path(args.cities))
    timezone = ZoneInfo(str(paths.get("timezone", "Asia/Shanghai")))
    runs_root = Path(str(paths["runs"]))
    if args.resume:
        run_id = args.resume
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run does not exist: {run_dir}")
    else:
        suffix = "_sample" if args.sample_rows else ""
        run_id = (
            datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
            + "_E01_standardized_events"
            + suffix
        )
        run_dir = runs_root / run_id
    data_dir = Path(str(paths["processed_data"])) / "e01" / run_id
    for directory in [
        run_dir / "logs",
        run_dir / "tables",
        run_dir / "figures",
        run_dir / "state",
        data_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(run_dir)
    selected = (
        {city.strip().upper() for city in args.city.split(",")}
        if args.city
        else {"DC", "NY", "VAN"}
    )
    unknown = selected.difference({"DC", "NY", "VAN"})
    if unknown:
        raise ValueError(f"Unknown cities: {sorted(unknown)}")
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        data_dir=data_dir,
        paths=paths,
        e01=e01,
        cities=cities,
        logger=logger,
        sample_rows=args.sample_rows,
        selected_cities=selected,
        station_dictionary_override=(
            Path(args.station_dictionary).resolve()
            if args.station_dictionary
            else None
        ),
    )


def run(args: argparse.Namespace) -> Path:
    started = datetime.now()
    ctx = _initialize_context(args)
    ctx.logger.info("Starting %s", ctx.run_id)
    command = " ".join([sys.executable, *sys.argv])
    (ctx.run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    _write_environment(ctx.run_dir / "environment.txt")
    config_snapshot = {
        "paths": ctx.paths,
        "e01": ctx.e01,
        "cities": ctx.cities,
        "selected_cities": sorted(ctx.selected_cities),
        "sample_rows": ctx.sample_rows,
        "station_dictionary_override": (
            str(ctx.station_dictionary_override)
            if ctx.station_dictionary_override
            else None
        ),
    }
    (ctx.run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    crime_files, bicycle_files, station_current = _input_files(ctx.paths)
    historical_cfg = ctx.e01["station_resolution"]["vancouver_historical"]
    historical_station = Path(str(historical_cfg["path"]))
    known_hashes = _e00_hashes(ctx.paths)
    manifest = _build_input_manifest(
        crime_files,
        bicycle_files,
        station_current,
        historical_station,
        known_hashes,
        ctx.logger,
    )
    if ctx.station_dictionary_override is not None:
        source = ctx.station_dictionary_override
        stat = source.stat()
        manifest = pd.concat(
            [
                manifest,
                pd.DataFrame(
                    [
                        {
                            "dataset": "derived_station_reference",
                            "city": "DC_NY",
                            "source_period": "2020-2022",
                            "absolute_path": str(source),
                            "size_bytes": stat.st_size,
                            "modified_at": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            "sha256": _sha256(source),
                            "e01_role": "verified_reusable_derived_reference",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    manifest.to_csv(ctx.run_dir / "tables" / "input_manifest.csv", index=False)
    manifest.to_csv(ctx.data_dir / "input_manifest.csv", index=False)
    manifest_hashes = dict(
        zip(manifest["absolute_path"], manifest["sha256"], strict=False)
    )

    historical = build_historical_station_dictionary(
        bicycle_files, manifest_hashes, ctx
    )
    resolver = StationResolver(
        historical=historical,
        vancouver_current_path=Path(
            str(
                ctx.e01["station_resolution"]["vancouver_current"]["path"]
            )
        ),
        vancouver_historical_path=historical_station,
        expected_current_hash=str(
            ctx.e01["station_resolution"]["vancouver_current"]["sha256"]
        ),
        non_public_patterns=list(
            ctx.e01["station_resolution"]["non_public_patterns"]
        ),
    )

    crime_reconciliation = standardize_crime(
        crime_files, manifest_hashes, ctx
    )
    bicycle_reconciliation = standardize_bicycle(
        bicycle_files, manifest_hashes, resolver, ctx
    )
    reconciliation = pd.concat(
        [crime_reconciliation, bicycle_reconciliation],
        ignore_index=True,
        sort=False,
    ).fillna(0)
    reconciliation.to_csv(
        ctx.run_dir / "tables" / "row_reconciliation.csv", index=False
    )
    summary = _aggregate_summary(reconciliation.copy())
    summary.to_csv(ctx.run_dir / "tables" / "city_summary.csv", index=False)
    unresolved = resolver.unresolved_table()
    unresolved.to_csv(
        ctx.run_dir / "tables" / "station_resolution_exceptions.csv",
        index=False,
    )

    schema_inventory = _schema_inventory(ctx.data_dir)
    schema_inventory.to_csv(
        ctx.run_dir / "tables" / "parquet_inventory.csv", index=False
    )
    parquet_rows = int(schema_inventory["rows"].sum()) if not schema_inventory.empty else 0
    ctx.logger.info(
        "Parquet files=%d, aggregate rows across datasets=%d",
        len(schema_inventory),
        parquet_rows,
    )

    if ctx.sample_rows is None and ctx.selected_cities == {"DC", "NY", "VAN"}:
        observed = _assert_expected(reconciliation, ctx)
        status = "complete"
    else:
        observed = {
            "crime_physical_rows": int(
                reconciliation.loc[
                    reconciliation["dataset"].eq("crime"), "physical_rows"
                ].sum()
            ),
            "crime_retained_rows": int(
                reconciliation.loc[
                    reconciliation["dataset"].eq("crime"), "retained_rows"
                ].sum()
            ),
            "crime_excluded_rows": int(
                reconciliation.loc[
                    reconciliation["dataset"].eq("crime"), "excluded_rows"
                ].sum()
            ),
            "bicycle_physical_rows": int(
                reconciliation.loc[
                    reconciliation["dataset"].eq("bicycle"), "physical_rows"
                ].sum()
            ),
            "bicycle_retained_rows": int(
                reconciliation.loc[
                    reconciliation["dataset"].eq("bicycle"), "retained_rows"
                ].sum()
            ),
            "bicycle_outside_source_month_rows": int(
                reconciliation.loc[
                    reconciliation["dataset"].eq("bicycle"),
                    "outside_source_period_rows",
                ].sum()
            ),
            "bicycle_blank_rows": int(
                reconciliation.loc[
                    reconciliation["dataset"].eq("bicycle"), "blank_rows"
                ].sum()
            ),
        }
        status = "sample_complete" if ctx.sample_rows is not None else "partial_complete"

    (ctx.run_dir / "tables" / "acceptance_counts.json").write_text(
        json.dumps(observed, indent=2), encoding="utf-8"
    )
    _write_interpretation(ctx, observed, summary, unresolved)
    elapsed = (datetime.now() - started).total_seconds()
    (ctx.run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "run_id": ctx.run_id,
                "status": status,
                "elapsed_seconds": elapsed,
                "completed_at": datetime.now().isoformat(),
                "data_directory": str(ctx.data_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _append_registry(ctx, elapsed, status)
    if status == "complete":
        Path(str(ctx.paths["runs"])).joinpath("latest_e01_run.txt").write_text(
            str(ctx.run_dir) + "\n", encoding="utf-8"
        )
        Path(str(ctx.paths["processed_data"])).joinpath(
            "e01", "latest_e01_data.txt"
        ).write_text(str(ctx.data_dir) + "\n", encoding="utf-8")
    ctx.logger.info("Completed %s in %.1f seconds", ctx.run_id, elapsed)
    return ctx.run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E01 standardized crime and bicycle event pipeline"
    )
    parser.add_argument("--paths", required=True, help="Path to paths.yaml")
    parser.add_argument("--e01", required=True, help="Path to e01.yaml")
    parser.add_argument("--cities", required=True, help="Path to cities.yaml")
    parser.add_argument(
        "--city",
        default="",
        help="Optional comma-separated subset: DC,NY,VAN",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Development-only maximum rows per archive member/source file",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Resume an incomplete E01 run identifier",
    )
    parser.add_argument(
        "--station-dictionary",
        default="",
        help="Optional verified historical station dictionary CSV from a prior run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = run(args)
    print(run_dir)


if __name__ == "__main__":
    main()
