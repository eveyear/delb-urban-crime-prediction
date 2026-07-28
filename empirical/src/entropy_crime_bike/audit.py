from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, TextIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ALLOWED_CRIME_TYPES = ("PROPERTY_THEFT", "VEHICLE_THEFT", "BURGLARY")
CHUNK_SIZE = 250_000


@dataclass
class Issue:
    dataset: str
    city: str
    source_file: str
    severity: str
    issue_code: str
    affected_rows: int | None
    description: str
    recommended_action: str


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _source_period(path: Path) -> str:
    match = re.search(r"(20\d{2})[-_]?([01]\d)", path.name)
    if match and 1 <= int(match.group(2)) <= 12:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.search(r"(20\d{2})", path.name)
    return match.group(1) if match else ""


def _city_from_path(path: Path) -> str:
    name = path.name.upper()
    if re.search(r"(?:^|_)DC(?:\.|_)", name):
        return "DC"
    if re.search(r"(?:^|_)NY(?:\.|_)", name):
        return "NY"
    if re.search(r"(?:^|_)VAN(?:\.|_)", name):
        return "VAN"
    for part in path.parts:
        upper = part.upper()
        if upper in {"DC", "NY", "VAN"}:
            return upper
    return "UNKNOWN"


def _configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e00")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(run_dir / "logs" / "e00_audit.log", encoding="utf-8")
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


def discover_inputs(crime_root: Path, bicycle_root: Path) -> list[Path]:
    crime_files = sorted(
        path for path in crime_root.glob("*.csv") if path.name != ".DS_Store"
    )
    bicycle_files = sorted(
        path
        for path in bicycle_root.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    )
    return crime_files + bicycle_files


def build_manifest(
    paths: Iterable[Path], crime_root: Path, bicycle_root: Path, hash_inputs: bool, logger: logging.Logger
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    paths = list(paths)
    for index, path in enumerate(paths, start=1):
        dataset = "crime" if crime_root in path.parents else "bicycle"
        stat = path.stat()
        logger.info("Manifest %d/%d: %s", index, len(paths), path.name)
        records.append(
            {
                "dataset": dataset,
                "city": _city_from_path(path),
                "source_period": _source_period(path),
                "absolute_path": str(path),
                "relative_path": str(
                    path.relative_to(crime_root if dataset == "crime" else bicycle_root)
                ),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "sha256": _sha256(path) if hash_inputs else "",
                "audit_role": (
                    "duplicate_candidate"
                    if dataset == "bicycle"
                    and path.suffix.lower() == ".csv"
                    and _city_from_path(path) == "NY"
                    else "canonical_input"
                ),
            }
        )
    return pd.DataFrame(records)


def audit_crime_file(path: Path) -> tuple[dict[str, object], pd.DataFrame, list[Issue]]:
    city = _city_from_path(path)
    year_match = re.match(r"(20\d{2})_", path.name)
    source_year = int(year_match.group(1)) if year_match else None
    frame = pd.read_csv(path, low_memory=False)
    row_count = len(frame)
    issues: list[Issue] = []

    required = {"datetime", "latitude", "longitude", "crime_type_raw", "crime_type_unified"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        issues.append(
            Issue(
                "crime",
                city,
                str(path),
                "BLOCKER",
                "MISSING_REQUIRED_COLUMNS",
                None,
                f"Missing required columns: {', '.join(missing_columns)}",
                "Resolve source schema before E01.",
            )
        )

    dates = pd.to_datetime(frame.get("datetime"), errors="coerce")
    invalid_dates = int(dates.isna().sum())
    if source_year is not None:
        period_start = pd.Timestamp(f"{source_year}-01-01")
        period_end = pd.Timestamp(f"{source_year + 1}-01-01")
        out_of_period = int(((dates < period_start) | (dates >= period_end)).fillna(False).sum())
    else:
        out_of_period = 0

    lat = pd.to_numeric(frame.get("latitude"), errors="coerce")
    lon = pd.to_numeric(frame.get("longitude"), errors="coerce")
    missing_lat = int(lat.isna().sum())
    missing_lon = int(lon.isna().sum())
    invalid_coords = int(
        ((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180)).fillna(False).sum()
    )
    exact_duplicates = int(frame.duplicated().sum())

    unified = frame.get("crime_type_unified", pd.Series(index=frame.index, dtype="object"))
    unified_text = unified.astype("string").str.strip()
    missing_unified = int(unified_text.isna().sum() + (unified_text == "").sum())
    unexpected_mask = (~unified_text.isin(ALLOWED_CRIME_TYPES)) & unified_text.notna() & (unified_text != "")
    unexpected_unified = int(unexpected_mask.sum())

    for code, count, severity, description, action in [
        (
            "INVALID_DATETIME",
            invalid_dates,
            "HIGH",
            "Datetime could not be parsed.",
            "Quarantine these records in E01.",
        ),
        (
            "OUTSIDE_SOURCE_YEAR",
            out_of_period,
            "MEDIUM",
            "Event datetime falls outside the year encoded in the filename.",
            "Filter to the filename year in E01 and retain excluded records in a QC table.",
        ),
        (
            "MISSING_COORDINATES",
            max(missing_lat, missing_lon),
            "HIGH",
            "Latitude or longitude is missing.",
            "Quarantine before spatial assignment.",
        ),
        (
            "INVALID_COORDINATE_RANGE",
            invalid_coords,
            "HIGH",
            "Coordinate is outside the valid WGS84 range.",
            "Quarantine before spatial assignment.",
        ),
        (
            "EXACT_DUPLICATE_ROWS",
            exact_duplicates,
            "MEDIUM",
            "Exact duplicate source rows were detected.",
            "Preserve and deduplicate deterministically in E01.",
        ),
        (
            "MISSING_UNIFIED_CRIME_TYPE",
            missing_unified,
            "HIGH",
            "The authoritative crime_type_unified field is missing.",
            "Exclude from the primary outcome and report separately.",
        ),
        (
            "UNEXPECTED_UNIFIED_CRIME_TYPE",
            unexpected_unified,
            "HIGH",
            "crime_type_unified contains values outside the three prespecified categories.",
            "Do not remap automatically; review and report separately.",
        ),
    ]:
        if count:
            issues.append(
                Issue("crime", city, str(path), severity, code, count, description, action)
            )

    counts = (
        unified_text.fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename_axis("crime_type_unified")
        .reset_index(name="row_count")
    )
    counts.insert(0, "source_year", source_year)
    counts.insert(0, "city", city)
    counts.insert(0, "source_file", str(path))

    record = {
        "source_file": str(path),
        "city": city,
        "source_year": source_year,
        "row_count": row_count,
        "column_count": len(frame.columns),
        "date_min": dates.min().isoformat() if dates.notna().any() else "",
        "date_max": dates.max().isoformat() if dates.notna().any() else "",
        "invalid_datetime_rows": invalid_dates,
        "outside_source_year_rows": out_of_period,
        "missing_latitude_rows": missing_lat,
        "missing_longitude_rows": missing_lon,
        "invalid_coordinate_rows": invalid_coords,
        "exact_duplicate_rows": exact_duplicates,
        "missing_unified_type_rows": missing_unified,
        "unexpected_unified_type_rows": unexpected_unified,
        "latitude_min": float(lat.min()) if lat.notna().any() else np.nan,
        "latitude_max": float(lat.max()) if lat.notna().any() else np.nan,
        "longitude_min": float(lon.min()) if lon.notna().any() else np.nan,
        "longitude_max": float(lon.max()) if lon.notna().any() else np.nan,
        "schema_signature": hashlib.sha256(
            "|".join(map(str, frame.columns)).encode("utf-8")
        ).hexdigest()[:16],
    }
    return record, counts, issues


def _zip_csv_members(path: Path) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(path) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and "/._" not in name
        ]
        ignored = [name for name in archive.namelist() if name not in csv_members]
    return csv_members, ignored


def _open_csv_streams(path: Path) -> Iterator[tuple[str, BinaryIO | str]]:
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        try:
            members, _ = _zip_csv_members(path)
            for member in members:
                with archive.open(member) as stream:
                    yield member, stream
        finally:
            archive.close()
    else:
        yield path.name, str(path)


def _normalized_columns(columns: Iterable[object]) -> dict[str, str]:
    return {
        re.sub(r"\s+", " ", str(column).replace("\ufeff", "").strip()).lower(): str(column)
        for column in columns
    }


def _first_existing(column_map: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        if name.lower() in column_map:
            return column_map[name.lower()]
    return None


def _chunk_hashes(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    usable = [column for column in columns if column in frame.columns]
    if not usable:
        return np.array([], dtype=np.uint64)
    return pd.util.hash_pandas_object(
        frame[usable].fillna("<NA>").astype("string"), index=False
    ).to_numpy(dtype=np.uint64)


def audit_bicycle_file(
    path: Path, logger: logging.Logger, full_audit: bool
) -> tuple[dict[str, object], list[dict[str, object]], list[Issue]]:
    city = _city_from_path(path)
    source_period = _source_period(path)
    issues: list[Issue] = []
    member_records: list[dict[str, object]] = []

    if city == "NY" and path.suffix.lower() == ".csv":
        issues.append(
            Issue(
                "bicycle",
                city,
                str(path),
                "INFO",
                "EXTRACTED_DUPLICATE_CANDIDATE",
                None,
                "An extracted monthly CSV is present alongside its canonical ZIP archive.",
                "Retain for provenance but exclude from E01 canonical ingestion.",
            )
        )
        return (
            {
                "source_file": str(path),
                "city": city,
                "source_month": source_period,
                "audit_role": "duplicate_candidate",
                "csv_member_count": 0,
                "ignored_archive_member_count": 0,
                "row_count": 0,
                "invalid_start_datetime_rows": 0,
                "outside_source_month_rows": 0,
                "missing_start_station_rows": 0,
                "missing_end_station_rows": 0,
                "missing_start_coordinate_rows": 0,
                "missing_end_coordinate_rows": 0,
                "invalid_coordinate_rows": 0,
                "nonpositive_duration_rows": 0,
                "duplicate_source_ride_id_rows": 0,
                "repeated_full_row_signature_rows": 0,
                "blank_source_rows": 0,
                "unique_ride_id_available": False,
                "start_datetime_min": "",
                "start_datetime_max": "",
                "schema_count": 0,
                "audit_status": "excluded_duplicate_candidate",
            },
            member_records,
            issues,
        )

    ignored_member_count = 0
    if path.suffix.lower() == ".zip":
        try:
            members, ignored = _zip_csv_members(path)
            ignored_member_count = len(ignored)
            if not members:
                raise ValueError("ZIP archive contains no canonical CSV member")
        except (zipfile.BadZipFile, ValueError) as exc:
            issues.append(
                Issue(
                    "bicycle",
                    city,
                    str(path),
                    "BLOCKER",
                    "UNREADABLE_ZIP_ARCHIVE",
                    None,
                    str(exc),
                    "Replace or repair the source archive before E01.",
                )
            )
            return (
                {
                    "source_file": str(path),
                    "city": city,
                    "source_month": source_period,
                    "audit_role": "canonical_input",
                    "csv_member_count": 0,
                    "ignored_archive_member_count": 0,
                    "row_count": 0,
                    "audit_status": "failed",
                },
                member_records,
                issues,
            )

    totals = Counter()
    schemas: set[str] = set()
    seen_ride_ids: set[int] = set()
    seen_row_signatures: set[int] = set()
    start_datetime_min: pd.Timestamp | None = None
    start_datetime_max: pd.Timestamp | None = None
    unique_ride_id_available = False
    audit_status = "complete"

    for member_name, stream in _open_csv_streams(path):
        member_totals = Counter()
        member_schema = ""
        try:
            reader = pd.read_csv(
                stream,
                chunksize=CHUNK_SIZE if full_audit else 50_000,
                nrows=None if full_audit else 50_000,
                dtype=str,
                encoding="utf-8-sig",
                encoding_errors="replace",
                low_memory=False,
                on_bad_lines="skip",
            )
            for chunk_index, chunk in enumerate(reader, start=1):
                physical_rows = len(chunk)
                blank_mask = chunk.isna().all(axis=1)
                blank_rows = int(blank_mask.sum())
                totals["physical_row_count"] += physical_rows
                totals["blank_source_rows"] += blank_rows
                member_totals["physical_row_count"] += physical_rows
                member_totals["blank_source_rows"] += blank_rows
                if blank_rows:
                    chunk = chunk.loc[~blank_mask].copy()
                if chunk_index == 1:
                    column_map = _normalized_columns(chunk.columns)
                    member_schema = hashlib.sha256(
                        "|".join(column_map.keys()).encode("utf-8")
                    ).hexdigest()[:16]
                    schemas.add(member_schema)
                    start_col = _first_existing(
                        column_map, ["started_at", "start date", "departure"]
                    )
                    end_col = _first_existing(
                        column_map, ["ended_at", "end date", "return"]
                    )
                    start_station_col = _first_existing(
                        column_map,
                        ["start_station_id", "start station number", "departure station"],
                    )
                    end_station_col = _first_existing(
                        column_map,
                        ["end_station_id", "end station number", "return station"],
                    )
                    start_lat_col = _first_existing(column_map, ["start_lat"])
                    start_lon_col = _first_existing(column_map, ["start_lng"])
                    end_lat_col = _first_existing(column_map, ["end_lat"])
                    end_lon_col = _first_existing(column_map, ["end_lng"])
                    duration_col = _first_existing(
                        column_map, ["duration", "duration (sec.)"]
                    )
                    ride_id_col = _first_existing(column_map, ["ride_id"])
                    unique_ride_id_available = unique_ride_id_available or ride_id_col is not None
                    bike_col = _first_existing(
                        column_map, ["bike number", "bike", "electric bike"]
                    )
                    required_missing = [
                        label
                        for label, column in [
                            ("start datetime", start_col),
                            ("end datetime", end_col),
                            ("start station", start_station_col),
                            ("end station", end_station_col),
                        ]
                        if column is None
                    ]
                    if required_missing:
                        issues.append(
                            Issue(
                                "bicycle",
                                city,
                                f"{path}::{member_name}",
                                "BLOCKER",
                                "MISSING_REQUIRED_BICYCLE_COLUMNS",
                                None,
                                f"Missing: {', '.join(required_missing)}",
                                "Add a reviewed schema alias before E01.",
                            )
                        )
                rows = len(chunk)
                totals["row_count"] += rows
                member_totals["row_count"] += rows

                start_dates = pd.to_datetime(chunk[start_col], errors="coerce") if start_col else pd.Series(pd.NaT, index=chunk.index)
                end_dates = pd.to_datetime(chunk[end_col], errors="coerce") if end_col else pd.Series(pd.NaT, index=chunk.index)
                invalid_start = int(start_dates.isna().sum())
                totals["invalid_start_datetime_rows"] += invalid_start
                member_totals["invalid_start_datetime_rows"] += invalid_start
                if start_dates.notna().any():
                    chunk_min = start_dates.min()
                    chunk_max = start_dates.max()
                    start_datetime_min = (
                        chunk_min
                        if start_datetime_min is None
                        else min(start_datetime_min, chunk_min)
                    )
                    start_datetime_max = (
                        chunk_max
                        if start_datetime_max is None
                        else max(start_datetime_max, chunk_max)
                    )

                if source_period:
                    period = pd.Period(source_period, freq="M")
                    start_bound = period.start_time
                    end_bound = (period + 1).start_time
                    outside = int(
                        ((start_dates < start_bound) | (start_dates >= end_bound))
                        .fillna(False)
                        .sum()
                    )
                else:
                    outside = 0
                totals["outside_source_month_rows"] += outside
                member_totals["outside_source_month_rows"] += outside

                missing_start_station = int(
                    chunk[start_station_col].isna().sum()
                    + (chunk[start_station_col].astype("string").str.strip() == "").sum()
                ) if start_station_col else rows
                missing_end_station = int(
                    chunk[end_station_col].isna().sum()
                    + (chunk[end_station_col].astype("string").str.strip() == "").sum()
                ) if end_station_col else rows
                totals["missing_start_station_rows"] += missing_start_station
                totals["missing_end_station_rows"] += missing_end_station
                member_totals["missing_start_station_rows"] += missing_start_station
                member_totals["missing_end_station_rows"] += missing_end_station

                coordinate_columns = [start_lat_col, start_lon_col, end_lat_col, end_lon_col]
                if all(coordinate_columns):
                    start_lat = pd.to_numeric(chunk[start_lat_col], errors="coerce")
                    start_lon = pd.to_numeric(chunk[start_lon_col], errors="coerce")
                    end_lat = pd.to_numeric(chunk[end_lat_col], errors="coerce")
                    end_lon = pd.to_numeric(chunk[end_lon_col], errors="coerce")
                    missing_start_coord = int((start_lat.isna() | start_lon.isna()).sum())
                    missing_end_coord = int((end_lat.isna() | end_lon.isna()).sum())
                    invalid_coord = int(
                        (
                            (start_lat < -90)
                            | (start_lat > 90)
                            | (end_lat < -90)
                            | (end_lat > 90)
                            | (start_lon < -180)
                            | (start_lon > 180)
                            | (end_lon < -180)
                            | (end_lon > 180)
                        )
                        .fillna(False)
                        .sum()
                    )
                else:
                    missing_start_coord = rows
                    missing_end_coord = rows
                    invalid_coord = 0
                totals["missing_start_coordinate_rows"] += missing_start_coord
                totals["missing_end_coordinate_rows"] += missing_end_coord
                totals["invalid_coordinate_rows"] += invalid_coord
                member_totals["missing_start_coordinate_rows"] += missing_start_coord
                member_totals["missing_end_coordinate_rows"] += missing_end_coord
                member_totals["invalid_coordinate_rows"] += invalid_coord

                if duration_col:
                    duration = pd.to_numeric(chunk[duration_col], errors="coerce")
                else:
                    duration = (end_dates - start_dates).dt.total_seconds()
                nonpositive = int((duration <= 0).fillna(False).sum())
                totals["nonpositive_duration_rows"] += nonpositive
                member_totals["nonpositive_duration_rows"] += nonpositive

                if ride_id_col:
                    hashes = _chunk_hashes(chunk, [ride_id_col])
                    if hashes.size:
                        unique_hashes, counts = np.unique(hashes, return_counts=True)
                        duplicates = int((counts - 1).sum())
                        duplicates += sum(
                            int(value) in seen_ride_ids for value in unique_hashes
                        )
                        seen_ride_ids.update(map(int, unique_hashes))
                        totals["duplicate_source_ride_id_rows"] += duplicates
                        member_totals["duplicate_source_ride_id_rows"] += duplicates
                else:
                    hashes = _chunk_hashes(chunk, list(chunk.columns))
                    if hashes.size:
                        unique_hashes, counts = np.unique(hashes, return_counts=True)
                        repetitions = int((counts - 1).sum())
                        repetitions += sum(
                            int(value) in seen_row_signatures for value in unique_hashes
                        )
                        seen_row_signatures.update(map(int, unique_hashes))
                        totals["repeated_full_row_signature_rows"] += repetitions
                        member_totals["repeated_full_row_signature_rows"] += repetitions

                if not full_audit:
                    break

            member_records.append(
                {
                    "source_file": str(path),
                    "archive_member": member_name,
                    "city": city,
                    "source_month": source_period,
                    "schema_signature": member_schema,
                    **member_totals,
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve source-level failures
            audit_status = "failed"
            issues.append(
                Issue(
                    "bicycle",
                    city,
                    f"{path}::{member_name}",
                    "BLOCKER",
                    "BICYCLE_PARSE_FAILURE",
                    None,
                    repr(exc),
                    "Inspect the source schema/encoding and add a reviewed parser rule.",
                )
            )
            logger.exception("Failed bicycle member %s::%s", path.name, member_name)

    row_count = totals["row_count"]
    checks = [
        (
            "INVALID_START_DATETIME",
            totals["invalid_start_datetime_rows"],
            "HIGH",
            "Trip start datetime could not be parsed.",
            "Quarantine in E01.",
        ),
        (
            "OUTSIDE_SOURCE_MONTH",
            totals["outside_source_month_rows"],
            "MEDIUM",
            "Trip start is outside the filename month.",
            "Filter to the filename month in E01 and retain exclusions.",
        ),
        (
            "MISSING_STATION",
            max(totals["missing_start_station_rows"], totals["missing_end_station_rows"]),
            "MEDIUM",
            "Start or end station identifier/name is missing.",
            "Retain for trip-level QC; exclude from affected spatial flow endpoint.",
        ),
        (
            "MISSING_TRIP_COORDINATES",
            max(
                totals["missing_start_coordinate_rows"],
                totals["missing_end_coordinate_rows"],
            ),
            "MEDIUM",
            "Trip endpoint coordinates are missing from the source schema or row.",
            "Resolve through the archived station lookup in E01.",
        ),
        (
            "INVALID_COORDINATE_RANGE",
            totals["invalid_coordinate_rows"],
            "HIGH",
            "Trip endpoint coordinate is outside valid WGS84 range.",
            "Quarantine before spatial assignment.",
        ),
        (
            "NONPOSITIVE_DURATION",
            totals["nonpositive_duration_rows"],
            "MEDIUM",
            "Trip duration is zero or negative.",
            "Flag and exclude from duration-based diagnostics; preserve flow decision separately.",
        ),
        (
            "DUPLICATE_SOURCE_RIDE_ID",
            totals["duplicate_source_ride_id_rows"],
            "MEDIUM",
            "A duplicate formal source ride identifier was detected.",
            "Deduplicate deterministically in E01.",
        ),
        (
            "REPEATED_FULL_ROW_SIGNATURE",
            totals["repeated_full_row_signature_rows"],
            "INFO",
            "A source without a unique ride ID contains repeated complete row signatures.",
            "Retain by default; do not deduplicate unless source documentation establishes duplication.",
        ),
        (
            "BLANK_SOURCE_ROWS",
            totals["blank_source_rows"],
            "MEDIUM",
            "The CSV contains physically present rows with all fields blank.",
            "Exclude blank rows in E01 and record the exclusion count.",
        ),
    ]
    for code, count, severity, description, action in checks:
        if count:
            issues.append(
                Issue("bicycle", city, str(path), severity, code, int(count), description, action)
            )

    record = {
        "source_file": str(path),
        "city": city,
        "source_month": source_period,
        "audit_role": "canonical_input",
        "csv_member_count": len(member_records),
        "ignored_archive_member_count": ignored_member_count,
        "physical_row_count": int(totals["physical_row_count"]),
        "row_count": int(row_count),
        "blank_source_rows": int(totals["blank_source_rows"]),
        "invalid_start_datetime_rows": int(totals["invalid_start_datetime_rows"]),
        "outside_source_month_rows": int(totals["outside_source_month_rows"]),
        "missing_start_station_rows": int(totals["missing_start_station_rows"]),
        "missing_end_station_rows": int(totals["missing_end_station_rows"]),
        "missing_start_coordinate_rows": int(totals["missing_start_coordinate_rows"]),
        "missing_end_coordinate_rows": int(totals["missing_end_coordinate_rows"]),
        "invalid_coordinate_rows": int(totals["invalid_coordinate_rows"]),
        "nonpositive_duration_rows": int(totals["nonpositive_duration_rows"]),
        "duplicate_source_ride_id_rows": int(totals["duplicate_source_ride_id_rows"]),
        "repeated_full_row_signature_rows": int(
            totals["repeated_full_row_signature_rows"]
        ),
        "unique_ride_id_available": unique_ride_id_available,
        "start_datetime_min": (
            start_datetime_min.isoformat() if start_datetime_min is not None else ""
        ),
        "start_datetime_max": (
            start_datetime_max.isoformat() if start_datetime_max is not None else ""
        ),
        "schema_count": len(schemas),
        "audit_status": audit_status if full_audit else "sampled",
    }
    return record, member_records, issues


def _make_summary(
    manifest: pd.DataFrame,
    crime_audit: pd.DataFrame,
    bicycle_audit: pd.DataFrame,
    issues: pd.DataFrame,
) -> pd.DataFrame:
    canonical_bike = bicycle_audit[bicycle_audit["audit_role"] == "canonical_input"]
    metrics = [
        ("input_files", len(manifest), "files"),
        ("input_size", int(manifest["size_bytes"].sum()), "bytes"),
        ("crime_files", len(crime_audit), "files"),
        ("crime_rows", int(crime_audit["row_count"].sum()), "rows"),
        (
            "crime_outside_source_year_rows",
            int(crime_audit["outside_source_year_rows"].sum()),
            "rows",
        ),
        (
            "crime_missing_or_unexpected_unified_rows",
            int(
                crime_audit["missing_unified_type_rows"].sum()
                + crime_audit["unexpected_unified_type_rows"].sum()
            ),
            "rows",
        ),
        ("bicycle_canonical_files", len(canonical_bike), "files"),
        ("bicycle_rows", int(canonical_bike["row_count"].sum()), "rows"),
        (
            "bicycle_outside_source_month_rows",
            int(canonical_bike["outside_source_month_rows"].sum()),
            "rows",
        ),
        (
            "bicycle_duplicate_source_ride_id_rows",
            int(canonical_bike["duplicate_source_ride_id_rows"].sum()),
            "rows",
        ),
        ("blocker_issues", int((issues["severity"] == "BLOCKER").sum()), "issues"),
        ("high_issues", int((issues["severity"] == "HIGH").sum()), "issues"),
        ("medium_issues", int((issues["severity"] == "MEDIUM").sum()), "issues"),
        ("informational_issues", int((issues["severity"] == "INFO").sum()), "issues"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "unit"])


def _write_markdown_report(
    run_id: str,
    summary: pd.DataFrame,
    crime_audit: pd.DataFrame,
    crime_counts: pd.DataFrame,
    bicycle_audit: pd.DataFrame,
    issues: pd.DataFrame,
    output_path: Path,
) -> None:
    totals_by_city = (
        crime_audit.groupby("city", as_index=False)["row_count"].sum().sort_values("city")
    )
    crime_pivot = (
        crime_counts[crime_counts["crime_type_unified"].isin(ALLOWED_CRIME_TYPES)]
        .groupby(["city", "source_year", "crime_type_unified"], as_index=False)["row_count"]
        .sum()
        .pivot_table(
            index=["city", "source_year"],
            columns="crime_type_unified",
            values="row_count",
            fill_value=0,
        )
        .reset_index()
    )
    bike_by_city = (
        bicycle_audit[bicycle_audit["audit_role"] == "canonical_input"]
        .groupby("city", as_index=False)
        .agg(files=("source_file", "count"), rows=("row_count", "sum"))
        .sort_values("city")
    )
    blocker_count = int((issues["severity"] == "BLOCKER").sum())
    status = "PASS WITH DOCUMENTED CLEANING ACTIONS" if blocker_count == 0 else "BLOCKED"
    lines = [
        "# E00 Data Integrity Audit",
        "",
        f"- Run ID: `{run_id}`",
        f"- Status: **{status}**",
        "- Raw inputs were read only; no source file was modified.",
        "- `crime_type_unified` was audited as the authoritative harmonized field.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Crime rows by city",
        "",
        totals_by_city.to_markdown(index=False),
        "",
        "## Harmonized crime categories",
        "",
        crime_pivot.to_markdown(index=False),
        "",
        "## Bicycle files and audited rows",
        "",
        bike_by_city.to_markdown(index=False),
        "",
        "## Issues requiring attention",
        "",
    ]
    if issues.empty:
        lines.append("No issues were detected.")
    else:
        display = issues[
            [
                "severity",
                "dataset",
                "city",
                "issue_code",
                "affected_rows",
                "description",
                "recommended_action",
            ]
        ].copy()
        lines.append(display.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## E01 gate",
            "",
            "E01 may begin only after this report is reviewed. All exclusions, month/year",
            "boundary filters, station-coordinate reconciliation, and deterministic",
            "deduplication must be written to separate QC tables rather than silently dropped.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _append_registry(empirical_root: Path, row: dict[str, object]) -> None:
    registry = empirical_root / "docs" / "experiment_registry.csv"
    new = pd.DataFrame([row])
    if registry.exists():
        old = pd.read_csv(registry)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(registry, index=False)


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    empirical_root = Path(config["empirical_root"])
    runs_root = Path(config["runs"])
    timezone = ZoneInfo(config.get("timezone", "Asia/Shanghai"))
    run_id = datetime.now(timezone).strftime("%Y%m%d_%H%M%S") + "_E00_data_audit"
    run_dir = runs_root / run_id
    for child in ["logs", "tables", "figures", "intermediate", "models"]:
        (run_dir / child).mkdir(parents=True, exist_ok=False)
    logger = _configure_logging(run_dir)
    start = time.time()
    logger.info("Starting %s", run_id)

    snapshot = {
        "run_id": run_id,
        "experiment": "E00",
        "started_at": datetime.now(timezone).isoformat(),
        "arguments": vars(args),
        "paths": config,
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]), encoding="utf-8"
    )
    _write_environment(run_dir / "environment.txt")

    crime_root = Path(config["crime_raw"])
    bicycle_root = Path(config["bicycle_raw"])
    input_paths = discover_inputs(crime_root, bicycle_root)
    manifest = build_manifest(input_paths, crime_root, bicycle_root, args.hash_inputs, logger)
    manifest.to_csv(run_dir / "tables" / "input_manifest.csv", index=False)

    crime_records: list[dict[str, object]] = []
    crime_count_frames: list[pd.DataFrame] = []
    issues: list[Issue] = []
    crime_files = sorted(crime_root.glob("*.csv"))
    for index, path in enumerate(crime_files, start=1):
        logger.info("Crime audit %d/%d: %s", index, len(crime_files), path.name)
        record, counts, file_issues = audit_crime_file(path)
        crime_records.append(record)
        crime_count_frames.append(counts)
        issues.extend(file_issues)
    crime_audit = pd.DataFrame(crime_records)
    crime_counts = pd.concat(crime_count_frames, ignore_index=True)
    crime_audit.to_csv(run_dir / "tables" / "crime_file_audit.csv", index=False)
    crime_counts.to_csv(run_dir / "tables" / "crime_type_counts.csv", index=False)

    bicycle_records: list[dict[str, object]] = []
    bicycle_member_records: list[dict[str, object]] = []
    bicycle_paths = sorted(
        path
        for path in bicycle_root.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    )
    for index, path in enumerate(bicycle_paths, start=1):
        logger.info(
            "Bicycle audit %d/%d: %s", index, len(bicycle_paths), path.relative_to(bicycle_root)
        )
        record, members, file_issues = audit_bicycle_file(path, logger, args.full_bike_audit)
        bicycle_records.append(record)
        bicycle_member_records.extend(members)
        issues.extend(file_issues)
    bicycle_audit = pd.DataFrame(bicycle_records)
    bicycle_members = pd.DataFrame(bicycle_member_records)
    bicycle_audit.to_csv(run_dir / "tables" / "bicycle_file_audit.csv", index=False)
    bicycle_members.to_csv(run_dir / "tables" / "bicycle_member_audit.csv", index=False)

    schema_catalog = (
        bicycle_members.groupby(
            ["city", "schema_signature"], as_index=False, dropna=False
        )
        .agg(
            members=("archive_member", "count"),
            rows=("row_count", "sum"),
            first_source=("source_file", "first"),
        )
        .sort_values(["city", "schema_signature"])
    )
    schema_catalog.to_csv(run_dir / "tables" / "bicycle_schema_catalog.csv", index=False)

    issue_frame = pd.DataFrame([asdict(issue) for issue in issues])
    if issue_frame.empty:
        issue_frame = pd.DataFrame(
            columns=[
                "dataset",
                "city",
                "source_file",
                "severity",
                "issue_code",
                "affected_rows",
                "description",
                "recommended_action",
            ]
        )
    issue_frame.to_csv(run_dir / "tables" / "issues.csv", index=False)
    summary = _make_summary(manifest, crime_audit, bicycle_audit, issue_frame)
    summary.to_csv(run_dir / "tables" / "audit_summary.csv", index=False)

    result_json = {
        "run_id": run_id,
        "summary": summary.to_dict(orient="records"),
        "status": (
            "blocked"
            if int((issue_frame["severity"] == "BLOCKER").sum()) > 0
            else "complete_with_documented_actions"
        ),
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (run_dir / "tables" / "audit_summary.json").write_text(
        json.dumps(result_json, indent=2), encoding="utf-8"
    )
    _write_markdown_report(
        run_id,
        summary,
        crime_audit,
        crime_counts,
        bicycle_audit,
        issue_frame,
        run_dir / "audit_report.md",
    )

    manifest_dir = Path(config["processed_data"]) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        run_dir / "tables" / "input_manifest.csv",
        manifest_dir / f"{run_id}_input_manifest.csv",
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "latest_e00_run.txt").write_text(str(run_dir), encoding="utf-8")
    _append_registry(
        empirical_root,
        {
            "run_id": run_id,
            "experiment": "E00",
            "status": result_json["status"],
            "started_at": snapshot["started_at"],
            "completed_at": datetime.now(timezone).isoformat(),
            "elapsed_seconds": result_json["elapsed_seconds"],
            "run_directory": str(run_dir),
            "notes": "Full input manifest and data-quality audit; raw data unchanged.",
        },
    )
    logger.info("Completed %s in %.1f seconds", run_id, time.time() - start)
    logger.info("Run directory: %s", run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E00 data-integrity audit.")
    default_config = Path(__file__).resolve().parents[2] / "config" / "paths.yaml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Calculate SHA-256 for every raw input file.",
    )
    parser.add_argument(
        "--full-bike-audit",
        action="store_true",
        help="Read every bicycle row; otherwise audit only the first 50,000 rows per member.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
