from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from entropy_crime_bike.standardize import (
    BICYCLE_SCHEMA,
    _aggregate_summary,
    _load_yaml,
    _sha256,
    _write_environment,
    _write_interpretation,
)


FALSE_POSITIVE_NON_PUBLIC_IDS = {
    "DC": {"32218"},
    "NY": {"5351.03", "5442.05", "6459.06", "HB202"},
}


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e01_patch")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e01_station_rule_patch.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _has_false_positive_station(path: Path, city: str) -> bool:
    identifiers = FALSE_POSITIVE_NON_PUBLIC_IDS.get(city)
    if not identifiers:
        return False
    table = pq.read_table(
        path, columns=["start_station_id", "end_station_id"]
    )
    frame = table.to_pandas()
    return bool(
        frame["start_station_id"].astype("string").isin(identifiers).any()
        or frame["end_station_id"].astype("string").isin(identifiers).any()
    )


def _clean_non_public_flag(flags: pd.Series) -> pd.Series:
    cleaned = flags.fillna("").astype("string")
    cleaned = cleaned.str.replace(
        r"(^|\|)non_public_endpoint(?=\||$)", "", regex=True
    )
    cleaned = cleaned.str.replace(r"\|+", "|", regex=True).str.strip("|")
    return cleaned


def _add_qc_flag(flags: pd.Series, flag: str) -> pd.Series:
    result = flags.fillna("").astype("string")
    already_present = result.str.split("|", regex=False).map(
        lambda values: flag in values if isinstance(values, list) else False
    )
    add = ~already_present
    result.loc[add & result.eq("")] = flag
    result.loc[add & result.ne("")] = result.loc[add & result.ne("")] + "|" + flag
    return result


def _patch_parquet(
    source: Path,
    destination: Path,
    city: str,
    compression: str,
    compression_level: int,
) -> tuple[dict[str, int], str, str]:
    identifiers = FALSE_POSITIVE_NON_PUBLIC_IDS[city]
    parquet = pq.ParquetFile(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        temporary.unlink()
    dictionary_columns = [
        field.name
        for field in BICYCLE_SCHEMA
        if pa.types.is_string(field.type)
        and field.name
        not in {
            "trip_record_id",
            "source_ride_id",
        }
    ]
    writer = pq.ParquetWriter(
        temporary,
        BICYCLE_SCHEMA,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=dictionary_columns,
        write_statistics=True,
    )
    deltas: Counter[str] = Counter()
    source_file = ""
    archive_member = ""
    try:
        for row_group_index in range(parquet.metadata.num_row_groups):
            frame = parquet.read_row_group(row_group_index).to_pandas()
            if frame.empty:
                continue
            if not source_file:
                source_file = str(frame["source_file"].iloc[0])
                archive_member = str(frame["archive_member"].iloc[0])

            start_false = (
                frame["start_station_id"].astype("string").isin(identifiers)
                & frame["start_coordinate_source"].eq("non_public_node")
            )
            end_false = (
                frame["end_station_id"].astype("string").isin(identifiers)
                & frame["end_coordinate_source"].eq("non_public_node")
            )
            old_start_eligible = frame["start_flow_eligible"].fillna(False).astype(bool)
            old_end_eligible = frame["end_flow_eligible"].fillna(False).astype(bool)

            frame.loc[start_false, "start_coordinate_source"] = "source"
            frame.loc[end_false, "end_coordinate_source"] = "source"
            true_non_public = frame["start_coordinate_source"].eq(
                "non_public_node"
            ) | frame["end_coordinate_source"].eq("non_public_node")

            valid_start = (
                frame["start_latitude"].notna()
                & frame["start_longitude"].notna()
                & frame["start_latitude"].between(-90, 90)
                & frame["start_longitude"].between(-180, 180)
                & ~frame["start_coordinate_source"].isin(
                    ["unresolved", "non_public_node"]
                )
            )
            valid_end = (
                frame["ended_at_local"].notna()
                & frame["end_latitude"].notna()
                & frame["end_longitude"].notna()
                & frame["end_latitude"].between(-90, 90)
                & frame["end_longitude"].between(-180, 180)
                & ~frame["end_coordinate_source"].isin(
                    ["unresolved", "non_public_node"]
                )
            )
            new_start_eligible = valid_start & ~true_non_public
            new_end_eligible = valid_end & ~true_non_public
            frame["start_flow_eligible"] = new_start_eligible
            frame["end_flow_eligible"] = new_end_eligible

            no_true_non_public = ~true_non_public
            frame.loc[no_true_non_public, "qc_flags"] = _clean_non_public_flag(
                frame.loc[no_true_non_public, "qc_flags"]
            )

            deltas["start_flow_eligible_rows"] += int(
                new_start_eligible.sum() - old_start_eligible.sum()
            )
            deltas["end_flow_eligible_rows"] += int(
                new_end_eligible.sum() - old_end_eligible.sum()
            )
            deltas["start_coordinate_source__non_public_node"] -= int(
                start_false.sum()
            )
            deltas["start_coordinate_source__source"] += int(start_false.sum())
            deltas["end_coordinate_source__non_public_node"] -= int(
                end_false.sum()
            )
            deltas["end_coordinate_source__source"] += int(end_false.sum())
            deltas["corrected_start_endpoint_rows"] += int(start_false.sum())
            deltas["corrected_end_endpoint_rows"] += int(end_false.sum())

            table = pa.Table.from_pandas(
                frame,
                schema=BICYCLE_SCHEMA,
                preserve_index=False,
                safe=False,
            )
            writer.write_table(table)
        writer.close()
        temporary.replace(destination)
    except Exception:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        raise
    return dict(deltas), source_file, archive_member


def _copy_non_bicycle_data(source_data: Path, destination_data: Path) -> None:
    for source in sorted(source_data.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_data)
        if relative.parts[0] == "bicycle_trips":
            continue
        if relative == Path("input_manifest.csv"):
            continue
        _link_or_copy(source, destination_data / relative)


def _repair_dc_legacy_temporary_station(
    data_dir: Path,
    run_dir: Path,
    deltas: pd.DataFrame,
    logger: logging.Logger,
    compression: str,
    compression_level: int,
) -> pd.DataFrame:
    dictionary = pd.read_csv(
        run_dir / "tables" / "historical_station_dictionary.csv",
        dtype={"station_id": str},
    )
    candidates = dictionary.loc[
        dictionary["city"].eq("DC") & dictionary["station_id"].eq("32218")
    ].sort_values("source_month")
    if candidates.empty:
        raise AssertionError("Station 32218 missing from historical dictionary")
    latitude = float(candidates.iloc[0]["latitude"])
    longitude = float(candidates.iloc[0]["longitude"])

    if "start_coordinate_source__historical_trip_station_nearest_month" not in deltas:
        deltas["start_coordinate_source__historical_trip_station_nearest_month"] = 0
    if "end_coordinate_source__historical_trip_station_nearest_month" not in deltas:
        deltas["end_coordinate_source__historical_trip_station_nearest_month"] = 0

    repaired_start = 0
    repaired_end = 0
    for path in sorted(
        (data_dir / "bicycle_trips" / "city=DC" / "year=2020").rglob(
            "*.parquet"
        )
    ):
        identifiers = pq.read_table(
            path,
            columns=[
                "start_station_id",
                "start_latitude",
                "start_longitude",
                "end_station_id",
                "end_latitude",
                "end_longitude",
            ],
        ).to_pandas()
        start_needs = (
            identifiers["start_station_id"].astype("string").eq("32218")
            & (
                identifiers["start_latitude"].isna()
                | identifiers["start_longitude"].isna()
            )
        )
        end_needs = (
            identifiers["end_station_id"].astype("string").eq("32218")
            & (
                identifiers["end_latitude"].isna()
                | identifiers["end_longitude"].isna()
            )
        )
        if not start_needs.any() and not end_needs.any():
            continue

        parquet = pq.ParquetFile(path)
        temporary = path.with_name(f".{path.name}.dc32218.partial")
        dictionary_columns = [
            field.name
            for field in BICYCLE_SCHEMA
            if pa.types.is_string(field.type)
            and field.name not in {"trip_record_id", "source_ride_id"}
        ]
        writer = pq.ParquetWriter(
            temporary,
            BICYCLE_SCHEMA,
            compression=compression,
            compression_level=compression_level,
            use_dictionary=dictionary_columns,
            write_statistics=True,
        )
        file_start = 0
        file_end = 0
        source_file = ""
        archive_member = ""
        try:
            for row_group_index in range(parquet.metadata.num_row_groups):
                frame = parquet.read_row_group(row_group_index).to_pandas()
                if not source_file:
                    source_file = str(frame["source_file"].iloc[0])
                    archive_member = str(frame["archive_member"].iloc[0])
                start_mask = (
                    frame["start_station_id"].astype("string").eq("32218")
                    & (
                        frame["start_latitude"].isna()
                        | frame["start_longitude"].isna()
                    )
                )
                end_mask = (
                    frame["end_station_id"].astype("string").eq("32218")
                    & (
                        frame["end_latitude"].isna()
                        | frame["end_longitude"].isna()
                    )
                )
                file_start += int(start_mask.sum())
                file_end += int(end_mask.sum())
                frame.loc[start_mask, "start_latitude"] = latitude
                frame.loc[start_mask, "start_longitude"] = longitude
                frame.loc[
                    start_mask, "start_coordinate_source"
                ] = "historical_trip_station_nearest_month"
                frame.loc[end_mask, "end_latitude"] = latitude
                frame.loc[end_mask, "end_longitude"] = longitude
                frame.loc[
                    end_mask, "end_coordinate_source"
                ] = "historical_trip_station_nearest_month"
                frame.loc[start_mask, "start_flow_eligible"] = True
                frame.loc[
                    end_mask & frame["ended_at_local"].notna(),
                    "end_flow_eligible",
                ] = True
                frame.loc[start_mask, "qc_flags"] = _add_qc_flag(
                    frame.loc[start_mask, "qc_flags"],
                    "start_coordinate_imputed",
                )
                frame.loc[end_mask, "qc_flags"] = _add_qc_flag(
                    frame.loc[end_mask, "qc_flags"],
                    "end_coordinate_imputed",
                )
                writer.write_table(
                    pa.Table.from_pandas(
                        frame,
                        schema=BICYCLE_SCHEMA,
                        preserve_index=False,
                        safe=False,
                    )
                )
            writer.close()
            temporary.replace(path)
        except Exception:
            writer.close()
            if temporary.exists():
                temporary.unlink()
            raise

        delta_mask = deltas["source_file"].eq(source_file) & deltas[
            "archive_member"
        ].eq(archive_member)
        if int(delta_mask.sum()) != 1:
            raise AssertionError(
                f"Cannot map DC 32218 repair to {source_file}::{archive_member}"
            )
        deltas.loc[delta_mask, "start_flow_eligible_rows"] += file_start
        deltas.loc[delta_mask, "end_flow_eligible_rows"] += file_end
        deltas.loc[
            delta_mask, "start_coordinate_source__source"
        ] -= file_start
        deltas.loc[
            delta_mask,
            "start_coordinate_source__historical_trip_station_nearest_month",
        ] += file_start
        deltas.loc[
            delta_mask, "end_coordinate_source__source"
        ] -= file_end
        deltas.loc[
            delta_mask,
            "end_coordinate_source__historical_trip_station_nearest_month",
        ] += file_end
        repaired_start += file_start
        repaired_end += file_end

    logger.info(
        "Repaired DC station 32218 legacy endpoints with historical coordinates: start=%d, end=%d",
        repaired_start,
        repaired_end,
    )
    deltas.to_csv(
        run_dir / "tables" / "station_rule_patch_deltas.csv", index=False
    )
    return deltas


def _patch_bicycle_data(
    source_data: Path,
    destination_data: Path,
    logger: logging.Logger,
    compression: str,
    compression_level: int,
) -> tuple[pd.DataFrame, int, int]:
    source_files = sorted((source_data / "bicycle_trips").rglob("*.parquet"))
    delta_by_member: defaultdict[
        tuple[str, str], Counter[str]
    ] = defaultdict(Counter)
    patched_files = 0
    linked_files = 0
    for index, source in enumerate(source_files, start=1):
        relative = source.relative_to(source_data)
        destination = destination_data / relative
        city_part = next(
            part for part in relative.parts if part.startswith("city=")
        )
        city = city_part.split("=", 1)[1]
        if not _has_false_positive_station(source, city):
            _link_or_copy(source, destination)
            linked_files += 1
        else:
            deltas, source_file, archive_member = _patch_parquet(
                source,
                destination,
                city,
                compression,
                compression_level,
            )
            delta_by_member[(source_file, archive_member)].update(deltas)
            patched_files += 1
        if index % 12 == 0 or index == len(source_files):
            logger.info(
                "Station-rule patch %d/%d (patched=%d, linked=%d)",
                index,
                len(source_files),
                patched_files,
                linked_files,
            )

    records: list[dict[str, object]] = []
    for (source_file, archive_member), deltas in delta_by_member.items():
        records.append(
            {
                "source_file": source_file,
                "archive_member": archive_member,
                **deltas,
            }
        )
    return pd.DataFrame(records), patched_files, linked_files


def _update_reconciliation(
    source_run: Path,
    run_dir: Path,
    deltas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reconciliation = pd.read_csv(
        source_run / "tables" / "row_reconciliation.csv"
    )
    for delta in deltas.to_dict("records"):
        mask = (
            reconciliation["source_file"].eq(delta["source_file"])
            & reconciliation["archive_member"].eq(delta["archive_member"])
        )
        if int(mask.sum()) != 1:
            raise AssertionError(
                "Could not map station correction to one reconciliation row: "
                f"{delta['source_file']}::{delta['archive_member']}"
            )
        for column, value in delta.items():
            if column in {"source_file", "archive_member"}:
                continue
            if column not in reconciliation.columns:
                reconciliation[column] = 0
            reconciliation.loc[mask, column] = (
                pd.to_numeric(
                    reconciliation.loc[mask, column], errors="coerce"
                ).fillna(0)
                + int(value)
            )
    reconciliation.to_csv(
        run_dir / "tables" / "row_reconciliation.csv", index=False
    )
    summary = _aggregate_summary(reconciliation.copy())
    summary.to_csv(run_dir / "tables" / "city_summary.csv", index=False)
    return reconciliation, summary


def _update_station_exceptions(source_run: Path, run_dir: Path) -> pd.DataFrame:
    exceptions = pd.read_csv(
        source_run / "tables" / "station_resolution_exceptions.csv",
        dtype={"station_id": str},
    )
    remove = pd.Series(False, index=exceptions.index)
    for city, identifiers in FALSE_POSITIVE_NON_PUBLIC_IDS.items():
        remove |= (
            exceptions["city"].eq(city)
            & exceptions["station_id"].astype("string").isin(identifiers)
            & exceptions["classification"].eq("non_public_node")
        )
    corrected = exceptions.loc[~remove].copy()
    corrected.to_csv(
        run_dir / "tables" / "station_resolution_exceptions.csv", index=False
    )
    return corrected


def _parquet_inventory(data_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(data_dir.rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        records.append(
            {
                "dataset": path.relative_to(data_dir).parts[0],
                "file": str(path),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "schema": str(parquet.schema_arrow),
            }
        )
    return pd.DataFrame(records)


def _validate_corrected_station_rules(data_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted((data_dir / "bicycle_trips").rglob("*.parquet")):
        relative = path.relative_to(data_dir)
        city = next(
            part.split("=", 1)[1]
            for part in relative.parts
            if part.startswith("city=")
        )
        identifiers = FALSE_POSITIVE_NON_PUBLIC_IDS.get(city)
        if not identifiers:
            continue
        table = pq.read_table(
            path,
            columns=[
                "start_station_id",
                "start_coordinate_source",
                "end_station_id",
                "end_coordinate_source",
            ],
        ).to_pandas()
        bad_start = (
            table["start_station_id"].astype("string").isin(identifiers)
            & table["start_coordinate_source"].eq("non_public_node")
        )
        bad_end = (
            table["end_station_id"].astype("string").isin(identifiers)
            & table["end_coordinate_source"].eq("non_public_node")
        )
        records.append(
            {
                "file": str(path),
                "city": city,
                "remaining_false_start_rows": int(bad_start.sum()),
                "remaining_false_end_rows": int(bad_end.sum()),
            }
        )
    result = pd.DataFrame(records)
    if not result.empty and (
        result["remaining_false_start_rows"].sum()
        + result["remaining_false_end_rows"].sum()
        != 0
    ):
        raise AssertionError("False-positive non-public station rows remain")
    return result


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    paths = _load_yaml(Path(args.paths))
    e01 = _load_yaml(Path(args.e01))
    source_run = Path(args.source_run).resolve()
    source_status = json.loads(
        (source_run / "run_status.json").read_text(encoding="utf-8")
    )
    if source_status.get("run_id") != source_run.name:
        raise ValueError("Source run identifier does not match directory")
    source_data = Path(source_status["data_directory"])
    timezone = ZoneInfo(str(paths.get("timezone", "Asia/Shanghai")))
    run_id = args.resume_run or (
        datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        + "_E01_station_rule_patch"
    )
    run_dir = Path(str(paths["runs"])) / run_id
    data_dir = Path(str(paths["processed_data"])) / "e01" / run_id
    resuming = bool(args.resume_run)
    if resuming and (not run_dir.is_dir() or not data_dir.is_dir()):
        raise FileNotFoundError(
            f"Cannot resume missing patch run or data directory: {run_id}"
        )
    for directory in [
        run_dir / "logs",
        run_dir / "tables",
        run_dir / "figures",
        data_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    if resuming:
        logger.info(
            "Resuming targeted E01 patch finalization for %s", run_id
        )
        (run_dir / "resume_command.txt").write_text(
            " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
        )
        deltas = pd.read_csv(
            run_dir / "tables" / "station_rule_patch_deltas.csv"
        )
        patched_files = len(deltas)
        total_bicycle_files = len(
            list((data_dir / "bicycle_trips").rglob("*.parquet"))
        )
        linked_files = total_bicycle_files - patched_files
    else:
        logger.info("Starting targeted E01 patch from %s", source_run.name)
        (run_dir / "command.txt").write_text(
            " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
        )
        _write_environment(run_dir / "environment.txt")
        (run_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "paths": paths,
                    "e01": e01,
                    "parent_run": str(source_run),
                    "false_positive_station_ids": {
                        city: sorted(values)
                        for city, values in FALSE_POSITIVE_NON_PUBLIC_IDS.items()
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "parent_run.txt").write_text(
            str(source_run) + "\n", encoding="utf-8"
        )

        for source in (source_run / "tables").glob("*"):
            if source.is_file():
                shutil.copy2(source, run_dir / "tables" / source.name)
        _copy_non_bicycle_data(source_data, data_dir)
        deltas, patched_files, linked_files = _patch_bicycle_data(
            source_data,
            data_dir,
            logger,
            str(e01["parquet"].get("compression", "zstd")),
            int(e01["parquet"].get("compression_level", 3)),
        )
        deltas.to_csv(
            run_dir / "tables" / "station_rule_patch_deltas.csv",
            index=False,
        )

        input_manifest = pd.read_csv(
            source_run / "tables" / "input_manifest.csv"
        )
        input_manifest = pd.concat(
            [
                input_manifest,
                pd.DataFrame(
                    [
                        {
                            "dataset": "parent_processed_version",
                            "city": "ALL",
                            "source_period": "2020-2022",
                            "absolute_path": str(source_data),
                            "size_bytes": 0,
                            "modified_at": datetime.fromtimestamp(
                                source_data.stat().st_mtime
                            ).isoformat(),
                            "sha256": _sha256(
                                source_run / "run_status.json"
                            ),
                            "e01_role": (
                                "targeted_station_rule_patch_parent"
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        input_manifest.to_csv(
            run_dir / "tables" / "input_manifest.csv", index=False
        )
        input_manifest.to_csv(data_dir / "input_manifest.csv", index=False)

    deltas = _repair_dc_legacy_temporary_station(
        data_dir,
        run_dir,
        deltas,
        logger,
        str(e01["parquet"].get("compression", "zstd")),
        int(e01["parquet"].get("compression_level", 3)),
    )

    reconciliation, summary = _update_reconciliation(
        source_run, run_dir, deltas
    )
    exceptions = _update_station_exceptions(source_run, run_dir)
    validation = _validate_corrected_station_rules(data_dir)
    validation.to_csv(
        run_dir / "tables" / "station_rule_patch_validation.csv", index=False
    )
    inventory = _parquet_inventory(data_dir)
    inventory.to_csv(
        run_dir / "tables" / "parquet_inventory.csv", index=False
    )

    expected = json.loads(
        (source_run / "tables" / "acceptance_counts.json").read_text(
            encoding="utf-8"
        )
    )
    (run_dir / "tables" / "acceptance_counts.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )
    expected_total_parquet_rows = (
        expected["crime_retained_rows"]
        + expected["bicycle_retained_rows"]
        + expected["crime_excluded_rows"]
        + expected["bicycle_outside_source_month_rows"]
        + expected["bicycle_blank_rows"]
    )
    event_datasets = {"crime_events", "bicycle_trips", "exclusions"}
    observed_total_parquet_rows = int(
        inventory.loc[
            inventory["dataset"].isin(event_datasets), "rows"
        ].sum()
    )
    if observed_total_parquet_rows != expected_total_parquet_rows:
        raise AssertionError(
            "Patched Parquet row count mismatch: "
            f"{observed_total_parquet_rows} != {expected_total_parquet_rows}"
        )

    ctx = type(
        "PatchContext",
        (),
        {"run_dir": run_dir},
    )()
    _write_interpretation(ctx, expected, summary, exceptions)
    interpretation = (run_dir / "interpretation.md").read_text(
        encoding="utf-8"
    )
    interpretation = interpretation.replace(
        "# E01 Interpretation and E02 Gate",
        "# E01 Interpretation and E02 Gate\n\n"
        "This is the final targeted station-rule correction of parent run "
        f"`{source_run.name}`. Unaffected Parquet files are byte-identical "
        "hard links; affected files were atomically rewritten and independently "
        "validated.",
    )
    (run_dir / "interpretation.md").write_text(
        interpretation, encoding="utf-8"
    )

    correction_rows = []
    for city, identifiers in FALSE_POSITIVE_NON_PUBLIC_IDS.items():
        for station_id in sorted(identifiers):
            correction_rows.append(
                {
                    "city": city,
                    "station_id": station_id,
                    "previous_classification": "non_public_node",
                    "corrected_classification": "public_station",
                    "reason": "Reviewed false-positive substring match",
                }
            )
    pd.DataFrame(correction_rows).to_csv(
        run_dir / "tables" / "station_rule_corrections.csv", index=False
    )

    elapsed = time.monotonic() - started
    status = {
        "run_id": run_id,
        "status": "complete_targeted_patch",
        "parent_run": source_run.name,
        "patched_parquet_files": patched_files,
        "linked_unaffected_parquet_files": linked_files,
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now().isoformat(),
        "data_directory": str(data_dir),
        "total_parquet_rows": observed_total_parquet_rows,
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    Path(str(paths["runs"])).joinpath("latest_e01_run.txt").write_text(
        str(run_dir) + "\n", encoding="utf-8"
    )
    Path(str(paths["processed_data"])).joinpath(
        "e01", "latest_e01_data.txt"
    ).write_text(str(data_dir) + "\n", encoding="utf-8")

    registry = Path(str(paths["empirical_root"])) / "docs" / "experiment_registry.csv"
    registry_frame = pd.read_csv(registry)
    if run_id not in set(registry_frame["run_id"].astype(str)):
        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "experiment": "E01",
                    "status": "complete_targeted_patch",
                    "started_at": datetime.fromtimestamp(
                        (run_dir / "command.txt").stat().st_mtime
                    ).isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "elapsed_seconds": round(elapsed, 3),
                    "run_directory": str(run_dir),
                    "notes": (
                        "Targeted city-specific station-rule correction; "
                        "raw data and unaffected Parquet files unchanged."
                    ),
                }
            ]
        ).to_csv(registry, mode="a", header=False, index=False)
    logger.info(
        "Completed targeted patch in %.1f seconds (patched=%d, linked=%d)",
        elapsed,
        patched_files,
        linked_files,
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Targeted E01 Parquet patch for reviewed station rules"
    )
    parser.add_argument("--paths", required=True)
    parser.add_argument("--e01", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument(
        "--resume-run",
        help=(
            "Finalize an existing targeted patch without reprocessing "
            "already-corrected Parquet files"
        ),
    )
    return parser


def main() -> None:
    run_dir = run(build_parser().parse_args())
    print(run_dir)


if __name__ == "__main__":
    main()
