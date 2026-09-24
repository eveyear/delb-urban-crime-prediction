"""Sensitivity deleting only the 217 repeated *complete raw-row* signatures.

The raw records have no ride identifier. This is a deterministic scenario,
not evidence that the repeated rows are duplicate journeys. The original
crime panel, frozen covered domain, 1 km grid, and estimator are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
EMPIRICAL = Path(os.environ.get("DELB_EMPIRICAL_DIR", ROOT / "empirical")).expanduser().resolve()
sys.path.insert(0, str(EMPIRICAL / "src"))
sys.path.insert(0, str(ROOT / "empirical/revision_audits"))
from entropy_crime_bike.panel import assign_projected_grid  # noqa: E402
from entropy_crime_bike.e09_robustness import _load_panel  # noqa: E402
from reviewer4_coordinate_sensitivity import estimate  # noqa: E402

RAW = Path(os.environ.get("DELB_VAN_RAW_DIR", ROOT / "data/Bicycle/VAN")).expanduser().resolve()
E01 = Path(os.environ.get(
    "DELB_E01_VAN_DIR",
    EMPIRICAL / "processed_data/e01/20260718_164130_E01_station_rule_patch/bicycle_trips/city=VAN",
)).expanduser().resolve()
E02 = Path(os.environ.get(
    "DELB_E02_PANEL_DIR",
    EMPIRICAL / "processed_data/e02/20260718_172231_E02_spatial_temporal_panel",
)).expanduser().resolve()
OUT = ROOT / "empirical/runs/reviewer1_exact_source_rows_20260924/tables"


def repeated_source_keys() -> tuple[set[tuple[str, int]], dict[str, str]]:
    if not RAW.is_dir():
        raise FileNotFoundError(
            f"Vancouver source CSV directory not found: {RAW}. "
            "Set DELB_VAN_RAW_DIR to the licensed source-file directory."
        )
    keys: set[tuple[str, int]] = set()
    source_hashes: dict[str, str] = {}
    for path in sorted(RAW.glob("Mobi_System_Data_20??-??.csv")):
        seen: set[int] = set()
        row_offset = 0
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                sha.update(block)
        source_hashes[path.name] = sha.hexdigest()
        for physical in pd.read_csv(
            path, chunksize=250_000, dtype=str, encoding="utf-8-sig",
            encoding_errors="replace", low_memory=False, on_bad_lines="skip",
        ):
            count = len(physical)
            nonblank_mask = ~physical.isna().all(axis=1)
            nonblank = physical.loc[nonblank_mask]
            hashes = pd.util.hash_pandas_object(
                nonblank.fillna("<NA>").astype("string"), index=False,
            ).to_numpy(dtype=np.uint64)
            earlier_in_chunk = pd.Series(hashes).duplicated().to_numpy()
            earlier_chunk = np.fromiter((int(h) in seen for h in hashes), dtype=bool)
            repeated = earlier_in_chunk | earlier_chunk
            for position in np.flatnonzero(nonblank_mask.to_numpy())[repeated]:
                keys.add((str(path), row_offset + int(position) + 1))
            seen.update(map(int, hashes))
            row_offset += count
    if len(keys) != 217:
        raise AssertionError(f"Expected the original 217 repeated source rows, found {len(keys)}")
    return keys, source_hashes


def selected_standardized_rows(keys: set[tuple[str, int]]) -> pd.DataFrame:
    columns = [
        "source_file", "source_row_number", "start_date", "start_longitude",
        "start_latitude", "start_flow_eligible", "end_date", "end_longitude",
        "end_latitude", "end_flow_eligible",
    ]
    selected = []
    for path in sorted(E01.rglob("*.parquet")):
        file = pq.ParquetFile(path)
        for batch in file.iter_batches(batch_size=250_000, columns=columns):
            frame = batch.to_pandas()
            matches = pd.MultiIndex.from_frame(frame[["source_file", "source_row_number"]]).isin(keys)
            if matches.any():
                selected.append(frame.loc[matches].copy())
    result = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)
    if result.duplicated(["source_file", "source_row_number"]).any():
        raise AssertionError("A source row appears more than once in the standardized data")
    if len(result) > len(keys):
        raise AssertionError("More standardized rows than selected raw records")
    return result


def main() -> None:
    for name, path in (("DELB_VAN_RAW_DIR", RAW), ("DELB_E01_VAN_DIR", E01),
                       ("DELB_E02_PANEL_DIR", E02)):
        if not path.is_dir():
            raise FileNotFoundError(
                f"Required input directory missing: {path}. Set {name} to the "
                "corresponding licensed source or frozen processed-data directory."
            )
    if not (EMPIRICAL / "config/cities.yaml").is_file():
        raise FileNotFoundError(f"Missing city configuration: {EMPIRICAL / 'config/cities.yaml'}")
    keys, hashes = repeated_source_keys()
    trips = selected_standardized_rows(keys)
    panel = _load_panel(E02, "VAN")
    panel["date"] = pd.to_datetime(panel["date"])
    crs = yaml.safe_load((EMPIRICAL / "config/cities.yaml").read_text())["cities"]["VAN"]["projected_crs"]
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    eligible_grids = set(zip(panel.x_index.astype(int), panel.y_index.astype(int)))
    parts = []
    for side in ("start", "end"):
        subset = trips.loc[trips[f"{side}_flow_eligible"].fillna(False)].copy()
        if subset.empty:
            continue
        x, y, valid = assign_projected_grid(
            subset[f"{side}_longitude"].reset_index(drop=True),
            subset[f"{side}_latitude"].reset_index(drop=True), transform, 1000,
        )
        inside = valid & np.fromiter(
            ((int(a), int(b)) in eligible_grids for a, b in zip(x, y)), dtype=bool,
        )
        parts.append(pd.DataFrame({
            "date": pd.to_datetime(subset[f"{side}_date"].to_numpy()[inside]),
            "x_index": x[inside].astype(int), "y_index": y[inside].astype(int),
            "removed_outflow": int(side == "start"),
            "removed_inflow": int(side == "end"),
        }))
    removed = pd.concat(parts).groupby(["date", "x_index", "y_index"], as_index=False)[
        ["removed_outflow", "removed_inflow"]
    ].sum()
    changed = panel.merge(removed, on=["date", "x_index", "y_index"], how="left", validate="one_to_one")
    for col in ("removed_outflow", "removed_inflow"):
        changed[col] = changed[col].fillna(0).astype(int)
    changed["bike_outflow"] -= changed["removed_outflow"]
    changed["bike_inflow"] -= changed["removed_inflow"]
    if not changed[["bike_outflow", "bike_inflow"]].ge(0).all().all():
        raise AssertionError("Endpoint subtraction exceeds original panel counts")
    changed["bike_total_flow"] = changed["bike_outflow"] + changed["bike_inflow"]
    changed["bike_net_flow"] = changed["bike_outflow"] - changed["bike_inflow"]
    rows = pd.DataFrame([
        estimate(panel, "VAN", "all_retained_source_rows"),
        estimate(changed, "VAN", "remove_217_repeated_complete_raw_rows"),
    ])
    rows["repeated_raw_rows_identified"] = len(keys)
    rows["repeated_raw_rows_retained_in_e01"] = len(trips)
    rows["in_domain_endpoint_contributions_removed"] = int(
        changed["removed_outflow"].sum() + changed["removed_inflow"].sum()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "primary_source_row_dedup.csv", index=False)
    (OUT / "source_provenance.json").write_text(json.dumps({
        "definition": "All-column equality within each raw monthly CSV; retain first row",
        "raw_rows_identified": len(keys),
        "raw_files_sha256": hashes,
        "selected_source_file_row_numbers": sorted([list(key) for key in keys]),
    }, indent=2) + "\n")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
