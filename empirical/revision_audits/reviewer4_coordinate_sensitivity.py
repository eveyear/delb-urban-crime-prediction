"""Source-coordinate-only sensitivity for DC and NY on the fixed E02 panel.

This subtracts eligible, in-domain bicycle endpoints whose coordinates came
from a deterministic station reference, retaining the original crime sample,
grid, split, and bicycle-covered domain. It then re-estimates the primary
Miller--Madow CMI and exact DELB with the original E09 estimator. Vancouver
cannot enter this source-only contrast: its trip files have no coordinates.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
EMPIRICAL = ROOT / "empirical"
sys.path.insert(0, str(EMPIRICAL / "src"))
from entropy_crime_bike.panel import assign_projected_grid  # noqa: E402
from entropy_crime_bike.e09_robustness import _load_panel, _prepare_spec, _bound_metrics  # noqa: E402
from entropy_crime_bike.conditional_information import ConditionalCodebook  # noqa: E402
from entropy_crime_bike.e10_placebo import (  # noqa: E402
    NULL_ORDER, _matrix_view, _null_bicycle_matrix,
    conditional_information, randomization_p_value,
)

E01 = EMPIRICAL / "processed_data/e01/20260718_164130_E01_station_rule_patch"
E02 = EMPIRICAL / "processed_data/e02/20260718_172231_E02_spatial_temporal_panel"
CONFIG = yaml.safe_load((EMPIRICAL / "config/e09.yaml").read_text())
CITIES = yaml.safe_load((EMPIRICAL / "config/cities.yaml").read_text())["cities"]
OUT = ROOT / "empirical/runs/reviewer4_sensitivity_20260923/tables"
REFERENCE_SOURCES = {
    "historical_trip_station_month",
    "historical_trip_station_nearest_month",
    "current_snapshot_retrofit",
}


def removed_endpoints(city: str, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    transformer = Transformer.from_crs("EPSG:4326", CITIES[city]["projected_crs"], always_xy=True)
    valid_grid = set(zip(panel["x_index"].astype(int), panel["y_index"].astype(int)))
    parts = []
    counts = {"reference_eligible_endpoints": 0, "reference_in_primary_domain": 0}
    files = sorted((E01 / "bicycle_trips" / f"city={city}").rglob("*.parquet"))
    assert files, city
    columns = [
        "start_date", "start_longitude", "start_latitude", "start_coordinate_source", "start_flow_eligible",
        "end_date", "end_longitude", "end_latitude", "end_coordinate_source", "end_flow_eligible",
    ]
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
            frame = batch.to_pandas()
            for endpoint in ("start", "end"):
                source = frame[f"{endpoint}_coordinate_source"]
                eligible = frame[f"{endpoint}_flow_eligible"].fillna(False).to_numpy(bool)
                selected = eligible & source.isin(REFERENCE_SOURCES).to_numpy(bool)
                if not selected.any():
                    continue
                counts["reference_eligible_endpoints"] += int(selected.sum())
                subset = frame.loc[selected, [f"{endpoint}_date", f"{endpoint}_longitude", f"{endpoint}_latitude"]]
                dates = pd.to_datetime(subset[f"{endpoint}_date"])
                period = dates.between("2020-01-01", "2022-12-31").to_numpy(bool)
                if not period.any():
                    continue
                subset = subset.loc[period]
                dates = dates.loc[period]
                x, y, valid = assign_projected_grid(
                    subset[f"{endpoint}_longitude"].reset_index(drop=True),
                    subset[f"{endpoint}_latitude"].reset_index(drop=True), transformer, 1000,
                )
                grid_pairs = list(zip(x.astype(int), y.astype(int)))
                inside = valid & np.fromiter((pair in valid_grid for pair in grid_pairs), dtype=bool)
                if not inside.any():
                    continue
                counts["reference_in_primary_domain"] += int(inside.sum())
                rows = pd.DataFrame({
                    "date": dates.to_numpy()[inside], "x_index": x[inside], "y_index": y[inside],
                    "removed_outflow": int(endpoint == "start"),
                    "removed_inflow": int(endpoint == "end"),
                })
                parts.append(rows.groupby(["date", "x_index", "y_index"], as_index=False)[["removed_outflow", "removed_inflow"]].sum())
    if parts:
        result = pd.concat(parts, ignore_index=True).groupby(
            ["date", "x_index", "y_index"], as_index=False
        )[["removed_outflow", "removed_inflow"]].sum()
    else:
        result = pd.DataFrame(columns=["date", "x_index", "y_index", "removed_outflow", "removed_inflow"])
    return result, counts


def estimate(frame: pd.DataFrame, city: str, design: str) -> dict:
    prepared, cuts = _prepare_spec(frame, "primary")
    prepared["one_block"] = 0
    codebook = ConditionalCodebook.from_frame(
        prepared, target=CONFIG["target"], baseline_state=CONFIG["information_sets"]["baseline_state"],
        bicycle_state_field="bicycle_lag_bin", block_field="one_block",
    )
    values = codebook.estimate(np.ones(1)).iloc[0]
    h0 = float(values["h0_miller_madow_bits"])
    hb = float(values["hb_miller_madow_bits"])
    return {
        "city": city, "design": design, "observations": len(prepared),
        "h0_miller_madow_bits": h0, "hb_miller_madow_bits": hb,
        "cmi_miller_madow_bits": h0 - hb, "cut_points": json.dumps(cuts),
        **_bound_metrics(h0, hb, CONFIG),
    }


def calibrate(frame: pd.DataFrame, city: str, design_name: str, null_design: str,
              replications: int = 199) -> dict:
    prepared, _ = _prepare_spec(frame, "primary")
    view = _matrix_view(prepared, baseline_state=CONFIG["information_sets"]["baseline_state"])
    y = np.asarray(view["target"], dtype=np.int64)
    s = np.asarray(view["baseline"], dtype=np.int64)
    b = np.asarray(view["bicycle_matrix"], dtype=np.int16)
    observed = conditional_information(y, s, b.ravel())[2]
    null_config = yaml.safe_load((EMPIRICAL / "config/e10.yaml").read_text())
    seed = 20260923 + zlib.crc32(f"reviewer4|{city}|{null_design}".encode())
    rng = np.random.default_rng(seed % 2**32)
    values = np.empty(replications, dtype=float)
    for i in range(len(values)):
        surrogate = _null_bicycle_matrix(null_design, b, rng, null_config)
        values[i] = conditional_information(y, s, surrogate.ravel())[2]
    return {
        "city": city, "design": design_name, "null_design": null_design,
        "replicates": len(values), "observed_cmi_bits": observed,
        "null_mean_cmi_bits": float(values.mean()),
        "observed_minus_null_mean_bits": float(observed - values.mean()),
        "randomization_p_value": randomization_p_value(observed, values),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []
    calibration = []
    confirmation = []
    for city in ("DC", "NY"):
        panel = _load_panel(E02, city)
        panel["date"] = pd.to_datetime(panel["date"])
        reference, counts = removed_endpoints(city, panel)
        comparison = panel.merge(reference, on=["date", "x_index", "y_index"], how="left", validate="one_to_one")
        for column in ("removed_outflow", "removed_inflow"):
            comparison[column] = comparison[column].fillna(0).astype("int32")
        original_total = int(panel["bike_total_flow"].sum())
        removed_total = int((comparison["removed_outflow"] + comparison["removed_inflow"]).sum())
        assert removed_total == counts["reference_in_primary_domain"]
        comparison["bike_outflow"] -= comparison["removed_outflow"]
        comparison["bike_inflow"] -= comparison["removed_inflow"]
        assert comparison[["bike_outflow", "bike_inflow"]].ge(0).all().all()
        comparison["bike_total_flow"] = comparison["bike_outflow"] + comparison["bike_inflow"]
        comparison["bike_net_flow"] = comparison["bike_outflow"] - comparison["bike_inflow"]
        assert len(comparison) == len(panel)
        rows.append(estimate(panel, city, "all_eligible_endpoints"))
        rows.append(estimate(comparison, city, "direct_source_endpoints_only"))
        original_prepared, _ = _prepare_spec(panel, "primary")
        source_prepared, _ = _prepare_spec(comparison, "primary")
        state_identical = np.array_equal(
            original_prepared["bicycle_lag_bin"].cat.codes.to_numpy(),
            source_prepared["bicycle_lag_bin"].cat.codes.to_numpy(),
        )
        counts["binned_state_identical"] = bool(state_identical)
        for null_design in NULL_ORDER:
            baseline_null = calibrate(panel, city, "all_eligible_endpoints", null_design)
            calibration.append(baseline_null)
            if state_identical:
                calibration.append({**baseline_null, "design": "direct_source_endpoints_only"})
            else:
                calibration.append(calibrate(comparison, city, "direct_source_endpoints_only", null_design))
        if city == "DC":
            for design_name, design_frame in (
                ("all_eligible_endpoints", panel),
                ("direct_source_endpoints_only", comparison),
            ):
                confirmation.append(calibrate(
                    design_frame, city, design_name, "circular_shift_surrogate", 999,
                ))
        audit.append({"city": city, **counts, "original_in_domain_flow": original_total,
                      "removed_in_domain_flow": removed_total,
                      "removed_share_pct": 100 * removed_total / original_total})
        print(city, audit[-1], flush=True)
    result = pd.DataFrame(rows)
    audit_table = pd.DataFrame(audit)
    calibration_table = pd.DataFrame(calibration)
    assert (result.groupby("city")["observations"].nunique() == 1).all()
    frozen = pd.read_csv(
        EMPIRICAL / "runs/20260719_190015_E10_placebo_null/tables/city_null_summary.csv"
    ).drop_duplicates("city")
    for city in ("DC", "NY"):
        baseline = result.loc[
            result["city"].eq(city) & result["design"].eq("all_eligible_endpoints")
        ].iloc[0]
        expected = frozen.loc[frozen["city"].eq(city)].iloc[0]
        assert abs(baseline["cmi_miller_madow_bits"] - expected["observed_delta_h_bits"]) < 1e-9
        assert abs(baseline["delta_l_exact_mse"] - expected["observed_delta_l_mse"]) < 1e-9
    result.to_csv(OUT / "primary_coordinate_source_sensitivity.csv", index=False, float_format="%.12g")
    audit_table.to_csv(OUT / "coordinate_source_sensitivity_audit.csv", index=False, float_format="%.9f")
    calibration_table.to_csv(OUT / "primary_coordinate_source_randomization.csv", index=False, float_format="%.12g")
    pd.DataFrame(confirmation).to_csv(
        OUT / "critical_circular_shift_confirmation.csv", index=False, float_format="%.12g",
    )
    print(result.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
