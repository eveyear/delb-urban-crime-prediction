"""Conservative Vancouver duplicate-signature stress test.

The E00 raw-file audit found 217 repeated complete source signatures, but the
raw files provide no stable trip ID. This deliberately stronger test removes
one row for every repeated *standardized* signature, which also captures
potentially legitimate identical trips. It must not be called exact dedup.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
EMPIRICAL = ROOT / "empirical"
sys.path.insert(0, str(EMPIRICAL / "src"))
sys.path.insert(0, str(ROOT / "empirical/revision_audits"))
from entropy_crime_bike.panel import assign_projected_grid  # noqa: E402
from entropy_crime_bike.e09_robustness import _load_panel  # noqa: E402
from reviewer4_coordinate_sensitivity import estimate  # noqa: E402

E01 = EMPIRICAL / "processed_data/e01/20260718_164130_E01_station_rule_patch/bicycle_trips/city=VAN"
E02 = EMPIRICAL / "processed_data/e02/20260718_172231_E02_spatial_temporal_panel"
OUT = ROOT / "empirical/runs/reviewer1_duplicate_stress_20260924/tables"
SIGNATURE = [
    "started_at_local", "ended_at_local", "start_station_id", "start_station_name",
    "end_station_id", "end_station_name", "duration_seconds", "rideable_type",
    "member_type",
]


def main() -> None:
    columns = SIGNATURE + [
        "start_date", "start_longitude", "start_latitude", "start_flow_eligible",
        "end_date", "end_longitude", "end_latitude", "end_flow_eligible",
    ]
    trips = ds.dataset(str(E01), format="parquet", partitioning="hive").to_table(columns=columns).to_pandas()
    repeated = trips.duplicated(subset=SIGNATURE, keep="first")
    candidates = trips.loc[repeated]
    panel = _load_panel(E02, "VAN")
    panel["date"] = pd.to_datetime(panel["date"])
    crs = yaml.safe_load((EMPIRICAL / "config/cities.yaml").read_text())["cities"]["VAN"]["projected_crs"]
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    eligible_grids = set(zip(panel.x_index.astype(int), panel.y_index.astype(int)))
    parts = []
    for side in ("start", "end"):
        subset = candidates.loc[candidates[f"{side}_flow_eligible"].fillna(False)].copy()
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
    assert changed[["bike_outflow", "bike_inflow"]].ge(0).all().all()
    changed["bike_total_flow"] = changed["bike_outflow"] + changed["bike_inflow"]
    changed["bike_net_flow"] = changed["bike_outflow"] - changed["bike_inflow"]
    baseline = estimate(panel, "VAN", "all_retained_trips")
    stress = estimate(changed, "VAN", "remove_repeated_standardized_signatures")
    rows = pd.DataFrame([baseline, stress])
    rows["repeated_standardized_signatures_removed"] = int(repeated.sum())
    rows["removed_in_domain_endpoint_contributions"] = int(
        changed["removed_outflow"].sum() + changed["removed_inflow"].sum()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "primary_duplicate_stress.csv", index=False)
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
