from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def run(run_dir: Path) -> None:
    status = json.loads(
        (run_dir / "run_status.json").read_text(encoding="utf-8")
    )
    data_dir = Path(status["data_directory"])
    frames = [
        pq.read_table(path).to_pandas()
        for path in sorted(
            (data_dir / "grid_day_panel").rglob("*.parquet")
        )
    ]
    panel = pd.concat(frames, ignore_index=True)
    city_summary = pd.read_csv(
        run_dir / "tables" / "city_panel_summary.csv"
    )
    panel.sort_values(["city", "grid_id", "date"], inplace=True)
    expected_previous_crime = panel.groupby(
        ["city", "grid_id"], sort=False
    )["crime_count_all"].shift(1)
    expected_previous_bike = panel.groupby(
        ["city", "grid_id"], sort=False
    )["bike_total_flow"].shift(1)
    nonfirst = expected_previous_crime.notna()
    grid_sizes = panel.groupby(["city", "grid_id"]).size()
    checks = [
        (
            "Panel metadata row count",
            int(status["panel_rows"]),
            len(panel),
        ),
        (
            "Unique city-grid-date keys",
            len(panel),
            int(
                panel[
                    ["city", "grid_id", "date"]
                ].drop_duplicates().shape[0]
            ),
        ),
        (
            "Each grid has 1,096 days",
            0,
            int(grid_sizes.ne(1096).sum()),
        ),
        (
            "Date starts at 2020-01-01",
            "2020-01-01",
            str(pd.Timestamp(panel["date"].min()).date()),
        ),
        (
            "Date ends at 2022-12-31",
            "2022-12-31",
            str(pd.Timestamp(panel["date"].max()).date()),
        ),
        (
            "Crime category sum identity failures",
            0,
            int(
                (
                    panel[
                        [
                            "crime_count_property_theft",
                            "crime_count_vehicle_theft",
                            "crime_count_burglary",
                        ]
                    ].sum(axis=1)
                    != panel["crime_count_all"]
                ).sum()
            ),
        ),
        (
            "Bicycle total-flow identity failures",
            0,
            int(
                (
                    panel["bike_outflow"] + panel["bike_inflow"]
                    != panel["bike_total_flow"]
                ).sum()
            ),
        ),
        (
            "Bicycle net-flow identity failures",
            0,
            int(
                (
                    panel["bike_inflow"] - panel["bike_outflow"]
                    != panel["bike_net_flow"]
                ).sum()
            ),
        ),
        (
            "Negative count or endpoint-flow rows",
            0,
            int(
                panel[
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
                .lt(0)
                .any(axis=1)
                .sum()
            ),
        ),
        (
            "First-day crime lag nonmissing rows",
            0,
            int(
                panel.loc[
                    ~nonfirst, "crime_count_lag1"
                ].notna().sum()
            ),
        ),
        (
            "Crime lag-1 identity failures",
            0,
            int(
                (
                    panel.loc[
                        nonfirst, "crime_count_lag1"
                    ].astype(np.int64)
                    != expected_previous_crime.loc[nonfirst].astype(
                        np.int64
                    )
                ).sum()
            ),
        ),
        (
            "Bicycle lag-1 identity failures",
            0,
            int(
                (
                    panel.loc[
                        nonfirst, "bike_total_flow_lag1"
                    ].astype(np.int64)
                    != expected_previous_bike.loc[nonfirst].astype(
                        np.int64
                    )
                ).sum()
            ),
        ),
        (
            "Panel crime sum",
            int(city_summary["crime_events_in_primary_domain"].sum()),
            int(panel["crime_count_all"].sum()),
        ),
        (
            "Panel bicycle endpoint sum",
            int(city_summary["bike_total_flow_in_panel"].sum()),
            int(panel["bike_total_flow"].sum()),
        ),
    ]
    frame = pd.DataFrame(
        checks, columns=["check", "expected", "observed"]
    )
    frame["status"] = np.where(
        frame["expected"].astype(str).eq(
            frame["observed"].astype(str)
        ),
        "PASS",
        "FAIL",
    )
    frame.to_csv(
        run_dir / "tables" / "panel_integrity_checks.csv", index=False
    )
    (run_dir / "panel_validation_status.json").write_text(
        json.dumps(
            {
                "status": (
                    "PASS"
                    if frame["status"].eq("PASS").all()
                    else "FAIL"
                ),
                "checks": len(frame),
                "completed_at": pd.Timestamp.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not frame["status"].eq("PASS").all():
        failures = frame.loc[frame["status"].ne("PASS"), "check"]
        raise AssertionError(
            "E02 panel integrity failed: " + ", ".join(failures)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run independent E02 panel-integrity checks"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run(args.run_dir.resolve())


if __name__ == "__main__":
    main()
