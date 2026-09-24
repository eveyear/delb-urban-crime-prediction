"""Summarize endpoint-coordinate provenance for the Reviewer 4 revision.

The script reads the frozen E01 coordinate-source audit and reports mutually
exclusive endpoint outcomes by city. It does not modify or re-estimate the
analytic panels.
"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "empirical/runs/20260718_164130_E01_station_rule_patch/tables/coordinate_source_summary.csv"
OUT = ROOT / "empirical/runs/reviewer4_revision/tables/coordinate_provenance_summary.csv"


def main() -> None:
    data = pd.read_csv(SOURCE)
    groups = {
        "direct_source": {"source"},
        "deterministic_reference": {
            "historical_trip_station_month",
            "historical_trip_station_nearest_month",
            "current_snapshot_retrofit",
        },
        "unresolved": {"unresolved"},
        "non_public": {"non_public_node"},
    }
    rows = []
    for (city, label), frame in data.groupby(["city", "city_label"], sort=False):
        total = int(frame["rows"].sum())
        row = {"city": city, "city_label": label, "retained_endpoints": total}
        for name, sources in groups.items():
            count = int(frame.loc[frame["coordinate_source"].isin(sources), "rows"].sum())
            row[f"{name}_n"] = count
            row[f"{name}_pct"] = 100.0 * count / total
        rows.append(row)
    out = pd.DataFrame(rows)
    assert (out[[c for c in out if c.endswith("_n")]].sum(axis=1) == out["retained_endpoints"]).all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.6f")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
