from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {
    "DC": "Washington, DC",
    "NY": "New York City",
    "VAN": "Vancouver",
}
COORDINATE_SOURCES = [
    ("source", "Source coordinates"),
    ("historical_trip_station_month", "Historical station, same month"),
    (
        "historical_trip_station_nearest_month",
        "Historical station, nearest month",
    ),
    ("current_snapshot_retrofit", "Current official snapshot"),
    ("unresolved", "Unresolved"),
    ("non_public_node", "Non-public node"),
]


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _monthly_summary(reconciliation: pd.DataFrame) -> pd.DataFrame:
    bicycle = reconciliation.loc[
        reconciliation["dataset"].eq("bicycle")
    ].copy()
    bicycle["month"] = pd.to_datetime(
        bicycle["source_period"], format="%Y-%m"
    )
    columns = [
        "physical_rows",
        "retained_rows",
        "excluded_rows",
        "outside_source_period_rows",
        "blank_rows",
    ]
    return (
        bicycle.groupby(["city", "month"], as_index=False)[columns]
        .sum()
        .sort_values(["city", "month"])
    )


def _coordinate_source_summary(city_summary: pd.DataFrame) -> pd.DataFrame:
    bicycle = city_summary.loc[
        city_summary["dataset"].eq("bicycle")
    ].set_index("city")
    records: list[dict[str, object]] = []
    for city in CITY_ORDER:
        retained = float(bicycle.loc[city, "retained_rows"])
        for endpoint in ["start", "end"]:
            for source, label in COORDINATE_SOURCES:
                column = f"{endpoint}_coordinate_source__{source}"
                rows = (
                    float(bicycle.loc[city, column])
                    if column in bicycle.columns
                    else 0.0
                )
                records.append(
                    {
                        "city": city,
                        "city_label": CITY_LABELS[city],
                        "endpoint": endpoint,
                        "coordinate_source": source,
                        "coordinate_source_label": label,
                        "rows": int(rows),
                        "share_of_retained_endpoints": (
                            rows / retained if retained else np.nan
                        ),
                    }
                )
    return pd.DataFrame(records)


def _acceptance_checklist(
    expected: dict[str, int],
    inventory: pd.DataFrame,
    validation: pd.DataFrame,
    corrections: pd.DataFrame,
    status: dict[str, object],
) -> pd.DataFrame:
    observed = inventory.groupby("dataset")["rows"].sum().to_dict()
    exclusion_rows = int(observed.get("exclusions", 0))
    expected_exclusions = (
        expected["crime_excluded_rows"]
        + expected["bicycle_outside_source_month_rows"]
        + expected["bicycle_blank_rows"]
    )
    false_rows = int(
        validation.get("remaining_false_start_rows", pd.Series(dtype=int)).sum()
        + validation.get("remaining_false_end_rows", pd.Series(dtype=int)).sum()
    )
    checks = [
        (
            "Crime retained rows",
            expected["crime_retained_rows"],
            int(observed.get("crime_events", 0)),
        ),
        (
            "Bicycle retained rows",
            expected["bicycle_retained_rows"],
            int(observed.get("bicycle_trips", 0)),
        ),
        ("Reason-coded exclusion rows", expected_exclusions, exclusion_rows),
        ("Remaining false non-public endpoints", 0, false_rows),
        (
            "Reviewed station IDs documented",
            sum(len(values) for values in {"DC": ["32218"], "NY": ["5351.03", "5442.05", "6459.06", "HB202"]}.values()),
            len(corrections),
        ),
        (
            "Run status complete",
            "complete_targeted_patch",
            str(status.get("status", "")),
        ),
    ]
    frame = pd.DataFrame(checks, columns=["check", "expected", "observed"])
    frame["difference"] = [
        observed_value - expected_value
        if isinstance(expected_value, (int, float))
        and isinstance(observed_value, (int, float))
        else ""
        for expected_value, observed_value in zip(
            frame["expected"], frame["observed"]
        )
    ]
    frame["status"] = np.where(
        frame["expected"].astype(str).eq(frame["observed"].astype(str)),
        "PASS",
        "FAIL",
    )
    return frame


def _plot_monthly(monthly: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(
        3, 1, figsize=(11.5, 10.5), sharex=True
    )
    figure.subplots_adjust(
        left=0.09, right=0.91, bottom=0.07, top=0.90, hspace=0.16
    )
    for axis, city in zip(axes, CITY_ORDER):
        subset = monthly.loc[monthly["city"].eq(city)]
        axis.plot(
            subset["month"],
            subset["physical_rows"],
            color="#1F4E78",
            linewidth=1.8,
            label="Physical rows",
        )
        axis.plot(
            subset["month"],
            subset["retained_rows"],
            color="#5B9BD5",
            linewidth=1.5,
            linestyle="--",
            label="Retained trips",
        )
        axis.set_title(CITY_LABELS[city], loc="left", fontweight="bold")
        axis.set_ylabel("Records")
        axis.grid(axis="y", color="#D9E2F3", linewidth=0.7)
        axis.ticklabel_format(axis="y", style="plain")
        if subset["excluded_rows"].max() > 0:
            secondary = axis.twinx()
            secondary.bar(
                subset["month"],
                subset["excluded_rows"],
                width=18,
                color="#C00000",
                alpha=0.25,
                label="Excluded rows",
            )
            secondary.set_ylabel("Excluded", color="#9C0006")
            secondary.tick_params(axis="y", colors="#9C0006")
        else:
            axis.text(
                0.985,
                0.94,
                "Excluded rows: 0",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="#9C0006",
                fontsize=9,
            )
    lines = axes[0].get_lines()
    figure.legend(
        [*lines, Patch(facecolor="#C00000", alpha=0.25)],
        ["Physical rows", "Retained trips", "Excluded rows"],
        loc="upper right",
        bbox_to_anchor=(0.91, 0.965),
        ncol=3,
        frameon=False,
    )
    axes[-1].set_xlabel("Source month")
    figure.suptitle(
        "E01 monthly bicycle record reconciliation",
        fontsize=15,
        fontweight="bold",
        x=0.09,
        ha="left",
        y=0.975,
    )
    _save_figure(figure, path)


def _plot_coordinate_mix(summary: pd.DataFrame, path: Path) -> None:
    combined = (
        summary.groupby(
            [
                "city",
                "city_label",
                "coordinate_source",
                "coordinate_source_label",
            ],
            as_index=False,
        )["rows"]
        .sum()
    )
    pivot = combined.pivot(
        index="city_label", columns="coordinate_source_label", values="rows"
    ).fillna(0)
    order = [CITY_LABELS[city] for city in CITY_ORDER]
    labels = [label for _, label in COORDINATE_SOURCES]
    pivot = pivot.reindex(index=order, columns=labels, fill_value=0)
    shares = pivot.div(pivot.sum(axis=1), axis=0) * 100
    colors = [
        "#1F4E78",
        "#70AD47",
        "#A5A5A5",
        "#5B9BD5",
        "#FFC000",
        "#C00000",
    ]
    figure, axis = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    left = np.zeros(len(shares))
    for label, color in zip(labels, colors):
        values = shares[label].to_numpy()
        axis.barh(
            shares.index,
            values,
            left=left,
            color=color,
            label=label,
            height=0.58,
        )
        left += values
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of retained trip endpoints (%)")
    axis.set_title(
        "E01 endpoint coordinate provenance by city",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#E7E6E6", linewidth=0.7)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=2,
        frameon=False,
    )
    _save_figure(figure, path)


def _plot_vancouver_resolution(summary: pd.DataFrame, path: Path) -> None:
    subset = (
        summary.loc[summary["city"].eq("VAN")]
        .groupby(
            ["coordinate_source", "coordinate_source_label"], as_index=False
        )["rows"]
        .sum()
    )
    subset = subset.loc[subset["rows"].gt(0)].sort_values("rows")
    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    colors = [
        "#C00000" if source == "non_public_node" else
        "#FFC000" if source == "unresolved" else "#5B9BD5"
        for source in subset["coordinate_source"]
    ]
    bars = axis.barh(
        subset["coordinate_source_label"], subset["rows"], color=colors
    )
    axis.set_xscale("log")
    axis.set_xlabel("Endpoint observations (log scale)")
    axis.set_title(
        "Vancouver endpoint resolution after official-station matching",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#E7E6E6", linewidth=0.7, which="both")
    for bar, value in zip(bars, subset["rows"]):
        axis.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
            fontsize=9,
        )
    _save_figure(figure, path)


def run(run_dir: Path) -> None:
    table_dir = run_dir / "tables"
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    reconciliation = pd.read_csv(table_dir / "row_reconciliation.csv")
    city_summary = pd.read_csv(table_dir / "city_summary.csv")
    inventory = pd.read_csv(table_dir / "parquet_inventory.csv")
    validation = pd.read_csv(
        table_dir / "station_rule_patch_validation.csv"
    )
    corrections = pd.read_csv(table_dir / "station_rule_corrections.csv")
    expected = json.loads(
        (table_dir / "acceptance_counts.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (run_dir / "run_status.json").read_text(encoding="utf-8")
    )

    monthly = _monthly_summary(reconciliation)
    coordinate = _coordinate_source_summary(city_summary)
    checklist = _acceptance_checklist(
        expected, inventory, validation, corrections, status
    )
    monthly.to_csv(table_dir / "monthly_qc_summary.csv", index=False)
    coordinate.to_csv(
        table_dir / "coordinate_source_summary.csv", index=False
    )
    checklist.to_csv(table_dir / "acceptance_checklist.csv", index=False)

    _plot_monthly(monthly, figure_dir / "e01_monthly_reconciliation.png")
    _plot_coordinate_mix(
        coordinate, figure_dir / "e01_coordinate_source_mix.png"
    )
    _plot_vancouver_resolution(
        coordinate, figure_dir / "e01_vancouver_station_resolution.png"
    )

    if not checklist["status"].eq("PASS").all():
        failures = checklist.loc[checklist["status"].ne("PASS"), "check"]
        raise AssertionError(
            "E01 reporting gate failed: " + ", ".join(failures)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create E01 QC tables and non-manuscript figures"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run(args.run_dir.resolve())


if __name__ == "__main__":
    main()
