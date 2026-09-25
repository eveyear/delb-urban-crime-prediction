from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from .figure_typography import normalize_chart_typography
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter


CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {
    "DC": "Washington, DC",
    "NY": "New York City",
    "VAN": "Vancouver",
}
WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(figure: plt.Figure, path: Path) -> None:
    normalize_chart_typography(figure)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _top_grid_share(values: pd.Series, fraction: float) -> float:
    numeric = values.astype(float).sort_values(ascending=False)
    total = numeric.sum()
    if total <= 0:
        return np.nan
    number = max(1, int(np.ceil(len(numeric) * fraction)))
    return float(numeric.iloc[:number].sum() / total)


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _derive_descriptive_tables(
    city_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    weekday: pd.DataFrame,
    spatial: pd.DataFrame,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enhanced = city_summary.copy()
    training_coverage = {}
    for city in CITY_ORDER:
        reference_path = (
            data_dir
            / "grid_reference"
            / f"city={city}"
            / "grid_reference.parquet"
        )
        reference = pd.read_parquet(
            reference_path,
            columns=["bike_coverage_training", "bike_coverage_ever"],
        )
        training_coverage[city] = float(
            reference["bike_coverage_training"].mean()
        )
    enhanced["bike_covered_training_grid_share"] = enhanced["city"].map(
        training_coverage
    )
    enhanced["crime_primary_domain_coverage"] = (
        enhanced["crime_events_in_primary_domain"]
        / enhanced["crime_events_e01"]
    )
    enhanced["bike_start_primary_domain_coverage"] = (
        enhanced["bike_start_endpoints_in_primary_domain"]
        / enhanced["bike_start_endpoints_e01_eligible"]
    )
    enhanced["bike_end_primary_domain_coverage"] = (
        enhanced["bike_end_endpoints_in_primary_domain"]
        / enhanced["bike_end_endpoints_e01_eligible"]
    )

    weekday_normalized = weekday.copy()
    weekday_normalized["crime_weekday_index"] = weekday_normalized.groupby(
        "city"
    )["mean_crime_count_per_grid_day"].transform(
        lambda values: 100 * values / values.mean()
    )
    weekday_normalized["bike_weekday_index"] = weekday_normalized.groupby(
        "city"
    )["mean_bike_total_flow_per_grid_day"].transform(
        lambda values: 100 * values / values.mean()
    )

    findings: list[dict[str, object]] = []
    for city in CITY_ORDER:
        city_monthly = monthly.loc[monthly["city"].eq(city)].copy()
        city_weekday = weekday.loc[weekday["city"].eq(city)].copy()
        city_spatial = spatial.loc[spatial["city"].eq(city)].copy()
        crime_peak = city_monthly.loc[
            city_monthly["crime_count_all"].idxmax()
        ]
        crime_trough = city_monthly.loc[
            city_monthly["crime_count_all"].idxmin()
        ]
        bike_peak = city_monthly.loc[
            city_monthly["bike_trips_any_endpoint"].idxmax()
        ]
        bike_trough = city_monthly.loc[
            city_monthly["bike_trips_any_endpoint"].idxmin()
        ]
        weekend = city_weekday["day_of_week"].isin([5, 6])
        findings.append(
            {
                "city": city,
                "crime_peak_month": crime_peak["month"],
                "crime_peak_count": int(crime_peak["crime_count_all"]),
                "crime_trough_month": crime_trough["month"],
                "crime_trough_count": int(crime_trough["crime_count_all"]),
                "bike_peak_month": bike_peak["month"],
                "bike_peak_trips": int(
                    bike_peak["bike_trips_any_endpoint"]
                ),
                "bike_trough_month": bike_trough["month"],
                "bike_trough_trips": int(
                    bike_trough["bike_trips_any_endpoint"]
                ),
                "weekend_to_weekday_crime_ratio": float(
                    city_weekday.loc[
                        weekend, "mean_crime_count_per_grid_day"
                    ].mean()
                    / city_weekday.loc[
                        ~weekend, "mean_crime_count_per_grid_day"
                    ].mean()
                ),
                "weekend_to_weekday_bike_ratio": float(
                    city_weekday.loc[
                        weekend, "mean_bike_total_flow_per_grid_day"
                    ].mean()
                    / city_weekday.loc[
                        ~weekend, "mean_bike_total_flow_per_grid_day"
                    ].mean()
                ),
                "monthly_crime_bike_rank_correlation": _rank_correlation(
                    city_monthly["crime_count_all"],
                    city_monthly["bike_trips_any_endpoint"],
                ),
                "spatial_crime_bike_rank_correlation": _rank_correlation(
                    np.log1p(city_spatial["crime_count_all"]),
                    np.log1p(city_spatial["bike_total_flow"]),
                ),
                "top_10pct_grids_crime_share": _top_grid_share(
                    city_spatial["crime_count_all"], 0.10
                ),
                "top_10pct_grids_bike_share": _top_grid_share(
                    city_spatial["bike_total_flow"], 0.10
                ),
            }
        )
    return enhanced, weekday_normalized, pd.DataFrame(findings)


def _plot_monthly(monthly: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11.2, 10.2), sharex=True)
    figure.subplots_adjust(
        left=0.09, right=0.90, bottom=0.12, top=0.92, hspace=0.18
    )
    for axis, city in zip(axes, CITY_ORDER):
        subset = monthly.loc[monthly["city"].eq(city)].copy()
        subset["month_date"] = pd.to_datetime(subset["month"])
        axis.plot(
            subset["month_date"],
            subset["crime_count_all"],
            color="#C00000",
            linewidth=2,
            label="Crime events",
        )
        axis.set_ylabel("Crime events", color="#9C0006")
        axis.tick_params(axis="y", colors="#9C0006")
        axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
        twin = axis.twinx()
        twin.plot(
            subset["month_date"],
            subset["bike_trips_any_endpoint"],
            color="#1F4E78",
            linewidth=1.8,
            label="Bicycle trips",
        )
        twin.set_ylabel("Bicycle trips", color="#1F4E78")
        twin.tick_params(axis="y", colors="#1F4E78")
        twin.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M")
        )
        axis.set_title(CITY_LABELS[city], loc="left", fontweight="bold")
        axis.axvline(
            pd.Timestamp("2022-01-01"),
            color="#7F7F7F",
            linestyle="--",
            linewidth=0.9,
        )
        axis.axvline(
            pd.Timestamp("2022-07-01"),
            color="#7F7F7F",
            linestyle=":",
            linewidth=0.9,
        )
    axes[-1].set_xlabel("Month")
    figure.suptitle(
        "Monthly crime and bicycle activity",
        x=0.09,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.50,
        0.025,
        "Primary 1 km domain; dashed: validation start; dotted: test start",
        ha="center",
        fontsize=9,
        color="#595959",
    )
    _save(figure, path)


def _plot_weekday(weekday: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(12.2, 4.3), sharey=True, constrained_layout=True
    )
    for axis, city in zip(axes, CITY_ORDER):
        subset = weekday.loc[weekday["city"].eq(city)].sort_values(
            "day_of_week"
        )
        axis.plot(
            subset["day_of_week"],
            subset["crime_weekday_index"],
            marker="o",
            color="#C00000",
            linewidth=1.8,
            label="Crime",
        )
        axis.plot(
            subset["day_of_week"],
            subset["bike_weekday_index"],
            marker="s",
            color="#1F4E78",
            linewidth=1.8,
            label="Bicycle flow",
        )
        axis.axhline(100, color="#A6A6A6", linewidth=0.8)
        axis.set_xticks(range(7), WEEKDAY_LABELS)
        axis.set_title(CITY_LABELS[city], fontweight="bold")
        axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
    axes[0].set_ylabel("Index (city daily mean = 100)")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle(
        "Day-of-week pattern of crime and bicycle activity",
        fontsize=14,
        fontweight="bold",
    )
    _save(figure, path)


def _plot_spatial(spatial: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(10.5, 13.0))
    figure.subplots_adjust(
        left=0.07, right=0.93, bottom=0.04, top=0.94, hspace=0.22, wspace=0.14
    )
    for row, city in enumerate(CITY_ORDER):
        subset = spatial.loc[spatial["city"].eq(city)].copy()
        for column, (variable, title, cmap) in enumerate(
            [
                ("crime_count_all", "Crime events", "Reds"),
                ("bike_total_flow", "Bicycle endpoints", "Blues"),
            ]
        ):
            axis = axes[row, column]
            values = np.log1p(subset[variable].astype(float))
            scatter = axis.scatter(
                subset["x_index"],
                subset["y_index"],
                c=values,
                cmap=cmap,
                norm=Normalize(vmin=0, vmax=max(values.max(), 1)),
                marker="s",
                s=18,
                linewidths=0,
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"{CITY_LABELS[city]} - {title}",
                loc="left",
                fontsize=11,
                fontweight="bold",
            )
            colorbar = figure.colorbar(
                scatter, ax=axis, fraction=0.045, pad=0.02
            )
            colorbar.set_label("log(1 + count)", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(
        "Three-year spatial distribution on the 1 km analysis grid",
        x=0.07,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    _save(figure, path)


def _plot_coverage(city_summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), constrained_layout=True
    )
    x_values = np.arange(len(CITY_ORDER))
    width = 0.24
    ordered = city_summary.set_index("city").loc[CITY_ORDER]
    measures = [
        ("crime_primary_domain_coverage", "Crime events", "#C00000"),
        (
            "bike_start_primary_domain_coverage",
            "Bicycle starts",
            "#1F4E78",
        ),
        (
            "bike_end_primary_domain_coverage",
            "Bicycle ends",
            "#5B9BD5",
        ),
    ]
    for offset, (column, label, color) in enumerate(measures):
        bars = axes[0].bar(
            x_values + (offset - 1) * width,
            100 * ordered[column],
            width,
            color=color,
            label=label,
        )
        for bar, value in zip(bars, ordered[column]):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                100 * value + 0.7,
                f"{100 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_xticks(x_values, [CITY_LABELS[value] for value in CITY_ORDER])
    axes[0].set_ylim(0, 106)
    axes[0].set_ylabel("Share retained in primary domain (%)")
    axes[0].set_title("Record and endpoint coverage", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", color="#E7E6E6", linewidth=0.7)

    training = 100 * ordered["bike_covered_training_grid_share"]
    ever = 100 * ordered["bike_covered_grid_share"]
    axes[1].bar(
        x_values - width / 2,
        training,
        width,
        color="#1F4E78",
        label="Training-period coverage",
    )
    axes[1].bar(
        x_values + width / 2,
        ever,
        width,
        color="#5B9BD5",
        label="Any-period coverage",
    )
    axes[1].set_xticks(x_values, [CITY_LABELS[value] for value in CITY_ORDER])
    axes[1].set_ylim(0, 106)
    axes[1].set_ylabel("Crime-domain grids with bicycle activity (%)")
    axes[1].set_title("Observed bicycle-system footprint", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#E7E6E6", linewidth=0.7)
    _save(figure, path)


def _plot_sparsity(city_summary: pd.DataFrame, path: Path) -> None:
    ordered = city_summary.set_index("city").loc[CITY_ORDER]
    matrix = 100 * ordered[
        ["zero_crime_grid_day_share", "zero_bike_flow_grid_day_share"]
    ].to_numpy()
    figure, axis = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100)
    axis.set_xticks([0, 1], ["Zero crime\ncount", "Zero bicycle\nflow"])
    axis.set_yticks(
        range(len(CITY_ORDER)),
        [CITY_LABELS[value] for value in CITY_ORDER],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.1f}%",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > 55 else "#1F1F1F",
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Zero-valued grid-days (%)")
    axis.set_title(
        "Sparsity of the complete 1 km daily panel",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    _save(figure, path)


def _write_findings(
    run_dir: Path,
    city_summary: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    lines = [
        "# E02 Descriptive Findings",
        "",
        "These results describe the standardized panel and do not establish "
        "conditional information gain, lower-bound reduction, predictive "
        "improvement, or causality.",
        "",
    ]
    for city in CITY_ORDER:
        summary = city_summary.loc[city_summary["city"].eq(city)].iloc[0]
        finding = findings.loc[findings["city"].eq(city)].iloc[0]
        lines.extend(
            [
                f"## {CITY_LABELS[city]}",
                "",
                f"- The primary domain contains {int(summary['active_grids']):,} "
                f"training-defined 1 km grids and retains "
                f"{100 * summary['crime_primary_domain_coverage']:.3f}% of "
                "all E01 crime events.",
                f"- Bicycle activity is observed in "
                f"{100 * summary['bike_covered_training_grid_share']:.1f}% "
                "of these grids during the training period.",
                f"- Crime peaks in {finding['crime_peak_month']} and reaches "
                f"its minimum in {finding['crime_trough_month']}; bicycle "
                f"trips peak in {finding['bike_peak_month']}.",
                f"- The top 10% of grids account for "
                f"{100 * finding['top_10pct_grids_crime_share']:.1f}% of "
                f"crime events and "
                f"{100 * finding['top_10pct_grids_bike_share']:.1f}% of "
                "observed bicycle endpoints.",
                f"- The descriptive spatial rank correlation between "
                f"three-year crime and bicycle totals is "
                f"{finding['spatial_crime_bike_rank_correlation']:.3f}.",
                "",
            ]
        )
    lines.extend(
        [
            "The observed bicycle footprint is not equivalent to total human "
            "mobility. In grids without system activity, a zero bicycle flow "
            "means no observed bike-system endpoint, not necessarily zero "
            "population movement.",
        ]
    )
    (run_dir / "descriptive_findings.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(run_dir: Path) -> None:
    table_dir = run_dir / "tables"
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    status = json.loads(
        (run_dir / "run_status.json").read_text(encoding="utf-8")
    )
    data_dir = Path(status["data_directory"])
    city_summary = pd.read_csv(table_dir / "city_panel_summary.csv")
    monthly = pd.read_csv(table_dir / "monthly_patterns.csv")
    weekday = pd.read_csv(table_dir / "weekday_patterns.csv")
    spatial = pd.read_csv(table_dir / "spatial_grid_totals.csv")
    enhanced, weekday_normalized, findings = _derive_descriptive_tables(
        city_summary, monthly, weekday, spatial, data_dir
    )
    enhanced.to_csv(table_dir / "city_panel_summary.csv", index=False)
    weekday_normalized.to_csv(
        table_dir / "weekday_patterns_normalized.csv", index=False
    )
    findings.to_csv(
        table_dir / "descriptive_findings_summary.csv", index=False
    )
    _plot_monthly(monthly, figure_dir / "e02_monthly_patterns.png")
    _plot_weekday(
        weekday_normalized, figure_dir / "e02_weekday_patterns.png"
    )
    _plot_spatial(spatial, figure_dir / "e02_spatial_patterns.png")
    _plot_coverage(enhanced, figure_dir / "e02_domain_coverage_qc.png")
    _plot_sparsity(enhanced, figure_dir / "e02_panel_sparsity.png")
    _write_findings(run_dir, enhanced, findings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate E02 descriptive tables and figures"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run(args.run_dir.resolve())


if __name__ == "__main__":
    main()
