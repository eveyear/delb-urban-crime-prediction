from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
import yaml


CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {
    "DC": "Washington, DC",
    "NY": "New York City",
    "VAN": "Vancouver",
}


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _format_float(value: float, digits: int = 3) -> str:
    numeric = float(value)
    if abs(numeric) < 0.5 * 10 ** (-digits):
        numeric = 0.0
    return f"{numeric:.{digits}f}"


def _format_count(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pointer(empirical_root: Path, relative: str) -> Path:
    pointer = empirical_root / relative
    target = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Missing pointer target: {target}")
    return target


def _source_acceptance(source_run: Path) -> tuple[int, int]:
    path = source_run / "tables" / "acceptance_checklist.csv"
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        passed = int(frame["status"].astype(str).eq("PASS").sum())
    elif "passed" in frame.columns:
        passed = int(frame["passed"].astype(bool).sum())
    else:
        raise ValueError(f"Missing status/passed column in {path}")
    return passed, int(len(frame))


def _write_table(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _data_summary_table(
    panel: pd.DataFrame,
    variables: pd.DataFrame,
) -> str:
    crime = variables.loc[variables["variable"].eq("crime_count_all")].set_index(
        "city"
    )
    bicycle = variables.loc[
        variables["variable"].eq("bike_total_flow")
    ].set_index("city")
    rows = []
    for city in CITY_ORDER:
        item = panel.set_index("city").loc[city]
        rows.append(
            " & ".join(
                [
                    latex_escape(CITY_LABELS[city]),
                    _format_count(item["active_grids"]),
                    _format_count(item["panel_rows"]),
                    _format_count(item["crime_events_in_primary_domain"]),
                    _format_count(item["bike_trips_any_endpoint_in_primary_domain"]),
                    _format_float(crime.loc[city, "mean"], 3),
                    _format_float(bicycle.loc[city, "mean"], 2),
                    f"{100 * float(crime.loc[city, 'zero_share']):.1f}\\%",
                ]
            )
            + r"\\"
        )
    return r"""
\begin{table}[H]
\caption{Descriptive statistics for the complete 1 km grid--day panels, 2020--2022.}
\label{tab:descriptive_statistics}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{City} & \textbf{Grids} & \textbf{Grid--days} &
\textbf{Crime events} & \textbf{Bike trips} &
\textbf{Mean crime} & \textbf{Mean bike flow} & \textbf{Zero crime}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustwidth}
\end{table}
"""


def _city_information_table(frame: pd.DataFrame) -> str:
    indexed = frame.set_index("city")
    rows = []
    for city in CITY_ORDER:
        item = indexed.loc[city]
        cmi = (
            f"{_format_float(item['delta_h_miller_madow_raw_bits'], 3)} "
            f"[{_format_float(item['delta_h_miller_madow_raw_bits_ci_low'], 3)}, "
            f"{_format_float(item['delta_h_miller_madow_raw_bits_ci_high'], 3)}]"
        )
        rows.append(
            " & ".join(
                [
                    latex_escape(CITY_LABELS[city]),
                    _format_count(item["sample_size"]),
                    _format_float(item["h0_miller_madow_bits"], 3),
                    _format_float(item["hb_miller_madow_bits"], 3),
                    cmi,
                    _format_float(item["l0_exact_mse"], 3),
                    _format_float(item["lb_exact_mse"], 3),
                    _format_float(item["delta_l_exact_mse"], 3),
                    f"{100 * float(item['relative_l_reduction']):.1f}\\%",
                ]
            )
            + r"\\"
        )
    return r"""
\begin{table}[H]
\caption{Pooled city-level conditional entropy and DELB results in the training bicycle-covered domain. Brackets report centered 95\% seven-day block-bootstrap intervals for CMI.}
\label{tab:city_entropy_results}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\begin{tabular}{lrrrrrrrr}
\toprule
\textbf{City} & \(\boldsymbol{N}\) & \(\boldsymbol{H^0}\) &
\(\boldsymbol{H^B}\) & \textbf{CMI [95\% CI]} &
\(\boldsymbol{L^0}\) & \(\boldsymbol{L^B}\) &
\(\boldsymbol{\Delta L}\) & \(\boldsymbol{R^L}\)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustwidth}
\end{table}
"""


def _local_information_table(
    summary: pd.DataFrame,
    moran: pd.DataFrame,
) -> str:
    summary_index = summary.set_index("city")
    moran_exact = moran.loc[
        moran["metric"].eq("delta_l_exact_mse")
    ].set_index("city")
    rows = []
    for city in CITY_ORDER:
        item = summary_index.loc[city]
        spatial = moran_exact.loc[city]
        screen = (
            f"{int(item['significant_positive_bound_reduction_grids'])}/"
            f"{int(item['eligible_grids'])}"
        )
        moran_text = (
            f"{_format_float(spatial['moran_i'], 3)} "
            f"({_format_float(spatial['permutation_p_value_two_sided'], 3)})"
        )
        rows.append(
            " & ".join(
                [
                    latex_escape(CITY_LABELS[city]),
                    f"{int(item['eligible_grids'])}/{int(item['covered_grids'])}",
                    _format_float(item["median_raw_cmi_bits"], 3),
                    _format_float(item["median_exact_delb_reduction_mse"], 3),
                    f"{100 * float(item['median_relative_bound_reduction']):.1f}\\%",
                    screen,
                    moran_text,
                ]
            )
            + r"\\"
        )
    return r"""
\begin{table}[H]
\caption{Spatially localized information value. The final column gives global Moran's \(I\) for DELB reduction with two-sided permutation \(p\)-values in parentheses.}
\label{tab:local_information_results}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
\textbf{City} & \textbf{Eligible/Covered} & \textbf{Median CMI} &
\textbf{Median \(\Delta L\)} & \textbf{Median \(R^L\)} &
\textbf{Positive screen} & \textbf{Moran's \(I\) (\(p\))}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{table}
"""


def _prediction_gap_table(frame: pd.DataFrame) -> str:
    model_order = [
        "state_mean",
        "poisson_glm",
        "negative_binomial_glm",
        "histogram_gradient_boosting_poisson",
    ]
    indexed = frame.set_index(["city", "model_family"])
    rows = []
    for city in CITY_ORDER:
        for model in model_order:
            item = indexed.loc[(city, model)]
            rows.append(
                " & ".join(
                    [
                        latex_escape(CITY_LABELS[city]),
                        latex_escape(item["model_label"]),
                        _format_float(item["mse_baseline_integer"], 3),
                        _format_float(item["mse_bicycle_integer"], 3),
                        _format_float(item["delta_mse_integer"], 3),
                        _format_float(item["delta_l_exact_mse"], 3),
                        _format_float(item["gap_narrowing"], 3),
                    ]
                )
                + r"\\"
            )
    return r"""
\begin{table}[H]
\caption{Matched held-out integer-prediction MSE and predictability-gap results for July--December 2022. Positive \(\Delta\mathrm{MSE}\) denotes lower error after adding bicycle information; gap narrowing equals \(\Delta\mathrm{MSE}-\Delta L\).}
\label{tab:prediction_gap_results}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\begin{tabular}{llrrrrr}
\toprule
\textbf{City} & \textbf{Model} & \(\boldsymbol{\mathrm{MSE}^0}\) &
\(\boldsymbol{\mathrm{MSE}^B}\) & \(\boldsymbol{\Delta\mathrm{MSE}}\) &
\(\boldsymbol{\Delta L}\) & \textbf{Gap narrowing}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustwidth}
\end{table}
"""


def _crime_type_table(frame: pd.DataFrame) -> str:
    subset = frame.loc[frame["period"].eq("pooled_2020_2022")].copy()
    category_order = ["PROPERTY_THEFT", "VEHICLE_THEFT", "BURGLARY"]
    indexed = subset.set_index(["city", "outcome"])
    rows = []
    for city in CITY_ORDER:
        for category in category_order:
            item = indexed.loc[(city, category)]
            interval = (
                f"[{_format_float(item['delta_h_raw_normal_ci_low'], 3)}, "
                f"{_format_float(item['delta_h_raw_normal_ci_high'], 3)}]"
            )
            rows.append(
                " & ".join(
                    [
                        latex_escape(CITY_LABELS[city]),
                        latex_escape(item["outcome_label"]),
                        _format_float(item["delta_h_miller_madow_raw_bits"], 3),
                        interval,
                        _format_float(item["delta_l_exact_mse"], 3),
                        f"{100 * float(item['relative_l_reduction']):.1f}\\%",
                    ]
                )
                + r"\\"
            )
    return r"""
\begin{table}[H]
\caption{Pooled crime-type heterogeneity in bicycle conditional information and DELB reduction. Intervals are centered 95\% seven-day block-bootstrap intervals for CMI.}
\label{tab:crime_type_heterogeneity}
\centering
\begin{tabular}{llrrrr}
\toprule
\textbf{City} & \textbf{Crime type} & \textbf{CMI} &
\textbf{95\% CI} & \(\boldsymbol{\Delta L}\) & \(\boldsymbol{R^L}\)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _robustness_table(
    theory: pd.DataFrame,
    rolling: pd.DataFrame,
) -> str:
    mm = theory.loc[theory["estimator"].eq("miller_madow")].copy()
    rows = []
    for city in CITY_ORDER:
        city_theory = mm.loc[mm["city"].eq(city)]
        city_rolling = rolling.loc[rolling["city"].eq(city)]
        positive_specs = int((city_theory["delta_l_exact_mse"] > 0).sum())
        positive_origins = int(
            round(
                (
                    city_rolling["origins"]
                    * city_rolling["positive_origin_share"]
                ).sum()
            )
        )
        hgb = city_rolling.loc[
            city_rolling["model_family"].eq(
                "histogram_gradient_boosting_poisson"
            )
        ].iloc[0]
        state = city_rolling.loc[
            city_rolling["model_family"].eq("state_mean")
        ].iloc[0]
        rows.append(
            " & ".join(
                [
                    latex_escape(CITY_LABELS[city]),
                    f"{positive_specs}/{len(city_theory)}",
                    f"{positive_origins}/{int(city_rolling['origins'].sum())}",
                    f"{100 * float(state['positive_origin_share']):.1f}\\%",
                    _format_float(state["mean_delta_mse"], 3),
                    f"{100 * float(hgb['positive_origin_share']):.1f}\\%",
                    _format_float(hgb["mean_delta_mse"], 3),
                ]
            )
            + r"\\"
        )
    return r"""
\begin{table}[H]
\caption{E09 robustness summary. Positive theory specifications count
Miller--Madow DELB reductions across 16 pre-specified alternatives.
Rolling forecasts comprise 12 monthly origins for each of four model
families. Positive \(\Delta\mathrm{MSE}\) favors the bicycle-aware model.}
\label{tab:robustness_summary}
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
\textbf{City} & \textbf{Positive theory} & \textbf{Positive origins} &
\multicolumn{2}{c}{\textbf{State mean}} &
\multicolumn{2}{c}{\textbf{Gradient boosting}}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
 & & & \textbf{Positive} & \textbf{Mean \(\Delta\)MSE} &
\textbf{Positive} & \textbf{Mean \(\Delta\)MSE}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\end{adjustwidth}
\end{table}
"""


def _null_calibration_table(
    city_nulls: pd.DataFrame,
    distant: pd.DataFrame,
) -> str:
    design_labels = {
        "temporal_block_permutation": "Temporal blocks",
        "spatial_series_permutation": "Spatial series",
        "circular_shift_surrogate": "Circular shift",
    }
    rows = []
    for city in CITY_ORDER:
        city_rows = city_nulls.loc[city_nulls["city"].eq(city)].set_index(
            "null_design"
        )
        city_distant = distant.loc[distant["city"].eq(city)].set_index(
            "distant_lag_days"
        )
        pvalues = [
            _format_float(
                city_rows.loc[design, "randomization_p_value"], 3
            )
            for design in design_labels
        ]
        recent_14 = city_distant.loc[14, "matched_recent_delta_h_bits"]
        recent_30 = city_distant.loc[30, "matched_recent_delta_h_bits"]
        ratio_14 = city_distant.loc[14, "distant_delta_h_bits"] / recent_14
        ratio_30 = city_distant.loc[30, "distant_delta_h_bits"] / recent_30
        rows.append(
            " & ".join(
                [
                    latex_escape(CITY_LABELS[city]),
                    *pvalues,
                    _format_float(ratio_14, 3),
                    _format_float(ratio_30, 3),
                ]
            )
            + r"\\"
        )
    return r"""
\begin{table}[H]
\caption{E10 null calibration and distant-lag diagnostics. Randomization
\(p\)-values are upper-tail tests of whether observed CMI exceeds the
specified null distribution. Distant-lag columns report
\(\widehat I_{\mathrm{distant}}/\widehat I_{\mathrm{recent}}\).}
\label{tab:null_calibration}
\centering
\begin{tabular}{lrrrrr}
\toprule
\textbf{City} & \multicolumn{3}{c}{\textbf{Randomization \(p\)-value}} &
\multicolumn{2}{c}{\textbf{Distant/recent CMI}}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
 & \textbf{Temporal} & \textbf{Spatial} & \textbf{Circular} &
\textbf{14 days} & \textbf{30 days}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def _write_macros(
    path: Path,
    panel: pd.DataFrame,
    city_information: pd.DataFrame,
    local_summary: pd.DataFrame,
    gaps: pd.DataFrame,
    crime_types: pd.DataFrame,
    city_nulls: pd.DataFrame,
    local_joint_null: pd.DataFrame,
    distant_lags: pd.DataFrame,
) -> None:
    primary = city_information.set_index("city")
    local = local_summary.set_index("city")
    total_crime = int(panel["crime_events_in_primary_domain"].sum())
    total_bicycle = int(panel["bike_trips_any_endpoint_in_primary_domain"].sum())
    positive_models = int(
        (
            (gaps["delta_mse_integer"] > 0)
            & (gaps["delta_mse_integer_ci_low"] > 0)
        ).sum()
    )
    pooled_crime_types = crime_types.loc[
        crime_types["period"].eq("pooled_2020_2022")
    ]
    crime_type_positive = int(
        pooled_crime_types["positive_information_screen"].astype(bool).sum()
    )
    city_null_rejections = int(
        city_nulls["separates_from_null"].astype(bool).sum()
    )
    local_null_rejections = int(
        local_joint_null["separates_all_three_nulls"].astype(bool).sum()
    )
    distant_ratios = (
        distant_lags["distant_delta_h_bits"]
        / distant_lags["matched_recent_delta_h_bits"]
    )
    lines = [
        "% Generated by E11. Do not edit manually.",
        rf"\newcommand{{\TotalCrimeEvents}}{{{total_crime:,}}}",
        rf"\newcommand{{\TotalBicycleTrips}}{{{total_bicycle:,}}}",
        rf"\newcommand{{\TotalPanelRows}}{{{int(panel['panel_rows'].sum()):,}}}",
        rf"\newcommand{{\TotalEligibleLocalGrids}}{{{int(local_summary['eligible_grids'].sum())}}}",
        rf"\newcommand{{\PositiveMatchedModels}}{{{positive_models}}}",
        rf"\newcommand{{\PositiveCrimeTypeTheory}}{{{crime_type_positive}}}",
        rf"\newcommand{{\CityNullRejections}}{{{city_null_rejections}}}",
        rf"\newcommand{{\LocalNullRejections}}{{{local_null_rejections}}}",
        rf"\newcommand{{\DistantRatioMin}}{{{distant_ratios.min():.3f}}}",
        rf"\newcommand{{\DistantRatioMax}}{{{distant_ratios.max():.3f}}}",
    ]
    for city, prefix in [("DC", "DC"), ("NY", "NY"), ("VAN", "VAN")]:
        lines.extend(
            [
                rf"\newcommand{{\{prefix}CMI}}{{{primary.loc[city, 'delta_h_miller_madow_raw_bits']:.3f}}}",
                rf"\newcommand{{\{prefix}DeltaL}}{{{primary.loc[city, 'delta_l_exact_mse']:.3f}}}",
                rf"\newcommand{{\{prefix}RelativeL}}{{{100 * primary.loc[city, 'relative_l_reduction']:.1f}\%}}",
                rf"\newcommand{{\{prefix}LocalMedianDeltaL}}{{{local.loc[city, 'median_exact_delb_reduction_mse']:.3f}}}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _workflow_figure(png_path: Path, pdf_path: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
        }
    )
    figure, axis = plt.subplots(figsize=(11.2, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.03, 0.63, 0.13, 0.19, "Crime events\n2020--2022", "#D9EAF7"),
        (0.03, 0.22, 0.13, 0.19, "Bicycle trips\n2020--2022", "#E2F0D9"),
        (0.22, 0.43, 0.15, 0.19, "1 km daily\ncomplete panel", "#EDEDED"),
        (0.43, 0.63, 0.15, 0.19, "Conditional entropy\nand CMI", "#FFF2CC"),
        (0.43, 0.22, 0.15, 0.19, "Matched integer\nprediction models", "#FCE4D6"),
        (0.65, 0.63, 0.14, 0.19, "Exact discrete\nerror floors", "#FFF2CC"),
        (0.65, 0.22, 0.14, 0.19, "Held-out MSE\nJuly--Dec. 2022", "#FCE4D6"),
        (
            0.82,
            0.63,
            0.16,
            0.19,
            "Predictability gap\nand heterogeneity",
            "#E4DFEC",
        ),
        (
            0.82,
            0.22,
            0.16,
            0.19,
            "Robustness and\nnull calibration",
            "#F4CCCC",
        ),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.0,
            edgecolor="#31546D",
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontweight="semibold",
            fontsize=7.8 if "Predictability" in label else 9,
        )
    arrows = [
        ((0.16, 0.725), (0.22, 0.555)),
        ((0.16, 0.315), (0.22, 0.505)),
        ((0.37, 0.555), (0.43, 0.725)),
        ((0.37, 0.495), (0.43, 0.315)),
        ((0.58, 0.725), (0.65, 0.725)),
        ((0.58, 0.315), (0.65, 0.315)),
        ((0.79, 0.725), (0.82, 0.725)),
        ((0.79, 0.315), (0.82, 0.315)),
    ]
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color="#31546D",
            )
        )
    axis.text(
        0.505,
        0.92,
        "Information-theoretic and predictive analysis workflow",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        0.505,
        0.06,
        "All bicycle variables are lagged and restricted to information available at forecast time.",
        ha="center",
        va="center",
        color="#555555",
    )
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _copy_figures(
    selected: list[dict[str, str]],
    source_runs: dict[str, Path],
    stable_directory: Path,
    run_directory: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for item in selected:
        source = source_runs[item["source"]] / "figures" / item["filename"]
        if not source.exists():
            raise FileNotFoundError(f"Missing selected figure: {source}")
        for destination_root in [stable_directory, run_directory]:
            destination = destination_root / item["filename"]
            shutil.copy2(source, destination)
        records.append(
            {
                "source_experiment": item["source"],
                "filename": item["filename"],
                "source_path": str(source),
                "sha256": _sha256(source),
            }
        )
    return records


def _append_registry(
    empirical_root: Path,
    run_id: str,
    started: datetime,
    completed: datetime,
    run_dir: Path,
) -> None:
    registry = empirical_root / "docs" / "experiment_registry.csv"
    record = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "experiment": "E11",
                "status": "complete",
                "started_at": started.isoformat(),
                "completed_at": completed.isoformat(),
                "elapsed_seconds": (completed - started).total_seconds(),
                "run_directory": str(run_dir),
                "notes": (
                    "Revised manuscript synthesis from accepted E02 and "
                    "E04--E10 results, including robustness and null "
                    "calibration."
                ),
            }
        ]
    )
    record.to_csv(registry, mode="a", header=False, index=False)


def run(config_path: Path) -> Path:
    started = datetime.now().astimezone()
    project_root = config_path.resolve().parents[2]
    empirical_root = project_root / "empirical"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = (
        started.strftime("%Y%m%d_%H%M%S")
        + "_E11_manuscript_synthesis"
    )
    run_dir = empirical_root / "runs" / run_id
    for relative in ["tables", "figures", "logs", "manuscript"]:
        (run_dir / relative).mkdir(parents=True, exist_ok=False)

    source_runs = {
        name: _read_pointer(empirical_root, pointer)
        for name, pointer in config["source_run_pointers"].items()
    }
    source_acceptance = {
        name: _source_acceptance(path)
        for name, path in source_runs.items()
    }
    failed_sources = {
        name: values
        for name, values in source_acceptance.items()
        if values[0] != values[1]
    }
    if failed_sources:
        raise AssertionError(f"Source acceptance failed: {failed_sources}")

    e02_tables = source_runs["E02"] / "tables"
    e04_tables = source_runs["E04"] / "tables"
    e05_tables = source_runs["E05"] / "tables"
    e07_tables = source_runs["E07"] / "tables"
    e08_tables = source_runs["E08"] / "tables"
    e09_tables = source_runs["E09"] / "tables"
    e10_tables = source_runs["E10"] / "tables"
    panel = pd.read_csv(e02_tables / "city_panel_summary.csv")
    variables = pd.read_csv(e02_tables / "variable_summary.csv")
    city_information = pd.read_csv(
        e04_tables / "primary_city_results.csv"
    )
    local_summary = pd.read_csv(e05_tables / "city_local_summary.csv")
    moran = pd.read_csv(e05_tables / "global_moran_results.csv")
    gaps = pd.read_csv(
        e07_tables / "city_model_predictability_gaps.csv"
    )
    crime_types = pd.read_csv(
        e08_tables / "crime_type_information_bounds.csv"
    )
    robustness = pd.read_csv(
        e09_tables / "theory_robustness_matrix.csv"
    )
    rolling = pd.read_csv(
        e09_tables / "rolling_origin_summary.csv"
    )
    city_nulls = pd.read_csv(
        e10_tables / "city_null_summary.csv"
    )
    local_joint_null = pd.read_csv(
        e10_tables / "local_joint_null_screen.csv"
    )
    distant_lags = pd.read_csv(
        e10_tables / "distant_lag_city.csv"
    )

    generated_directory = project_root / config["manuscript"][
        "generated_section_directory"
    ]
    stable_figure_directory = project_root / config["manuscript"][
        "stable_figure_directory"
    ]
    generated_directory.mkdir(parents=True, exist_ok=True)
    stable_figure_directory.mkdir(parents=True, exist_ok=True)

    generated = {
        "e11_macros.tex": None,
        "table_data_summary.tex": _data_summary_table(panel, variables),
        "table_city_information.tex": _city_information_table(city_information),
        "table_local_information.tex": _local_information_table(
            local_summary, moran
        ),
        "table_prediction_gap.tex": _prediction_gap_table(gaps),
        "table_crime_type.tex": _crime_type_table(crime_types),
        "table_robustness.tex": _robustness_table(robustness, rolling),
        "table_null_calibration.tex": _null_calibration_table(
            city_nulls, distant_lags
        ),
    }
    _write_macros(
        generated_directory / "e11_macros.tex",
        panel,
        city_information,
        local_summary,
        gaps,
        crime_types,
        city_nulls,
        local_joint_null,
        distant_lags,
    )
    for filename, content in generated.items():
        if content is not None:
            _write_table(generated_directory / filename, content)
        shutil.copy2(
            generated_directory / filename,
            run_dir / "manuscript" / filename,
        )

    figure_manifest = _copy_figures(
        config["selected_figures"],
        source_runs,
        stable_figure_directory,
        run_dir / "figures",
    )
    workflow_png = stable_figure_directory / "e11_analysis_workflow.png"
    workflow_pdf = stable_figure_directory / "e11_analysis_workflow.pdf"
    _workflow_figure(
        workflow_png,
        workflow_pdf,
        int(config["figure_dpi"]),
    )
    shutil.copy2(workflow_png, run_dir / "figures" / workflow_png.name)
    shutil.copy2(workflow_pdf, run_dir / "figures" / workflow_pdf.name)

    primary_crime_types = crime_types.loc[
        crime_types["period"].eq("pooled_2020_2022")
    ]
    checks = [
        (
            "Accepted source experiments",
            int(config["acceptance"]["expected_source_experiments"]),
            len(source_runs),
        ),
        (
            "Cities",
            int(config["acceptance"]["expected_cities"]),
            panel["city"].nunique(),
        ),
        (
            "City information rows",
            int(config["acceptance"]["expected_city_information_rows"]),
            len(city_information),
        ),
        (
            "Eligible local grids",
            int(config["acceptance"]["expected_local_grids"]),
            int(local_summary["eligible_grids"].sum()),
        ),
        (
            "Model gap rows",
            int(config["acceptance"]["expected_model_gap_rows"]),
            len(gaps),
        ),
        (
            "Pooled crime-type rows",
            int(config["acceptance"]["expected_crime_type_rows"]),
            len(primary_crime_types),
        ),
        (
            "Selected source figures",
            int(config["acceptance"]["expected_selected_figures"]),
            len(figure_manifest),
        ),
        (
            "Theory robustness rows",
            int(config["acceptance"]["expected_theory_robustness_rows"]),
            len(robustness),
        ),
        (
            "Positive Miller--Madow specifications",
            int(
                config["acceptance"][
                    "expected_positive_miller_madow_specs"
                ]
            ),
            int(
                (
                    robustness["estimator"].eq("miller_madow")
                    & (robustness["delta_l_exact_mse"] > 0)
                ).sum()
            ),
        ),
        (
            "City null rows",
            int(config["acceptance"]["expected_city_null_rows"]),
            len(city_nulls),
        ),
        (
            "City null rejections",
            int(config["acceptance"]["expected_city_null_rejections"]),
            int(city_nulls["separates_from_null"].astype(bool).sum()),
        ),
        (
            "Local joint-null rows",
            int(config["acceptance"]["expected_local_joint_null_rows"]),
            len(local_joint_null),
        ),
        (
            "Local joint-null rejections",
            int(
                config["acceptance"][
                    "expected_local_joint_null_rejections"
                ]
            ),
            int(
                local_joint_null["separates_all_three_nulls"]
                .astype(bool)
                .sum()
            ),
        ),
        (
            "Distant-lag rows",
            int(config["acceptance"]["expected_distant_lag_rows"]),
            len(distant_lags),
        ),
        ("Generated LaTeX files", 8, len(list(generated_directory.glob("*.tex")))),
        ("Generated workflow figure formats", 2, int(workflow_png.exists()) + int(workflow_pdf.exists())),
    ]
    acceptance = pd.DataFrame(
        [
            {
                "criterion": name,
                "expected": expected,
                "observed": observed,
                "status": "PASS" if expected == observed else "FAIL",
            }
            for name, expected, observed in checks
        ]
    )
    acceptance.to_csv(
        run_dir / "tables" / "acceptance_checklist.csv",
        index=False,
    )
    if acceptance["status"].ne("PASS").any():
        raise AssertionError("E11 acceptance gate failed")

    figure_manifest_frame = pd.DataFrame(figure_manifest)
    figure_manifest_frame.to_csv(
        run_dir / "tables" / "figure_manifest.csv",
        index=False,
    )
    source_manifest = pd.DataFrame(
        [
            {
                "experiment": name,
                "run_directory": str(path),
                "acceptance_passed": source_acceptance[name][0],
                "acceptance_total": source_acceptance[name][1],
            }
            for name, path in source_runs.items()
        ]
    )
    source_manifest.to_csv(
        run_dir / "tables" / "source_run_manifest.csv",
        index=False,
    )
    summary = {
        "run_id": run_id,
        "sources": {name: str(path) for name, path in source_runs.items()},
        "skipped_experiments": config["skipped_experiments"],
        "crime_events": int(panel["crime_events_in_primary_domain"].sum()),
        "bicycle_trips": int(
            panel["bike_trips_any_endpoint_in_primary_domain"].sum()
        ),
        "eligible_local_grids": int(local_summary["eligible_grids"].sum()),
        "matched_model_comparisons": int(len(gaps)),
        "crime_type_theory_rows": int(len(primary_crime_types)),
        "robustness_rows": int(len(robustness)),
        "city_null_rows": int(len(city_nulls)),
        "local_null_rows": int(len(local_joint_null)),
    }
    (run_dir / "tables" / "synthesis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    distant_ratios = (
        distant_lags["distant_delta_h_bits"]
        / distant_lags["matched_recent_delta_h_bits"]
    )
    interpretation = (
        "# E11 Revised Manuscript Synthesis\n\n"
        "E11 integrates the accepted E02 and E04--E10 outputs. The exact "
        "DELB theorem and numerical inversion remain valid independently "
        "of the empirical mobility application.\n\n"
        "All 48 city-by-Miller--Madow robustness specifications retained "
        "a positive DELB-reduction point estimate. However, 0/9 city-level "
        "null comparisons and 0/339 local joint-null screens separated "
        "the observed statistic from the pre-specified E10 nulls. The "
        f"distant/recent CMI ratios ranged from {distant_ratios.min():.3f} "
        f"to {distant_ratios.max():.3f}. The revised manuscript therefore "
        "reports robust point estimates but does not claim that recent "
        "local bicycle mobility has been shown to reduce irreducible "
        "crime-prediction error.\n\n"
        "All manuscript figures are generated by versioned Python code; "
        "none is manually edited.\n"
    )
    (run_dir / "interpretation.md").write_text(
        interpretation,
        encoding="utf-8",
    )
    shutil.copy2(config_path, run_dir / "config_snapshot.yaml")
    (run_dir / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    environment = subprocess.run(
        [sys.executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (run_dir / "environment.txt").write_text(
        f"python={environment}\nexecutable={sys.executable}\n",
        encoding="utf-8",
    )
    completed = datetime.now().astimezone()
    status = {
        "run_id": run_id,
        "status": "complete",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    pointer = empirical_root / "runs" / "latest_e11_run.txt"
    pointer.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    _append_registry(
        empirical_root,
        run_id,
        started,
        completed,
        run_dir.resolve(),
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize accepted E02 and E04--E10 results for E11."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("empirical/config/e11.yaml"),
    )
    args = parser.parse_args()
    result = run(args.config)
    print(result.resolve())


if __name__ == "__main__":
    main()
