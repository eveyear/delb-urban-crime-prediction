from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .figure_typography import normalize_chart_typography


SCALES = ["500m_day", "1km_day", "2km_week"]
SCALE_LABELS = {
    "500m_day": "500 m–day",
    "1km_day": "1 km–day",
    "2km_week": "2 km–week",
}
ESTIMATORS = ["plugin", "miller_madow", "jeffreys_dirichlet"]
ESTIMATOR_LABELS = {
    "plugin": "Plugin",
    "miller_madow": "Miller–Madow",
    "jeffreys_dirichlet": "Jeffreys–Dirichlet",
}
COLORS = {
    "plugin": "#4472C4",
    "miller_madow": "#ED7D31",
    "jeffreys_dirichlet": "#70AD47",
    "500m_day": "#4472C4",
    "1km_day": "#ED7D31",
    "2km_week": "#70AD47",
}
SAMPLE_COLORS = {90: "#A5A5A5", 150: "#4472C4", 365: "#ED7D31", 730: "#70AD47"}


def _load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def _save(figure: plt.Figure, stem: Path, *, dpi: int) -> list[Path]:
    normalize_chart_typography(figure)
    outputs = []
    for suffix in [".pdf", ".svg"]:
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    png = stem.with_suffix(".png")
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    outputs.append(png)
    plt.close(figure)
    return outputs


def _convergence_plot(
    summary: pd.DataFrame, output: Path, *, dpi: int
) -> tuple[list[Path], pd.DataFrame]:
    positive = summary.loc[summary["target_cmi_bits"].gt(0)].copy()
    plot_data = (
        positive.groupby(
            ["scale", "sample_length_periods", "estimator"],
            observed=True,
            as_index=False,
        )
        .agg(
            cmi_rmse_median=("cmi_rmse_bits", "median"),
            cmi_rmse_min=("cmi_rmse_bits", "min"),
            cmi_rmse_max=("cmi_rmse_bits", "max"),
            delb_rrmse_median=("delb_relative_rmse", "median"),
            delb_rrmse_min=("delb_relative_rmse", "min"),
            delb_rrmse_max=("delb_relative_rmse", "max"),
        )
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(12.2, 6.8), sharex="col", sharey="row",
        constrained_layout=True
    )
    for column, scale in enumerate(SCALES):
        part = plot_data.loc[plot_data["scale"].eq(scale)]
        for estimator in ESTIMATORS:
            line = part.loc[part["estimator"].eq(estimator)].sort_values(
                "sample_length_periods"
            )
            x = line["sample_length_periods"].to_numpy(float)
            for row, median, lower, upper in [
                (0, "cmi_rmse_median", "cmi_rmse_min", "cmi_rmse_max"),
                (1, "delb_rrmse_median", "delb_rrmse_min", "delb_rrmse_max"),
            ]:
                axes[row, column].plot(
                    x,
                    line[median],
                    marker="o",
                    linewidth=1.7,
                    markersize=4.5,
                    color=COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                )
                axes[row, column].fill_between(
                    x,
                    line[lower].to_numpy(float),
                    line[upper].to_numpy(float),
                    color=COLORS[estimator],
                    alpha=0.10,
                    linewidth=0,
                )
        axes[0, column].set_title(
            f"({chr(97 + column)}) {SCALE_LABELS[scale]}", loc="left"
        )
        for row in range(2):
            axes[row, column].set_xticks([90, 150, 365, 730])
            axes[row, column].set_xticklabels(["90", "150", "365", "730"])
            axes[row, column].grid(alpha=0.25, linewidth=0.6)
        axes[1, column].tick_params(axis="x", labelrotation=45)
    axes[0, 0].set_ylabel("CMI RMSE (bits)")
    axes[1, 0].set_ylabel("DELB relative RMSE")
    fig.supxlabel("Generated periods per spatial unit, $N$")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    outputs = _save(fig, output / "main_e19_truth_convergence", dpi=dpi)
    return outputs, plot_data


def _power_mdcmi_plot(
    power: pd.DataFrame, mdcmi: pd.DataFrame, output: Path, *, dpi: int
) -> tuple[list[Path], pd.DataFrame]:
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.55), constrained_layout=True)
    for column, scale in enumerate(SCALES):
        part = power.loc[power["scale"].eq(scale)]
        for periods in [90, 150, 365, 730]:
            line = part.loc[
                part["sample_length_periods"].eq(periods)
            ].sort_values("target_cmi_bits")
            axes[column].plot(
                line["target_cmi_bits"],
                line["power_or_type1_error"],
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=SAMPLE_COLORS[periods],
                label=f"$N={periods}$",
            )
        axes[column].axhline(0.8, color="#666666", linestyle="--", linewidth=0.9)
        axes[column].set(
            title=f"({chr(97 + column)}) {SCALE_LABELS[scale]}",
            xlabel="Population CMI (bits)",
            ylim=(-0.02, 1.04),
            xticks=[0, 0.03, 0.06, 0.12],
        )
        axes[column].grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Rejection probability")
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    axis = axes[3]
    for scale in SCALES:
        line = mdcmi.loc[mdcmi["scale"].eq(scale)].sort_values(
            "sample_length_periods"
        )
        axis.plot(
            line["sample_length_periods"],
            line["mdcmi80_interpolated_bits"],
            marker="o",
            linewidth=1.7,
            color=COLORS[scale],
            label=SCALE_LABELS[scale],
        )
    axis.set_xticks([90, 150, 365, 730])
    axis.set_xticklabels(["90", "150", "365", "730"])
    axis.tick_params(axis="x", labelrotation=45)
    axis.set(
        title="(d) Detectable CMI",
        xlabel="Periods, $N$",
        ylabel="MDCMI (bits)",
        ylim=(0, 0.125),
    )
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False, fontsize=7.5)
    outputs = _save(fig, output / "main_e19_power_mdcmi", dpi=dpi)
    combined = power.copy()
    combined["plot_component"] = "power"
    return outputs, combined


def _support_plot(
    support: pd.DataFrame, power: pd.DataFrame, output: Path, *, dpi: int
) -> tuple[list[Path], pd.DataFrame]:
    movable = power[
        ["scale", "sample_length_periods", "movable_share"]
    ].drop_duplicates()
    data = support.merge(
        movable, on=["scale", "sample_length_periods"], validate="one_to_one"
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    specifications = [
        ("baseline_support_atoms", "Occupied baseline states", "(a) Baseline support"),
        ("joint_support_atoms", "Occupied baseline–bicycle states", "(b) Joint support"),
        (
            "observations_per_joint_atom",
            "Observations per occupied joint state",
            "(c) Support density",
        ),
        ("movable_share", "Matched-permutation movable share", "(d) Reference mobility"),
    ]
    for axis, (metric, ylabel, title) in zip(axes.ravel(), specifications, strict=True):
        for scale in SCALES:
            line = data.loc[data["scale"].eq(scale)].sort_values(
                "sample_length_periods"
            )
            axis.plot(
                line["sample_length_periods"],
                line[metric],
                marker="o",
                linewidth=1.7,
                color=COLORS[scale],
                label=SCALE_LABELS[scale],
            )
        axis.set_xticks([90, 150, 365, 730])
        axis.set_xticklabels(["90", "150", "365", "730"])
        axis.set(title=title, xlabel="Generated periods per spatial unit, $N$", ylabel=ylabel)
        axis.grid(alpha=0.23, linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    outputs = _save(fig, output / "supp_e19_support_complexity", dpi=dpi)
    return outputs, data


def _heatmap_plot(
    summary: pd.DataFrame, output: Path, *, dpi: int
) -> tuple[list[Path], pd.DataFrame]:
    data = (
        summary.loc[summary["target_cmi_bits"].gt(0)]
        .groupby(
            ["scale", "sample_length_periods", "estimator"],
            observed=True,
            as_index=False,
        )
        .agg(
            cmi_rmse_bits=("cmi_rmse_bits", "median"),
            delb_relative_rmse=("delb_relative_rmse", "median"),
        )
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.9), constrained_layout=True)
    ranges = {
        metric: (
            float(data[metric].min()),
            float(data[metric].max()),
        )
        for metric in ["cmi_rmse_bits", "delb_relative_rmse"]
    }
    row_images = {}
    for column, estimator in enumerate(ESTIMATORS):
        part = data.loc[data["estimator"].eq(estimator)]
        for row, metric in enumerate(["cmi_rmse_bits", "delb_relative_rmse"]):
            matrix = (
                part.pivot(
                    index="scale", columns="sample_length_periods", values=metric
                )
                .reindex(index=SCALES, columns=[90, 150, 365, 730])
            )
            vmin, vmax = ranges[metric]
            image = axes[row, column].imshow(
                matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
            )
            row_images[row] = image
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    value = float(matrix.iloc[y, x])
                    axes[row, column].text(
                        x,
                        y,
                        f"{value:.3f}" if row == 0 else f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if value > (vmin + vmax) / 2 else "black",
                    )
            axes[row, column].set_xticks(range(4), ["90", "150", "365", "730"])
            axes[row, column].set_yticks(
                range(3), [SCALE_LABELS[x] for x in SCALES]
            )
            axes[row, column].set_xlabel("Generated periods, $N$")
            if column > 0:
                axes[row, column].tick_params(labelleft=False)
        axes[0, column].set_title(
            f"({chr(97 + column)}) {ESTIMATOR_LABELS[estimator]}", loc="left"
        )
    axes[0, 0].set_ylabel("CMI RMSE (bits)\nScale")
    axes[1, 0].set_ylabel("DELB relative RMSE\nScale")
    for row in range(2):
        fig.colorbar(
            row_images[row],
            ax=axes[row, :].tolist(),
            fraction=0.018,
            pad=0.02,
        )
    outputs = _save(fig, output / "supp_e19_scale_sample_heatmaps", dpi=dpi)
    return outputs, data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path) -> Path:
    config = _load_yaml(config_path)
    root = config_path.resolve().parents[2]
    cfg = config["stage_6_figures"]
    sources = {
        stage: root / "empirical" / str(cfg[f"source_stage_{stage}_run"])
        for stage in [2, 3, 4, 5]
    }
    for stage, source in sources.items():
        status = json.loads((source / f"stage{stage}_status.json").read_text())
        if status.get("status") != "complete":
            raise ValueError(f"Stage 6 requires accepted Stage {stage}.")
    convergence = pd.read_csv(sources[3] / "convergence_summary.csv")
    support = pd.read_csv(sources[2] / "support_complexity.csv")
    power = pd.read_csv(sources[4] / "power_summary.csv")
    mdcmi = pd.read_csv(sources[5] / "minimum_detectable_cmi.csv")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_dir = root / "empirical" / "runs" / f"{stamp}_E19_MVP_stage6_figures"
    figure_dir = run_dir / "figures"
    data_dir = run_dir / "plot_data"
    figure_dir.mkdir(parents=True, exist_ok=False)
    data_dir.mkdir()
    dpi = int(cfg["dpi"])
    files = []
    output, data = _convergence_plot(convergence, figure_dir, dpi=dpi)
    files.extend(output)
    data.to_csv(data_dir / "truth_convergence.csv", index=False)
    output, data = _power_mdcmi_plot(power, mdcmi, figure_dir, dpi=dpi)
    files.extend(output)
    data.to_csv(data_dir / "power_curves.csv", index=False)
    mdcmi.to_csv(data_dir / "mdcmi.csv", index=False)
    output, data = _support_plot(support, power, figure_dir, dpi=dpi)
    files.extend(output)
    data.to_csv(data_dir / "support_complexity.csv", index=False)
    output, data = _heatmap_plot(convergence, figure_dir, dpi=dpi)
    files.extend(output)
    data.to_csv(data_dir / "scale_sample_heatmaps.csv", index=False)
    manifest = pd.DataFrame(
        {
            "file": [path.name for path in files],
            "bytes": [path.stat().st_size for path in files],
            "sha256": [_sha256(path) for path in files],
        }
    )
    manifest.to_csv(run_dir / "figure_manifest.csv", index=False)
    status = {
        "experiment_id": "E19",
        "stage": "MVP_stage_6",
        "status": "complete",
        "figure_concepts": 4,
        "figure_files": len(files),
        "plot_data_files": 5,
        "manuscript_modified": False,
    }
    (run_dir / "stage6_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
