from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from .figure_typography import normalize_chart_typography
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle


CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {
    "DC": "Washington, DC",
    "NY": "New York City",
    "VAN": "Vancouver",
}
LAYOUT_TOLERANCE = 0.005


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 14,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _square_limits(frame: pd.DataFrame, padding: float = 0.06) -> tuple[float, ...]:
    x_min = float(frame["x_index"].min()) - 0.5
    x_max = float(frame["x_index"].max()) + 0.5
    y_min = float(frame["y_index"].min()) - 0.5
    y_max = float(frame["y_index"].max()) + 0.5
    side = max(x_max - x_min, y_max - y_min) * (1.0 + 2.0 * padding)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    return (
        x_mid - side / 2.0,
        x_mid + side / 2.0,
        y_mid - side / 2.0,
        y_mid + side / 2.0,
    )


def _fix_map_axis(axis: plt.Axes, frame: pd.DataFrame) -> None:
    x_min, x_max, y_min, y_max = _square_limits(frame)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_box_aspect(1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _axis_geometry(
    figure: plt.Figure, axes: list[plt.Axes], figure_name: str
) -> pd.DataFrame:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    records = []
    for index, axis in enumerate(axes, start=1):
        box = axis.get_window_extent(renderer=renderer)
        records.append(
            {
                "figure": figure_name,
                "panel": index,
                "width_px": float(box.width),
                "height_px": float(box.height),
            }
        )
    result = pd.DataFrame(records)
    for dimension in ["width_px", "height_px"]:
        values = result[dimension].to_numpy(float)
        relative_range = (values.max() - values.min()) / values.mean()
        if relative_range > LAYOUT_TOLERANCE:
            raise AssertionError(
                f"{figure_name} {dimension} differs by {relative_range:.3%}; "
                f"tolerance is {LAYOUT_TOLERANCE:.3%}."
            )
    return result


def _save_both(figure: plt.Figure, base_path: Path) -> list[Path]:
    normalize_chart_typography(figure)
    outputs = [base_path.with_suffix(".pdf"), base_path.with_suffix(".png")]
    figure.savefig(outputs[0], facecolor="white")
    figure.savefig(outputs[1], dpi=300, facecolor="white")
    plt.close(figure)
    return outputs


def _plot_figure_3(spatial: pd.DataFrame, output_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    figure = plt.figure(figsize=(10.8, 12.6), layout="constrained")
    grid = figure.add_gridspec(
        3,
        4,
        width_ratios=[1.0, 0.045, 1.0, 0.045],
        wspace=0.05,
        hspace=0.10,
    )
    map_axes: list[plt.Axes] = []
    specifications = [
        ("crime_count_all", "Crime events", "Reds"),
        ("bike_total_flow", "Bicycle endpoints", "Blues"),
    ]
    for row, city in enumerate(CITY_ORDER):
        subset = spatial.loc[spatial["city"].eq(city)].copy()
        for column, (variable, title, cmap) in enumerate(specifications):
            axis = figure.add_subplot(grid[row, column * 2])
            color_axis = figure.add_subplot(grid[row, column * 2 + 1])
            values = np.log1p(subset[variable].astype(float))
            scatter = axis.scatter(
                subset["x_index"],
                subset["y_index"],
                c=values,
                cmap=cmap,
                norm=colors.Normalize(vmin=0, vmax=max(float(values.max()), 1.0)),
                marker="s",
                s=18,
                linewidths=0,
            )
            _fix_map_axis(axis, subset)
            axis.set_title(
                f"{CITY_LABELS[city]} - {title}",
                loc="left",
                fontweight="bold",
            )
            colorbar = figure.colorbar(scatter, cax=color_axis)
            colorbar.set_label("log(1 + count)", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
            map_axes.append(axis)
    figure.suptitle(
        "Three-year spatial distribution on the 1 km analysis grid",
        fontweight="bold",
    )
    geometry = _axis_geometry(figure, map_axes, "Figure 3")
    outputs = _save_both(figure, output_dir / "e02_spatial_patterns")
    return outputs, geometry


def _rectangles(frame: pd.DataFrame) -> list[Rectangle]:
    return [
        Rectangle((float(row.x_index) - 0.5, float(row.y_index) - 0.5), 1, 1)
        for row in frame.itertuples()
    ]


def _plot_figure_10(
    catalog: pd.DataFrame,
    eligibility: pd.DataFrame,
    estimates: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], pd.DataFrame]:
    values = estimates["delta_l_exact_mse"].to_numpy(float)
    maximum = max(float(np.quantile(values, 0.98)), np.finfo(float).eps)
    normalization = colors.Normalize(vmin=0.0, vmax=maximum)
    figure = plt.figure(figsize=(11.4, 4.5), layout="constrained")
    grid = figure.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.06)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    color_axis = figure.add_subplot(grid[0, 3])
    eligible_ids = set(estimates["grid_id"].astype(str))
    covered_ids = set(eligibility["grid_id"].astype(str))
    last_collection = None
    for axis, city in zip(axes, CITY_ORDER, strict=True):
        city_catalog = catalog.loc[catalog["city"].eq(city)].copy()
        outside = city_catalog.loc[
            ~city_catalog["grid_id"].astype(str).isin(covered_ids)
        ]
        excluded = city_catalog.loc[
            city_catalog["grid_id"].astype(str).isin(covered_ids)
            & ~city_catalog["grid_id"].astype(str).isin(eligible_ids)
        ]
        city_values = estimates.loc[estimates["city"].eq(city)]
        if len(outside):
            axis.add_collection(
                PatchCollection(
                    _rectangles(outside),
                    facecolor="#F2F2F2",
                    edgecolor="#FFFFFF",
                    linewidth=0.15,
                )
            )
        if len(excluded):
            axis.add_collection(
                PatchCollection(
                    _rectangles(excluded),
                    facecolor="#C9C9C9",
                    edgecolor="#FFFFFF",
                    linewidth=0.15,
                )
            )
        collection = PatchCollection(
            _rectangles(city_values),
            cmap="viridis",
            norm=normalization,
            edgecolor="#FFFFFF",
            linewidth=0.15,
        )
        collection.set_array(city_values["delta_l_exact_mse"].to_numpy(float))
        axis.add_collection(collection)
        last_collection = collection
        _fix_map_axis(axis, city_catalog)
        axis.set_title(
            f"{CITY_LABELS[city]}\n$n={len(city_values)}$ eligible grids",
            fontweight="bold",
        )
    figure.suptitle(
        "Local reduction in the exact discrete prediction bound",
        fontweight="bold",
    )
    colorbar = figure.colorbar(last_collection, cax=color_axis)
    colorbar.set_label("Exact DELB reduction (MSE units)")
    figure.legend(
        handles=[
            Patch(facecolor="#F2F2F2", label="Outside bicycle footprint"),
            Patch(facecolor="#C9C9C9", label="Covered but ineligible"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    geometry = _axis_geometry(figure, axes, "Figure 10")
    outputs = _save_both(
        figure, output_dir / "e05_local_delb_reduction_maps"
    )
    return outputs, geometry


def _validate_inputs(
    spatial: pd.DataFrame,
    eligibility: pd.DataFrame,
    estimates: pd.DataFrame,
) -> None:
    required_spatial = {
        "city",
        "grid_id",
        "x_index",
        "y_index",
        "crime_count_all",
        "bike_total_flow",
    }
    required_eligibility = {"city", "grid_id", "x_index", "y_index", "eligible"}
    required_estimates = {
        "city",
        "grid_id",
        "x_index",
        "y_index",
        "delta_l_exact_mse",
    }
    for name, frame, required in [
        ("spatial", spatial, required_spatial),
        ("eligibility", eligibility, required_eligibility),
        ("estimates", estimates, required_estimates),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} input is missing columns: {sorted(missing)}")
    if len(spatial) != 1133 or len(eligibility) != 413 or len(estimates) != 339:
        raise AssertionError(
            "Unexpected accepted input row counts: "
            f"spatial={len(spatial)}, eligibility={len(eligibility)}, "
            f"estimates={len(estimates)}"
        )


def run(project_root: Path, e02_run: Path, e05_run: Path) -> Path:
    inputs = {
        "e02_spatial_grid_totals": e02_run / "tables" / "spatial_grid_totals.csv",
        "e05_grid_eligibility": e05_run / "tables" / "grid_eligibility.csv",
        "e05_local_information_estimates": e05_run
        / "tables"
        / "local_information_estimates.csv",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    spatial = pd.read_csv(inputs["e02_spatial_grid_totals"])
    eligibility = pd.read_csv(inputs["e05_grid_eligibility"])
    estimates = pd.read_csv(inputs["e05_local_information_estimates"])
    _validate_inputs(spatial, eligibility, estimates)
    _publication_style()

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_id = f"{started.strftime('%Y%m%d_%H%M%S')}_manuscript_figure_layout"
    run_dir = project_root / "empirical" / "runs" / run_id
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)

    figure_3, geometry_3 = _plot_figure_3(spatial, figure_dir)
    catalog = spatial[["city", "grid_id", "x_index", "y_index"]].copy()
    figure_10, geometry_10 = _plot_figure_10(
        catalog, eligibility, estimates, figure_dir
    )
    geometry = pd.concat([geometry_3, geometry_10], ignore_index=True)
    geometry.to_csv(run_dir / "panel_geometry.csv", index=False)

    command = " ".join(sys.argv)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "status": "complete",
        "purpose": "Figure 3 and Figure 10 equal-panel layout revision",
        "layout_tolerance": LAYOUT_TOLERANCE,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "input_rows": {
            "spatial_grid_totals": len(spatial),
            "grid_eligibility": len(eligibility),
            "local_information_estimates": len(estimates),
        },
        "outputs": [
            {"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)}
            for path in figure_3 + figure_10
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "README.md").write_text(
        "# Manuscript figure-layout revision\n\n"
        "This lightweight Python run redraws Figure 3 and Figure 10 from "
        "accepted E02/E05 tables. It changes panel geometry only and does "
        "not rerun an experiment or modify any numeric result.\n",
        encoding="utf-8",
    )

    manuscript_dir = project_root / "figures" / "e11"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    for path in figure_3 + figure_10:
        shutil.copy2(path, manuscript_dir / path.name)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw manuscript Figures 3 and 10 with equal panel sizes."
    )
    default_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--e02-run", type=Path, required=True)
    parser.add_argument("--e05-run", type=Path, required=True)
    args = parser.parse_args()
    run_dir = run(
        args.project_root.resolve(),
        args.e02_run.resolve(),
        args.e05_run.resolve(),
    )
    print(run_dir)
