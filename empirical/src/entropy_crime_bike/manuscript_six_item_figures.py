from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .e06_predictive_models import CITY_COLORS, CITY_LABELS, CITY_ORDER
from .figure_typography import normalize_chart_typography


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(figure: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    normalize_chart_typography(figure)
    outputs = []
    for suffix in [".pdf", ".svg"]:
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    png = stem.with_suffix(".png")
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    outputs.append(png)
    return outputs


def _combined_city_figure(
    estimates: pd.DataFrame, figure_dir: Path, dpi: int
) -> tuple[list[Path], pd.DataFrame]:
    subset = estimates.loc[
        estimates["domain"].eq("bike_covered_training")
        & estimates["period"].eq("pooled_2020_2022")
    ].set_index("city").loc[CITY_ORDER]
    if len(subset) != 3:
        raise AssertionError("Expected one pooled primary row per city.")

    figure, axes = plt.subplots(
        1, 2, figsize=(12.2, 4.25), constrained_layout=True
    )
    x = np.arange(len(CITY_ORDER), dtype=float)

    cmi = subset["delta_h_miller_madow_raw_bits"].to_numpy(float)
    cmi_lower = cmi - subset[
        "delta_h_miller_madow_raw_bits_ci_low"
    ].to_numpy(float)
    cmi_upper = subset[
        "delta_h_miller_madow_raw_bits_ci_high"
    ].to_numpy(float) - cmi
    axes[0].bar(
        x,
        cmi,
        color=[CITY_COLORS[city] for city in CITY_ORDER],
        width=0.62,
    )
    axes[0].errorbar(
        x,
        cmi,
        yerr=np.vstack([cmi_lower, cmi_upper]),
        fmt="none",
        ecolor="#262626",
        capsize=4,
        linewidth=1.0,
    )
    axes[0].axhline(0, color="#7F7F7F", linewidth=0.8)
    axes[0].set_ylabel("Conditional information gain (bits)")
    axes[0].set_title("(a) Estimated conditional information", loc="left")

    width = 0.34
    for offset, metric, label, color in [
        (-width / 2, "l0_exact_mse", "Baseline DELB", "#A5A5A5"),
        (width / 2, "lb_exact_mse", "Bicycle-aware DELB", "#1F4E78"),
    ]:
        values = subset[metric].to_numpy(float)
        lower = values - subset[f"{metric}_ci_low"].to_numpy(float)
        upper = subset[f"{metric}_ci_high"].to_numpy(float) - values
        axes[1].bar(
            x + offset, values, width, label=label, color=color
        )
        axes[1].errorbar(
            x + offset,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#262626",
            capsize=3,
            linewidth=1.0,
        )
    for index, city in enumerate(CITY_ORDER):
        axes[1].text(
            index,
            max(
                subset.loc[city, "l0_exact_mse_ci_high"],
                subset.loc[city, "lb_exact_mse_ci_high"],
            )
            * 1.04,
            rf"$\Delta L={subset.loc[city, 'delta_l_exact_mse']:.4f}$",
            ha="center",
            fontsize=8.0,
        )
    axes[1].set_ylim(
        top=1.23 * max(
            float(subset["l0_exact_mse_ci_high"].max()),
            float(subset["lb_exact_mse_ci_high"].max()),
        )
    )
    axes[1].set_ylabel("MSE lower bound")
    axes[1].set_title("(b) DELB under matched information sets", loc="left")
    axes[1].legend(frameon=False, fontsize=8.5)

    for axis in axes:
        axis.set_xticks(x, [CITY_LABELS[city] for city in CITY_ORDER])
        axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
        axis.tick_params(axis="both", labelsize=9)
    outputs = _save(
        figure, figure_dir / "e17_city_cmi_delb_combined", dpi
    )
    figure.canvas.draw()
    geometry = []
    for index, axis in enumerate(axes):
        bounds = axis.get_position().bounds
        geometry.append(
            {
                "panel": index,
                "x0": bounds[0],
                "y0": bounds[1],
                "width": bounds[2],
                "height": bounds[3],
            }
        )
    plt.close(figure)
    geometry_frame = pd.DataFrame(geometry)
    if (
        geometry_frame["width"].max() / geometry_frame["width"].min() - 1
        > 0.005
        or geometry_frame["height"].max()
        / geometry_frame["height"].min()
        - 1
        > 0.005
    ):
        raise AssertionError("Combined city panels differ in physical size.")
    return outputs, geometry_frame


def run(project_root: Path, e04_run: Path, e17_run: Path) -> Path:
    inputs = {
        "e04_estimates": e04_run / "tables" / "city_information_estimates.csv",
        "e04_domain_figure_pdf": e04_run
        / "figures"
        / "e04_domain_sensitivity.pdf",
        "e04_domain_figure_png": e04_run
        / "figures"
        / "e04_domain_sensitivity.png",
        "e17_estimator_figure_pdf": e17_run
        / "figures"
        / "e17_estimator_null_sensitivity.pdf",
        "e17_estimator_figure_png": e17_run
        / "figures"
        / "e17_estimator_null_sensitivity.png",
        "e17_estimator_figure_svg": e17_run
        / "figures"
        / "e17_estimator_null_sensitivity.svg",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    estimates = pd.read_csv(inputs["e04_estimates"])

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_id = started.strftime("%Y%m%d_%H%M%S_manuscript_six_item_figures")
    run_dir = project_root / "empirical" / "runs" / run_id
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)

    combined, geometry = _combined_city_figure(estimates, figure_dir, 300)
    geometry.to_csv(run_dir / "panel_geometry.csv", index=False)
    copied = []
    for key in [
        "e04_domain_figure_pdf",
        "e04_domain_figure_png",
        "e17_estimator_figure_pdf",
        "e17_estimator_figure_png",
        "e17_estimator_figure_svg",
    ]:
        source = inputs[key]
        destination = figure_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)

    manuscript_dir = project_root / "figures" / "e17"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    for path in [*combined, *copied]:
        shutil.copy2(path, manuscript_dir / path.name)

    manifest = {
        "run_id": run_id,
        "status": "complete",
        "purpose": "Six-item manuscript revision figures",
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "outputs": [
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": _sha256(path),
            }
            for path in [*combined, *copied]
        ],
        "panel_size_tolerance": 0.005,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "README.md").write_text(
        "# Six-item manuscript figure revision\n\n"
        "The combined city figure is redrawn from the accepted E04 table. "
        "The fixed-domain and estimator-sensitivity figures are copied from "
        "their Python-generated formal runs. No numeric result is edited.\n",
        encoding="utf-8",
    )
    (project_root / "empirical" / "runs" / "latest_six_item_figure_run.txt").write_text(
        str(run_dir) + "\n", encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate figures for the six-item manuscript revision."
    )
    default_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--e04-run", type=Path, required=True)
    parser.add_argument("--e17-run", type=Path, required=True)
    args = parser.parse_args()
    print(
        run(
            args.project_root.resolve(),
            args.e04_run.resolve(),
            args.e17_run.resolve(),
        )
    )


if __name__ == "__main__":
    main()
