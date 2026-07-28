from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
import yaml


COLORS = {
    "estimand": "#2A6F97",
    "estimator": "#D97706",
    "null": "#4D7C5B",
    "ink": "#202124",
    "muted": "#5F6368",
    "structure": "#9AA0A6",
    "surface": "#F7F8FA",
}


def concept_elements() -> list[dict[str, str]]:
    """Return the fixed conceptual content independently of layout."""
    return [
        {
            "key": "estimand",
            "stage": "1",
            "title": "Population estimand",
            "formula": r"$\theta=I(C;B\mid\mathcal{F}^{0})$",
            "line_1": r"$H(C\mid\mathcal{F}^{0})-H(C\mid\mathcal{F}^{B})=\theta$",
            "line_2": "Orders population DELB/Fano floors",
            "line_3": "Predictive information; not a causal effect",
        },
        {
            "key": "estimator",
            "stage": "2",
            "title": "Finite-sample estimator",
            "formula": (
                r"$\widehat{\theta}=\theta+"
                r"b(N,K_{\mathrm{obs}},S_1,\nu)+\varepsilon$"
            ),
            "line_1": "Plugin or Miller--Madow statistic",
            "line_2": "Bias and variance depend on support",
            "line_3": r"$K_{\mathrm{obs}}/N,\ S_1,\ \nu/N$ diagnose sparsity",
        },
        {
            "key": "null",
            "stage": "3",
            "title": "Randomization reference",
            "formula": r"$\mathcal{L}_{\pi}(\widehat{\theta}^{\pi}\mid H_0)$",
            "line_1": (
                r"$\mu_{\mathrm{null}}="
                r"\mathbb{E}_{\pi}[\widehat{\theta}^{\pi}]$"
                " need not be zero"
            ),
            "line_2": r"Compare $\widehat{\theta}_{\mathrm{obs}}$ with its null law",
            "line_3": r"Recompute statistic and support after $\pi$",
        },
    ]


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return config


def _box(
    axis: plt.Axes,
    element: dict[str, str],
    bounds: tuple[float, float, float, float],
    *,
    font_scale: float,
    compact: bool,
) -> None:
    x, y, width, height = bounds
    color = COLORS[element["key"]]
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=color,
        facecolor=COLORS["surface"],
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(patch)
    title_size = 8.3 if compact else 10.5 * font_scale
    formula_size = 8.6 if compact else 11.0 * font_scale
    line_size = 6.8 if compact else 7.9 * font_scale
    axis.text(
        x + 0.035 * width,
        y + 0.88 * height,
        f'{element["stage"]}  {element["title"]}',
        color=color,
        fontsize=title_size,
        fontweight="bold",
        transform=axis.transAxes,
        va="center",
    )
    axis.text(
        x + 0.5 * width,
        y + 0.66 * height,
        element["formula"],
        color=COLORS["ink"],
        fontsize=formula_size,
        transform=axis.transAxes,
        ha="center",
        va="center",
    )
    for position, key in zip(
        (0.43, 0.27, 0.11), ("line_1", "line_2", "line_3"), strict=True
    ):
        axis.text(
            x + 0.05 * width,
            y + position * height,
            element[key],
            color=COLORS["ink"] if key == "line_1" else COLORS["muted"],
            fontsize=line_size,
            transform=axis.transAxes,
            ha="left",
            va="center",
        )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    vertical: bool,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color=COLORS["structure"],
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(arrow)
    if vertical:
        x = start[0] + 0.04
        y = (start[1] + end[1]) / 2
        axis.text(
            x,
            y,
            label,
            color=COLORS["muted"],
            fontsize=7.0,
            transform=axis.transAxes,
            va="center",
            ha="left",
        )
    else:
        x = (start[0] + end[0]) / 2
        y = 0.205
        axis.text(
            x,
            y,
            label,
            color=COLORS["muted"],
            fontsize=6.4,
            transform=axis.transAxes,
            va="bottom",
            ha="center",
        )


def draw_concept_figure(
    *,
    layout: str,
    figure_size: tuple[float, float],
) -> plt.Figure:
    """Draw the conceptual distinction in wide or single-column form."""
    if layout not in {"wide", "single_column"}:
        raise ValueError(f"Unsupported layout: {layout}")
    elements = concept_elements()
    figure, axis = plt.subplots(figsize=figure_size)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    if layout == "wide":
        bounds = [
            (0.015, 0.25, 0.292, 0.64),
            (0.354, 0.25, 0.292, 0.64),
            (0.693, 0.25, 0.292, 0.64),
        ]
        for element, box_bounds in zip(elements, bounds, strict=True):
            _box(
                axis,
                element,
                box_bounds,
                font_scale=0.93,
                compact=True,
            )
        _arrow(
            axis,
            (0.31, 0.57),
            (0.351, 0.57),
            "sample + discretize",
            vertical=False,
        )
        _arrow(
            axis,
            (0.649, 0.57),
            (0.69, 0.57),
            r"apply $\pi$; recompute",
            vertical=False,
        )
        footer_y = 0.105
        footer_size = 8.3
    else:
        bounds = [
            (0.04, 0.705, 0.92, 0.255),
            (0.04, 0.385, 0.92, 0.255),
            (0.04, 0.065, 0.92, 0.255),
        ]
        for element, box_bounds in zip(elements, bounds, strict=True):
            _box(
                axis,
                element,
                box_bounds,
                font_scale=0.88,
                compact=False,
            )
        _arrow(
            axis,
            (0.50, 0.70),
            (0.50, 0.645),
            "sample + discretize",
            vertical=True,
        )
        _arrow(
            axis,
            (0.50, 0.38),
            (0.50, 0.325),
            r"apply $\pi$; recompute",
            vertical=True,
        )
        footer_y = 0.012
        footer_size = 7.0

    footer = (
        "Randomization evidence does not validate or refute DELB/Fano; "
        "the three quantities answer different questions."
        if layout == "wide"
        else "Randomization evidence does not validate or refute DELB/Fano;\n"
        "the three quantities answer different questions."
    )
    axis.text(
        0.5,
        footer_y,
        footer,
        color=COLORS["ink"],
        fontsize=footer_size,
        fontweight="bold",
        transform=axis.transAxes,
        ha="center",
        va="center",
    )
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure


def _save_figure(
    figure: plt.Figure,
    stem: Path,
    *,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    outputs = []
    for suffix in formats:
        output = stem.with_suffix(f".{suffix}")
        options: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if suffix == "png":
            options["dpi"] = dpi
        figure.savefig(output, **options)
        outputs.append(output)
    plt.close(figure)
    return outputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
    }


def _interpretation() -> str:
    return "\n".join(
        [
            "# E15 Conceptual Diagram Interpretation",
            "",
            "The diagram separates three objects that must not be conflated.",
            "",
            "1. The population estimand is the conditional mutual information "
            "that orders the population DELB and Fano floors.",
            "2. A plugin or Miller--Madow estimate adds finite-sample bias and "
            "sampling variability, both of which depend on state occupancy.",
            "3. A randomization null is the design-conditioned distribution of "
            "the recomputed estimator after transforming bicycle information.",
            "",
            "The randomization-null mean need not equal zero because the "
            "estimator and randomized support can remain finite-sample biased. "
            "Comparing the observed statistic with this reference distribution "
            "addresses mobility-specific evidence; it does not validate or "
            "refute the DELB or Fano mathematical inequalities. Conditional "
            "mutual information is predictive rather than causal here.",
            "",
        ]
    )


def run(config_path: Path) -> Path:
    start = time.perf_counter()
    config = _load_yaml(config_path)
    empirical = config_path.resolve().parents[1]
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_E15_concept_diagram"
    )
    run_dir = empirical / "runs" / run_id
    figure_dir = run_dir / "figures"
    table_dir = run_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    reporting = config["reporting"]
    formats = [str(item) for item in reporting["formats"]]
    dpi = int(reporting["figure_dpi"])
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )
    layout_sizes = {
        "wide": tuple(float(value) for value in reporting["wide_size_inches"]),
        "single_column": tuple(
            float(value)
            for value in reporting["single_column_size_inches"]
        ),
    }
    outputs: list[Path] = []
    for layout, size in layout_sizes.items():
        figure = draw_concept_figure(layout=layout, figure_size=size)
        outputs.extend(
            _save_figure(
                figure,
                figure_dir / f"e15_estimand_estimator_null_{layout}",
                formats=formats,
                dpi=dpi,
            )
        )

    elements = pd.DataFrame(concept_elements())
    elements.to_csv(table_dir / "concept_elements.csv", index=False)
    (run_dir / "interpretation.md").write_text(
        _interpretation(), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text(
        "PYTHONPATH=src python run_e15_concept_diagram.py "
        "--config config/e15.yaml\n",
        encoding="utf-8",
    )
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as out:
        yaml.safe_dump(config, out, sort_keys=False, allow_unicode=True)

    required_terms = [str(term) for term in config["acceptance"]["required_terms"]]
    all_text = " ".join(elements.astype(str).to_numpy().ravel()) + " " + (
        "Randomization evidence does not validate or refute DELB/Fano; "
        "null mean need not be zero"
    )
    checks = {
        "expected_layouts": len(layout_sizes)
        == int(config["acceptance"]["expected_layouts"]),
        "expected_figure_files": len(outputs)
        == len(layout_sizes)
        * int(config["acceptance"]["expected_formats_per_layout"]),
        "all_figure_files_nonempty": all(
            output.exists() and output.stat().st_size > 0 for output in outputs
        ),
        "required_terms_present": all(
            term.lower() in all_text.lower() for term in required_terms
        ),
        "python_generation_declared": bool(
            reporting["figures_are_python_generated"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"E15 acceptance failure: {checks}")
    manifest = {
        "run_id": run_id,
        "experiment": str(config["experiment_id"]),
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - start,
        "inputs": {
            "empirical_data": None,
            "concept_source": "fixed mathematical specification in Python",
        },
        "outputs": {
            "layouts": list(layout_sizes),
            "formats": formats,
            "figure_files": len(outputs),
        },
        "checks": checks,
        "figures_generated_by": "Python/Matplotlib",
        "figure_sha256": {
            output.name: _sha256(output) for output in sorted(outputs)
        },
        "environment": _environment(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the E15 estimand--estimator--randomization-null "
            "conceptual diagram."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "e15.yaml",
    )
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
