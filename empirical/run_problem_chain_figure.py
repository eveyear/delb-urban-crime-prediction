#!/usr/bin/env python3
"""Generate the manuscript problem-chain figure.

This script has no data inputs.  It produces the same conceptual figure in
PDF, SVG, and PNG so that every formal manuscript visual remains generated
by Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


STAGES = [
    (
        "RQ1  Population lower bound",
        "Integer target + causal information set\n"
        r"$\longrightarrow$ DELB and attainability",
        "#DCEAF7",
        "#2C6EAA",
    ),
    (
        "RQ2  Finite-sample evidence",
        "CMI estimate + sparse-state bias\n"
        r"$\longrightarrow$ design-conditioned null",
        "#FCE8CE",
        "#D97917",
    ),
    (
        "RQ3  Urban application",
        "Baseline versus bicycle information\n"
        r"$\longrightarrow$ calibrated bound change",
        "#E2F0D9",
        "#57923F",
    ),
    (
        "Prediction consequence",
        "Estimated floor versus held-out error\n"
        r"$\longrightarrow$ Predictability Gap",
        "#E8E1F2",
        "#73559A",
    ),
]


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(11.2, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centers = [0.14, 0.38, 0.62, 0.86]
    width, height, y = 0.205, 0.47, 0.47
    for idx, (title, body, fill, edge) in enumerate(STAGES):
        x = centers[idx] - width / 2
        box = FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(
            centers[idx],
            y + 0.105,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#202020",
        )
        ax.text(
            centers[idx],
            y - 0.055,
            body,
            ha="center",
            va="center",
            fontsize=9.2,
            linespacing=1.45,
            color="#303030",
        )
        if idx < len(STAGES) - 1:
            arrow = FancyArrowPatch(
                (centers[idx] + width / 2 + 0.008, y),
                (centers[idx + 1] - width / 2 - 0.008, y),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.3,
                color="#555555",
            )
            ax.add_patch(arrow)

    ax.text(
        0.5,
        0.88,
        "Unified question-to-evidence chain",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#202020",
    )
    ax.text(
        0.5,
        0.10,
        "A lower estimated floor, a calibrated information gain, and improved "
        "realized prediction are distinct claims.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(pad=0.6)
    return fig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/problem_chain"),
        help="Directory for PDF, SVG, PNG, and manifest outputs.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "unified_problem_chain"
    fig = build_figure()
    outputs = {
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
        "png": stem.with_suffix(".png"),
    }
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(
        outputs["png"], dpi=args.dpi, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    manifest = {
        "generator": str(Path(__file__).resolve()),
        "data_inputs": [],
        "dpi": args.dpi,
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = args.output_dir / "unified_problem_chain_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
