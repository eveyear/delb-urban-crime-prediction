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
        "1  DELB",
        "Integer-MSE floor",
        "Entropy and integer-lattice\nenvelope",
        "Assessment of floor change",
        "#DCEAF7",
        "#2C6EAA",
    ),
    (
        "2  CMI +\nrandomization",
        "Added information",
        "CMI and matched\nreferences",
        "Evidence beyond\nfinite-sample bias",
        "#FCE8CE",
        "#D97917",
    ),
    (
        "3  Known-truth\nexperiments",
        "Bias and power",
        "Injected CMI and\nreplicates",
        "Detectable effects",
        "#E2F0D9",
        "#57923F",
    ),
    (
        "4  Held-out MSE\n+ Gap",
        "Realized model error",
        "Test MSE and paired\nDELB",
        "Realized opportunity",
        "#E8E1F2",
        "#73559A",
    ),
]


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 7.083))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centers = [(0.255, 0.70), (0.745, 0.70), (0.255, 0.31), (0.745, 0.31)]
    width, height = 0.455, 0.335
    card_text = []
    card_patches = []
    for idx, (title, meaning, calculation, conclusion, fill, edge) in enumerate(STAGES):
        cx, cy = centers[idx]
        x = cx - width / 2
        box = FancyBboxPatch(
            (x, cy - height / 2),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        card_patches.append(box)
        texts = []
        texts.append(ax.text(
            cx,
            cy + 0.120,
            title,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#202020",
        ))
        texts.append(ax.text(
            cx,
            cy + 0.052,
            "Meaning: " + meaning,
            ha="center",
            va="center",
            fontsize=8.0,
            color="#303030",
        ))
        texts.append(ax.text(
            cx, cy - 0.026, "From: " + calculation,
            ha="center", va="center", fontsize=8.0,
            linespacing=1.12, color="#303030",
        ))
        texts.append(ax.text(
            cx, cy - 0.112, "Supports:\n" + conclusion,
            ha="center", va="center", fontsize=8.0,
            linespacing=1.12, color="#303030",
        ))
        card_text.append(texts)

    ax.text(
        0.5,
        0.965,
        "Evidence layers in the analysis",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#202020",
    )
    ax.text(
        0.5,
        0.06,
        "The four layers answer related but different questions.",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#444444",
    )
    fig.tight_layout(pad=0.25)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for box, texts in zip(card_patches, card_text):
        left, bottom, right, top = box.get_window_extent(renderer).extents
        bounds = (left + 4, bottom + 3, right - 4, top - 3)
        for item in texts:
            x0, y0, x1, y1 = item.get_window_extent(renderer).extents
            if not (bounds[0] <= x0 and bounds[1] <= y0 and x1 <= bounds[2] and y1 <= bounds[3]):
                raise ValueError(f"Text exceeds card boundary: {item.get_text()}")
        for first, second in zip(texts, texts[1:]):
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                raise ValueError(f"Text overlaps within card: {first.get_text()}")
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
    parser.add_argument("--dpi", type=int, default=600)
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
        "generator": "empirical/run_problem_chain_figure.py",
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
