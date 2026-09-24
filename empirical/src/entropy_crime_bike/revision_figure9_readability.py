"""Redraw manuscript Figure 9 from frozen E06 held-out contrasts.

The plotted point estimates and normal confidence limits are not recomputed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CITIES = ("DC", "NY", "VAN")
CITY_LABELS = {"DC": "Washington, DC", "NY": "New York City", "VAN": "Vancouver"}
CITY_COLORS = {"DC": "#365B89", "NY": "#D77A36", "VAN": "#6A9B54"}
MODELS = (
    ("state_mean", "State mean"),
    ("poisson_glm", "Poisson GLM"),
    ("negative_binomial_glm", "Neg. binomial"),
    ("histogram_gradient_boosting_poisson", "Gradient boost"),
)


def redraw(input_csv: Path, output_pdf: Path) -> None:
    data = pd.read_csv(input_csv)
    assert len(data) == 12
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 10.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.35), sharey=True)
    y = np.arange(len(MODELS))
    for ax, city in zip(axes, CITIES):
        subset = data.loc[data["city"].eq(city)].set_index("model_family").loc[
            [key for key, _ in MODELS]
        ]
        values = subset["delta_mse_integer"].to_numpy(float)
        low = subset["normal_ci_low"].to_numpy(float)
        high = subset["normal_ci_high"].to_numpy(float)
        ax.errorbar(
            values,
            y,
            xerr=np.vstack((values - low, high - values)),
            fmt="o",
            markersize=5,
            color=CITY_COLORS[city],
            ecolor="#505050",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
        )
        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.set_title(CITY_LABELS[city], pad=7)
        ax.set_xlabel(r"$\Delta$MSE", labelpad=5)
        ax.grid(axis="x", alpha=0.2, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", pad=3)
        padding = 0.08 * (high.max() - low.min())
        ax.set_xlim(low.min() - padding, high.max() + padding)

    axes[0].set_yticks(y, labels=[label for _, label in MODELS])
    fig.subplots_adjust(left=0.17, right=0.99, top=0.88, bottom=0.22, wspace=0.18)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()
    redraw(args.input_csv, args.output_pdf)


if __name__ == "__main__":
    main()
