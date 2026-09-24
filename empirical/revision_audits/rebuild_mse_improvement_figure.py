"""Rebuild manuscript Figure 9 from the frozen held-out contrast table.

Only the layout and typography change; point estimates and normal confidence
intervals are read without alteration from the accepted E06 output.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "empirical/runs/20260718_193815_E06_predictive_models/tables/paired_mse_contrasts.csv"
OUTPUT = ROOT / "figures/e11/e06_mse_improvement.pdf"
CITIES = [("DC", "Washington, DC", "#4472C4"), ("NY", "New York City", "#ED7D31"), ("VAN", "Vancouver", "#70AD47")]
MODELS = ["state_mean", "poisson_glm", "negative_binomial_glm", "histogram_gradient_boosting_poisson"]


def main() -> None:
    data = pd.read_csv(SOURCE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5,
                         "axes.titlesize": 13, "axes.labelsize": 12.5,
                         "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.6), sharey=True, constrained_layout=True)
    y = np.arange(len(MODELS))
    for axis, (city, title, color) in zip(axes, CITIES):
        subset = data.loc[data.city.eq(city)].set_index("model_family").loc[MODELS]
        values = subset.delta_mse_integer.to_numpy(float)
        lower = values - subset.normal_ci_low.to_numpy(float)
        upper = subset.normal_ci_high.to_numpy(float) - values
        axis.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="o",
                      color=color, ecolor="#555555", capsize=3, markersize=6)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel(r"$\Delta$MSE")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, ["State mean", "Poisson GLM", "Neg. binomial", "Gradient boost"])
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
