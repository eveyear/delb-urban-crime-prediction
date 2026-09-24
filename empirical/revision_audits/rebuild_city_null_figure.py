"""Rebuild the city-level randomization figure from frozen E10 tabular results.

Data, axes, bins, colors, and p-values match the original generator. Figure
dimensions and typography are enlarged for legibility at manuscript size.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "empirical/runs/20260719_190015_E10_placebo_null/tables"
OUTPUT = ROOT / "figures/e11/e10_city_null_distributions"
CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {"DC": "Washington, DC", "NY": "New York City", "VAN": "Vancouver"}
OBSERVED_COLOR = "#7A1731"
HISTOGRAM_COLOR = "#C9824B"
NULL_ORDER = ["temporal_block_permutation", "spatial_series_permutation", "circular_shift_surrogate"]
NULL_LABELS = {
    "temporal_block_permutation": "Temporal blocks",
    "spatial_series_permutation": "Spatial series",
    "circular_shift_surrogate": "Circular surrogate",
}

def main() -> None:
    nulls = pd.read_parquet(SOURCE / "city_null_replicates.parquet")
    summary = pd.read_csv(SOURCE / "city_null_summary.csv")
    assert len(summary) == 9
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13,
        "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 7.2), sharex=False, constrained_layout=True)
    for row, design in enumerate(NULL_ORDER):
        for col, city in enumerate(CITY_ORDER):
            axis = axes[row, col]
            values = nulls.loc[nulls["city"].eq(city) & nulls["null_design"].eq(design), "delta_h_raw_bits"]
            record = summary.loc[summary["city"].eq(city) & summary["null_design"].eq(design)].iloc[0]
            assert len(values) == 999, (city, design, len(values))
            axis.hist(values, bins=35, color=HISTOGRAM_COLOR,
                      edgecolor="#6E351F", linewidth=0.25)
            axis.axvline(record["observed_delta_h_bits"], color=OBSERVED_COLOR,
                         lw=2.7, label="Observed CMI")
            if row == 0:
                axis.set_title(CITY_LABELS[city])
            if col == 0:
                axis.set_ylabel(NULL_LABELS[design] + "\nFrequency")
            if row == 2:
                axis.set_xlabel("Conditional information (bits)")
            axis.text(0.98, 0.92, f"p={record['randomization_p_value']:.3f}",
                      transform=axis.transAxes, ha="right", va="top", fontsize=11.5)
    fig.suptitle("Observed conditional information versus randomization reference distributions")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))

if __name__ == "__main__":
    main()
