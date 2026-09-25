"""Redraw manuscript Figure 6 from frozen E10 randomization outputs.

Only typography and plotting style change; all 8,991 replicate statistics and
the nine reported randomization p-values come directly from the E10 tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CITY_ORDER = ("DC", "NY", "VAN")
CITY_LABELS = {"DC": "Washington, DC", "NY": "New York City", "VAN": "Vancouver"}
DESIGNS = (
    ("temporal_block_permutation", "Temporal blocks"),
    ("spatial_series_permutation", "Spatial series"),
    ("circular_shift_surrogate", "Circular surrogate"),
)


def redraw(table_dir: Path, output_pdf: Path) -> None:
    replicates = pd.read_parquet(table_dir / "city_null_replicates.parquet")
    summary = pd.read_csv(table_dir / "city_null_summary.csv")
    assert len(replicates) == 8991
    assert len(summary) == 9

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(8.3, 6.45))
    for row, (design, label) in enumerate(DESIGNS):
        for column, city in enumerate(CITY_ORDER):
            ax = axes[row, column]
            values = replicates.loc[
                replicates["city"].eq(city) & replicates["null_design"].eq(design),
                "delta_h_raw_bits",
            ].to_numpy(float)
            record = summary.loc[
                summary["city"].eq(city) & summary["null_design"].eq(design)
            ].iloc[0]
            assert len(values) == 999
            ax.hist(values, bins=35, color="#C8754A", edgecolor="#82452F", linewidth=0.3)
            ax.axvline(float(record["observed_delta_h_bits"]), color="#7D183C", linewidth=2.0)
            if row == 0:
                ax.set_title(CITY_LABELS[city], pad=6)
            if column == 0:
                ax.set_ylabel(f"{label}\nFrequency", labelpad=3)
            ax.text(
                0.97,
                0.94,
                f"p={record['randomization_p_value']:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", length=2.5, pad=2)
            ax.locator_params(axis="x", nbins=3)

    fig.suptitle("Observed CMI versus randomization references", fontsize=13.5, y=0.985)
    fig.supxlabel("Conditional information (bits)", fontsize=12, y=0.025)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.91, bottom=0.12, wspace=0.28, hspace=0.38)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()
    redraw(args.table_dir, args.output_pdf)


if __name__ == "__main__":
    main()
