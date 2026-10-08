"""Rebuild the main sparse-state CMI figure from frozen experiment summaries.

This redraws the figure only; it does not rerun simulations or change results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from entropy_crime_bike.e12_cmi_simulation import _plot_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tables = args.tables.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _plot_main(
        pd.read_csv(tables / "estimator_summary.csv"),
        pd.read_parquet(tables / "randomization_dataset_summary.parquet"),
        pd.read_csv(tables / "randomization_summary.csv"),
        output,
        300,
    )
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
