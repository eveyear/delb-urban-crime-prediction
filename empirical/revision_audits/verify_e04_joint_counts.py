"""Recalculate primary E04 entropy, CMI, and DELB values from archived counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from export_e04_joint_counts import REPO, _verify_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.counts_dir / "validation_summary.csv")
    config = yaml.safe_load((REPO / "config/e04.yaml").read_text(encoding="utf-8"))
    if len(summary) != 24 or summary["spec_id"].nunique() != 24:
        raise AssertionError("Expected 24 distinct E04 specifications")
    max_difference = 0.0
    for _, expected in summary.iterrows():
        table = pd.read_csv(args.counts_dir / f"{expected['spec_id']}.csv.gz")
        actual = _verify_counts(table, config, expected)
        max_difference = max(max_difference, actual["max_absolute_difference"])
    print(f"Validated 24 primary E04 specifications from archived counts; maximum difference {max_difference:.3g}")


if __name__ == "__main__":
    main()
