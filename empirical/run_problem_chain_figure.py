#!/usr/bin/env python3
"""Rebuild final Figure 1 from the canonical accepted-paper implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from entropy_crime_bike.editor_framework_figures import evidence_map  # noqa: E402


def build_figure():
    return evidence_map()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=HERE.parent / "figures/problem_chain")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "unified_problem_chain"
    fig = build_figure()
    outputs = {extension: stem.with_suffix("." + extension)
               for extension in ["pdf", "svg", "png"]}
    for path in outputs.values():
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {
        "generator": "empirical/src/entropy_crime_bike/editor_framework_figures.py",
        "manuscript_version": "production-proof-20261008",
        "data_inputs": [],
        "dpi": args.dpi,
        "outputs": {extension: {"path": str(path),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for extension, path in outputs.items()},
    }
    (args.output_dir / "unified_problem_chain_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
