"""Rebuild the final data-free Figure 1, Figure 2, and Figure 3 assets.

Run from any working directory. The canonical implementations are in
empirical/src/entropy_crime_bike. Figures 1 and 2 share one implementation;
the Figure 1-only command is a wrapper around that same implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import matplotlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from entropy_crime_bike.editor_framework_figures import save_framework_figures  # noqa: E402
from entropy_crime_bike.e15_concept_diagram import draw_concept_figure, _save_figure  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=HERE.parent / "figures",
        help="Root containing e11/ and s7/ figure folders.",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    e11 = root / "e11"
    s7 = root / "s7"
    e11.mkdir(parents=True, exist_ok=True)
    s7.mkdir(parents=True, exist_ok=True)

    framework_files = save_framework_figures(root, dpi=400)

    stem = s7 / "e15_estimand_estimator_null_wide"
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.unicode_minus": True,
    })
    concept = draw_concept_figure(layout="wide", figure_size=(6.8, 6.015))
    concept_files = _save_figure(concept, stem, formats=["pdf", "svg", "png"], dpi=600)
    outputs = [*framework_files, *concept_files]
    print(json.dumps({str(path.relative_to(root)): sha256(path) for path in outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
