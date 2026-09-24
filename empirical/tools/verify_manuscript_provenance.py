#!/usr/bin/env python3
"""Verify manuscript figure/table provenance against active LaTeX and files."""

from __future__ import annotations

import csv
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
FIGURE_MANIFEST = ROOT / "figure_code" / "figure_code_manifest.csv"
TABLE_MANIFEST = ROOT / "empirical" / "docs" / "manuscript_table_provenance.csv"
ACTIVE_FIGURE_SOURCES = (
    ROOT / "sections" / "data_methods.tex",
    ROOT / "sections" / "results.tex",
    ROOT / "sections" / "supplement_results.tex",
)
AUX_FILES = (ROOT / "applsci-4501207.aux", ROOT / "supplement.aux")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def active_figures() -> set[str]:
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    return {
        match
        for source in ACTIVE_FIGURE_SOURCES
        for match in pattern.findall(source.read_text(encoding="utf-8"))
    }


def active_table_labels() -> set[str]:
    pattern = re.compile(r"\\newlabel\{(tab:[^}@]+)\}")
    labels: set[str] = set()
    for aux in AUX_FILES:
        if not aux.exists():
            raise FileNotFoundError(f"Compile manuscripts first; missing {aux}")
        labels.update(pattern.findall(aux.read_text(encoding="utf-8")))
    return labels


def verify_paths(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    sentinels = {"NONE", "AUTHOR_MAINTAINED"}
    for row in rows:
        identity = row.get("figure_label") or row.get("table_label") or "unknown"
        for field in fields:
            value = row[field]
            if value in sentinels:
                continue
            if not (ROOT / value).exists():
                errors.append(f"{identity}: missing {field}={value}")
    return errors


def main() -> int:
    figure_rows = read_csv(FIGURE_MANIFEST)
    table_rows = read_csv(TABLE_MANIFEST)
    manifest_figures = {row["latex_asset"] for row in figure_rows}
    manuscript_figures = active_figures()
    manifest_tables = {row["table_label"] for row in table_rows}
    manuscript_tables = active_table_labels()

    errors: list[str] = []
    for item in sorted(manuscript_figures - manifest_figures):
        errors.append(f"active figure missing from manifest: {item}")
    for item in sorted(manifest_figures - manuscript_figures):
        errors.append(f"manifest figure not used by active English manuscripts: {item}")
    for item in sorted(manuscript_tables - manifest_tables):
        errors.append(f"active table missing from manifest: {item}")
    for item in sorted(manifest_tables - manuscript_tables):
        errors.append(f"manifest table not used by active English manuscripts: {item}")

    errors.extend(
        verify_paths(
            figure_rows,
            ("latex_asset", "latex_source", "canonical_python", "entry_point", "config"),
        )
    )
    errors.extend(
        verify_paths(
            table_rows,
            ("latex_asset", "latex_source", "canonical_python", "entry_point", "config"),
        )
    )

    if errors:
        print("PROVENANCE CHECK FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(
        "PROVENANCE CHECK PASSED: "
        f"{len(manuscript_figures)} figures and {len(manuscript_tables)} tables "
        "are mapped to existing LaTeX assets, generators, entry points, and configs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
