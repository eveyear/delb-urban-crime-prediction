#!/usr/bin/env python3
"""Order the manual English/Chinese bibliographies by first citation."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGLISH = ROOT / "template.tex"
CHINESE_REFS = ROOT / "sections" / "references_Chinese.tex"

SOURCE_ORDER = [
    ENGLISH,
    ROOT / "sections" / "theoretical_framework.tex",
    ROOT / "sections" / "table_delb_theorem_package.tex",
    ROOT / "sections" / "theoretical_extensions.tex",
    ROOT / "sections" / "data_methods.tex",
    ROOT / "sections" / "application_specialization.tex",
    ROOT / "sections" / "results.tex",
    ROOT / "sections" / "discussion_conclusion.tex",
    ROOT / "sections" / "theoretical_proofs.tex",
    ROOT / "sections" / "empirical_appendix.tex",
]

BIBITEM = re.compile(r"(?=\\bibitem\{[^}]+\})")
CITATION = re.compile(r"\\cite[tp]?\s*\{([^}]+)\}", re.DOTALL)


def active_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path == ENGLISH:
        text = text.split("\\iffalse", 1)[0]
    return re.sub(r"(?m)(?<!\\)%.*$", "", text)


def citation_order() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in SOURCE_ORDER:
        for match in CITATION.finditer(active_text(path)):
            for key in match.group(1).split(","):
                key = key.strip()
                if key and key not in seen:
                    ordered.append(key)
                    seen.add(key)
    return ordered


def parse_entries(text: str) -> tuple[str, dict[str, str], list[str], str]:
    begin = "\\begin{thebibliography}{999}"
    end = "\\end{thebibliography}"
    before, remainder = text.split(begin, 1)
    body, after = remainder.split(end, 1)
    entries: dict[str, str] = {}
    original: list[str] = []
    for block in BIBITEM.split(body.strip()):
        if not block.strip():
            continue
        match = re.match(r"\\bibitem\{([^}]+)\}", block)
        if match is None:
            raise ValueError(f"Unparsed bibliography block: {block[:80]!r}")
        key = match.group(1)
        entries[key] = block.strip()
        original.append(key)
    return before + begin, entries, original, end + after


def ordered_bibliography(source: Path, order: list[str]) -> str:
    text = source.read_text(encoding="utf-8")
    prefix, entries, original, suffix = parse_entries(text)
    missing = [key for key in order if key not in entries]
    if missing:
        raise KeyError(f"Citations without bibliography entries: {missing}")
    final_order = order + [key for key in original if key not in set(order)]
    if len(final_order) != len(entries):
        raise AssertionError("Duplicate or lost bibliography entries")
    body = "\n\n".join(entries[key] for key in final_order)
    return f"{prefix}\n\n{body}\n\n{suffix}"


def main() -> None:
    order = citation_order()
    english_text = ordered_bibliography(ENGLISH, order)
    ENGLISH.write_text(english_text, encoding="utf-8")

    _, english_entries, english_original, _ = parse_entries(english_text)
    chinese_text = CHINESE_REFS.read_text(encoding="utf-8")
    _, chinese_entries, _, _ = parse_entries(chinese_text)
    if set(english_entries) != set(chinese_entries):
        raise AssertionError("English and Chinese bibliography keys differ")
    chinese_prefix, _, _, chinese_suffix = parse_entries(chinese_text)
    chinese_body = "\n\n".join(chinese_entries[key] for key in english_original)
    CHINESE_REFS.write_text(
        f"{chinese_prefix}\n\n{chinese_body}\n\n{chinese_suffix}",
        encoding="utf-8",
    )

    print(f"Ordered {len(english_entries)} entries; {len(order)} are cited.")


if __name__ == "__main__":
    main()
