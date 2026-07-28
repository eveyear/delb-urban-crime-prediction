from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from entropy_crime_bike.e15_concept_diagram import (
    concept_elements,
    draw_concept_figure,
)


def test_concept_has_three_distinct_layers() -> None:
    elements = concept_elements()
    assert [element["key"] for element in elements] == [
        "estimand",
        "estimator",
        "null",
    ]
    text = " ".join(
        value for element in elements for value in element.values()
    )
    assert "I(C;B" in text
    assert "K_{\\mathrm{obs}}" in text
    assert "need not be zero" in text
    assert "not a causal effect" in text


def test_both_layouts_render(tmp_path: Path) -> None:
    for layout, size in [
        ("wide", (7.2, 4.0)),
        ("single_column", (3.5, 7.3)),
    ]:
        figure = draw_concept_figure(layout=layout, figure_size=size)
        output = tmp_path / f"{layout}.png"
        figure.savefig(output, dpi=100)
        plt.close(figure)
        assert output.exists()
        assert output.stat().st_size > 10_000


def test_unsupported_layout_is_rejected() -> None:
    try:
        draw_concept_figure(layout="square", figure_size=(4.0, 4.0))
    except ValueError as error:
        assert "Unsupported layout" in str(error)
    else:
        raise AssertionError("Unsupported layout was not rejected.")
