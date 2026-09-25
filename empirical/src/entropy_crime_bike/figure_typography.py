"""Publication-size typography for manuscript and supplementary charts.

Sizes below are intended points after a figure is placed at its manuscript
width.  Multiplying by the source-to-placement width ratio keeps a 12-inch
multi-panel export as legible as a 7-inch export on the journal page.
"""

from __future__ import annotations

from matplotlib.figure import Figure


PRINT_WIDTH_IN = 6.0
TITLE_PT = 9.0
AXIS_PT = 8.5
TICK_PT = 7.5
LEGEND_PT = 7.5
ANNOTATION_PT = 7.5


def normalize_chart_typography(figure: Figure) -> None:
    """Set text by semantic role without changing data, limits, or marks."""
    scale = float(figure.get_size_inches()[0]) / PRINT_WIDTH_IN
    for axis in figure.axes:
        for title in (axis.title, axis._left_title, axis._right_title):
            title.set_fontsize(TITLE_PT * scale)
        axis.xaxis.label.set_fontsize(AXIS_PT * scale)
        axis.yaxis.label.set_fontsize(AXIS_PT * scale)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontsize(TICK_PT * scale)
        for label in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
            label.set_fontsize(TICK_PT * scale)
        for label in axis.texts:
            label.set_fontsize(ANNOTATION_PT * scale)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(LEGEND_PT * scale)
            if legend.get_title().get_text():
                legend.get_title().set_fontsize(LEGEND_PT * scale)
    for label in figure.texts:
        if label is figure._suptitle:
            size = TITLE_PT
        elif label in (figure._supxlabel, figure._supylabel):
            size = AXIS_PT
        else:
            size = ANNOTATION_PT
        label.set_fontsize(size * scale)
    for legend in figure.legends:
        for label in legend.get_texts():
            label.set_fontsize(LEGEND_PT * scale)
        if legend.get_title().get_text():
            legend.get_title().set_fontsize(LEGEND_PT * scale)
