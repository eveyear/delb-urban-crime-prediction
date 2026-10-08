#!/usr/bin/env python3
"""Canonical data-free Figure 1 and Figure 2 builders for the accepted paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#23384A"
LINE = "#637B8E"
BLUE = "#E4F0FA"
ORANGE = "#FFF0DC"
GREEN = "#E9F4E5"
PURPLE = "#EEEAF7"


def box(ax, x, y, w, h, title, detail, fill, title_size=9.2, detail_size=8.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1, edgecolor=LINE, facecolor=fill,
    ))
    title_offset = 0.026 if h < 0.09 else 0.032
    detail_offset = 0.057 if h < 0.09 else 0.075
    ax.text(x + 0.018, y + h - title_offset, title, ha="left", va="top",
            fontsize=title_size, weight="bold", color=INK)
    ax.text(x + 0.018, y + h - detail_offset, detail, ha="left", va="top",
            fontsize=detail_size, color=INK, linespacing=1.18)


def arrow(ax, a, b, color=LINE):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.05, color=color,
                                 connectionstyle="arc3,rad=0"))


def evidence_map():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.5, 0.976, "Four evidence layers across three research questions",
            ha="center", va="top", fontsize=10.2, weight="bold", color=INK)
    ax.text(0.035, 0.916, "EVIDENCE LAYER", fontsize=8, weight="bold", color=LINE)
    ax.text(0.74, 0.916, "QUESTION ADDRESSED", fontsize=8, weight="bold", color=LINE)
    rows = [
        (0.71, "1  DELB: integer-MSE floor",
         "Population bound and attainability;\npaired estimated floors in application", BLUE,
         "RQ1 + RQ3"),
        (0.51, "2  CMI + randomization",
         "Added-information estimate; comparison\nwith design-matched null references", ORANGE,
         "RQ2 + RQ3"),
        (0.31, "3  Known-truth experiments",
         "Known CMI, replicate estimates, bias,\npower and detection limits", GREEN,
         "RQ2"),
        (0.11, "4  Held-out MSE + Gap",
         "Matched model errors and DELB;\nrealized use of prediction opportunity", PURPLE,
         "RQ3"),
    ]
    for y, title, detail, fill, rq in rows:
        box(ax, 0.035, y, 0.67, 0.155, title, detail, fill)
        ax.add_patch(FancyBboxPatch((0.745, y + 0.039), 0.22, 0.08,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor="white", edgecolor=LINE, linewidth=0.9))
        ax.text(0.855, y + 0.079, rq,
                ha="center", va="center", fontsize=8.7, weight="bold", color=INK)
        arrow(ax, (0.708, y + 0.078), (0.737, y + 0.078))
    ax.text(0.5, 0.032,
            "Layer 3 assesses behavior under tested designs; it is not a city-specific test.",
            ha="center", va="center", fontsize=8.1, color=LINE)
    return fig


def workflow():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 7.2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.5, 1.015, "Forecast-time information to calibrated and realized evidence",
            ha="center", va="top", fontsize=10.0, weight="bold", color=INK)
    box(ax, 0.09, 0.875, 0.82, 0.072, "Crime events + bicycle trips (2020-2022)",
        "Harmonize events; construct the complete 1-km daily panel", BLUE, 8.5, 7.5)
    arrow(ax, (0.5, 0.869), (0.5, 0.847))
    box(ax, 0.09, 0.760, 0.82, 0.082, "Lagged predictors and fixed temporal split",
        r"Baseline $\mathcal{S}^0$, bicycle-augmented $\mathcal{S}^B$; train / validate / test", BLUE, 8.5, 7.5)
    arrow(ax, (0.32, 0.752), (0.25, 0.707))
    arrow(ax, (0.68, 0.752), (0.75, 0.707))
    box(ax, 0.035, 0.586, 0.43, 0.113, "Entropy, CMI and DELB",
        "Compare both states: CMI and\npaired DELB for each stated window", ORANGE, 8.3, 7.4)
    box(ax, 0.535, 0.586, 0.43, 0.113, "Paired model training",
        "Same target/split; validation tuning;\nbaseline and bicycle models", GREEN, 8.3, 7.4)
    arrow(ax, (0.25, 0.578), (0.25, 0.548))
    arrow(ax, (0.75, 0.578), (0.75, 0.548))
    box(ax, 0.035, 0.425, 0.43, 0.116, "CMI null calibration (RQ2)",
        "Recompute on design-matched\nsurrogates; compare observed CMI", ORANGE, 8.3, 7.4)
    box(ax, 0.535, 0.425, 0.43, 0.116, "Held-out integer MSE (RQ3)",
        "Same test grid-days and target;\ncompute paired model MSE", GREEN, 8.3, 7.4)
    ax.plot([0.035, 0.016, 0.016, 0.112], [0.642, 0.642, 0.268, 0.268],
            color=LINE, linewidth=1.05, clip_on=False)
    arrow(ax, (0.10, 0.268), (0.116, 0.268))
    ax.text(0.04, 0.328, "test-window DELB to Gap", fontsize=7.8,
            ha="left", va="bottom", color=INK)
    ax.plot([0.25, 0.25], [0.416, 0.361], color=LINE, linewidth=0.95,
            linestyle="--")
    ax.add_patch(FancyArrowPatch((0.25, 0.361), (0.327, 0.361),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.95, linestyle="--", color=LINE))
    ax.text(0.34, 0.361, "RQ3: calibrated evidence", fontsize=7.8,
            ha="left", va="center", color=INK)
    arrow(ax, (0.75, 0.414), (0.64, 0.327))
    box(ax, 0.12, 0.215, 0.76, 0.106, "Align DELB and MSE; calculate Predictability Gap",
        "Same state and test grid-days: G = MSE - L; compare the two gaps", PURPLE,
        8.5, 7.5)
    ax.text(0.5, 0.179, "RQ1: DELB foundation  |  RQ3: calibrated information and realized gain",
            ha="center", fontsize=7.9, color=INK)
    ax.add_patch(FancyBboxPatch((0.12, 0.05), 0.76, 0.083,
                                boxstyle="round,pad=0.006,rounding_size=0.009",
                                facecolor="#F7F8F9", edgecolor=LINE, linewidth=0.8,
                                linestyle="--"))
    ax.text(0.5, 0.106, "Parallel assessment: known-truth experiments",
            ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
    ax.text(0.5, 0.071, "Check CMI bias, calibration, power and detection limits (RQ2)",
            ha="center", va="center", fontsize=8.0, color=INK)
    return fig


def save_framework_figures(output_root: Path, dpi: int = 400):
    outputs = []
    for build, relative in [(evidence_map, "problem_chain/unified_problem_chain"),
                            (workflow, "e11/e11_analysis_workflow")]:
        stem = output_root / relative
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig = build()
        for extension in ["pdf", "png", "svg"]:
            path = stem.with_suffix("." + extension)
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)
    return outputs
