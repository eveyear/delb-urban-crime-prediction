from __future__ import annotations

from pathlib import Path

import pytest

from entropy_crime_bike.s7_manuscript_synthesis import (
    _fano_summary,
    _simulation_summary,
    latex_escape,
)


EMPIRICAL = Path(__file__).resolve().parents[1]


def _accepted_run(experiment: str) -> Path:
    pointer = EMPIRICAL / "runs" / f"latest_{experiment}_run.txt"
    if not pointer.is_file():
        pytest.skip(f"Accepted {experiment} run is not redistributed")
    run_id = pointer.read_text(encoding="utf-8").strip()
    run = EMPIRICAL / "runs" / run_id
    if not run.is_dir():
        pytest.skip(f"Accepted {experiment} run directory is not redistributed")
    return run


def test_latex_escape_handles_table_characters() -> None:
    assert latex_escape("A&B_1%") == r"A\&B\_1\%"


def test_simulation_summary_has_both_estimators() -> None:
    frame, macros = _simulation_summary(_accepted_run("e12"))
    assert frame["Estimator"].tolist() == ["Plugin", "Miller--Madow"]
    assert float(macros["SimPluginNullBias"]) > 0
    assert float(macros["SimMMNullCenter"]) > 0


def test_fano_summary_preserves_loss_direction() -> None:
    frame, macros = _fano_summary(_accepted_run("e14"))
    assert len(frame) == 6
    assert frame["Floor reduction"].gt(0).all()
    test_change = (
        frame["Baseline test error"] - frame["Bicycle test error"]
    )
    assert test_change.lt(0).all()
    assert macros["FanoEstimatedViolations"] == "0"
