from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attainability import attainability_decomposition
from .discrete_bound import discrete_gaussian_pmf


def _validation_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    support_q, probability_q, _ = discrete_gaussian_pmf(
        0.8, tail_tolerance=1e-15
    )
    return [
        (
            "degenerate_zero",
            np.array([0], dtype=int),
            np.array([[0.25, 0.75]], dtype=float),
        ),
        (
            "independent_discrete_gaussian",
            support_q,
            np.outer(probability_q, np.array([0.2, 0.3, 0.5])),
        ),
        (
            "independent_non_gaussian",
            np.array([-1, 0, 1], dtype=int),
            np.outer(np.array([0.1, 0.8, 0.1]), np.array([0.4, 0.6])),
        ),
        (
            "dependent_non_gaussian",
            np.array([-2, -1, 0, 1, 2], dtype=int),
            np.array(
                [
                    [0.04, 0.01],
                    [0.18, 0.02],
                    [0.20, 0.20],
                    [0.02, 0.18],
                    [0.01, 0.14],
                ]
            ),
        ),
    ]


def main() -> None:
    empirical_root = Path(__file__).resolve().parents[2]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_S2_attainability")
    run_dir = empirical_root / "runs" / run_id
    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, object]] = []
    for case_name, support, joint in _validation_cases():
        result = attainability_decomposition(support, joint)
        record: dict[str, object] = {"case": case_name}
        record.update(result.as_record())
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(table_dir / "attainability_identity_checks.csv", index=False)

    tolerance = 2e-10
    checks = {
        "all_slacks_nonnegative": bool(
            (
                frame[
                    [
                        "history_dependence_bits",
                        "shape_mismatch_bits",
                        "predictor_specific_slack_bits",
                        "universal_slack_bits",
                    ]
                ]
                >= -tolerance
            ).all().all()
        ),
        "identity_residual_within_tolerance": bool(
            (frame["identity_residual_bits"].abs() <= tolerance).all()
        ),
        "discrete_gaussian_case_zero_slack": bool(
            frame.loc[
                frame["case"].eq("independent_discrete_gaussian"),
                "universal_slack_bits",
            ].abs().max()
            <= tolerance
        ),
        "dependent_case_has_both_slacks": bool(
            (
                frame.loc[
                    frame["case"].eq("dependent_non_gaussian"),
                    ["history_dependence_bits", "shape_mismatch_bits"],
                ]
                > tolerance
            ).all().all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"S2 validation failed: {checks}")

    manifest = {
        "run_id": run_id,
        "stage": "S2",
        "purpose": "Numerical validation of the exact DELB attainability decomposition",
        "tolerance_bits": tolerance,
        "checks": checks,
        "input": "Analytically specified probability tables; no empirical data",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "interpretation.md").write_text(
        "\n".join(
            [
                "# S2 Attainability Validation",
                "",
                "All four analytic cases satisfy the entropy-domain identity",
                "within the recorded numerical tolerance.",
                "",
                "- The degenerate zero-error case has zero total slack.",
                "- The independent discrete-Gaussian case has zero slack up to truncation and floating-point tolerance.",
                "- The independent non-Gaussian case has shape mismatch but no history-dependence slack.",
                "- The dependent non-Gaussian case has both nonnegative components.",
                "",
                "These checks validate the implementation; they are not empirical evidence about crime or bicycle mobility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
