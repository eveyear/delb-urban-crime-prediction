from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "examples" / "minimal_delb_cmi_simulation.py"
SPEC = importlib.util.spec_from_file_location("minimal_delb_cmi", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_minimal_example_population_ordering() -> None:
    rows = MODULE.run_example(seed=1234, sample_size=200)
    assert [row["scenario"] for row in rows] == [
        "population_null",
        "positive_cmi",
    ]
    assert abs(float(rows[0]["population_cmi_bits"])) < 1.0e-10
    assert float(rows[1]["population_cmi_bits"]) > 0
    assert float(rows[1]["population_baseline_delb_mse"]) > float(
        rows[1]["population_augmented_delb_mse"]
    )
