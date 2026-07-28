#!/usr/bin/env python3
"""Minimal known-truth DELB/CMI simulation using the manuscript code."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "empirical" / "src"))

from entropy_crime_bike.discrete_bound import inverse_entropy_envelope  # noqa: E402
from entropy_crime_bike.e12_cmi_simulation import (  # noqa: E402
    build_population_model,
    estimate_cmi,
    simulate_series,
)


DGP = {
    "dispersion": 2.5,
    "minimum_mean": 0.35,
    "maximum_mean": 2.0,
    "tail_probability_tolerance": 1.0e-12,
    "markov_persistence": 0.85,
}


def run_example(seed: int = 20260723, sample_size: int = 365) -> list[dict[str, float | int | str]]:
    """Return one null and one positive-CMI known-truth comparison."""

    rows: list[dict[str, float | int | str]] = []
    for scenario, effect in (("population_null", 0.0), ("positive_cmi", 0.5)):
        model = build_population_model(
            baseline_count=4,
            bicycle_count=3,
            balance_regime="balanced",
            effect_strength=effect,
            dgp_config=DGP,
        )
        rng = np.random.default_rng(seed + (0 if effect == 0 else 1))
        target, baseline, additional = simulate_series(
            model,
            sample_size=sample_size,
            temporal_structure="iid",
            dgp_config=DGP,
            rng=rng,
        )
        estimate = estimate_cmi(target, baseline, additional)

        population_hb = model.conditional_entropy_bits
        population_h0 = population_hb + model.population_cmi_bits
        population_l0 = inverse_entropy_envelope(population_h0)
        population_lb = inverse_entropy_envelope(population_hb)

        rows.append(
            {
                "scenario": scenario,
                "sample_size": sample_size,
                "population_cmi_bits": model.population_cmi_bits,
                "plugin_cmi_bits": estimate["cmi_plugin_bits"],
                "miller_madow_cmi_bits": estimate["cmi_miller_madow_bits"],
                "population_baseline_delb_mse": population_l0,
                "population_augmented_delb_mse": population_lb,
                "population_delb_reduction_mse": population_l0 - population_lb,
                "rounded_oracle_predictor_mse": model.rounded_predictor_mse,
            }
        )

    assert abs(float(rows[0]["population_cmi_bits"])) < 1.0e-10
    assert abs(float(rows[0]["population_delb_reduction_mse"])) < 1.0e-10
    assert float(rows[1]["population_cmi_bits"]) > 0
    assert float(rows[1]["population_delb_reduction_mse"]) > 0
    assert all(
        float(row["rounded_oracle_predictor_mse"])
        >= float(row["population_augmented_delb_mse"])
        for row in rows
    )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a two-scenario known-truth DELB/CMI example."
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--sample-size", type=int, default=365)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "examples" / "minimal_delb_cmi_results.csv",
    )
    args = parser.parse_args()
    rows = run_example(args.seed, args.sample_size)
    write_csv(rows, args.output)
    for row in rows:
        print(
            f"{row['scenario']}: population CMI="
            f"{float(row['population_cmi_bits']):.6f} bits, "
            f"MM estimate={float(row['miller_madow_cmi_bits']):.6f} bits, "
            f"population DELB reduction="
            f"{float(row['population_delb_reduction_mse']):.6f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
