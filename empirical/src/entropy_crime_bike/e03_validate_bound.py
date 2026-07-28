from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
import yaml

from entropy_crime_bike.discrete_bound import (
    closed_form_delb,
    discrete_gaussian_pmf,
    discrete_gaussian_stats,
    entropy_envelope,
    inverse_entropy_envelope,
    lambda_for_entropy,
    shannon_entropy,
)


CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {
    "DC": "Washington, DC",
    "NY": "New York City",
    "VAN": "Vancouver",
}
OUTCOME_LABELS = {
    "crime_count_all": "All harmonized crime",
    "crime_count_property_theft": "Property theft",
    "crime_count_vehicle_theft": "Vehicle theft",
    "crime_count_burglary": "Burglary",
}
COLORS = {
    "exact": "#17365D",
    "closed": "#C00000",
    "gap": "#5B9BD5",
    "DC": "#4472C4",
    "NY": "#ED7D31",
    "VAN": "#70AD47",
}


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return loaded


def _configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e03_bound_validation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e03_bound_validation.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _write_environment(path: Path) -> None:
    lines = [
        f"timestamp={datetime.now().isoformat()}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
        f"scipy={scipy.__version__}",
        f"matplotlib={plt.matplotlib.__version__}",
        f"pyarrow={pq.__version__ if hasattr(pq, '__version__') else 'via pyarrow'}",
    ]
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        lines.extend(["", "[pip-freeze]", freeze])
    except subprocess.SubprocessError as exc:
        lines.append(f"pip_freeze_error={exc}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(figure: plt.Figure, png_path: Path, dpi: int) -> None:
    figure.savefig(
        png_path, dpi=dpi, bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _envelope_grid(
    config: dict[str, object],
) -> pd.DataFrame:
    numerics = config["numerics"]
    entropy_values = np.linspace(
        float(numerics["entropy_grid_min_bits"]),
        float(numerics["entropy_grid_max_bits"]),
        int(numerics["entropy_grid_points"]),
    )
    tail_tolerance = float(numerics["lattice_tail_tolerance"])
    root_tolerance = float(numerics["root_tolerance"])
    records = []
    for entropy_bits in entropy_values:
        exact = inverse_entropy_envelope(
            float(entropy_bits),
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        closed = closed_form_delb(float(entropy_bits))
        records.append(
            {
                "entropy_bits": entropy_bits,
                "exact_delb_mse": exact,
                "closed_form_delb_mse": closed,
                "exact_minus_closed_mse": exact - closed,
                "closed_to_exact_ratio": (
                    closed / exact if exact > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _inverse_recovery(
    config: dict[str, object],
) -> pd.DataFrame:
    numerics = config["numerics"]
    lambdas = np.geomspace(
        float(numerics["lambda_grid_min"]),
        float(numerics["lambda_grid_max"]),
        int(numerics["lambda_grid_points"]),
    )
    tail_tolerance = float(numerics["lattice_tail_tolerance"])
    root_tolerance = float(numerics["root_tolerance"])
    records = []
    for lambda_value in lambdas:
        statistics = discrete_gaussian_stats(
            float(lambda_value), tail_tolerance
        )
        recovered_moment = inverse_entropy_envelope(
            statistics.entropy_bits,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        recovered_entropy = entropy_envelope(
            statistics.second_moment,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        records.append(
            {
                "lambda": lambda_value,
                "summation_method": statistics.summation_method,
                "support_radius": statistics.support_radius,
                "omitted_sum_bound": statistics.omitted_sum_bound,
                "partition_function": statistics.partition,
                "true_second_moment": statistics.second_moment,
                "entropy_bits": statistics.entropy_bits,
                "inverse_recovered_second_moment": recovered_moment,
                "moment_absolute_error": abs(
                    recovered_moment - statistics.second_moment
                ),
                "moment_relative_error": abs(
                    recovered_moment - statistics.second_moment
                )
                / max(statistics.second_moment, np.finfo(float).tiny),
                "envelope_recovered_entropy_bits": recovered_entropy,
                "entropy_absolute_error_bits": abs(
                    recovered_entropy - statistics.entropy_bits
                ),
            }
        )
    return pd.DataFrame(records)


def _distribution_records(
    config: dict[str, object],
) -> pd.DataFrame:
    numerics = config["numerics"]
    tail_tolerance = float(numerics["lattice_tail_tolerance"])
    root_tolerance = float(numerics["root_tolerance"])
    rng = np.random.default_rng(int(config["random_seed"]))
    distributions: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        (
            "two_point_0_1",
            np.array([0, 1]),
            np.array([0.5, 0.5]),
            "non_extremal",
        ),
        (
            "symmetric_three_point",
            np.array([-1, 0, 1]),
            np.array([0.25, 0.5, 0.25]),
            "non_extremal",
        ),
        (
            "rademacher",
            np.array([-1, 1]),
            np.array([0.5, 0.5]),
            "non_extremal",
        ),
    ]
    laplace_support = np.arange(-30, 31)
    laplace_weights = np.exp(-0.7 * np.abs(laplace_support))
    distributions.append(
        (
            "truncated_discrete_laplace",
            laplace_support,
            laplace_weights / laplace_weights.sum(),
            "non_extremal",
        )
    )
    poisson_support = np.arange(0, 31)
    poisson_log_weights = np.array(
        [
            -3.0 + value * math.log(3.0) - math.lgamma(value + 1)
            for value in poisson_support
        ]
    )
    poisson_weights = np.exp(poisson_log_weights)
    distributions.append(
        (
            "truncated_poisson_3",
            poisson_support,
            poisson_weights / poisson_weights.sum(),
            "non_extremal",
        )
    )
    for index in range(100):
        support = np.arange(-8, 9)
        probabilities = rng.dirichlet(
            np.full(len(support), 0.35 + 0.01 * (index % 5))
        )
        distributions.append(
            (
                f"random_finite_{index + 1:03d}",
                support,
                probabilities,
                "non_extremal",
            )
        )
    for entropy_bits in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
        lambda_value = lambda_for_entropy(
            entropy_bits,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        support, probabilities, _ = discrete_gaussian_pmf(
            lambda_value, tail_tolerance
        )
        distributions.append(
            (
                f"discrete_gaussian_H{entropy_bits:g}",
                support,
                probabilities,
                "extremizer",
            )
        )

    records = []
    for name, support, probabilities, distribution_class in distributions:
        normalized = probabilities / probabilities.sum()
        entropy_bits = shannon_entropy(normalized)
        second_moment = float(
            np.dot(support.astype(float) ** 2, normalized)
        )
        maximum_entropy = entropy_envelope(
            second_moment,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        records.append(
            {
                "distribution": name,
                "distribution_class": distribution_class,
                "support_min": int(support.min()),
                "support_max": int(support.max()),
                "support_size": len(support),
                "second_moment": second_moment,
                "entropy_bits": entropy_bits,
                "maximum_entropy_at_moment_bits": maximum_entropy,
                "entropy_slack_bits": maximum_entropy - entropy_bits,
            }
        )
    return pd.DataFrame(records)


def _simulation(
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numerics = config["numerics"]
    simulation = config["simulation"]
    tail_tolerance = float(numerics["lattice_tail_tolerance"])
    root_tolerance = float(numerics["root_tolerance"])
    rng = np.random.default_rng(int(config["random_seed"]) + 3)
    replicate_records = []
    for entropy_bits in simulation["target_entropies_bits"]:
        lambda_value = lambda_for_entropy(
            float(entropy_bits),
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        statistics = discrete_gaussian_stats(
            lambda_value, tail_tolerance
        )
        support, probabilities, _ = discrete_gaussian_pmf(
            lambda_value, tail_tolerance
        )
        for sample_size in simulation["sample_sizes"]:
            for repetition in range(1, int(simulation["repetitions"]) + 1):
                sample = rng.choice(
                    support, size=int(sample_size), p=probabilities
                )
                unique, counts = np.unique(sample, return_counts=True)
                empirical_probabilities = counts / counts.sum()
                plugin_entropy = shannon_entropy(empirical_probabilities)
                miller_madow_entropy = plugin_entropy + (
                    len(unique) - 1
                ) / (2.0 * int(sample_size) * math.log(2.0))
                replicate_records.append(
                    {
                        "target_entropy_bits": float(entropy_bits),
                        "lambda": lambda_value,
                        "sample_size": int(sample_size),
                        "repetition": repetition,
                        "true_second_moment": statistics.second_moment,
                        "sample_second_moment": float(
                            np.mean(sample.astype(float) ** 2)
                        ),
                        "plugin_entropy_bits": plugin_entropy,
                        "miller_madow_entropy_bits": miller_madow_entropy,
                        "observed_atoms": len(unique),
                    }
                )
    replicates = pd.DataFrame(replicate_records)
    summary = (
        replicates.groupby(
            ["target_entropy_bits", "lambda", "sample_size"],
            as_index=False,
        )
        .agg(
            true_second_moment=("true_second_moment", "first"),
            mean_sample_second_moment=("sample_second_moment", "mean"),
            sd_sample_second_moment=("sample_second_moment", "std"),
            mean_plugin_entropy_bits=("plugin_entropy_bits", "mean"),
            sd_plugin_entropy_bits=("plugin_entropy_bits", "std"),
            mean_miller_madow_entropy_bits=(
                "miller_madow_entropy_bits",
                "mean",
            ),
            sd_miller_madow_entropy_bits=(
                "miller_madow_entropy_bits",
                "std",
            ),
            repetitions=("repetition", "count"),
        )
    )
    summary["moment_relative_error"] = (
        summary["mean_sample_second_moment"]
        - summary["true_second_moment"]
    ).abs() / summary["true_second_moment"]
    summary["plugin_entropy_error_bits"] = (
        summary["mean_plugin_entropy_bits"] - summary["target_entropy_bits"]
    )
    summary["miller_madow_entropy_error_bits"] = (
        summary["mean_miller_madow_entropy_bits"]
        - summary["target_entropy_bits"]
    )
    return replicates, summary


def _plugin_entropy(values: pd.Series) -> tuple[float, float, int]:
    counts = values.value_counts(dropna=False).to_numpy(dtype=float)
    probabilities = counts / counts.sum()
    plugin = shannon_entropy(probabilities)
    miller_madow = plugin + (len(counts) - 1) / (
        2.0 * len(values) * math.log(2.0)
    )
    return plugin, miller_madow, len(counts)


def _e02_range_calibration(
    config: dict[str, object], empirical_root: Path
) -> tuple[pd.DataFrame, Path]:
    pointer = empirical_root / str(
        config["source_e02_data_pointer"]
    )
    data_dir = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    outcomes = list(config["range_calibration"]["outcomes"])
    split_names = list(config["range_calibration"]["splits"])
    city_frames = {}
    for city in CITY_ORDER:
        files = sorted(
            (data_dir / "grid_day_panel" / f"city={city}").rglob(
                "*.parquet"
            )
        )
        city_frames[city] = pd.concat(
            [
                pd.read_parquet(
                    path, columns=["split", *outcomes]
                )
                for path in files
            ],
            ignore_index=True,
        )

    records = []
    for city in CITY_ORDER:
        city_frame = city_frames[city]
        for split in split_names:
            subset = (
                city_frame
                if split == "full"
                else city_frame.loc[city_frame["split"].eq(split)]
            )
            for outcome in outcomes:
                plugin, corrected, observed_atoms = _plugin_entropy(
                    subset[outcome]
                )
                exact = inverse_entropy_envelope(corrected)
                closed = closed_form_delb(corrected)
                records.append(
                    {
                        "city": city,
                        "city_label": CITY_LABELS[city],
                        "split": split,
                        "outcome": outcome,
                        "outcome_label": OUTCOME_LABELS[outcome],
                        "observations": len(subset),
                        "observed_atoms": observed_atoms,
                        "maximum_count": int(subset[outcome].max()),
                        "plugin_unconditional_entropy_bits": plugin,
                        "miller_madow_unconditional_entropy_bits": corrected,
                        "exact_delb_at_unconditional_entropy_mse": exact,
                        "closed_delb_at_unconditional_entropy_mse": closed,
                        "calibration_only": True,
                    }
                )
    return pd.DataFrame(records), data_dir


def _acceptance_checks(
    config: dict[str, object],
    envelope: pd.DataFrame,
    recovery: pd.DataFrame,
    distributions: pd.DataFrame,
    simulation: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    validation = config["validation"]
    tail_tolerance = float(
        config["numerics"]["lattice_tail_tolerance"]
    )
    final_size = int(max(config["simulation"]["sample_sizes"]))
    final_simulation = simulation.loc[
        simulation["sample_size"].eq(final_size)
    ]
    extremizers = distributions.loc[
        distributions["distribution_class"].eq("extremizer")
    ]
    non_extremizers = distributions.loc[
        distributions["distribution_class"].eq("non_extremal")
    ]
    calibration_max = float(
        calibration["miller_madow_unconditional_entropy_bits"].max()
    )
    configured_max = float(
        config["numerics"]["entropy_grid_max_bits"]
    )
    checks = [
        {
            "check": "Zero-entropy exact inverse",
            "criterion": "equals 0",
            "observed": inverse_entropy_envelope(0.0),
            "pass": inverse_entropy_envelope(0.0) == 0.0,
        },
        {
            "check": "Exact inverse monotonicity",
            "criterion": "minimum first difference >= 0",
            "observed": float(np.diff(envelope["exact_delb_mse"]).min()),
            "pass": bool(np.diff(envelope["exact_delb_mse"]).min() >= 0),
        },
        {
            "check": "Closed-form bound is conservative",
            "criterion": (
                "minimum exact-minus-closed >= "
                f"-{validation['conservative_slack_tolerance']}"
            ),
            "observed": float(
                envelope["exact_minus_closed_mse"].min()
            ),
            "pass": bool(
                envelope["exact_minus_closed_mse"].min()
                >= -float(validation["conservative_slack_tolerance"])
            ),
        },
        {
            "check": "Lambda-to-moment strict monotonicity",
            "criterion": "all descending in ascending lambda",
            "observed": int(
                (np.diff(recovery["true_second_moment"]) < 0).sum()
            ),
            "pass": bool(
                (np.diff(recovery["true_second_moment"]) < 0).all()
            ),
        },
        {
            "check": "Lambda-to-entropy strict monotonicity",
            "criterion": "all descending in ascending lambda",
            "observed": int(
                (np.diff(recovery["entropy_bits"]) < 0).sum()
            ),
            "pass": bool(
                (np.diff(recovery["entropy_bits"]) < 0).all()
            ),
        },
        {
            "check": "Moment inverse recovery",
            "criterion": (
                "maximum relative error <= "
                f"{validation['inverse_relative_tolerance']}"
            ),
            "observed": float(
                recovery["moment_relative_error"].max()
            ),
            "pass": bool(
                recovery["moment_relative_error"].max()
                <= float(validation["inverse_relative_tolerance"])
            ),
        },
        {
            "check": "Entropy envelope recovery",
            "criterion": (
                "maximum absolute error <= "
                f"{validation['entropy_absolute_tolerance_bits']} bits"
            ),
            "observed": float(
                recovery["entropy_absolute_error_bits"].max()
            ),
            "pass": bool(
                recovery["entropy_absolute_error_bits"].max()
                <= float(validation["entropy_absolute_tolerance_bits"])
            ),
        },
        {
            "check": "Numerical lattice-tail control",
            "criterion": f"maximum omitted sum bound <= {tail_tolerance}",
            "observed": float(recovery["omitted_sum_bound"].max()),
            "pass": bool(
                recovery["omitted_sum_bound"].max()
                <= tail_tolerance * (1 + 1e-12)
            ),
        },
        {
            "check": "Discrete-Gaussian extremizer equality",
            "criterion": "maximum absolute entropy slack <= 1e-9 bits",
            "observed": float(
                extremizers["entropy_slack_bits"].abs().max()
            ),
            "pass": bool(
                extremizers["entropy_slack_bits"].abs().max() <= 1e-9
            ),
        },
        {
            "check": "Non-extremal maximum-entropy inequality",
            "criterion": "minimum entropy slack >= -1e-9 bits",
            "observed": float(
                non_extremizers["entropy_slack_bits"].min()
            ),
            "pass": bool(
                non_extremizers["entropy_slack_bits"].min() >= -1e-9
            ),
        },
        {
            "check": "Monte Carlo moment convergence",
            "criterion": (
                "maximum final mean relative error <= "
                f"{validation['simulation_final_relative_moment_tolerance']}"
            ),
            "observed": float(
                final_simulation["moment_relative_error"].max()
            ),
            "pass": bool(
                final_simulation["moment_relative_error"].max()
                <= float(
                    validation[
                        "simulation_final_relative_moment_tolerance"
                    ]
                )
            ),
        },
        {
            "check": "Monte Carlo entropy convergence",
            "criterion": (
                "maximum final Miller-Madow error <= "
                f"{validation['simulation_final_entropy_tolerance_bits']} bits"
            ),
            "observed": float(
                final_simulation[
                    "miller_madow_entropy_error_bits"
                ].abs().max()
            ),
            "pass": bool(
                final_simulation[
                    "miller_madow_entropy_error_bits"
                ].abs().max()
                <= float(
                    validation["simulation_final_entropy_tolerance_bits"]
                )
            ),
        },
        {
            "check": "E02 entropy range covered by numerical grid",
            "criterion": f"maximum calibration entropy <= {configured_max} bits",
            "observed": calibration_max,
            "pass": calibration_max <= configured_max,
        },
        {
            "check": "All numerical outputs finite",
            "criterion": "no non-finite required values",
            "observed": 0,
            "pass": bool(
                np.isfinite(
                    envelope[
                        [
                            "entropy_bits",
                            "exact_delb_mse",
                            "closed_form_delb_mse",
                        ]
                    ].to_numpy()
                ).all()
                and np.isfinite(
                    recovery[
                        [
                            "true_second_moment",
                            "entropy_bits",
                            "inverse_recovered_second_moment",
                        ]
                    ].to_numpy()
                ).all()
            ),
        },
    ]
    frame = pd.DataFrame(checks)
    frame["status"] = np.where(frame["pass"], "PASS", "FAIL")
    return frame.drop(columns="pass")


def _plot_exact_and_closed(
    envelope: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.1), constrained_layout=True
    )
    axes[0].plot(
        envelope["entropy_bits"],
        envelope["exact_delb_mse"],
        color=COLORS["exact"],
        linewidth=2.2,
        label="Exact inverse-envelope DELB",
    )
    axes[0].plot(
        envelope["entropy_bits"],
        envelope["closed_form_delb_mse"],
        color=COLORS["closed"],
        linewidth=1.8,
        linestyle="--",
        label="Closed-form lattice bound",
    )
    axes[0].set_xlabel("Entropy, $h$ (bits)")
    axes[0].set_ylabel("MSE lower bound")
    axes[0].set_title("(a) Natural scale", loc="left", fontweight="bold")
    axes[0].grid(color="#E7E6E6", linewidth=0.7)
    axes[0].legend(frameon=False)

    positive = envelope["entropy_bits"] > 0
    axes[1].semilogy(
        envelope.loc[positive, "entropy_bits"],
        envelope.loc[positive, "exact_delb_mse"],
        color=COLORS["exact"],
        linewidth=2.2,
        label="Exact inverse-envelope DELB",
    )
    closed_positive = envelope["closed_form_delb_mse"] > 0
    axes[1].semilogy(
        envelope.loc[closed_positive, "entropy_bits"],
        envelope.loc[closed_positive, "closed_form_delb_mse"],
        color=COLORS["closed"],
        linewidth=1.8,
        linestyle="--",
        label="Closed-form lattice bound",
    )
    axes[1].set_xlabel("Entropy, $h$ (bits)")
    axes[1].set_ylabel("MSE lower bound (log scale)")
    axes[1].set_title("(b) Logarithmic scale", loc="left", fontweight="bold")
    axes[1].grid(color="#E7E6E6", linewidth=0.7, which="both")
    figure.suptitle(
        "Exact and closed-form discrete entropy lower bounds",
        fontsize=13,
        fontweight="bold",
    )
    _save_figure(figure, path, dpi)


def _plot_lattice_gap(
    envelope: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.0), constrained_layout=True
    )
    axes[0].plot(
        envelope["entropy_bits"],
        envelope["exact_minus_closed_mse"],
        color=COLORS["gap"],
        linewidth=2,
    )
    axes[0].axhline(0, color="#7F7F7F", linewidth=0.8)
    axes[0].set_xlabel("Entropy, $h$ (bits)")
    axes[0].set_ylabel("Exact minus closed-form bound")
    axes[0].set_title(
        "(a) Absolute conservatism", loc="left", fontweight="bold"
    )
    axes[0].grid(color="#E7E6E6", linewidth=0.7)

    valid = envelope["closed_to_exact_ratio"].notna()
    axes[1].plot(
        envelope.loc[valid, "entropy_bits"],
        envelope.loc[valid, "closed_to_exact_ratio"],
        color=COLORS["closed"],
        linewidth=2,
    )
    axes[1].axhline(1, color="#7F7F7F", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Entropy, $h$ (bits)")
    axes[1].set_ylabel("Closed-form / exact DELB")
    axes[1].set_title(
        "(b) Relative tightness", loc="left", fontweight="bold"
    )
    axes[1].grid(color="#E7E6E6", linewidth=0.7)
    figure.suptitle(
        "Conservatism induced by the closed-form relaxation",
        fontsize=13,
        fontweight="bold",
    )
    _save_figure(figure, path, dpi)


def _plot_pmfs(
    config: dict[str, object], path: Path, dpi: int
) -> None:
    target_entropies = [0.5, 1.0, 2.0, 3.0]
    tail_tolerance = float(
        config["numerics"]["lattice_tail_tolerance"]
    )
    root_tolerance = float(config["numerics"]["root_tolerance"])
    figure, axes = plt.subplots(
        2, 2, figsize=(9.2, 6.6), constrained_layout=True
    )
    for axis, entropy_bits in zip(axes.flat, target_entropies):
        lambda_value = lambda_for_entropy(
            entropy_bits,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        support, probabilities, _ = discrete_gaussian_pmf(
            lambda_value, tail_tolerance
        )
        visible = probabilities >= 1e-5
        axis.bar(
            support[visible],
            probabilities[visible],
            color=COLORS["exact"],
            width=0.78,
        )
        axis.set_xlabel("Integer error, $k$")
        axis.set_ylabel("$q_\\lambda(k)$")
        axis.set_title(
            f"$H={entropy_bits:.1f}$ bits, $\\lambda={lambda_value:.3g}$",
            loc="left",
            fontweight="bold",
        )
        axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
    figure.suptitle(
        "Maximum-entropy distributions on the integer lattice",
        fontsize=13,
        fontweight="bold",
    )
    _save_figure(figure, path, dpi)


def _plot_simulation(
    simulation: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.1), constrained_layout=True
    )
    palette = plt.get_cmap("viridis")
    entropies = sorted(simulation["target_entropy_bits"].unique())
    for index, entropy_bits in enumerate(entropies):
        subset = simulation.loc[
            simulation["target_entropy_bits"].eq(entropy_bits)
        ].sort_values("sample_size")
        color = palette(index / max(1, len(entropies) - 1))
        axes[0].plot(
            subset["sample_size"],
            subset["moment_relative_error"],
            marker="o",
            color=color,
            label=f"$H={entropy_bits:g}$",
        )
        axes[1].plot(
            subset["sample_size"],
            subset["miller_madow_entropy_error_bits"].abs(),
            marker="s",
            color=color,
            label=f"$H={entropy_bits:g}$",
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Monte Carlo sample size")
        axis.grid(color="#E7E6E6", linewidth=0.7, which="both")
    axes[0].set_ylabel("Relative second-moment error")
    axes[0].set_title(
        "(a) Second-moment recovery", loc="left", fontweight="bold"
    )
    axes[1].set_ylabel("Absolute entropy error (bits)")
    axes[1].set_title(
        "(b) Miller–Madow entropy recovery",
        loc="left",
        fontweight="bold",
    )
    axes[1].legend(frameon=False, ncol=2)
    figure.suptitle(
        "Monte Carlo recovery under attainable discrete-Gaussian errors",
        fontsize=13,
        fontweight="bold",
    )
    _save_figure(figure, path, dpi)


def _plot_calibration(
    calibration: pd.DataFrame, path: Path, dpi: int
) -> None:
    subset = calibration.loc[
        calibration["split"].eq("full")
        & calibration["outcome"].eq("crime_count_all")
    ].copy()
    figure, axis = plt.subplots(
        figsize=(7.4, 4.5), constrained_layout=True
    )
    x = np.arange(len(subset))
    width = 0.34
    axis.bar(
        x - width / 2,
        subset["exact_delb_at_unconditional_entropy_mse"],
        width,
        color=COLORS["exact"],
        label="Exact inverse-envelope value",
    )
    axis.bar(
        x + width / 2,
        subset["closed_delb_at_unconditional_entropy_mse"],
        width,
        color=COLORS["closed"],
        label="Closed-form value",
    )
    axis.set_xticks(x, subset["city_label"])
    axis.set_ylabel("MSE-scale calibration value")
    axis.set_title(
        "Numerical range check using unconditional E02 count entropy",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
    axis.legend(frameon=False)
    axis.text(
        0.0,
        -0.22,
        "Calibration only: these values are not conditional DELB estimates "
        "and do not measure bicycle information.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#595959",
    )
    _save_figure(figure, path, dpi)


def _write_interpretation(
    run_dir: Path,
    acceptance: pd.DataFrame,
    envelope: pd.DataFrame,
    recovery: pd.DataFrame,
    distributions: pd.DataFrame,
    simulation: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    threshold = 0.5 * math.log2(math.pi * math.e / 6.0)
    final_size = int(simulation["sample_size"].max())
    final_simulation = simulation.loc[
        simulation["sample_size"].eq(final_size)
    ]
    lines = [
        "# E03 Numerical Validation of the Discrete Entropy Lower Bound",
        "",
        "## Scope",
        "",
        "E03 validates the numerical implementation of the integer-lattice "
        "maximum-entropy envelope and its generalized inverse. It does not "
        "estimate city-level conditional mutual information; that estimand "
        "begins in E04.",
        "",
        "## Acceptance result",
        "",
        f"- {int(acceptance['status'].eq('PASS').sum())} of "
        f"{len(acceptance)} prespecified numerical checks passed.",
        f"- Maximum inverse moment relative error: "
        f"{recovery['moment_relative_error'].max():.3e}.",
        f"- Maximum entropy round-trip error: "
        f"{recovery['entropy_absolute_error_bits'].max():.3e} bits.",
        f"- Minimum tested exact-minus-closed slack: "
        f"{envelope['exact_minus_closed_mse'].min():.3e}.",
        f"- The closed-form positive-part threshold is "
        f"{threshold:.4f} bits; the exact inverse is positive for every "
        "strictly positive entropy.",
        "",
        "## Extremal and simulation checks",
        "",
        f"- Maximum absolute entropy slack for tested discrete-Gaussian "
        f"extremizers: "
        f"{distributions.loc[distributions['distribution_class'].eq('extremizer'), 'entropy_slack_bits'].abs().max():.3e} bits.",
        f"- Minimum entropy slack among 105 non-extremal finite-support "
        f"distributions: "
        f"{distributions.loc[distributions['distribution_class'].eq('non_extremal'), 'entropy_slack_bits'].min():.3e} bits.",
        f"- At n={final_size:,}, maximum mean second-moment relative error "
        f"across simulated target entropies: "
        f"{final_simulation['moment_relative_error'].max():.3%}.",
        f"- At n={final_size:,}, maximum absolute mean Miller--Madow entropy "
        f"error: "
        f"{final_simulation['miller_madow_entropy_error_bits'].abs().max():.4f} bits.",
        "",
        "## E02 range calibration",
        "",
        "Unconditional count entropies from the accepted E02 panel were used "
        "only to verify that the configured numerical domain covers the "
        "observed integer-count range. These are not conditional entropy "
        "estimates, are not matched prediction bounds, and provide no evidence "
        "about bicycle information value.",
        "",
    ]
    for city in CITY_ORDER:
        row = calibration.loc[
            calibration["city"].eq(city)
            & calibration["split"].eq("full")
            & calibration["outcome"].eq("crime_count_all")
        ].iloc[0]
        lines.append(
            f"- {CITY_LABELS[city]}: unconditional Miller--Madow entropy "
            f"{row['miller_madow_unconditional_entropy_bits']:.4f} bits; "
            f"exact inverse-envelope calibration value "
            f"{row['exact_delb_at_unconditional_entropy_mse']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The implemented solver reproduces the discrete-Gaussian "
            "attainability curve, respects the zero-entropy boundary and "
            "monotonicity, and confirms numerically that the dither-based "
            "closed form is conservative over the tested domain. E04 may "
            "therefore use the exact inverse-envelope implementation as its "
            "primary MSE-scale transformation.",
        ]
    )
    (run_dir / "interpretation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _append_registry(
    empirical_root: Path,
    run_id: str,
    run_dir: Path,
    started_at: datetime,
    completed_at: datetime,
    elapsed: float,
    status: str,
) -> None:
    registry = empirical_root / "docs" / "experiment_registry.csv"
    existing = pd.read_csv(registry)
    if run_id in set(existing["run_id"].astype(str)):
        return
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "experiment": "E03",
                "status": status,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "run_directory": str(run_dir),
                "notes": (
                    "Numerical validation of the exact integer-lattice "
                    "entropy envelope, inverse DELB, and closed-form bound."
                ),
            }
        ]
    ).to_csv(registry, mode="a", header=False, index=False)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    empirical_root = Path(__file__).resolve().parents[2]
    timezone = ZoneInfo("Asia/Shanghai")
    started_at = datetime.now(timezone)
    run_id = (
        f"{started_at.strftime('%Y%m%d_%H%M%S')}_"
        "E03_discrete_bound_validation"
    )
    run_dir = empirical_root / "runs" / run_id
    for directory in ["logs", "tables", "figures"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(run_dir)
    command = (
        f"{sys.executable} empirical/run_e03_bound_validation.py "
        f"--config {config_path}"
    )
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_environment(run_dir / "environment.txt")
    started_clock = time.monotonic()
    logger.info("Starting E03 numerical validation")

    try:
        envelope = _envelope_grid(config)
        envelope.to_csv(
            run_dir / "tables" / "entropy_envelope_grid.csv", index=False
        )
        logger.info("Evaluated %d entropy-grid values", len(envelope))

        recovery = _inverse_recovery(config)
        recovery.to_csv(
            run_dir / "tables" / "inverse_recovery.csv", index=False
        )
        logger.info("Completed %d inverse round-trip checks", len(recovery))

        distributions = _distribution_records(config)
        distributions.to_csv(
            run_dir / "tables" / "maximum_entropy_checks.csv", index=False
        )
        logger.info(
            "Compared %d integer distributions with the entropy envelope",
            len(distributions),
        )

        simulation_replicates, simulation = _simulation(config)
        simulation_replicates.to_csv(
            run_dir / "tables" / "simulation_replicates.csv", index=False
        )
        simulation.to_csv(
            run_dir / "tables" / "simulation_convergence.csv", index=False
        )
        logger.info(
            "Completed %d Monte Carlo replicates",
            len(simulation_replicates),
        )

        calibration, e02_data_dir = _e02_range_calibration(
            config, empirical_root
        )
        calibration.to_csv(
            run_dir / "tables" / "e02_entropy_range_calibration.csv",
            index=False,
        )
        logger.info("Calibrated numerical range against accepted E02 panel")

        acceptance = _acceptance_checks(
            config,
            envelope,
            recovery,
            distributions,
            simulation,
            calibration,
        )
        acceptance.to_csv(
            run_dir / "tables" / "acceptance_checklist.csv", index=False
        )

        _publication_style()
        dpi = int(config["reporting"]["figure_dpi"])
        figure_dir = run_dir / "figures"
        _plot_exact_and_closed(
            envelope, figure_dir / "e03_exact_vs_closed_delb.png", dpi
        )
        _plot_lattice_gap(
            envelope, figure_dir / "e03_closed_form_conservatism.png", dpi
        )
        _plot_pmfs(
            config, figure_dir / "e03_discrete_gaussian_pmfs.png", dpi
        )
        _plot_simulation(
            simulation, figure_dir / "e03_simulation_convergence.png", dpi
        )
        _plot_calibration(
            calibration, figure_dir / "e03_e02_range_calibration.png", dpi
        )
        _write_interpretation(
            run_dir,
            acceptance,
            envelope,
            recovery,
            distributions,
            simulation,
            calibration,
        )

        failed = acceptance.loc[acceptance["status"].ne("PASS")]
        if len(failed):
            raise AssertionError(
                "E03 acceptance gate failed: "
                + ", ".join(failed["check"].astype(str))
            )

        elapsed = time.monotonic() - started_clock
        completed_at = datetime.now(timezone)
        status = {
            "run_id": run_id,
            "status": "complete",
            "acceptance_status": "PASS",
            "acceptance_checks": len(acceptance),
            "source_e02_data_directory": str(e02_data_dir),
            "elapsed_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "article_figures_generated_by": "Python matplotlib",
        }
        (run_dir / "run_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        (empirical_root / "runs" / "latest_e03_run.txt").write_text(
            str(run_dir) + "\n", encoding="utf-8"
        )
        _append_registry(
            empirical_root,
            run_id,
            run_dir,
            started_at,
            completed_at,
            elapsed,
            "complete",
        )
        logger.info("E03 passed all %d checks in %.1f seconds", len(acceptance), elapsed)
        return run_dir
    except Exception:
        elapsed = time.monotonic() - started_clock
        (run_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "elapsed_seconds": elapsed,
                    "started_at": started_at.isoformat(),
                    "completed_at": datetime.now(timezone).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.exception("E03 failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Numerically validate the discrete entropy lower bound"
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
