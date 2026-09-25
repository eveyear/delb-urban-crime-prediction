from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from .figure_typography import normalize_chart_typography
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml
from scipy.special import digamma

from .e09_robustness import _load_panel, _prepare_spec
from .e10_placebo import (
    CITY_COLORS,
    CITY_LABELS,
    CITY_ORDER,
    NULL_LABELS,
    NULL_ORDER,
    _bound_reduction,
    _matrix_view,
    _null_bicycle_matrix,
    _seed,
    conditional_information,
    randomization_p_value,
)
from .spatial_information import BoundInterpolator, benjamini_hochberg


LOG_TWO = math.log(2.0)
ESTIMATORS = ["plugin", "miller_madow", "jeffreys_dirichlet"]


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entropy_from_counts(counts: np.ndarray) -> float:
    positive = np.asarray(counts, dtype=float)
    positive = positive[positive > 0]
    probabilities = positive / positive.sum()
    return float(-np.dot(probabilities, np.log2(probabilities)))


def _dirichlet_expected_entropy(parameters: np.ndarray) -> float:
    values = np.asarray(parameters, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        raise ValueError("Dirichlet parameter vector cannot be empty.")
    total = float(values.sum())
    return float(
        (
            digamma(total + 1.0)
            - np.dot(values / total, digamma(values + 1.0))
        )
        / LOG_TWO
    )


def _plugin_and_jeffreys(
    target: np.ndarray,
    baseline: np.ndarray,
    bicycle: np.ndarray,
    *,
    alpha: float,
) -> dict[str, tuple[float, float]]:
    y = np.asarray(target, dtype=np.int64)
    s = np.asarray(baseline, dtype=np.int64)
    b = np.asarray(bicycle, dtype=np.int64)
    if not (y.shape == s.shape == b.shape):
        raise ValueError("Entropy vectors must be aligned.")
    if np.any(y < 0) or np.any(s < 0) or np.any(b < 0):
        raise ValueError("Entropy codes must be nonnegative.")

    y_alphabet = int(y.max()) + 1
    b_alphabet = int(b.max()) + 1
    sb = s * b_alphabet + b
    sy = s * y_alphabet + y
    sby = sb * y_alphabet + y

    s_counts = np.bincount(s)
    sy_counts = np.bincount(sy)
    sb_counts = np.bincount(sb)
    sby_counts = np.bincount(sby)
    h0_plugin = _entropy_from_counts(sy_counts) - _entropy_from_counts(s_counts)
    hb_plugin = _entropy_from_counts(sby_counts) - _entropy_from_counts(sb_counts)

    full_indices = np.flatnonzero(sby_counts)
    full_parameters = sby_counts[full_indices].astype(float) + float(alpha)
    sb_atom_codes = full_indices // y_alphabet
    y_atom_codes = full_indices % y_alphabet
    s_atom_codes = sb_atom_codes // b_alphabet
    sy_atom_codes = s_atom_codes * y_alphabet + y_atom_codes

    sb_parameters = np.bincount(sb_atom_codes, weights=full_parameters)
    sy_parameters = np.bincount(sy_atom_codes, weights=full_parameters)
    s_parameters = np.bincount(s_atom_codes, weights=full_parameters)
    h0_jeffreys = _dirichlet_expected_entropy(
        sy_parameters
    ) - _dirichlet_expected_entropy(s_parameters)
    hb_jeffreys = _dirichlet_expected_entropy(
        full_parameters
    ) - _dirichlet_expected_entropy(sb_parameters)
    return {
        "plugin": (float(h0_plugin), float(hb_plugin)),
        "jeffreys_dirichlet": (
            float(h0_jeffreys),
            float(hb_jeffreys),
        ),
    }


def _estimate_all(
    target: np.ndarray,
    baseline: np.ndarray,
    bicycle: np.ndarray,
    *,
    alpha: float,
    interpolator: BoundInterpolator,
) -> list[dict[str, float | str]]:
    pair = _plugin_and_jeffreys(
        target, baseline, bicycle, alpha=alpha
    )
    h0_mm, hb_mm, _ = conditional_information(target, baseline, bicycle)
    pair["miller_madow"] = (float(h0_mm), float(hb_mm))
    rows = []
    for estimator in ESTIMATORS:
        h0, hb = pair[estimator]
        rows.append(
            {
                "estimator": estimator,
                "h0_bits": h0,
                "hb_bits": hb,
                "delta_h_raw_bits": h0 - hb,
                "delta_l_exact_mse": float(
                    _bound_reduction(h0, hb, interpolator)
                ),
            }
        )
    return rows


def _e10_compatible_config(config: dict[str, object]) -> dict[str, object]:
    randomization = config["randomization"]
    return {
        "random_seed": int(config["random_seed"]),
        "frozen_null_matrix": {
            "repetitions": int(randomization["repetitions"]),
            "temporal_block_permutation": {
                "block_days": int(randomization["temporal_block_days"])
            },
            "spatial_series_permutation": {},
            "circular_shift_surrogate": {
                "minimum_absolute_shift_days": int(
                    randomization["circular_minimum_shift_days"]
                )
            },
        },
    }


def _summarize(
    replicates: pd.DataFrame, observed: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    keys = ["city", "null_design", "estimator"]
    for key, group in replicates.groupby(keys, observed=True, sort=False):
        city, design, estimator = key
        values = group["delta_h_raw_bits"].to_numpy(float)
        point = observed.loc[
            observed["city"].eq(city)
            & observed["null_design"].eq(design)
            & observed["estimator"].eq(estimator)
        ].iloc[0]
        rows.append(
            {
                **point.to_dict(),
                "null_mean_delta_h_bits": float(values.mean()),
                "null_sd_delta_h_bits": float(values.std(ddof=1)),
                "null_q025_delta_h_bits": float(np.quantile(values, 0.025)),
                "null_q975_delta_h_bits": float(np.quantile(values, 0.975)),
                "observed_minus_null_mean_bits": float(
                    point["observed_delta_h_bits"] - values.mean()
                ),
                "standardized_separation": float(
                    (point["observed_delta_h_bits"] - values.mean())
                    / values.std(ddof=1)
                ),
                "randomization_p_value": randomization_p_value(
                    float(point["observed_delta_h_bits"]), values
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_within_estimator"] = np.nan
    for estimator, indices in result.groupby(
        "estimator", observed=True
    ).groups.items():
        result.loc[indices, "q_value_within_estimator"] = benjamini_hochberg(
            result.loc[indices, "randomization_p_value"]
        )
    result["q_value_global_27"] = benjamini_hochberg(
        result["randomization_p_value"]
    )
    result["separates_within_estimator"] = result[
        "q_value_within_estimator"
    ].le(0.05)
    result["separates_global_27"] = result["q_value_global_27"].le(0.05)
    result["q_value_bh"] = result["q_value_within_estimator"]
    result["separates_from_null"] = result["separates_global_27"]
    return result.sort_values(keys).reset_index(drop=True)


def _e09_summary(e09: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        e09.groupby(["city", "estimator"], observed=True)
        .agg(
            specifications=("spec_id", "nunique"),
            positive_delb_reductions=(
                "delta_l_exact_mse",
                lambda x: int(np.count_nonzero(np.asarray(x) > 0)),
            ),
            minimum_delta_l_mse=("delta_l_exact_mse", "min"),
            maximum_delta_l_mse=("delta_l_exact_mse", "max"),
        )
        .reset_index()
    )
    primary = e09.loc[
        e09["spec_id"].eq("primary"),
        [
            "city",
            "city_label",
            "estimator",
            "h0_bits",
            "hb_bits",
            "delta_h_raw_bits",
            "l0_exact_mse",
            "lb_exact_mse",
            "delta_l_exact_mse",
        ],
    ].copy()
    return counts, primary


def _domain_summary(e04: pd.DataFrame) -> pd.DataFrame:
    pooled = e04.loc[
        e04["period"].eq("pooled_2020_2022"),
        [
            "city",
            "city_label",
            "domain",
            "domain_label",
            "grids",
            "sample_size",
            "h0_miller_madow_bits",
            "hb_miller_madow_bits",
            "delta_h_miller_madow_raw_bits",
            "l0_exact_mse",
            "lb_exact_mse",
            "delta_l_exact_mse",
            "relative_l_reduction",
        ],
    ].copy()
    wide = pooled.pivot(
        index=["city", "city_label"],
        columns="domain",
        values=[
            "grids",
            "sample_size",
            "delta_h_miller_madow_raw_bits",
            "delta_l_exact_mse",
            "relative_l_reduction",
        ],
    )
    wide.columns = [f"{metric}__{domain}" for metric, domain in wide.columns]
    comparison = wide.reset_index()
    comparison["complete_to_service_cmi_ratio"] = (
        comparison[
            "delta_h_miller_madow_raw_bits__complete_crime_domain"
        ]
        / comparison[
            "delta_h_miller_madow_raw_bits__bike_covered_training"
        ]
    )
    comparison["complete_to_service_delb_ratio"] = (
        comparison["delta_l_exact_mse__complete_crime_domain"]
        / comparison["delta_l_exact_mse__bike_covered_training"]
    )
    return pooled.sort_values(["city", "domain"]), comparison


def _plot_estimator_sensitivity(
    summary: pd.DataFrame, output: Path, dpi: int
) -> None:
    labels = {
        "plugin": "Plug-in",
        "miller_madow": "Miller–Madow",
        "jeffreys_dirichlet": "Jeffreys–Dirichlet",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), sharey=True)
    x = np.arange(len(NULL_ORDER), dtype=float)
    offsets = {"DC": -0.22, "NY": 0.0, "VAN": 0.22}
    for axis, estimator in zip(axes, ESTIMATORS, strict=True):
        subset = summary.loc[summary["estimator"].eq(estimator)]
        for city in CITY_ORDER:
            city_rows = subset.loc[subset["city"].eq(city)].set_index(
                "null_design"
            ).loc[NULL_ORDER]
            axis.plot(
                x + offsets[city],
                city_rows["observed_minus_null_mean_bits"],
                marker="o",
                linewidth=1.2,
                color=CITY_COLORS[city],
                label=CITY_LABELS[city],
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["Temporal", "Spatial", "Circular"], rotation=25, ha="right"
        )
        axis.set_title(labels[estimator])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Observed CMI - null mean (bits)")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    normalize_chart_typography(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in [".pdf", ".svg"]:
        fig.savefig(output.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(
        output.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


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
        f"pyarrow={pyarrow.__version__}",
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


def run(config_path: Path, repetitions_override: int | None = None) -> Path:
    root = Path.cwd().resolve()
    empirical = root / "empirical"
    config = _load_yaml(config_path)
    if repetitions_override is not None:
        config["randomization"]["repetitions"] = int(repetitions_override)

    started = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_id = started.strftime("%Y%m%d_%H%M%S_E17_estimator_domain_sensitivity")
    run_dir = empirical / "runs" / run_id
    for directory in ["tables", "figures", "logs"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    clock = time.monotonic()

    e02 = root / str(config["source_e02_data"])
    e04_run = root / str(config["source_e04_run"])
    e09_run = root / str(config["source_e09_run"])
    e10_run = root / str(config["source_e10_run"])
    source_files = {
        "config": config_path.resolve(),
        "e04_estimates": e04_run / "tables" / "city_information_estimates.csv",
        "e09_matrix": e09_run / "tables" / "theory_robustness_matrix.csv",
        "e10_replicates": e10_run
        / "tables"
        / "city_null_replicates.parquet",
        "e10_summary": e10_run / "tables" / "city_null_summary.csv",
    }
    for path in [e02, e04_run, e09_run, e10_run, *source_files.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    e10_config = _e10_compatible_config(config)
    interpolator = BoundInterpolator.build(
        float(config["numerics"]["bound_interpolation_max_entropy_bits"]),
        tail_tolerance=float(config["numerics"]["lattice_tail_tolerance"]),
        root_tolerance=float(config["numerics"]["root_tolerance"]),
    )
    alpha = float(config["estimation"]["jeffreys_alpha"])
    repetitions = int(config["randomization"]["repetitions"])
    replicate_parts: list[pd.DataFrame] = []
    observed_rows: list[dict[str, object]] = []

    for city in CITY_ORDER:
        panel = _load_panel(e02, city)
        prepared, _ = _prepare_spec(panel, "primary")
        view = _matrix_view(
            prepared,
            baseline_state=list(config["information_sets"]["baseline_state"]),
        )
        target = np.asarray(view["target"], dtype=np.int64)
        baseline = np.asarray(view["baseline"], dtype=np.int64)
        observed_matrix = np.asarray(view["bicycle_matrix"], dtype=np.int16)
        observed_estimates = _estimate_all(
            target,
            baseline,
            observed_matrix.ravel(),
            alpha=alpha,
            interpolator=interpolator,
        )
        for design in config["randomization"]["designs"]:
            for item in observed_estimates:
                observed_rows.append(
                    {
                        "city": city,
                        "city_label": CITY_LABELS[city],
                        "null_design": design,
                        "estimator": item["estimator"],
                        "observed_h0_bits": item["h0_bits"],
                        "observed_hb_bits": item["hb_bits"],
                        "observed_delta_h_bits": item["delta_h_raw_bits"],
                        "observed_delta_l_mse": item["delta_l_exact_mse"],
                    }
                )
            rng = np.random.default_rng(
                _seed(
                    int(config["random_seed"]),
                    f"E10|city|{city}|{design}",
                )
            )
            rows = []
            for replicate in range(1, repetitions + 1):
                null_matrix = _null_bicycle_matrix(
                    str(design), observed_matrix, rng, e10_config
                )
                for item in _estimate_all(
                    target,
                    baseline,
                    null_matrix.ravel(),
                    alpha=alpha,
                    interpolator=interpolator,
                ):
                    rows.append(
                        {
                            "city": city,
                            "city_label": CITY_LABELS[city],
                            "null_design": design,
                            "replicate": replicate,
                            **item,
                        }
                    )
            replicate_parts.append(pd.DataFrame(rows))

    replicates = pd.concat(replicate_parts, ignore_index=True)
    observed = pd.DataFrame(observed_rows)
    summary = _summarize(replicates, observed)

    accepted_e10 = pd.read_parquet(source_files["e10_replicates"])
    mm = replicates.loc[
        replicates["estimator"].eq("miller_madow"),
        ["city", "null_design", "replicate", "delta_h_raw_bits"],
    ].rename(columns={"delta_h_raw_bits": "e17_delta_h_bits"})
    reconciliation = mm.merge(
        accepted_e10[
            ["city", "null_design", "replicate", "delta_h_raw_bits"]
        ].rename(columns={"delta_h_raw_bits": "e10_delta_h_bits"}),
        on=["city", "null_design", "replicate"],
        validate="one_to_one",
    )
    reconciliation["absolute_difference_bits"] = (
        reconciliation["e17_delta_h_bits"]
        - reconciliation["e10_delta_h_bits"]
    ).abs()

    e09 = pd.read_csv(source_files["e09_matrix"])
    e09_counts, e09_primary = _e09_summary(e09)
    e04 = pd.read_csv(source_files["e04_estimates"])
    domain_long, domain_comparison = _domain_summary(e04)

    expected_rows = int(
        config["acceptance"]["expected_replicates_per_estimator"]
    )
    production = repetitions == int(
        _load_yaml(config_path)["randomization"]["repetitions"]
    )
    checks = [
        {
            "check": "Three estimators are present",
            "passed": set(replicates["estimator"]) == set(ESTIMATORS),
            "detail": sorted(replicates["estimator"].unique()),
        },
        {
            "check": "Each estimator has the expected production row count",
            "passed": (not production)
            or replicates.groupby("estimator").size().eq(expected_rows).all(),
            "detail": replicates.groupby("estimator").size().to_dict(),
        },
        {
            "check": "Summary contains nine tests per estimator and 27 globally",
            "passed": len(summary) == 27
            and summary.groupby("estimator").size().eq(9).all(),
            "detail": summary.groupby("estimator").size().to_dict(),
        },
        {
            "check": "Miller--Madow reproduces E10",
            "passed": float(reconciliation["absolute_difference_bits"].max())
            <= float(
                config["numerics"]["e10_reconciliation_tolerance_bits"]
            ),
            "detail": float(reconciliation["absolute_difference_bits"].max()),
        },
        {
            "check": "All E09 city-estimator groups contain 16 specifications",
            "passed": e09_counts["specifications"]
            .eq(
                int(
                    config["acceptance"][
                        "expected_e09_specs_per_city_estimator"
                    ]
                )
            )
            .all(),
            "detail": e09_counts["specifications"].tolist(),
        },
        {
            "check": "All E09 DELB reductions are positive",
            "passed": e09_counts["positive_delb_reductions"]
            .eq(e09_counts["specifications"])
            .all(),
            "detail": e09_counts[
                ["city", "estimator", "positive_delb_reductions"]
            ].to_dict("records"),
        },
        {
            "check": "Both fixed domains are present for every city",
            "passed": domain_long.groupby("city")["domain"].nunique().eq(2).all(),
            "detail": domain_long.groupby("city")["domain"].nunique().to_dict(),
        },
    ]
    acceptance = pd.DataFrame(checks)
    if not acceptance["passed"].all():
        raise AssertionError(
            acceptance.loc[~acceptance["passed"]].to_dict("records")
        )

    replicates.to_parquet(
        run_dir / "tables" / "estimator_null_replicates.parquet",
        index=False,
        compression="zstd",
    )
    for name, frame in [
        ("estimator_null_summary.csv", summary),
        ("e10_reconciliation.csv", reconciliation),
        ("e09_estimator_specification_summary.csv", e09_counts),
        ("e09_primary_estimator_sensitivity.csv", e09_primary),
        ("fixed_domain_results.csv", domain_long),
        ("fixed_domain_comparison.csv", domain_comparison),
        ("acceptance_checklist.csv", acceptance),
    ]:
        frame.to_csv(run_dir / "tables" / name, index=False)
    _plot_estimator_sensitivity(
        summary,
        run_dir / "figures" / "e17_estimator_null_sensitivity",
        int(config["reporting"]["figure_dpi"]),
    )

    source_manifest = pd.DataFrame(
        [
            {
                "source": name,
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in source_files.items()
            if path.is_file()
        ]
    )
    source_manifest.to_csv(
        run_dir / "tables" / "input_manifest.csv", index=False
    )
    figure_manifest = pd.DataFrame(
        [
            {
                "file": path.name,
                "sha256": _sha256(path),
            }
            for path in sorted((run_dir / "figures").iterdir())
        ]
    )
    figure_manifest.to_csv(
        run_dir / "tables" / "figure_manifest.csv", index=False
    )

    significant_within = summary.groupby(
        "estimator", observed=True
    )["separates_within_estimator"].sum().to_dict()
    significant_global = summary.groupby(
        "estimator", observed=True
    )["separates_global_27"].sum().to_dict()
    interpretation = [
        "# E17 estimator and fixed-domain sensitivity",
        "",
        "The E09 point-estimate sign was positive in all 16 prespecified "
        "specifications for every city and each of the plug-in, "
        "Miller--Madow, and coherent observed-support Jeffreys--Dirichlet "
        "estimators.",
        "",
        "City-level randomization results were recalculated on identical "
        "surrogates. The number of FDR-adjusted separations under the "
        f"within-estimator nine-test correction was: {significant_within}; "
        "under the global 27-test correction it was: "
        f"{significant_global}. A nonzero null mean "
        "is design- and estimator-conditioned; it is not population CMI.",
        "",
        "The fixed-domain comparison uses the training-period-frozen bicycle "
        "service domain and the complete training crime-supported domain. "
        "Differences between them describe changes in the target spatial "
        "population and are not causal bicycle effects.",
    ]
    (run_dir / "interpretation.md").write_text(
        "\n".join(interpretation) + "\n", encoding="utf-8"
    )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text(
        f"{sys.executable} empirical/run_e17_estimator_domain_sensitivity.py "
        f"--config {config_path}\n",
        encoding="utf-8",
    )
    _write_environment(run_dir / "environment.txt")

    completed = datetime.now(ZoneInfo("Asia/Shanghai"))
    status = {
        "experiment": "E17",
        "status": "complete",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": time.monotonic() - clock,
        "repetitions_per_city_design": repetitions,
        "replicate_rows": len(replicates),
        "checks_passed": int(acceptance["passed"].sum()),
        "checks_total": len(acceptance),
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    if production:
        (empirical / "runs" / "latest_e17_run.txt").write_text(
            str(run_dir) + "\n", encoding="utf-8"
        )
    return run_dir


def finalize_existing_run(config_path: Path, run_dir: Path) -> Path:
    """Apply current summary and multiplicity rules without regenerating surrogates."""
    config = _load_yaml(config_path)
    replicates = pd.read_parquet(
        run_dir / "tables" / "estimator_null_replicates.parquet"
    )
    previous = pd.read_csv(
        run_dir / "tables" / "estimator_null_summary.csv"
    )
    observed_columns = [
        "city",
        "city_label",
        "null_design",
        "estimator",
        "observed_h0_bits",
        "observed_hb_bits",
        "observed_delta_h_bits",
        "observed_delta_l_mse",
    ]
    observed = previous[observed_columns].drop_duplicates()
    summary = _summarize(replicates, observed)
    summary.to_csv(
        run_dir / "tables" / "estimator_null_summary.csv", index=False
    )

    acceptance_path = run_dir / "tables" / "acceptance_checklist.csv"
    acceptance = pd.read_csv(acceptance_path)
    acceptance = acceptance.loc[
        ~acceptance["check"].eq(
            "Summary contains nine tests per estimator and 27 globally"
        )
    ]
    acceptance = pd.concat(
        [
            acceptance,
            pd.DataFrame(
                [
                    {
                        "check": (
                            "Summary contains nine tests per estimator "
                            "and 27 globally"
                        ),
                        "passed": len(summary) == 27
                        and summary.groupby("estimator")
                        .size()
                        .eq(9)
                        .all(),
                        "detail": json.dumps(
                            summary.groupby("estimator").size().to_dict()
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    acceptance.to_csv(acceptance_path, index=False)
    if not acceptance["passed"].astype(bool).all():
        raise AssertionError("E17 finalization acceptance failed.")

    significant_within = summary.groupby(
        "estimator", observed=True
    )["separates_within_estimator"].sum().to_dict()
    significant_global = summary.groupby(
        "estimator", observed=True
    )["separates_global_27"].sum().to_dict()
    (run_dir / "interpretation.md").write_text(
        "# E17 estimator and training-period-frozen analysis-domain sensitivity\n\n"
        "The E09 point-estimate sign was positive in all 16 prespecified "
        "specifications for every city and each estimator.\n\n"
        "City-level randomization results were recalculated on identical "
        "surrogates. FDR-adjusted separations under the within-estimator "
        f"nine-test correction were {significant_within}; under the global "
        f"27-test correction they were {significant_global}. A nonzero null "
        "mean is design- and estimator-conditioned; it is not population "
        "CMI.\n\n"
        "The analysis-domain comparison uses the training-period-frozen "
        "bicycle service domain and the complete training crime-supported "
        "domain. Differences describe target-population changes and are not "
        "causal bicycle effects.\n",
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with (run_dir / "command.txt").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{sys.executable} empirical/run_e17_estimator_domain_sensitivity.py "
            f"--config {config_path} --finalize-run {run_dir}\n"
        )
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["summary_finalized_at"] = datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).isoformat()
    status["multiplicity_families"] = [
        "within_estimator_nine",
        "global_twenty_seven",
    ]
    status["checks_passed"] = int(acceptance["passed"].astype(bool).sum())
    status["checks_total"] = len(acceptance)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run city-level entropy-estimator and fixed-domain sensitivity."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("empirical/config/e17.yaml")
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Development override; production uses the YAML value.",
    )
    parser.add_argument(
        "--finalize-run",
        type=Path,
        help="Recompute summaries and multiplicity corrections from an existing run.",
    )
    args = parser.parse_args()
    run_dir = (
        finalize_existing_run(args.config, args.finalize_run.resolve())
        if args.finalize_run
        else run(args.config, args.repetitions)
    )
    print(run_dir)


if __name__ == "__main__":
    main()
