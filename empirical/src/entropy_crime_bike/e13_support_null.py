from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import statsmodels
import statsmodels.formula.api as smf
import yaml

from .e06_predictive_models import (
    CITY_COLORS,
    CITY_LABELS,
    CITY_ORDER,
)
from .e09_robustness import _load_panel, _prepare_spec
from .e10_placebo import (
    NULL_LABELS,
    NULL_ORDER,
    _local_components,
    _matrix_view,
    _null_bicycle_matrix,
    _seed,
)


LOG_TWO = math.log(2.0)
SUPPORT_METRICS = (
    "support_atoms_per_observation",
    "singleton_observation_share",
    "effective_df_per_observation",
)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def _read_pointer(root: Path, relative: str) -> Path:
    pointer = root / relative
    result = Path(pointer.read_text(encoding="utf-8").strip())
    if not result.is_absolute():
        relative_to_pointer = pointer.parent / result
        result = (
            relative_to_pointer
            if relative_to_pointer.exists()
            else root / result
        )
    result = result.resolve()
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def _entropy_rows(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    rows = values.reshape(values.shape[0], -1)
    totals = rows.sum(axis=1).astype(float)
    if np.any(totals <= 0):
        raise ValueError("Each entropy row must contain observations.")
    positive = np.where(rows > 0, rows, 1)
    return np.log2(totals) - (
        np.sum(rows * np.log2(positive), axis=1) / totals
    )


def panel_cmi_support(
    target_matrix: np.ndarray,
    baseline_matrix: np.ndarray,
    bicycle_matrix: np.ndarray,
) -> pd.DataFrame:
    """Compute Miller--Madow CMI and support diagnostics for many series.

    Each row of the three aligned matrices is one city-grid series. Baseline
    codes may be local to a row because the grid index is included in the
    dense count key.
    """
    y = np.asarray(target_matrix, dtype=np.int64)
    z = np.asarray(baseline_matrix, dtype=np.int64)
    b = np.asarray(bicycle_matrix, dtype=np.int64)
    if not (y.shape == z.shape == b.shape) or y.ndim != 2:
        raise ValueError("Target, baseline, and bicycle matrices must align.")
    if np.any(y < 0) or np.any(z < 0) or np.any(b < 0):
        raise ValueError("All information-state codes must be nonnegative.")
    grids, observations = y.shape
    y_count = int(y.max()) + 1
    z_count = int(z.max()) + 1
    b_count = int(b.max()) + 1
    grid = np.arange(grids, dtype=np.int64)[:, None]
    grid_state = grid * z_count + z
    atom = ((grid_state * b_count + b) * y_count + y).ravel()
    counts = np.bincount(
        atom,
        minlength=grids * z_count * b_count * y_count,
    ).reshape(grids, z_count, b_count, y_count)
    counts_zb = counts.sum(axis=3)
    counts_zy = counts.sum(axis=2)
    counts_z = counts_zy.sum(axis=2)

    h_yz = _entropy_rows(counts_zy)
    h_z = _entropy_rows(counts_z)
    h_yzb = _entropy_rows(counts)
    h_zb = _entropy_rows(counts_zb)
    k_yz = np.count_nonzero(counts_zy, axis=(1, 2))
    k_z = np.count_nonzero(counts_z, axis=1)
    k_yzb = np.count_nonzero(counts, axis=(1, 2, 3))
    k_zb = np.count_nonzero(counts_zb, axis=(1, 2))
    h0_plugin = h_yz - h_z
    hb_plugin = h_yzb - h_zb
    h0_mm = h0_plugin + (k_yz - k_z) / (
        2.0 * observations * LOG_TWO
    )
    hb_mm = hb_plugin + (k_yzb - k_zb) / (
        2.0 * observations * LOG_TWO
    )

    target_states_by_z = np.count_nonzero(counts_zy, axis=2)
    bicycle_states_by_z = np.count_nonzero(counts_zb, axis=2)
    observed_z = counts_z > 0
    effective_df = np.sum(
        np.where(
            observed_z,
            (target_states_by_z - 1) * (bicycle_states_by_z - 1),
            0,
        ),
        axis=1,
    )
    singleton_atoms = np.sum(counts == 1, axis=(1, 2, 3))
    return pd.DataFrame(
        {
            "observations": observations,
            "support_atoms": k_yzb,
            "support_atoms_per_observation": k_yzb / observations,
            "singleton_atom_count": singleton_atoms,
            "singleton_observation_share": singleton_atoms / observations,
            "effective_df": effective_df,
            "effective_df_per_observation": effective_df / observations,
            "h0_miller_madow_bits": h0_mm,
            "hb_miller_madow_bits": hb_mm,
            "delta_h_raw_bits": h0_mm - hb_mm,
        }
    )


def _city_matrices(view: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(view["target"], dtype=np.int64)[None, :],
        np.asarray(view["baseline"], dtype=np.int64)[None, :],
    )


def _local_matrices(
    components: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = components["groups"]
    return (
        np.vstack([group["target"] for group in groups]),
        np.vstack([group["baseline"] for group in groups]),
        [str(group["grid_id"]) for group in groups],
    )


def _surrogate_records(
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    observed_bicycle: np.ndarray,
    city: str,
    design: str,
    level: str,
    grid_ids: list[str],
    e10_config: dict[str, object],
    repetitions: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        _seed(
            int(e10_config["random_seed"]),
            f"E10|{level}|{city}|{design}",
        )
    )
    records: list[pd.DataFrame] = []
    for repetition in range(1, repetitions + 1):
        surrogate = _null_bicycle_matrix(
            design, observed_bicycle, rng, e10_config
        )
        metrics = panel_cmi_support(
            target, baseline, surrogate.reshape(target.shape)
        )
        metrics.insert(0, "replicate", repetition)
        metrics.insert(0, "null_design", design)
        metrics.insert(0, "grid_id", grid_ids)
        metrics.insert(0, "city", city)
        records.append(metrics)
    return pd.concat(records, ignore_index=True)


def _observed_records(
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    bicycle: np.ndarray,
    city: str,
    grid_ids: list[str],
) -> pd.DataFrame:
    result = panel_cmi_support(target, baseline, bicycle)
    result.insert(0, "grid_id", grid_ids)
    result.insert(0, "city", city)
    return result


def _add_source_reconciliation(
    generated: pd.DataFrame,
    source: pd.DataFrame,
    keys: list[str],
) -> tuple[pd.DataFrame, float]:
    reference = source[keys + ["delta_h_raw_bits"]].rename(
        columns={"delta_h_raw_bits": "e10_delta_h_raw_bits"}
    )
    merged = generated.merge(
        reference, on=keys, how="left", validate="one_to_one"
    )
    if merged["e10_delta_h_raw_bits"].isna().any():
        raise AssertionError("E10 reconciliation produced missing rows.")
    merged["cmi_reconciliation_difference_bits"] = (
        merged["delta_h_raw_bits"] - merged["e10_delta_h_raw_bits"]
    )
    return (
        merged,
        float(merged["cmi_reconciliation_difference_bits"].abs().max()),
    )


def _summarize_support(
    surrogate: pd.DataFrame,
    observed: pd.DataFrame,
    e10_summary: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "null_mean_recomputed_bits": ("delta_h_raw_bits", "mean"),
    }
    for metric in SUPPORT_METRICS:
        aggregations[f"mean_null_{metric}"] = (metric, "mean")
        aggregations[f"sd_null_{metric}"] = (metric, "std")
    summary = (
        surrogate.groupby(keys, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    observed_columns = keys[:-1] + [
        "delta_h_raw_bits",
        *SUPPORT_METRICS,
    ]
    observed_unique = observed[observed_columns].drop_duplicates(keys[:-1])
    observed_unique = observed_unique.rename(
        columns={
            "delta_h_raw_bits": "observed_delta_h_recomputed_bits",
            **{
                metric: f"observed_{metric}"
                for metric in SUPPORT_METRICS
            },
        }
    )
    summary = summary.merge(
        observed_unique, on=keys[:-1], how="left", validate="many_to_one"
    )
    source_columns = keys + [
        "observed_delta_h_bits",
        "null_mean_delta_h_bits",
        "observed_minus_null_mean_bits",
    ]
    summary = summary.merge(
        e10_summary[source_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    for metric in SUPPORT_METRICS:
        summary[f"null_minus_observed_{metric}"] = (
            summary[f"mean_null_{metric}"]
            - summary[f"observed_{metric}"]
        )
    summary["null_mean_reconciliation_difference_bits"] = (
        summary["null_mean_recomputed_bits"]
        - summary["null_mean_delta_h_bits"]
    )
    return summary


def _spearman_row(
    *,
    source: str,
    outcome: str,
    predictor: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    rho, p_value = spearmanr(frame[predictor], frame[outcome])
    return {
        "source": source,
        "outcome": outcome,
        "predictor": predictor,
        "observations": len(frame),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
    }


def _relationship_tables(
    local_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    e12_run: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    e12 = pd.read_csv(e12_run / "tables" / "estimator_summary.csv")
    simulation_null = e12.loc[e12["effect_strength"].eq(0)].copy()
    for estimator in ["plugin", "miller_madow"]:
        for metric in SUPPORT_METRICS:
            rows.append(
                _spearman_row(
                    source="E12_population_null",
                    outcome=f"bias_{estimator}_bits",
                    predictor=f"mean_{metric}",
                    frame=simulation_null,
                )
            )
    for level, frame in [
        ("E13_empirical_local", local_summary),
        ("E13_empirical_city", city_summary),
    ]:
        for metric in SUPPORT_METRICS:
            rows.append(
                _spearman_row(
                    source=level,
                    outcome="null_mean_delta_h_bits",
                    predictor=f"mean_null_{metric}",
                    frame=frame,
                )
            )
            rows.append(
                _spearman_row(
                    source=level,
                    outcome="observed_minus_null_mean_bits",
                    predictor=f"null_minus_observed_{metric}",
                    frame=frame,
                )
            )
    spearman = pd.DataFrame(rows)

    regression_frame = local_summary.copy()
    regression_frame["cluster_id"] = (
        regression_frame["city"].astype(str)
        + ":"
        + regression_frame["grid_id"].astype(str)
    )
    regression_rows: list[dict[str, object]] = []
    specifications = []
    for metric in SUPPORT_METRICS:
        specifications.append(
            (
                f"null_mean_univariate_{metric}",
                "null_mean_delta_h_bits",
                [f"mean_null_{metric}"],
            )
        )
        specifications.append(
            (
                f"observed_minus_null_univariate_{metric}",
                "observed_minus_null_mean_bits",
                [f"null_minus_observed_{metric}"],
            )
        )
    specifications.extend(
        [
            (
                "null_mean_combined",
                "null_mean_delta_h_bits",
                [f"mean_null_{metric}" for metric in SUPPORT_METRICS],
            ),
            (
                "observed_minus_null_combined",
                "observed_minus_null_mean_bits",
                [
                    f"null_minus_observed_{metric}"
                    for metric in SUPPORT_METRICS
                ],
            ),
        ]
    )
    for model_id, outcome, predictors in specifications:
        formula = (
            outcome
            + " ~ "
            + " + ".join(predictors)
            + " + C(city) + C(null_design)"
        )
        fitted = smf.ols(formula, data=regression_frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": regression_frame["cluster_id"]},
        )
        for predictor in predictors:
            regression_rows.append(
                {
                    "model_id": model_id,
                    "outcome": outcome,
                    "predictor": predictor,
                    "coefficient": float(fitted.params[predictor]),
                    "clustered_standard_error": float(fitted.bse[predictor]),
                    "p_value": float(fitted.pvalues[predictor]),
                    "observations": int(fitted.nobs),
                    "clusters": regression_frame["cluster_id"].nunique(),
                    "city_fixed_effects": True,
                    "null_design_fixed_effects": True,
                    "r_squared": float(fitted.rsquared),
                }
            )
    return spearman, pd.DataFrame(regression_rows)


def _nine_result_table(city_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "city",
        "null_design",
        "observed_delta_h_bits",
        "null_mean_delta_h_bits",
        "observed_minus_null_mean_bits",
    ]
    result = city_summary[columns].copy()
    for metric in SUPPORT_METRICS:
        result[f"observed_{metric}"] = city_summary[
            f"observed_{metric}"
        ]
        result[f"null_mean_{metric}"] = city_summary[
            f"mean_null_{metric}"
        ]
        result[f"null_minus_observed_{metric}"] = city_summary[
            f"null_minus_observed_{metric}"
        ]
    result["interpretation"] = np.where(
        result["observed_minus_null_mean_bits"] < 0,
        "Observed CMI below transformation-specific null mean",
        "Observed CMI at or above transformation-specific null mean",
    )
    return result


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_results(
    local_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    e12_run: Path,
    figure_dir: Path,
    dpi: int,
) -> None:
    _publication_style()
    e12 = pd.read_csv(e12_run / "tables" / "estimator_summary.csv")
    e12 = e12.loc[e12["effect_strength"].eq(0)]
    markers = {
        "temporal_block_permutation": "o",
        "spatial_series_permutation": "s",
        "circular_shift_surrogate": "^",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    axes[0, 0].scatter(
        e12["mean_singleton_observation_share"],
        e12["bias_plugin_bits"],
        s=23,
        alpha=0.65,
        color="#D55E00",
        label="Plugin",
    )
    axes[0, 0].scatter(
        e12["mean_singleton_observation_share"],
        e12["bias_miller_madow_bits"],
        s=23,
        alpha=0.65,
        color="#0072B2",
        label="Miller--Madow",
    )
    axes[0, 0].set(
        title="(a) Simulation: singleton occupancy and null bias",
        xlabel="Singleton-observation share",
        ylabel="CMI bias (bits)",
    )
    axes[0, 0].legend(frameon=False)

    for city in CITY_ORDER:
        for design in NULL_ORDER:
            subset = local_summary.loc[
                local_summary["city"].eq(city)
                & local_summary["null_design"].eq(design)
            ]
            axes[0, 1].scatter(
                subset["mean_null_singleton_observation_share"],
                subset["null_mean_delta_h_bits"],
                color=CITY_COLORS[city],
                marker=markers[design],
                s=22,
                alpha=0.55,
            )
    axes[0, 1].set(
        title="(b) Empirical reference center and singleton occupancy",
        xlabel="Mean surrogate singleton-observation share",
        ylabel="Reference mean (bits)",
    )
    city_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=CITY_COLORS[city],
            label=CITY_LABELS[city],
            markersize=5,
        )
        for city in CITY_ORDER
    ]
    design_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[design],
            linestyle="",
            color="0.35",
            label=NULL_LABELS[design],
            markersize=5,
        )
        for design in NULL_ORDER
    ]
    axes[0, 1].legend(
        handles=city_handles + design_handles,
        frameon=False,
        fontsize=6.5,
        ncol=2,
        loc="upper left",
    )

    for city in CITY_ORDER:
        for design in NULL_ORDER:
            subset = local_summary.loc[
                local_summary["city"].eq(city)
                & local_summary["null_design"].eq(design)
            ]
            axes[1, 0].scatter(
                subset[
                    "null_minus_observed_singleton_observation_share"
                ],
                subset["observed_minus_null_mean_bits"],
                color=CITY_COLORS[city],
                marker=markers[design],
                s=22,
                alpha=0.55,
            )
    axes[1, 0].axhline(0, color="0.3", linewidth=0.8)
    axes[1, 0].axvline(0, color="0.3", linewidth=0.8)
    axes[1, 0].set(
        title="(c) Support change and observed-minus-null CMI",
        xlabel="Null mean minus observed singleton share",
        ylabel="Observed CMI minus null mean (bits)",
    )

    positions = np.arange(len(city_summary))
    ordered = city_summary.sort_values(["city", "null_design"]).reset_index(
        drop=True
    )
    axes[1, 1].bar(
        positions - 0.18,
        ordered["observed_delta_h_bits"],
        width=0.36,
        color="#4C78A8",
        label="Observed",
    )
    axes[1, 1].bar(
        positions + 0.18,
        ordered["null_mean_delta_h_bits"],
        width=0.36,
        color="#B0B0B0",
        label="Null mean",
    )
    design_codes = {
        "circular_shift_surrogate": "C",
        "spatial_series_permutation": "S",
        "temporal_block_permutation": "T",
    }
    labels = [
        f"{row.city}\n{design_codes[row.null_design]}"
        for row in ordered.itertuples(index=False)
    ]
    axes[1, 1].set_xticks(positions, labels, fontsize=6.5)
    axes[1, 1].set(
        title="(d) Nine city-level randomization comparisons",
        xlabel="C = circular; S = spatial; T = temporal blocks",
        ylabel="Conditional information (bits)",
    )
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel():
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "e13_support_null_main", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for city in CITY_ORDER:
        subset = local_summary.loc[local_summary["city"].eq(city)]
        axes[0].scatter(
            subset["mean_null_support_atoms_per_observation"],
            subset["null_mean_delta_h_bits"],
            color=CITY_COLORS[city],
            s=22,
            alpha=0.55,
            label=CITY_LABELS[city],
        )
        axes[1].scatter(
            subset["mean_null_effective_df_per_observation"],
            subset["null_mean_delta_h_bits"],
            color=CITY_COLORS[city],
            s=22,
            alpha=0.55,
        )
    axes[0].set(
        title="(a) Observed support atoms",
        xlabel=r"Mean surrogate $K_{\mathrm{obs}}/N$",
        ylabel="Reference mean (bits)",
    )
    axes[1].set(
        title="(b) Effective conditional degrees of freedom",
        xlabel=r"Mean surrogate $\nu/N$",
        ylabel="Reference mean (bits)",
    )
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "e13_support_metrics_supplement", dpi)


def _interpretation(
    spearman: pd.DataFrame,
    regressions: pd.DataFrame,
    city_summary: pd.DataFrame,
) -> str:
    simulation = spearman.loc[
        spearman["source"].eq("E12_population_null")
        & spearman["predictor"].eq(
            "mean_singleton_observation_share"
        )
    ]
    empirical = spearman.loc[
        spearman["source"].eq("E13_empirical_local")
        & spearman["outcome"].eq("null_mean_delta_h_bits")
        & spearman["predictor"].eq(
            "mean_null_singleton_observation_share"
        )
    ].iloc[0]
    coefficient = regressions.loc[
        regressions["model_id"].eq(
            "null_mean_univariate_singleton_observation_share"
        )
    ].iloc[0]
    lines = [
        "# E13 Interpretation",
        "",
        "## Simulation mechanism",
        "",
    ]
    for row in simulation.itertuples(index=False):
        estimator = row.outcome.replace("bias_", "").replace("_bits", "")
        lines.append(
            f"- Under the E12 population null, singleton share and "
            f"{estimator} bias have Spearman rho={row.spearman_rho:.3f} "
            f"(p={row.p_value:.3g})."
        )
    lines.extend(
        [
            "",
            "## Empirical E10 association",
            "",
            f"- Across 1,017 city-grid-null-design summaries, surrogate "
            f"singleton share and the randomization-reference mean have Spearman "
            f"rho={empirical.spearman_rho:.3f} "
            f"(p={empirical.p_value:.3g}).",
            f"- In the univariate specification with city and null-design "
            f"fixed effects and city-grid clustered standard errors, the "
            f"singleton coefficient is {coefficient.coefficient:.6f} "
            f"(SE={coefficient.clustered_standard_error:.6f}, "
            f"p={coefficient.p_value:.3g}).",
            "- Spearman p-values are descriptive because each grid appears "
            "under three null designs; inferential emphasis is placed on the "
            "fixed-effect models with city-grid clustered standard errors.",
            "",
            "## Nine city-level comparisons",
            "",
        ]
    )
    for row in city_summary.sort_values(
        ["city", "null_design"]
    ).itertuples(index=False):
        lines.append(
            f"- {CITY_LABELS[row.city]}, {NULL_LABELS[row.null_design]}: "
            f"observed={row.observed_delta_h_bits:.6f}, "
            f"null mean={row.null_mean_delta_h_bits:.6f}, "
            f"observed-minus-null={row.observed_minus_null_mean_bits:.6f} bits; "
            f"surrogate-minus-observed singleton share="
            f"{row.null_minus_observed_singleton_observation_share:.6f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Support changes are associated with the finite-sample center of "
            "the E10 statistic, but these descriptive and fixed-effect "
            "relationships are not causal. The randomization-null mean is "
            "neither population CMI nor estimator bias alone: it also depends "
            "on the transformation-specific state structure. These results "
            "reinterpret E10's positive null centers without changing its "
            "upper-tail non-rejection or the mathematical DELB.",
            "",
        ]
    )
    return "\n".join(lines)


def _environment() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }


def run(config_path: Path, *, quick: bool = False) -> Path:
    start = time.perf_counter()
    config = _load_yaml(config_path)
    project = config_path.resolve().parents[2]
    empirical = project / "empirical"
    e02_data = _read_pointer(
        empirical, str(config["source_e02_data_pointer"])
    )
    e05_run = _read_pointer(
        empirical, str(config["source_e05_run_pointer"])
    )
    e10_run = _read_pointer(
        empirical, str(config["source_e10_run_pointer"])
    )
    e12_run = _read_pointer(
        empirical, str(config["source_e12_run_pointer"])
    )
    repetitions = (
        5 if quick else int(config["frozen_e10"]["repetitions"])
    )
    suffix = "_development" if quick else ""
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_E13_support_null"
        + suffix
    )
    run_dir = empirical / "runs" / run_id
    table_dir = run_dir / "tables"
    figure_dir = run_dir / "figures"
    state_dir = run_dir / "state"
    for path in [table_dir, figure_dir, state_dir]:
        path.mkdir(parents=True, exist_ok=True)

    e10_config = _load_yaml(e10_run / "config_snapshot.yaml")["e10"]
    eligibility = pd.read_csv(
        e05_run / "tables" / "grid_eligibility.csv"
    )
    source_city_null = pd.read_parquet(
        e10_run / "tables" / "city_null_replicates.parquet"
    )
    source_local_null = pd.read_parquet(
        e10_run / "tables" / "local_null_replicates.parquet"
    )
    source_city_summary = pd.read_csv(
        e10_run / "tables" / "city_null_summary.csv"
    )
    source_local_summary = pd.read_csv(
        e10_run / "tables" / "local_null_summary.csv"
    )

    city_parts: list[pd.DataFrame] = []
    local_parts: list[pd.DataFrame] = []
    city_observed_parts: list[pd.DataFrame] = []
    local_observed_parts: list[pd.DataFrame] = []
    reconciliation_rows: list[dict[str, object]] = []
    cities = CITY_ORDER[:1] if quick else CITY_ORDER
    designs = NULL_ORDER[:1] if quick else NULL_ORDER
    for city in cities:
        panel = _load_panel(e02_data, city)
        prepared, _ = _prepare_spec(panel, "primary")
        city_view = _matrix_view(
            prepared,
            baseline_state=list(
                config["information_sets"]["city_baseline_state"]
            ),
        )
        city_target, city_baseline = _city_matrices(city_view)
        city_bicycle = np.asarray(
            city_view["bicycle_matrix"], dtype=np.int16
        ).reshape(1, -1)
        city_observed = _observed_records(
            target=city_target,
            baseline=city_baseline,
            bicycle=city_bicycle,
            city=city,
            grid_ids=["__CITY__"],
        )
        city_observed_parts.append(city_observed)

        eligible_ids = sorted(
            eligibility.loc[
                eligibility["city"].eq(city) & eligibility["eligible"],
                "grid_id",
            ].astype(str)
        )
        components = _local_components(
            prepared,
            eligible_ids,
            list(config["information_sets"]["local_baseline_state"]),
        )
        local_target, local_baseline, grid_ids = _local_matrices(components)
        local_bicycle = np.asarray(
            components["bicycle_matrix"], dtype=np.int16
        )
        local_observed = _observed_records(
            target=local_target,
            baseline=local_baseline,
            bicycle=local_bicycle,
            city=city,
            grid_ids=grid_ids,
        )
        local_observed_parts.append(local_observed)

        for design in designs:
            city_generated = _surrogate_records(
                target=city_target,
                baseline=city_baseline,
                observed_bicycle=np.asarray(
                    city_view["bicycle_matrix"], dtype=np.int16
                ),
                city=city,
                design=design,
                level="city",
                grid_ids=["__CITY__"],
                e10_config=e10_config,
                repetitions=repetitions,
            )
            city_source = source_city_null.loc[
                source_city_null["city"].eq(city)
                & source_city_null["null_design"].eq(design)
                & source_city_null["replicate"].le(repetitions)
            ]
            city_generated, city_difference = _add_source_reconciliation(
                city_generated,
                city_source,
                ["city", "null_design", "replicate"],
            )
            city_generated.to_parquet(
                state_dir / f"{city}_{design}_city_support.parquet",
                index=False,
                compression=str(config["reporting"]["parquet_compression"]),
            )
            city_parts.append(city_generated)

            local_generated = _surrogate_records(
                target=local_target,
                baseline=local_baseline,
                observed_bicycle=local_bicycle,
                city=city,
                design=design,
                level="local",
                grid_ids=grid_ids,
                e10_config=e10_config,
                repetitions=repetitions,
            )
            local_source = source_local_null.loc[
                source_local_null["city"].eq(city)
                & source_local_null["null_design"].eq(design)
                & source_local_null["replicate"].le(repetitions)
            ]
            local_generated, local_difference = _add_source_reconciliation(
                local_generated,
                local_source,
                ["city", "grid_id", "null_design", "replicate"],
            )
            local_generated.to_parquet(
                state_dir / f"{city}_{design}_local_support.parquet",
                index=False,
                compression=str(config["reporting"]["parquet_compression"]),
            )
            local_parts.append(local_generated)
            reconciliation_rows.append(
                {
                    "city": city,
                    "null_design": design,
                    "city_maximum_absolute_cmi_difference_bits": city_difference,
                    "local_maximum_absolute_cmi_difference_bits": local_difference,
                }
            )

    city_surrogate = pd.concat(city_parts, ignore_index=True)
    local_surrogate = pd.concat(local_parts, ignore_index=True)
    city_observed = pd.concat(city_observed_parts, ignore_index=True)
    local_observed = pd.concat(local_observed_parts, ignore_index=True)
    city_observed = pd.concat(
        [
            city_observed.assign(null_design=design)
            for design in designs
        ],
        ignore_index=True,
    )
    local_observed = pd.concat(
        [
            local_observed.assign(null_design=design)
            for design in designs
        ],
        ignore_index=True,
    )

    selected_city_summary = source_city_summary.loc[
        source_city_summary["city"].isin(cities)
        & source_city_summary["null_design"].isin(designs)
    ]
    selected_local_summary = source_local_summary.loc[
        source_local_summary["city"].isin(cities)
        & source_local_summary["null_design"].isin(designs)
    ]
    city_summary = _summarize_support(
        city_surrogate,
        city_observed,
        selected_city_summary,
        ["city", "null_design"],
    )
    local_summary = _summarize_support(
        local_surrogate,
        local_observed,
        selected_local_summary,
        ["city", "grid_id", "null_design"],
    )
    reconciliation = pd.DataFrame(reconciliation_rows)

    if quick:
        spearman = pd.DataFrame()
        regressions = pd.DataFrame()
        nine_results = _nine_result_table(city_summary)
    else:
        spearman, regressions = _relationship_tables(
            local_summary, city_summary, e12_run
        )
        nine_results = _nine_result_table(city_summary)
        _plot_results(
            local_summary,
            city_summary,
            e12_run,
            figure_dir,
            int(config["reporting"]["figure_dpi"]),
        )
        (run_dir / "interpretation.md").write_text(
            _interpretation(
                spearman, regressions, city_summary
            ),
            encoding="utf-8",
        )

    compression = str(config["reporting"]["parquet_compression"])
    city_surrogate.to_parquet(
        table_dir / "city_surrogate_support.parquet",
        index=False,
        compression=compression,
    )
    local_surrogate.to_parquet(
        table_dir / "local_surrogate_support.parquet",
        index=False,
        compression=compression,
    )
    for name, frame in [
        ("city_observed_support.csv", city_observed),
        ("local_observed_support.csv", local_observed),
        ("city_support_summary.csv", city_summary),
        ("local_support_summary.csv", local_summary),
        ("cmi_reconciliation.csv", reconciliation),
        ("spearman_relationships.csv", spearman),
        ("fixed_effect_regressions.csv", regressions),
        ("e10_nine_result_reinterpretation.csv", nine_results),
    ]:
        frame.to_csv(table_dir / name, index=False)

    tolerance = float(
        config["acceptance"]["cmi_reconciliation_tolerance_bits"]
    )
    checks = {
        "city_cmi_reconciles": bool(
            reconciliation[
                "city_maximum_absolute_cmi_difference_bits"
            ].max()
            <= tolerance
        ),
        "local_cmi_reconciles": bool(
            reconciliation[
                "local_maximum_absolute_cmi_difference_bits"
            ].max()
            <= tolerance
        ),
        "null_mean_reconciles": bool(
            quick
            or max(
                    city_summary[
                        "null_mean_reconciliation_difference_bits"
                    ].abs().max(),
                    local_summary[
                        "null_mean_reconciliation_difference_bits"
                    ].abs().max(),
                )
                <= tolerance
        ),
        "support_metrics_nonnegative": bool(
            (
                city_surrogate[list(SUPPORT_METRICS)] >= 0
            ).all().all()
            and (
                local_surrogate[list(SUPPORT_METRICS)] >= 0
            ).all().all()
        ),
        "support_ratios_bounded": bool(
            (
                city_surrogate[
                    [
                        "support_atoms_per_observation",
                        "singleton_observation_share",
                    ]
                ]
                <= 1
            ).all().all()
            and (
                local_surrogate[
                    [
                        "support_atoms_per_observation",
                        "singleton_observation_share",
                    ]
                ]
                <= 1
            ).all().all()
        ),
        "city_row_reconciliation": bool(
            quick
            or len(city_surrogate)
            == int(config["acceptance"]["expected_city_surrogate_rows"])
        ),
        "local_row_reconciliation": bool(
            quick
            or len(local_surrogate)
            == int(config["acceptance"]["expected_local_surrogate_rows"])
        ),
        "summary_row_reconciliation": bool(
            quick
            or (
                len(city_summary)
                == int(config["acceptance"]["expected_city_summary_rows"])
                and len(local_summary)
                == int(config["acceptance"]["expected_local_summary_rows"])
            )
        ),
        "local_grid_reconciliation": bool(
            quick
            or local_summary["grid_id"].nunique()
            == int(config["acceptance"]["expected_local_grids"])
        ),
        "all_formal_figures_exist": bool(
            quick
            or all(
                path.exists()
                for path in [
                    figure_dir / "e13_support_null_main.pdf",
                    figure_dir / "e13_support_null_main.png",
                    figure_dir / "e13_support_metrics_supplement.pdf",
                    figure_dir / "e13_support_metrics_supplement.png",
                ]
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError(
            "E13 acceptance failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    config_text = yaml.safe_dump(config, sort_keys=False)
    (run_dir / "config_snapshot.yaml").write_text(
        config_text, encoding="utf-8"
    )
    command = (
        "MPLCONFIGDIR=/private/tmp/e13_mplconfig "
        "python run_e13_support_null.py"
        + (" --quick" if quick else "")
    )
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - start
    manifest = {
        "run_id": run_id,
        "experiment": "E13",
        "status": "complete",
        "development": quick,
        "elapsed_seconds": elapsed,
        "inputs": {
            "e02_data": str(e02_data),
            "e05_run": str(e05_run),
            "e10_run": str(e10_run),
            "e12_run": str(e12_run),
        },
        "rows": {
            "city_surrogates": len(city_surrogate),
            "local_surrogates": len(local_surrogate),
            "city_summaries": len(city_summary),
            "local_summaries": len(local_summary),
        },
        "checks": checks,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "environment": _environment(),
        "figures_generated_by": "Python/Matplotlib",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E13 surrogate support and E10 null-center analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "e13.yaml",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(run(args.config, quick=args.quick))


if __name__ == "__main__":
    main()
