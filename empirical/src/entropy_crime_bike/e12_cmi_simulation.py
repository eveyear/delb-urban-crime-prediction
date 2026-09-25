from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import zlib

import matplotlib.pyplot as plt
from .figure_typography import normalize_chart_typography
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.stats import nbinom, spearmanr
import yaml

from .discrete_bound import inverse_entropy_envelope
from .e10_placebo import (
    circular_shift_surrogate,
    conditional_information,
    randomization_p_value,
    spatial_series_permutation,
    temporal_block_permutation,
)


LOG_TWO = math.log(2.0)
ESTIMATORS = ("plugin", "miller_madow")
NULL_DESIGNS = (
    "temporal_block_permutation",
    "spatial_series_permutation",
    "circular_shift_surrogate",
)
NULL_LABELS = {
    "temporal_block_permutation": "Temporal blocks",
    "spatial_series_permutation": "Spatial series",
    "circular_shift_surrogate": "Circular shift",
}
ESTIMATOR_LABELS = {
    "plugin": "Plugin",
    "miller_madow": "Miller–Madow",
}


@dataclass(frozen=True)
class PopulationModel:
    baseline_count: int
    bicycle_count: int
    balance_regime: str
    effect_strength: float
    joint_state_probability: np.ndarray
    count_mean: np.ndarray
    count_probability: np.ndarray
    count_support: np.ndarray
    population_cmi_bits: float
    conditional_entropy_bits: float
    rounded_predictor_mse: float
    exact_delb_mse: float
    binary_conditional_entropy_bits: float
    binary_bayes_error: float
    binary_fano_floor: float
    maximum_tail_probability: float


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return config


def _seed(base: int, label: str) -> int:
    return int((base + zlib.crc32(label.encode("utf-8"))) % 2**32)


def _entropy(probability: np.ndarray) -> float:
    p = np.asarray(probability, dtype=float)
    p = p[p > 0]
    return float(-np.dot(p, np.log2(p)))


def _binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 0.0), 1.0)
    if p in (0.0, 1.0):
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def _binary_entropy_inverse(entropy_bits: float) -> float:
    h = min(max(float(entropy_bits), 0.0), 1.0)
    if h <= 0:
        return 0.0
    if h >= 1:
        return 0.5
    return float(brentq(lambda p: _binary_entropy(p) - h, 0.0, 0.5))


def _state_probabilities(
    baseline_count: int,
    bicycle_count: int,
    balance_regime: str,
    *,
    bicycle_rotation: int = 0,
) -> np.ndarray:
    if balance_regime == "balanced":
        baseline_probability = np.full(
            baseline_count, 1.0 / baseline_count
        )
        conditional_bicycle = np.full(
            (baseline_count, bicycle_count), 1.0 / bicycle_count
        )
    elif balance_regime == "skewed":
        ranks = np.arange(1, baseline_count + 1, dtype=float)
        baseline_probability = ranks ** -1.25
        baseline_probability /= baseline_probability.sum()
        conditional_bicycle = np.full(
            (baseline_count, bicycle_count),
            0.30 / max(1, bicycle_count - 1),
        )
        for state in range(baseline_count):
            dominant = (state + bicycle_rotation) % bicycle_count
            conditional_bicycle[state, dominant] = 0.70
        conditional_bicycle /= conditional_bicycle.sum(axis=1, keepdims=True)
    else:
        raise ValueError(balance_regime)
    return baseline_probability[:, None] * conditional_bicycle


def _negative_binomial_probabilities(
    means: np.ndarray,
    dispersion: float,
    tail_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    means_flat = np.asarray(means, dtype=float).ravel()
    success = dispersion / (dispersion + means_flat)
    quantiles = nbinom.ppf(
        1.0 - tail_tolerance / max(1, len(means_flat)),
        dispersion,
        success,
    )
    maximum_count = int(np.nanmax(quantiles)) + 2
    support = np.arange(maximum_count + 1, dtype=np.int64)
    probability = nbinom.pmf(
        support[None, :],
        dispersion,
        success[:, None],
    )
    tails = nbinom.sf(maximum_count, dispersion, success)
    probability /= probability.sum(axis=1, keepdims=True)
    return support, probability, float(tails.max())


def build_population_model(
    baseline_count: int,
    bicycle_count: int,
    balance_regime: str,
    effect_strength: float,
    dgp_config: dict[str, object],
    *,
    bicycle_rotation: int = 0,
) -> PopulationModel:
    joint_state = _state_probabilities(
        baseline_count,
        bicycle_count,
        balance_regime,
        bicycle_rotation=bicycle_rotation,
    )
    baseline_scale = np.linspace(
        float(dgp_config["minimum_mean"]),
        float(dgp_config["maximum_mean"]),
        baseline_count,
    )
    bicycle_score = np.linspace(-1.0, 1.0, bicycle_count)
    means = baseline_scale[:, None] * np.exp(
        float(effect_strength) * bicycle_score[None, :]
    )
    support, count_probability_flat, maximum_tail = (
        _negative_binomial_probabilities(
            means,
            float(dgp_config["dispersion"]),
            float(dgp_config["tail_probability_tolerance"]),
        )
    )
    count_probability = count_probability_flat.reshape(
        baseline_count, bicycle_count, -1
    )

    baseline_probability = joint_state.sum(axis=1)
    conditional_bicycle = joint_state / baseline_probability[:, None]
    count_given_baseline = np.einsum(
        "zb,zbc->zc", conditional_bicycle, count_probability
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            count_probability > 0,
            np.log2(count_probability / count_given_baseline[:, None, :]),
            0.0,
        )
    population_cmi = float(
        np.sum(joint_state[:, :, None] * count_probability * log_ratio)
    )
    conditional_entropy = float(
        np.sum(
            joint_state
            * np.apply_along_axis(_entropy, 2, count_probability)
        )
    )

    rounded_prediction = np.rint(means).astype(int)
    squared_error = (
        support[None, None, :] - rounded_prediction[:, :, None]
    ) ** 2
    rounded_mse = float(
        np.sum(joint_state[:, :, None] * count_probability * squared_error)
    )
    exact_delb = float(inverse_entropy_envelope(conditional_entropy))

    probability_positive = 1.0 - count_probability[:, :, 0]
    binary_entropy = float(
        np.sum(
            joint_state
            * np.vectorize(_binary_entropy)(probability_positive)
        )
    )
    binary_error = float(
        np.sum(
            joint_state
            * np.minimum(probability_positive, 1.0 - probability_positive)
        )
    )
    binary_fano = _binary_entropy_inverse(binary_entropy)

    return PopulationModel(
        baseline_count=baseline_count,
        bicycle_count=bicycle_count,
        balance_regime=balance_regime,
        effect_strength=float(effect_strength),
        joint_state_probability=joint_state,
        count_mean=means,
        count_probability=count_probability,
        count_support=support,
        population_cmi_bits=max(0.0, population_cmi),
        conditional_entropy_bits=conditional_entropy,
        rounded_predictor_mse=rounded_mse,
        exact_delb_mse=exact_delb,
        binary_conditional_entropy_bits=binary_entropy,
        binary_bayes_error=binary_error,
        binary_fano_floor=binary_fano,
        maximum_tail_probability=maximum_tail,
    )


def _draw_states(
    probability: np.ndarray,
    sample_size: int,
    temporal_structure: str,
    persistence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_probability = np.asarray(probability, dtype=float).ravel()
    if temporal_structure == "iid":
        return rng.choice(
            len(flat_probability), size=sample_size, p=flat_probability
        )
    if temporal_structure != "markov":
        raise ValueError(temporal_structure)
    innovations = rng.random(sample_size) >= persistence
    innovations[0] = True
    fresh = rng.choice(
        len(flat_probability), size=sample_size, p=flat_probability
    )
    states = np.empty(sample_size, dtype=np.int64)
    states[0] = fresh[0]
    for index in range(1, sample_size):
        states[index] = fresh[index] if innovations[index] else states[index - 1]
    return states


def simulate_series(
    model: PopulationModel,
    sample_size: int,
    temporal_structure: str,
    dgp_config: dict[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_codes = _draw_states(
        model.joint_state_probability,
        sample_size,
        temporal_structure,
        float(dgp_config["markov_persistence"]),
        rng,
    )
    baseline = joint_codes // model.bicycle_count
    bicycle = joint_codes % model.bicycle_count
    mean = model.count_mean[baseline, bicycle]
    dispersion = float(dgp_config["dispersion"])
    success = dispersion / (dispersion + mean)
    target = rng.negative_binomial(dispersion, success).astype(np.int64)
    return target, baseline.astype(np.int64), bicycle.astype(np.int64)


def _plugin_conditional_entropy(
    target: np.ndarray, state: np.ndarray
) -> float:
    target = np.asarray(target, dtype=np.int64)
    state = np.asarray(state, dtype=np.int64)
    alphabet = int(target.max()) + 1
    joint_counts = np.bincount(state * alphabet + target)
    state_counts = np.bincount(state)
    return _entropy(joint_counts / joint_counts.sum()) - _entropy(
        state_counts / state_counts.sum()
    )


def estimate_cmi(
    target: np.ndarray,
    baseline: np.ndarray,
    bicycle: np.ndarray,
) -> dict[str, float]:
    enhanced = (
        np.asarray(baseline, dtype=np.int64)
        * (int(np.max(bicycle)) + 1)
        + np.asarray(bicycle, dtype=np.int64)
    )
    h0_plugin = _plugin_conditional_entropy(target, baseline)
    hb_plugin = _plugin_conditional_entropy(target, enhanced)
    h0_mm, hb_mm, cmi_mm = conditional_information(
        target, baseline, bicycle
    )
    return {
        "h0_plugin_bits": h0_plugin,
        "hb_plugin_bits": hb_plugin,
        "cmi_plugin_bits": h0_plugin - hb_plugin,
        "h0_miller_madow_bits": h0_mm,
        "hb_miller_madow_bits": hb_mm,
        "cmi_miller_madow_bits": cmi_mm,
    }


def support_diagnostics(
    target: np.ndarray,
    baseline: np.ndarray,
    bicycle: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int64)
    z = np.asarray(baseline, dtype=np.int64)
    b = np.asarray(bicycle, dtype=np.int64)
    bicycle_alphabet = int(b.max()) + 1
    target_alphabet = int(y.max()) + 1
    enhanced = z * bicycle_alphabet + b
    atom = enhanced * target_alphabet + y
    atom_counts = np.bincount(atom)
    baseline_atom_counts = np.bincount(z * target_alphabet + y)
    support_atoms = int(np.count_nonzero(atom_counts))
    singleton_atoms = int(np.count_nonzero(atom_counts == 1))
    effective_df = 0
    for state in np.unique(z):
        selected = z == state
        b_count = len(np.unique(b[selected]))
        y_count = len(np.unique(y[selected]))
        effective_df += max(0, b_count - 1) * max(0, y_count - 1)
    return {
        "baseline_joint_atoms": int(np.count_nonzero(baseline_atom_counts)),
        "enhanced_joint_atoms": support_atoms,
        "support_atoms_per_observation": support_atoms / len(y),
        "singleton_atom_count": singleton_atoms,
        "singleton_observation_share": singleton_atoms / len(y),
        "effective_df": int(effective_df),
        "effective_df_per_observation": effective_df / len(y),
    }


def _estimator_scenarios(config: dict[str, object]) -> list[dict[str, object]]:
    grid = config["estimator_grid"]
    scenarios = []
    for n in grid["sample_sizes"]:
        for z_count in grid["baseline_state_counts"]:
            for b_count in grid["bicycle_state_counts"]:
                for balance in grid["balance_regimes"]:
                    for effect in grid["effect_strengths"]:
                        for structure in grid["temporal_structures"]:
                            scenarios.append(
                                {
                                    "sample_size": int(n),
                                    "baseline_state_count": int(z_count),
                                    "bicycle_state_count": int(b_count),
                                    "balance_regime": str(balance),
                                    "effect_strength": float(effect),
                                    "temporal_structure": str(structure),
                                }
                            )
    return scenarios


def run_estimator_grid(
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_seed = int(config["random_seed"])
    repetitions = int(config["estimator_grid"]["repetitions"])
    dgp = config["dgp"]
    records: list[dict[str, object]] = []
    population_records: list[dict[str, object]] = []
    model_cache: dict[tuple[object, ...], PopulationModel] = {}
    for scenario_index, scenario in enumerate(_estimator_scenarios(config), 1):
        key = (
            scenario["baseline_state_count"],
            scenario["bicycle_state_count"],
            scenario["balance_regime"],
            scenario["effect_strength"],
        )
        model = model_cache.get(key)
        if model is None:
            model = build_population_model(*key, dgp)
            model_cache[key] = model
            population_records.append(
                {
                    "population_id": "|".join(map(str, key)),
                    "baseline_state_count": model.baseline_count,
                    "bicycle_state_count": model.bicycle_count,
                    "balance_regime": model.balance_regime,
                    "effect_strength": model.effect_strength,
                    "population_cmi_bits": model.population_cmi_bits,
                    "conditional_entropy_bits": model.conditional_entropy_bits,
                    "rounded_predictor_mse": model.rounded_predictor_mse,
                    "exact_delb_mse": model.exact_delb_mse,
                    "binary_conditional_entropy_bits": model.binary_conditional_entropy_bits,
                    "binary_bayes_error": model.binary_bayes_error,
                    "binary_fano_floor": model.binary_fano_floor,
                    "maximum_tail_probability": model.maximum_tail_probability,
                }
            )
        label = "|".join(f"{key}={value}" for key, value in scenario.items())
        for repetition in range(1, repetitions + 1):
            rng = np.random.default_rng(
                _seed(base_seed, f"estimator|{label}|{repetition}")
            )
            target, baseline, bicycle = simulate_series(
                model,
                int(scenario["sample_size"]),
                str(scenario["temporal_structure"]),
                dgp,
                rng,
            )
            estimates = estimate_cmi(target, baseline, bicycle)
            support = support_diagnostics(target, baseline, bicycle)
            record = {
                "scenario_id": scenario_index,
                "repetition": repetition,
                **scenario,
                "population_cmi_bits": model.population_cmi_bits,
                **estimates,
                **support,
            }
            record["bias_plugin_bits"] = (
                record["cmi_plugin_bits"] - model.population_cmi_bits
            )
            record["bias_miller_madow_bits"] = (
                record["cmi_miller_madow_bits"] - model.population_cmi_bits
            )
            records.append(record)
    return pd.DataFrame(records), pd.DataFrame(population_records)


def summarize_estimators(replicates: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scenario_id",
        "sample_size",
        "baseline_state_count",
        "bicycle_state_count",
        "balance_regime",
        "effect_strength",
        "temporal_structure",
        "population_cmi_bits",
    ]
    rows = []
    for values, group in replicates.groupby(keys, observed=True, sort=True):
        row = dict(zip(keys, values, strict=True))
        row.update(
            {
                "repetitions": len(group),
                "mean_support_atoms_per_observation": group[
                    "support_atoms_per_observation"
                ].mean(),
                "mean_singleton_observation_share": group[
                    "singleton_observation_share"
                ].mean(),
                "mean_effective_df_per_observation": group[
                    "effective_df_per_observation"
                ].mean(),
            }
        )
        for estimator in ESTIMATORS:
            estimate = group[f"cmi_{estimator}_bits"].to_numpy(float)
            truth = float(row["population_cmi_bits"])
            row[f"mean_{estimator}_bits"] = estimate.mean()
            row[f"bias_{estimator}_bits"] = estimate.mean() - truth
            row[f"rmse_{estimator}_bits"] = float(
                np.sqrt(np.mean((estimate - truth) ** 2))
            )
            row[f"positive_share_{estimator}"] = float(
                np.mean(estimate > 0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _panel_population_truth(
    grid_count: int,
    baseline_count: int,
    bicycle_count: int,
    balance: str,
    effect: float,
    dgp: dict[str, object],
) -> tuple[list[PopulationModel], float]:
    models = [
        build_population_model(
            baseline_count,
            bicycle_count,
            balance,
            effect,
            dgp,
            bicycle_rotation=grid,
        )
        for grid in range(grid_count)
    ]
    return models, float(np.mean([model.population_cmi_bits for model in models]))


def _randomization_scenarios(
    config: dict[str, object],
) -> list[dict[str, object]]:
    grid = config["randomization_grid"]
    scenarios = []
    for days in grid["days_per_grid"]:
        for balance in grid["balance_regimes"]:
            for effect in grid["effect_strengths"]:
                for structure in grid["temporal_structures"]:
                    scenarios.append(
                        {
                            "days_per_grid": int(days),
                            "balance_regime": str(balance),
                            "effect_strength": float(effect),
                            "temporal_structure": str(structure),
                        }
                    )
    return scenarios


def _simulate_panel(
    models: list[PopulationModel],
    days: int,
    temporal_structure: str,
    dgp: dict[str, object],
    base_seed: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = []
    baselines = []
    bicycles = []
    baseline_count = models[0].baseline_count
    for grid, model in enumerate(models):
        rng = np.random.default_rng(_seed(base_seed, f"{label}|grid={grid}"))
        target, baseline, bicycle = simulate_series(
            model, days, temporal_structure, dgp, rng
        )
        targets.append(target)
        baselines.append(grid * baseline_count + baseline)
        bicycles.append(bicycle)
    return (
        np.concatenate(targets),
        np.concatenate(baselines),
        np.vstack(bicycles),
    )


def _null_matrix(
    design: str,
    matrix: np.ndarray,
    rng: np.random.Generator,
    grid_config: dict[str, object],
) -> np.ndarray:
    if design == "temporal_block_permutation":
        return temporal_block_permutation(
            matrix, rng, int(grid_config["temporal_block_days"])
        )
    if design == "spatial_series_permutation":
        return spatial_series_permutation(matrix, rng)
    if design == "circular_shift_surrogate":
        return circular_shift_surrogate(
            matrix, rng, int(grid_config["circular_minimum_shift_days"])
        )
    raise ValueError(design)


def run_randomization_grid(
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_config = config["randomization_grid"]
    dgp = config["dgp"]
    base_seed = int(config["random_seed"])
    grid_count = int(grid_config["grid_count"])
    baseline_count = int(grid_config["baseline_state_count_per_grid"])
    bicycle_count = int(grid_config["bicycle_state_count"])
    monte_carlo = int(grid_config["monte_carlo_repetitions"])
    null_repetitions = int(grid_config["null_repetitions"])
    dataset_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    for scenario_id, scenario in enumerate(
        _randomization_scenarios(config), 1
    ):
        models, truth = _panel_population_truth(
            grid_count,
            baseline_count,
            bicycle_count,
            str(scenario["balance_regime"]),
            float(scenario["effect_strength"]),
            dgp,
        )
        scenario_label = "|".join(
            f"{key}={value}" for key, value in scenario.items()
        )
        for repetition in range(1, monte_carlo + 1):
            dataset_label = (
                f"randomization|{scenario_label}|rep={repetition}"
            )
            target, baseline, bicycle_matrix = _simulate_panel(
                models,
                int(scenario["days_per_grid"]),
                str(scenario["temporal_structure"]),
                dgp,
                base_seed,
                dataset_label,
            )
            observed = estimate_cmi(
                target, baseline, bicycle_matrix.ravel()
            )
            observed_support = support_diagnostics(
                target, baseline, bicycle_matrix.ravel()
            )
            for design in NULL_DESIGNS:
                null_by_estimator = {name: [] for name in ESTIMATORS}
                rng = np.random.default_rng(
                    _seed(base_seed, f"{dataset_label}|{design}")
                )
                for null_repetition in range(1, null_repetitions + 1):
                    matrix = _null_matrix(
                        design, bicycle_matrix, rng, grid_config
                    )
                    estimate = estimate_cmi(
                        target, baseline, matrix.ravel()
                    )
                    null_support = support_diagnostics(
                        target, baseline, matrix.ravel()
                    )
                    for estimator in ESTIMATORS:
                        value = float(estimate[f"cmi_{estimator}_bits"])
                        null_by_estimator[estimator].append(value)
                        null_rows.append(
                            {
                                "scenario_id": scenario_id,
                                "repetition": repetition,
                                "null_design": design,
                                "null_repetition": null_repetition,
                                "estimator": estimator,
                                **scenario,
                                "population_cmi_bits": truth,
                                "null_cmi_bits": value,
                                "null_support_atoms_per_observation": null_support[
                                    "support_atoms_per_observation"
                                ],
                                "null_singleton_observation_share": null_support[
                                    "singleton_observation_share"
                                ],
                                "null_effective_df_per_observation": null_support[
                                    "effective_df_per_observation"
                                ],
                            }
                        )
                for estimator in ESTIMATORS:
                    values = np.asarray(
                        null_by_estimator[estimator], dtype=float
                    )
                    observed_value = float(
                        observed[f"cmi_{estimator}_bits"]
                    )
                    dataset_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "repetition": repetition,
                            "null_design": design,
                            "estimator": estimator,
                            **scenario,
                            "population_cmi_bits": truth,
                            "observed_cmi_bits": observed_value,
                            "observed_bias_bits": observed_value - truth,
                            "null_mean_bits": values.mean(),
                            "null_sd_bits": values.std(ddof=1),
                            "observed_minus_null_mean_bits": observed_value
                            - values.mean(),
                            "randomization_p_value": randomization_p_value(
                                observed_value, values
                            ),
                            "observed_support_atoms_per_observation": observed_support[
                                "support_atoms_per_observation"
                            ],
                            "observed_singleton_observation_share": observed_support[
                                "singleton_observation_share"
                            ],
                            "observed_effective_df_per_observation": observed_support[
                                "effective_df_per_observation"
                            ],
                        }
                    )
    return pd.DataFrame(dataset_rows), pd.DataFrame(null_rows)


def summarize_randomization(
    datasets: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    keys = [
        "scenario_id",
        "days_per_grid",
        "balance_regime",
        "effect_strength",
        "temporal_structure",
        "null_design",
        "estimator",
        "population_cmi_bits",
    ]
    result = (
        datasets.groupby(keys, observed=True, sort=True)
        .agg(
            monte_carlo_repetitions=("repetition", "nunique"),
            mean_observed_cmi_bits=("observed_cmi_bits", "mean"),
            mean_observed_bias_bits=("observed_bias_bits", "mean"),
            mean_null_bits=("null_mean_bits", "mean"),
            mean_observed_minus_null_bits=(
                "observed_minus_null_mean_bits",
                "mean",
            ),
            rejection_rate=(
                "randomization_p_value",
                lambda value: float(np.mean(value <= alpha)),
            ),
            mean_support_atoms_per_observation=(
                "observed_support_atoms_per_observation",
                "mean",
            ),
            mean_singleton_observation_share=(
                "observed_singleton_observation_share",
                "mean",
            ),
        )
        .reset_index()
    )
    result["evidence_metric"] = np.where(
        result["effect_strength"].eq(0),
        "type_i_error",
        "power",
    )
    return result


def _plot_main(
    estimator_summary: pd.DataFrame,
    randomization_datasets: pd.DataFrame,
    randomization_summary: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    colors = {"plugin": "#D55E00", "miller_madow": "#0072B2"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    null_estimator = estimator_summary.loc[
        estimator_summary["effect_strength"].eq(0)
    ]
    for estimator in ESTIMATORS:
        axes[0, 0].scatter(
            null_estimator["mean_support_atoms_per_observation"],
            null_estimator[f"bias_{estimator}_bits"],
            s=20,
            alpha=0.65,
            color=colors[estimator],
            label=ESTIMATOR_LABELS[estimator],
        )
        axes[0, 1].scatter(
            null_estimator["mean_singleton_observation_share"],
            null_estimator[f"bias_{estimator}_bits"],
            s=20,
            alpha=0.65,
            color=colors[estimator],
            label=ESTIMATOR_LABELS[estimator],
        )
    axes[0, 0].axhline(0, color="0.25", linewidth=0.8)
    axes[0, 1].axhline(0, color="0.25", linewidth=0.8)
    axes[0, 0].set(
        xlabel="Observed support atoms / N",
        ylabel="Mean CMI bias (bits)",
        title="(a) Sparse support and null bias",
    )
    axes[0, 1].set(
        xlabel="Singleton-observation share",
        ylabel="Mean CMI bias (bits)",
        title="(b) Singleton occupancy and null bias",
    )
    axes[0, 0].legend(frameon=False)

    sample = randomization_datasets.loc[
        randomization_datasets["effect_strength"].eq(0)
    ]
    for estimator in ESTIMATORS:
        selected = sample.loc[sample["estimator"].eq(estimator)]
        axes[1, 0].scatter(
            selected["observed_singleton_observation_share"],
            selected["null_mean_bits"],
            s=12,
            alpha=0.35,
            color=colors[estimator],
            label=ESTIMATOR_LABELS[estimator],
        )
    axes[1, 0].set(
        xlabel="Observed singleton-observation share",
        ylabel="Randomization-null mean (bits)",
        title="(c) A randomization null need not center at zero",
    )

    grouped = (
        randomization_summary.groupby(
            ["effect_strength", "estimator"], observed=True
        )["rejection_rate"]
        .mean()
        .reset_index()
    )
    positions = np.arange(len(ESTIMATORS), dtype=float)
    width = 0.34
    for offset, effect in zip((-width / 2, width / 2), (0.0, 0.5)):
        values = [
            grouped.loc[
                grouped["effect_strength"].eq(effect)
                & grouped["estimator"].eq(estimator),
                "rejection_rate",
            ].mean()
            for estimator in ESTIMATORS
        ]
        axes[1, 1].bar(
            positions + offset,
            values,
            width,
            color="#999999" if effect == 0 else "#009E73",
            label="Null: rejection rate" if effect == 0 else "Alternative: power",
        )
    axes[1, 1].axhline(
        0.05, color="#D55E00", linestyle="--", linewidth=1, label=r"$\alpha=0.05$"
    )
    axes[1, 1].set_xticks(positions, [ESTIMATOR_LABELS[x] for x in ESTIMATORS])
    axes[1, 1].set(
        ylabel="Mean rejection rate",
        ylim=(0, 1),
        title="(d) Randomization calibration and detection",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    normalize_chart_typography(fig)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_supplement(
    estimator_summary: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {"plugin": "#D55E00", "miller_madow": "#0072B2"}
    markers = {"plugin": "o", "miller_madow": "x"}
    singleton_min = estimator_summary[
        "mean_singleton_observation_share"
    ].min()
    singleton_max = estimator_summary[
        "mean_singleton_observation_share"
    ].max()
    singleton_map = None
    for estimator in ESTIMATORS:
        singleton_map = axes[0].scatter(
            estimator_summary["population_cmi_bits"],
            estimator_summary[f"bias_{estimator}_bits"],
            c=estimator_summary["mean_singleton_observation_share"],
            cmap="viridis",
            vmin=singleton_min,
            vmax=singleton_max,
            marker=markers[estimator],
            s=22,
            alpha=0.7,
            label=ESTIMATOR_LABELS[estimator],
        )
        axes[1].scatter(
            estimator_summary["mean_effective_df_per_observation"],
            estimator_summary[f"rmse_{estimator}_bits"],
            color=colors[estimator],
            s=22,
            alpha=0.65,
            label=ESTIMATOR_LABELS[estimator],
        )
    axes[0].axhline(0, color="0.25", linewidth=0.8)
    axes[0].set(
        xlabel="Population CMI (bits)",
        ylabel="Bias (bits)",
        title="(a) Bias across null and alternatives",
    )
    axes[1].set(
        xlabel="Effective degrees of freedom / N",
        ylabel="RMSE (bits)",
        title="(b) Sparse degrees of freedom and RMSE",
    )
    colorbar = fig.colorbar(singleton_map, ax=axes[0], pad=0.02)
    colorbar.set_label("Singleton-observation share")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.legend(frameon=False)
    normalize_chart_typography(fig)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _interpretation(
    estimator_summary: pd.DataFrame,
    randomization_summary: pd.DataFrame,
) -> str:
    null = estimator_summary.loc[
        estimator_summary["effect_strength"].eq(0)
    ]
    lines = ["# E12 Interpretation", ""]
    for estimator in ESTIMATORS:
        bias = null[f"bias_{estimator}_bits"]
        rho, p_value = spearmanr(
            null["mean_singleton_observation_share"], bias
        )
        lines.extend(
            [
                f"## {ESTIMATOR_LABELS[estimator]} under population CMI = 0",
                "",
                f"- Mean bias across design cells: {bias.mean():.6f} bits.",
                f"- Bias range: {bias.min():.6f} to {bias.max():.6f} bits.",
                f"- Spearman association between singleton-observation share and bias: rho={rho:.3f}, p={p_value:.3g}.",
                "",
            ]
        )
    for estimator in ESTIMATORS:
        selected = randomization_summary.loc[
            randomization_summary["estimator"].eq(estimator)
        ]
        null_rows = selected.loc[selected["effect_strength"].eq(0)]
        alternative_rows = selected.loc[selected["effect_strength"].gt(0)]
        lines.extend(
            [
                f"## E10-style behavior: {ESTIMATOR_LABELS[estimator]}",
                "",
                f"- Mean null-distribution center under the population null: {null_rows['mean_null_bits'].mean():.6f} bits.",
                f"- Mean randomization rejection rate under the population null: {null_rows['rejection_rate'].mean():.3f}.",
                f"- Mean randomization rejection rate under the positive-CMI alternative: {alternative_rows['rejection_rate'].mean():.3f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The population estimand, finite-sample estimator bias, and the center of a transformation-specific randomization distribution are different quantities. A positive null mean is therefore not itself evidence of positive population CMI. Conversely, simulation behavior does not overturn the empirical E10 result; it explains conditions under which an E10 statistic can be centered away from zero.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate(
    population: pd.DataFrame,
    estimator_replicates: pd.DataFrame,
    randomization_datasets: pd.DataFrame,
    randomization_nulls: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, bool]:
    tolerance = float(config["reporting"]["identity_tolerance_bits"])
    expected_estimator_rows = (
        len(_estimator_scenarios(config))
        * int(config["estimator_grid"]["repetitions"])
    )
    expected_randomization_datasets = (
        len(_randomization_scenarios(config))
        * int(config["randomization_grid"]["monte_carlo_repetitions"])
        * len(NULL_DESIGNS)
        * len(ESTIMATORS)
    )
    expected_null_rows = (
        expected_randomization_datasets
        * int(config["randomization_grid"]["null_repetitions"])
    )
    return {
        "population_null_is_zero": bool(
            (
                population.loc[
                    population["effect_strength"].eq(0),
                    "population_cmi_bits",
                ].abs()
                <= tolerance
            ).all()
        ),
        "population_alternatives_are_positive": bool(
            (
                population.loc[
                    population["effect_strength"].gt(0),
                    "population_cmi_bits",
                ]
                > tolerance
            ).all()
        ),
        "negative_binomial_tail_controlled": bool(
            (
                population["maximum_tail_probability"]
                <= float(config["dgp"]["tail_probability_tolerance"])
            ).all()
        ),
        "oracle_delb_holds": bool(
            (
                population["rounded_predictor_mse"] + tolerance
                >= population["exact_delb_mse"]
            ).all()
        ),
        "oracle_binary_fano_holds": bool(
            (
                population["binary_bayes_error"] + tolerance
                >= population["binary_fano_floor"]
            ).all()
        ),
        "estimator_row_reconciliation": len(estimator_replicates)
        == expected_estimator_rows,
        "randomization_dataset_reconciliation": len(randomization_datasets)
        == expected_randomization_datasets,
        "randomization_null_reconciliation": len(randomization_nulls)
        == expected_null_rows,
        "all_estimates_finite": bool(
            estimator_replicates[
                ["cmi_plugin_bits", "cmi_miller_madow_bits"]
            ]
            .apply(np.isfinite)
            .all()
            .all()
            and randomization_nulls["null_cmi_bits"].map(np.isfinite).all()
        ),
    }


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "e12.yaml",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced development grid without publishing official outputs.",
    )
    args = parser.parse_args()
    config = _load_yaml(args.config.resolve())
    if args.quick:
        config["estimator_grid"]["sample_sizes"] = [365]
        config["estimator_grid"]["baseline_state_counts"] = [4, 16]
        config["estimator_grid"]["bicycle_state_counts"] = [2, 4]
        config["estimator_grid"]["effect_strengths"] = [0.0, 0.5]
        config["estimator_grid"]["temporal_structures"] = ["iid"]
        config["estimator_grid"]["repetitions"] = 3
        config["randomization_grid"]["days_per_grid"] = [365]
        config["randomization_grid"]["balance_regimes"] = ["balanced"]
        config["randomization_grid"]["effect_strengths"] = [0.0, 0.5]
        config["randomization_grid"]["temporal_structures"] = ["iid"]
        config["randomization_grid"]["monte_carlo_repetitions"] = 2
        config["randomization_grid"]["null_repetitions"] = 3

    empirical_root = Path(__file__).resolve().parents[2]
    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_E12_sparse_cmi"
        + ("_development" if args.quick else "")
    )
    run_dir = empirical_root / "runs" / run_id
    table_dir = run_dir / "tables"
    figure_dir = run_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()

    estimator_replicates, population = run_estimator_grid(config)
    estimator_summary = summarize_estimators(estimator_replicates)
    randomization_datasets, randomization_nulls = run_randomization_grid(
        config
    )
    randomization_summary = summarize_randomization(
        randomization_datasets,
        float(config["randomization_grid"]["alpha"]),
    )
    checks = _validate(
        population,
        estimator_replicates,
        randomization_datasets,
        randomization_nulls,
        config,
    )
    if not all(checks.values()):
        raise RuntimeError(f"E12 validation failed: {checks}")

    compression = str(config["reporting"]["parquet_compression"])
    estimator_replicates.to_parquet(
        table_dir / "estimator_replicates.parquet",
        index=False,
        compression=compression,
    )
    randomization_nulls.to_parquet(
        table_dir / "randomization_null_replicates.parquet",
        index=False,
        compression=compression,
    )
    randomization_datasets.to_parquet(
        table_dir / "randomization_dataset_summary.parquet",
        index=False,
        compression=compression,
    )
    population.to_csv(table_dir / "population_truth.csv", index=False)
    estimator_summary.to_csv(
        table_dir / "estimator_summary.csv", index=False
    )
    randomization_summary.to_csv(
        table_dir / "randomization_summary.csv", index=False
    )
    _plot_main(
        estimator_summary,
        randomization_datasets,
        randomization_summary,
        figure_dir / "e12_sparse_cmi_main",
        int(config["reporting"]["figure_dpi"]),
    )
    _plot_supplement(
        estimator_summary,
        figure_dir / "e12_bias_rmse_supplement",
        int(config["reporting"]["figure_dpi"]),
    )
    (run_dir / "interpretation.md").write_text(
        _interpretation(estimator_summary, randomization_summary),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - start
    config_text = yaml.safe_dump(config, sort_keys=False)
    (run_dir / "config_snapshot.yaml").write_text(
        config_text, encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "experiment": "E12",
        "status": "complete",
        "development": bool(args.quick),
        "elapsed_seconds": elapsed,
        "input": "Fully specified synthetic probability models; no empirical data",
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "checks": checks,
        "environment": _environment(),
        "figures_generated_by": "Python/Matplotlib",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(run_dir)


if __name__ == "__main__":
    main()
