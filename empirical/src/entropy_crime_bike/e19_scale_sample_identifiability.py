from __future__ import annotations

import argparse
import json
import math
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.stats import nbinom
import yaml

from .conditional_information import (
    apply_bicycle_bins,
    crime_lag_bins,
    fit_positive_bicycle_quantiles,
)
from .discrete_bound import inverse_entropy_envelope
from .e10_placebo import conditional_information, miller_madow_conditional_entropy
from .e17_estimator_domain_sensitivity import _plugin_and_jeffreys
from .spatial_information import BoundInterpolator


SCALE_KEYS = ("500m_day", "1km_day", "2km_week")
SAMPLE_LENGTHS = (90, 150, 365, 730)
TARGET_CMI_BITS = (0.0, 0.03, 0.06, 0.12)
ESTIMATORS = ("plugin", "miller_madow", "jeffreys_dirichlet")


def _seed(base: int, label: str) -> int:
    return int((base + zlib.crc32(label.encode("utf-8"))) % 2**32)


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, object]) -> None:
    formal = config["formal_design"]
    scale_keys = tuple(item["key"] for item in formal["scales"])
    checks = {
        "experiment": config.get("experiment_id") == "E19",
        "fixed design": config["estimand"]["design"] == "fixed_design_semi_synthetic",
        "scales": scale_keys == SCALE_KEYS,
        "sample lengths": tuple(formal["sample_lengths"]) == SAMPLE_LENGTHS,
        "CMI targets": tuple(float(x) for x in formal["target_cmi_bits"])
        == TARGET_CMI_BITS,
        "estimators": tuple(formal["estimators"]) == ESTIMATORS,
        "replicates": int(formal["monte_carlo_replicates"]) == 200,
        "matched reference": formal["randomization"]["reference"]
        == "baseline_state_matched",
        "surrogates": (
            int(formal["randomization"]["routine_surrogates"]) == 199
            and int(formal["randomization"]["confirmation_surrogates"]) == 999
        ),
        "N definition": config["estimand"]["sample_length_definition"]
        == "periods_per_spatial_unit",
        "formal blocked": all(bool(x) for x in config["stage_1_prohibitions"].values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("E19 design-freeze checks failed: " + ", ".join(failed))


def _entropy(probability: np.ndarray) -> float:
    values = np.asarray(probability, dtype=float)
    values = values[values > 0]
    return float(-np.dot(values, np.log2(values)))


def population_truth(
    baseline: np.ndarray,
    bicycle: np.ndarray,
    base_mean: np.ndarray,
    dispersion: float,
    beta: float,
) -> dict[str, float]:
    s = np.asarray(baseline, dtype=np.int64)
    b = np.asarray(bicycle, dtype=np.int64)
    mu0 = np.asarray(base_mean, dtype=float)
    centered = b - float(np.mean(b))
    means = np.clip(mu0 * np.exp(beta * centered), 0.02, 50.0)
    success = dispersion / (dispersion + means)
    maximum = int(np.nanmax(nbinom.ppf(1 - 1e-10, dispersion, success))) + 2
    support = np.arange(maximum + 1)
    probabilities = nbinom.pmf(
        support[None, :], dispersion, success[:, None]
    )
    tail = nbinom.sf(maximum, dispersion, success)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    def conditional_entropy(state: np.ndarray) -> float:
        result = 0.0
        for code in np.unique(state):
            mask = state == code
            result += float(mask.mean()) * _entropy(probabilities[mask].mean(axis=0))
        return result

    pair = s * (int(b.max()) + 1) + b
    h0 = conditional_entropy(s)
    hb = conditional_entropy(pair)
    return {
        "h0_true_bits": h0,
        "hb_true_bits": hb,
        "cmi_true_bits": max(0.0, h0 - hb),
        "maximum_tail_probability": float(np.max(tail)),
    }


def add_delb_truth(
    truth: dict[str, float], *, tail_tolerance: float, root_tolerance: float
) -> dict[str, float]:
    h0 = float(truth["h0_true_bits"])
    hb = min(max(0.0, float(truth["hb_true_bits"])), max(0.0, h0))
    l0 = float(
        inverse_entropy_envelope(
            max(0.0, h0),
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
    )
    lb = float(
        inverse_entropy_envelope(
            hb,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
    )
    return {
        **truth,
        "l0_true_mse": l0,
        "lb_true_mse": lb,
        "delta_l_true_mse": l0 - lb,
    }


def calibrate_beta(
    target_bits: float,
    baseline: np.ndarray,
    bicycle: np.ndarray,
    base_mean: np.ndarray,
    dispersion: float,
) -> tuple[float, dict[str, float]]:
    if target_bits == 0:
        truth = population_truth(baseline, bicycle, base_mean, dispersion, 0.0)
        return 0.0, truth

    def objective(beta: float) -> float:
        return (
            population_truth(baseline, bicycle, base_mean, dispersion, beta)[
                "cmi_true_bits"
            ]
            - target_bits
        )

    upper = 0.25
    while objective(upper) < 0 and upper < 8:
        upper *= 2
    if objective(upper) < 0:
        raise ValueError(f"Target CMI {target_bits} bits is not attainable.")
    beta = float(brentq(objective, 0.0, upper))
    return beta, population_truth(baseline, bicycle, base_mean, dispersion, beta)


def estimate_all(
    target: np.ndarray,
    baseline: np.ndarray,
    bicycle: np.ndarray,
    *,
    jeffreys_alpha: float,
) -> list[dict[str, float | str]]:
    pairs = _plugin_and_jeffreys(
        target, baseline, bicycle, alpha=jeffreys_alpha
    )
    h0_mm, hb_mm, _ = conditional_information(target, baseline, bicycle)
    pairs["miller_madow"] = (float(h0_mm), float(hb_mm))
    return [
        {
            "estimator": estimator,
            "h0_hat_bits": float(pairs[estimator][0]),
            "hb_hat_bits": float(pairs[estimator][1]),
            "cmi_hat_bits": float(pairs[estimator][0] - pairs[estimator][1]),
        }
        for estimator in ESTIMATORS
    ]


def synthetic_fixed_design(scale_key: str, periods: int) -> tuple[np.ndarray, ...]:
    grid_counts = {"500m_day": 8, "1km_day": 5, "2km_week": 3}
    if scale_key not in grid_counts:
        raise ValueError(scale_key)
    grids = grid_counts[scale_key]
    grid = np.repeat(np.arange(grids), periods)
    time = np.tile(np.arange(periods), grids)
    crime_lag = (grid + time // 3) % 4
    calendar = time % (7 if scale_key != "2km_week" else 4)
    baseline = ((grid * 4 + crime_lag) * (calendar.max() + 1) + calendar).astype(int)
    bicycle = ((2 * grid + time // 5 + time % 3) % 4).astype(int)
    base_mean = 0.35 + 0.10 * (grid % 3) + 0.12 * crime_lag
    return baseline, bicycle, base_mean.astype(float)


def _scale_spec(scale_key: str) -> tuple[int, str]:
    values = {
        "500m_day": (500, "day"),
        "1km_day": (1000, "day"),
        "2km_week": (2000, "week"),
    }
    if scale_key not in values:
        raise ValueError(scale_key)
    return values[scale_key]


def _load_scale_city(
    data_dir: Path,
    city: str,
    scale_key: str,
    *,
    quantiles: list[float],
) -> pd.DataFrame:
    resolution, temporal = _scale_spec(scale_key)
    location = (
        data_dir
        / "common_core"
        / f"resolution={resolution}m"
        / f"temporal={temporal}"
        / f"city={city}"
    )
    files = sorted(location.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(location)
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[
        frame["bike_coverage_training"].astype(bool)
        & frame["crime_count_lag1"].notna()
        & frame["bike_total_flow_lag1"].notna()
    ].copy()
    frame["crime_lag_bin"] = crime_lag_bins(frame["crime_count_lag1"])
    cut_points = fit_positive_bicycle_quantiles(
        frame.loc[frame["split"].eq("train"), "bike_total_flow_lag1"], quantiles
    )
    frame["bicycle_lag_bin"] = apply_bicycle_bins(
        frame["bike_total_flow_lag1"], cut_points
    )
    frame["city_grid_id"] = city + "|" + frame["grid_id"].astype(str)
    return frame.sort_values(["grid_id", "date"]).reset_index(drop=True)


def empirical_fixed_design(
    data_dir: Path,
    scale_key: str,
    periods: int,
    *,
    cities: list[str],
    representative_grids_per_city: int,
    quantiles: list[float],
    mean_shrinkage: float,
    dispersion_bounds: tuple[float, float],
    minimum_calibration_dispersion: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    _, temporal = _scale_spec(scale_key)
    pieces = []
    source_periods = []
    for city in cities:
        frame = _load_scale_city(data_dir, city, scale_key, quantiles=quantiles)
        grid_score = (
            frame.groupby("grid_id", observed=True)["bike_total_flow_lag1"]
            .mean()
            .sort_values()
        )
        positions = np.linspace(
            0,
            len(grid_score) - 1,
            min(representative_grids_per_city, len(grid_score)),
        ).round().astype(int)
        selected = grid_score.index[np.unique(positions)].tolist()
        for grid_id in selected:
            part = frame.loc[frame["grid_id"].eq(grid_id)].sort_values("date")
            source_periods.append(int(len(part)))
            take = np.arange(periods) % len(part)
            expanded = part.iloc[take].copy().reset_index(drop=True)
            expanded["generated_period"] = np.arange(periods)
            expanded["source_replay_cycle"] = np.arange(periods) // len(part)
            pieces.append(expanded)
    design = pd.concat(pieces, ignore_index=True)
    state_fields = ["city_grid_id", "crime_lag_bin", "season"]
    if temporal == "day":
        state_fields.extend(["day_of_week", "is_holiday"])
    else:
        state_fields.append("contains_holiday")
    baseline, _ = pd.factorize(
        pd.MultiIndex.from_frame(design[state_fields].astype(object)), sort=False
    )
    bicycle = design["bicycle_lag_bin"].cat.codes.to_numpy(np.int64)
    observed_y = design["crime_count_all"].to_numpy(float)
    counts = np.bincount(baseline)
    sums = np.bincount(baseline, weights=observed_y)
    global_mean = float(observed_y.mean())
    state_mean = (sums + mean_shrinkage * global_mean) / (
        counts + mean_shrinkage
    )
    base_mean = np.maximum(state_mean[baseline], 0.02)
    empirical_variance = float(np.var(observed_y))
    empirical_dispersion = global_mean**2 / max(
        empirical_variance - global_mean, 1e-3
    )
    empirical_dispersion = float(np.clip(empirical_dispersion, *dispersion_bounds))
    dispersion = max(empirical_dispersion, float(minimum_calibration_dispersion))
    design = design[
        [
            "city",
            "grid_id",
            "date",
            "generated_period",
            "source_replay_cycle",
            "crime_count_all",
        ]
    ].copy()
    design["baseline_code"] = baseline
    design["bicycle_code"] = bicycle
    design["base_mean"] = base_mean
    pair = baseline * (int(bicycle.max()) + 1) + bicycle
    metadata: dict[str, float | int | str] = {
        "scale": scale_key,
        "sample_length_periods": periods,
        "spatial_units": int(design[["city", "grid_id"]].drop_duplicates().shape[0]),
        "observations": int(len(design)),
        "minimum_source_periods": int(min(source_periods)),
        "maximum_replay_cycle": int(design["source_replay_cycle"].max()),
        "baseline_support_atoms": int(np.unique(baseline).size),
        "joint_support_atoms": int(np.unique(pair).size),
        "observations_per_joint_atom": float(len(pair) / np.unique(pair).size),
        "observed_zero_share": float(np.mean(observed_y == 0)),
        "empirical_dispersion": empirical_dispersion,
        "dispersion": dispersion,
        "dispersion_regularized": bool(dispersion > empirical_dispersion),
        "extension_rule": "deterministic_within_grid_periodic_replay",
    }
    return design, metadata


def build_stage2_populations(
    config: dict[str, object], root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    validate_config(config)
    cfg = config["stage_2_population_construction"]
    data_dir = root / "empirical" / str(cfg["source_multiscale_data"])
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    truth_rows = []
    support_rows = []
    designs = {}
    for scale in SCALE_KEYS:
        for periods in SAMPLE_LENGTHS:
            design, metadata = empirical_fixed_design(
                data_dir,
                scale,
                periods,
                cities=list(cfg["cities"]),
                representative_grids_per_city=int(
                    cfg["representative_grids_per_city"]
                ),
                quantiles=[float(x) for x in cfg["positive_bicycle_quantiles"]],
                mean_shrinkage=float(cfg["conditional_mean_shrinkage"]),
                dispersion_bounds=tuple(
                    float(x) for x in cfg["dispersion_bounds"]
                ),
                minimum_calibration_dispersion=float(
                    cfg["minimum_calibration_dispersion"]
                ),
            )
            designs[(scale, periods)] = design
            support_rows.append(metadata)
            baseline = design["baseline_code"].to_numpy(np.int64)
            bicycle = design["bicycle_code"].to_numpy(np.int64)
            base_mean = design["base_mean"].to_numpy(float)
            for target in TARGET_CMI_BITS:
                beta, truth = calibrate_beta(
                    target,
                    baseline,
                    bicycle,
                    base_mean,
                    float(metadata["dispersion"]),
                )
                truth = add_delb_truth(
                    truth,
                    tail_tolerance=float(cfg["lattice_tail_tolerance"]),
                    root_tolerance=float(cfg["lattice_root_tolerance"]),
                )
                truth_rows.append(
                    {
                        **metadata,
                        "target_cmi_bits": target,
                        "beta": beta,
                        **truth,
                        "calibration_error_bits": truth["cmi_true_bits"] - target,
                    }
                )
    return pd.DataFrame(truth_rows), pd.DataFrame(support_rows), designs


def run_stage2(config_path: Path) -> Path:
    config = load_config(config_path)
    root = config_path.resolve().parents[2]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_dir = root / "empirical" / "runs" / f"{stamp}_E19_MVP_stage2_population"
    (run_dir / "fixed_designs").mkdir(parents=True, exist_ok=False)
    truth, support, designs = build_stage2_populations(config, root)
    tolerance = float(
        config["stage_2_population_construction"]["calibration_tolerance_bits"]
    )
    if float(truth["calibration_error_bits"].abs().max()) > tolerance:
        raise AssertionError("Stage 2 population calibration missed tolerance.")
    if (truth["delta_l_true_mse"] < -1e-10).any():
        raise AssertionError("Population DELB reduction must be nonnegative.")
    truth.to_csv(run_dir / "population_truth.csv", index=False)
    support.to_csv(run_dir / "support_complexity.csv", index=False)
    for (scale, periods), design in designs.items():
        design.to_parquet(
            run_dir / "fixed_designs" / f"{scale}_N{periods}.parquet", index=False
        )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    status = {
        "experiment_id": "E19",
        "stage": "MVP_stage_2",
        "status": "complete",
        "population_cells": int(len(truth)),
        "fixed_design_cells": int(len(designs)),
        "maximum_calibration_error_bits": float(
            truth["calibration_error_bits"].abs().max()
        ),
        "randomization_surrogates_run": 0,
        "monte_carlo_datasets_generated": 0,
        "manuscript_modified": False,
    }
    (run_dir / "stage2_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def summarize_stage3(
    replicates: pd.DataFrame, *, relative_error_floor: float
) -> pd.DataFrame:
    rows = []
    keys = ["scale", "sample_length_periods", "target_cmi_bits", "estimator"]
    for key, group in replicates.groupby(keys, observed=True, sort=False):
        cmi_error = group["cmi_hat_bits"] - group["cmi_true_bits"]
        delb_error = group["delta_l_hat_mse"] - group["delta_l_true_mse"]
        target = float(group["cmi_true_bits"].iloc[0])
        delb_target = float(group["delta_l_true_mse"].iloc[0])
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "replicates": int(len(group)),
                "cmi_mean_hat_bits": float(group["cmi_hat_bits"].mean()),
                "cmi_bias_bits": float(cmi_error.mean()),
                "cmi_rmse_bits": float(np.sqrt(np.mean(cmi_error**2))),
                "cmi_relative_rmse": (
                    float(np.sqrt(np.mean(cmi_error**2)) / target)
                    if target > relative_error_floor
                    else np.nan
                ),
                "cmi_monte_carlo_sd_bits": float(
                    group["cmi_hat_bits"].std(ddof=1)
                ),
                "cmi_positive_rate": float(group["cmi_hat_bits"].gt(0).mean()),
                "cmi_sign_correct_rate": (
                    float(group["cmi_hat_bits"].gt(0).mean())
                    if target > 0
                    else np.nan
                ),
                "delb_mean_hat_mse": float(group["delta_l_hat_mse"].mean()),
                "delb_bias_mse": float(delb_error.mean()),
                "delb_rmse_mse": float(np.sqrt(np.mean(delb_error**2))),
                "delb_relative_rmse": (
                    float(np.sqrt(np.mean(delb_error**2)) / delb_target)
                    if delb_target > relative_error_floor
                    else np.nan
                ),
                "delb_monte_carlo_sd_mse": float(
                    group["delta_l_hat_mse"].std(ddof=1)
                ),
                "projection_rate": float(group["projection_applied"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_stage3(config_path: Path) -> Path:
    config = load_config(config_path)
    root = config_path.resolve().parents[2]
    cfg = config["stage_3_monte_carlo"]
    source = root / "empirical" / str(cfg["source_stage_2_run"])
    status = json.loads((source / "stage2_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("population_cells") != 48:
        raise ValueError("Stage 3 requires an accepted 48-cell Stage 2 run.")
    truth = pd.read_csv(source / "population_truth.csv")
    if len(truth) != 48:
        raise AssertionError("Stage 2 truth table must contain 48 rows.")
    repetitions = int(cfg["replicates"])
    if repetitions != 200:
        raise ValueError("The frozen Stage 3 design requires 200 replicates.")

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_dir = root / "empirical" / "runs" / f"{stamp}_E19_MVP_stage3_monte_carlo"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    parts = []
    for cell_index, cell in enumerate(truth.itertuples(index=False), start=1):
        design = pd.read_parquet(
            source
            / "fixed_designs"
            / f"{cell.scale}_N{int(cell.sample_length_periods)}.parquet"
        )
        baseline = design["baseline_code"].to_numpy(np.int64)
        bicycle = design["bicycle_code"].to_numpy(np.int64)
        base_mean = design["base_mean"].to_numpy(float)
        centered = bicycle - float(np.mean(bicycle))
        means = np.clip(base_mean * np.exp(float(cell.beta) * centered), 0.02, 50.0)
        probability = float(cell.dispersion) / (float(cell.dispersion) + means)
        rows = []
        for replicate in range(1, repetitions + 1):
            label = (
                f"E19S3|{cell.scale}|{int(cell.sample_length_periods)}|"
                f"{float(cell.target_cmi_bits)}|{replicate}"
            )
            rng = np.random.default_rng(_seed(int(config["random_seed"]), label))
            y = rng.negative_binomial(float(cell.dispersion), probability).astype(int)
            for estimate in estimate_all(
                y,
                baseline,
                bicycle,
                jeffreys_alpha=float(
                    config["stage_1_micro_validation"]["jeffreys_alpha"]
                ),
            ):
                rows.append(
                    {
                        "scale": cell.scale,
                        "sample_length_periods": int(cell.sample_length_periods),
                        "observations": int(len(y)),
                        "target_cmi_bits": float(cell.target_cmi_bits),
                        "cmi_true_bits": float(cell.cmi_true_bits),
                        "h0_true_bits": float(cell.h0_true_bits),
                        "hb_true_bits": float(cell.hb_true_bits),
                        "l0_true_mse": float(cell.l0_true_mse),
                        "lb_true_mse": float(cell.lb_true_mse),
                        "delta_l_true_mse": float(cell.delta_l_true_mse),
                        "beta": float(cell.beta),
                        "dispersion": float(cell.dispersion),
                        "replicate": replicate,
                        **estimate,
                    }
                )
        part = pd.DataFrame(rows)
        path = checkpoint_dir / (
            f"{cell.scale}_N{int(cell.sample_length_periods)}"
            f"_I{float(cell.target_cmi_bits):.2f}.parquet"
        )
        part.to_parquet(path, index=False)
        parts.append(part)
        print(f"E19 stage 3 cell {cell_index}/48 complete: {path.stem}", flush=True)

    replicates = pd.concat(parts, ignore_index=True)
    interpolation_cfg = cfg["delb_interpolation"]
    maximum_entropy = max(
        float(interpolation_cfg["minimum_maximum_entropy_bits"]),
        float(replicates[["h0_hat_bits", "hb_hat_bits"]].max().max()) + 0.05,
    )
    interpolator = BoundInterpolator.build(
        maximum_entropy,
        tail_tolerance=float(interpolation_cfg["lattice_tail_tolerance"]),
        root_tolerance=float(interpolation_cfg["lattice_root_tolerance"]),
    )
    h0 = np.maximum(replicates["h0_hat_bits"].to_numpy(float), 0.0)
    hb_raw = np.maximum(replicates["hb_hat_bits"].to_numpy(float), 0.0)
    hb = np.minimum(hb_raw, h0)
    replicates["projection_applied"] = hb_raw > h0 + 1e-14
    replicates["cmi_projected_bits"] = h0 - hb
    replicates["l0_hat_mse"] = interpolator.transform(h0)
    replicates["lb_hat_mse"] = interpolator.transform(hb)
    replicates["delta_l_hat_mse"] = (
        replicates["l0_hat_mse"] - replicates["lb_hat_mse"]
    )
    tolerance = float(cfg["near_truth_tolerance_bits"])
    replicates["cmi_within_tolerance"] = (
        replicates["cmi_hat_bits"] - replicates["cmi_true_bits"]
    ).abs().le(tolerance)
    summary = summarize_stage3(
        replicates, relative_error_floor=float(cfg["relative_error_floor"])
    )
    replicates.to_parquet(run_dir / "monte_carlo_replicates.parquet", index=False)
    summary.to_csv(run_dir / "convergence_summary.csv", index=False)
    pd.DataFrame(
        {
            "entropy_bits": interpolator.entropy_grid,
            "exact_delb_mse": interpolator.exact_bound_grid,
        }
    ).to_csv(run_dir / "delb_interpolation_grid.csv", index=False)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    expected_rows = 48 * repetitions * len(ESTIMATORS)
    if len(replicates) != expected_rows or len(summary) != 48 * len(ESTIMATORS):
        raise AssertionError("Stage 3 output dimensions do not match frozen design.")
    if not np.isfinite(
        replicates[
            [
                "h0_hat_bits",
                "hb_hat_bits",
                "cmi_hat_bits",
                "l0_hat_mse",
                "lb_hat_mse",
                "delta_l_hat_mse",
            ]
        ].to_numpy(float)
    ).all():
        raise AssertionError("Stage 3 produced non-finite estimates.")
    status = {
        "experiment_id": "E19",
        "stage": "MVP_stage_3",
        "status": "complete",
        "population_cells": 48,
        "datasets_generated": 48 * repetitions,
        "estimators_per_dataset": len(ESTIMATORS),
        "replicate_rows": int(len(replicates)),
        "summary_rows": int(len(summary)),
        "randomization_surrogates_run": 0,
        "manuscript_modified": False,
    }
    (run_dir / "stage3_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def _matched_groups(baseline: np.ndarray, bicycle: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(baseline, kind="stable")
    split = np.flatnonzero(np.diff(baseline[order])) + 1
    groups = np.split(order, split)
    return [
        group
        for group in groups
        if len(group) > 1 and np.unique(bicycle[group]).size > 1
    ]


def baseline_state_matched_permutation(
    bicycle: np.ndarray,
    groups: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(bicycle, dtype=np.int64).copy()
    for group in groups:
        result[group] = rng.permutation(result[group])
    return result


def _stage4_cell(
    cell: dict[str, object],
    *,
    source_stage2: Path,
    checkpoint_dir: Path,
    config: dict[str, object],
) -> str:
    scale = str(cell["scale"])
    periods = int(cell["sample_length_periods"])
    target_bits = float(cell["target_cmi_bits"])
    cfg = config["stage_4_power"]
    key = periods in [int(x) for x in cfg["confirmation_sample_lengths"]] and any(
        math.isclose(target_bits, float(x), abs_tol=1e-12)
        for x in cfg["confirmation_target_cmi_bits"]
    )
    surrogates = (
        int(cfg["confirmation_surrogates"])
        if key
        else int(cfg["routine_surrogates"])
    )
    design = pd.read_parquet(
        source_stage2 / "fixed_designs" / f"{scale}_N{periods}.parquet"
    )
    baseline = design["baseline_code"].to_numpy(np.int64)
    bicycle = design["bicycle_code"].to_numpy(np.int64)
    base_mean = design["base_mean"].to_numpy(float)
    groups = _matched_groups(baseline, bicycle)
    movable = int(sum(len(group) for group in groups))
    rng_reference = np.random.default_rng(
        _seed(
            int(config["random_seed"]),
            f"E19S4|REFERENCE|{scale}|{periods}|{target_bits}|{surrogates}",
        )
    )
    reference_matrix = np.empty((surrogates, len(bicycle)), dtype=np.int8)
    for index in range(surrogates):
        reference_matrix[index] = baseline_state_matched_permutation(
            bicycle, groups, rng_reference
        )
    centered = bicycle - float(np.mean(bicycle))
    means = np.clip(
        base_mean * np.exp(float(cell["beta"]) * centered), 0.02, 50.0
    )
    dispersion = float(cell["dispersion"])
    probability = dispersion / (dispersion + means)
    rows = []
    for replicate in range(1, 201):
        label = f"E19S3|{scale}|{periods}|{target_bits}|{replicate}"
        rng_y = np.random.default_rng(_seed(int(config["random_seed"]), label))
        y = rng_y.negative_binomial(dispersion, probability).astype(int)
        h0 = miller_madow_conditional_entropy(y, baseline)
        observed = h0 - miller_madow_conditional_entropy(
            y, baseline * 4 + bicycle
        )
        null = np.empty(surrogates, dtype=float)
        for index in range(surrogates):
            null[index] = h0 - miller_madow_conditional_entropy(
                y, baseline * 4 + reference_matrix[index]
            )
        p_value = (1 + int(np.sum(null >= observed))) / (surrogates + 1)
        rows.append(
            {
                "scale": scale,
                "sample_length_periods": periods,
                "target_cmi_bits": target_bits,
                "replicate": replicate,
                "surrogates": surrogates,
                "confirmation_cell": key,
                "observed_cmi_bits": observed,
                "null_mean_bits": float(null.mean()),
                "null_sd_bits": float(null.std(ddof=1)),
                "p_value": p_value,
                "reject_005": p_value <= float(cfg["alpha"]),
                "matched_groups": len(groups),
                "movable_observations": movable,
                "movable_share": movable / len(bicycle),
            }
        )
    output = checkpoint_dir / (
        f"{scale}_N{periods}_I{target_bits:.2f}_S{surrogates}.parquet"
    )
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output.name


def run_stage4(config_path: Path) -> Path:
    config = load_config(config_path)
    root = config_path.resolve().parents[2]
    cfg = config["stage_4_power"]
    source2 = root / "empirical" / str(cfg["source_stage_2_run"])
    source3 = root / "empirical" / str(cfg["source_stage_3_run"])
    status2 = json.loads((source2 / "stage2_status.json").read_text())
    status3 = json.loads((source3 / "stage3_status.json").read_text())
    if status2.get("status") != "complete" or status3.get("status") != "complete":
        raise ValueError("Stage 4 requires accepted Stage 2 and Stage 3 runs.")
    truth = pd.read_csv(source2 / "population_truth.csv")
    if len(truth) != 48:
        raise AssertionError("Stage 4 requires exactly 48 population cells.")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_dir = root / "empirical" / "runs" / f"{stamp}_E19_MVP_stage4_power"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    cells = truth.to_dict("records")
    completed = Parallel(
        n_jobs=int(cfg["worker_processes"]),
        backend=str(cfg["parallel_backend"]),
        verbose=10,
    )(
        delayed(_stage4_cell)(
            cell,
            source_stage2=source2,
            checkpoint_dir=checkpoint_dir,
            config=config,
        )
        for cell in cells
    )
    parts = [pd.read_parquet(checkpoint_dir / name) for name in completed]
    replicates = pd.concat(parts, ignore_index=True)
    summary = (
        replicates.groupby(
            ["scale", "sample_length_periods", "target_cmi_bits"],
            observed=True,
            as_index=False,
        )
        .agg(
            replicates=("replicate", "size"),
            surrogates=("surrogates", "first"),
            confirmation_cell=("confirmation_cell", "first"),
            power_or_type1_error=("reject_005", "mean"),
            mean_observed_cmi_bits=("observed_cmi_bits", "mean"),
            mean_null_cmi_bits=("null_mean_bits", "mean"),
            movable_share=("movable_share", "first"),
        )
    )
    summary["metric"] = np.where(
        summary["target_cmi_bits"].eq(0), "type_I_error", "power"
    )
    replicates.to_parquet(
        run_dir / "matched_reference_replicates.parquet", index=False
    )
    summary.to_csv(run_dir / "power_summary.csv", index=False)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    if len(replicates) != 48 * 200 or len(summary) != 48:
        raise AssertionError("Stage 4 output dimensions do not match frozen design.")
    confirmation = summary["confirmation_cell"]
    if not (summary.loc[confirmation, "surrogates"] == 999).all():
        raise AssertionError("A confirmation cell did not use 999 surrogates.")
    if not (summary.loc[~confirmation, "surrogates"] == 199).all():
        raise AssertionError("A routine cell did not use 199 surrogates.")
    status = {
        "experiment_id": "E19",
        "stage": "MVP_stage_4",
        "status": "complete",
        "population_cells": 48,
        "replicate_tests": int(len(replicates)),
        "routine_cells": int((~confirmation).sum()),
        "confirmation_cells": int(confirmation.sum()),
        "routine_surrogates": 199,
        "confirmation_surrogates": 999,
        "statistic": str(cfg["statistic"]),
        "reference": str(cfg["reference"]),
        "manuscript_modified": False,
    }
    (run_dir / "stage4_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def minimum_detectable_cmi(
    curve: pd.DataFrame, *, target_power: float
) -> dict[str, float | bool | str]:
    ordered = curve.sort_values("target_cmi_bits")
    cmi = ordered["target_cmi_bits"].to_numpy(float)
    raw_power = ordered["power_or_type1_error"].to_numpy(float)
    if len(cmi) < 2 or not np.all(np.diff(cmi) > 0):
        raise ValueError("MDCMI requires at least two ordered, unique CMI levels.")
    monotone_power = np.maximum.accumulate(raw_power)
    adjusted = not np.allclose(monotone_power, raw_power, atol=1e-12, rtol=0)
    reached = np.flatnonzero(monotone_power >= target_power)
    if len(reached) == 0:
        return {
            "mdcmi80_discrete_bits": np.nan,
            "mdcmi80_interpolated_bits": np.nan,
            "threshold_status": f"greater_than_{cmi[-1]:.2f}_bits",
            "monotonicity_adjusted": adjusted,
            "lower_bracket_cmi_bits": cmi[-1],
            "upper_bracket_cmi_bits": np.nan,
            "lower_bracket_power": monotone_power[-1],
            "upper_bracket_power": np.nan,
        }
    upper_index = int(reached[0])
    discrete = float(cmi[upper_index])
    if upper_index == 0:
        interpolated = discrete
        lower_index = upper_index
    else:
        lower_index = upper_index - 1
        x0, x1 = cmi[lower_index], cmi[upper_index]
        p0, p1 = monotone_power[lower_index], monotone_power[upper_index]
        interpolated = (
            float(x1)
            if math.isclose(p1, p0)
            else float(x0 + (target_power - p0) * (x1 - x0) / (p1 - p0))
        )
    return {
        "mdcmi80_discrete_bits": discrete,
        "mdcmi80_interpolated_bits": interpolated,
        "threshold_status": "reached_within_tested_grid",
        "monotonicity_adjusted": adjusted,
        "lower_bracket_cmi_bits": float(cmi[lower_index]),
        "upper_bracket_cmi_bits": float(cmi[upper_index]),
        "lower_bracket_power": float(monotone_power[lower_index]),
        "upper_bracket_power": float(monotone_power[upper_index]),
    }


def run_stage5(config_path: Path) -> Path:
    config = load_config(config_path)
    root = config_path.resolve().parents[2]
    cfg = config["stage_5_minimum_detectable_cmi"]
    source = root / "empirical" / str(cfg["source_stage_4_run"])
    status4 = json.loads((source / "stage4_status.json").read_text())
    if status4.get("status") != "complete":
        raise ValueError("Stage 5 requires an accepted Stage 4 run.")
    power = pd.read_csv(source / "power_summary.csv")
    if len(power) != 48:
        raise AssertionError("Stage 5 requires the complete 48-cell power grid.")
    rows = []
    for (scale, periods), curve in power.groupby(
        ["scale", "sample_length_periods"], observed=True, sort=False
    ):
        positive = curve.loc[curve["target_cmi_bits"].gt(0)]
        rows.append(
            {
                "scale": scale,
                "sample_length_periods": int(periods),
                "target_power": float(cfg["target_power"]),
                "type_I_error": float(
                    curve.loc[
                        curve["target_cmi_bits"].eq(0), "power_or_type1_error"
                    ].iloc[0]
                ),
                "maximum_tested_power": float(
                    positive["power_or_type1_error"].max()
                ),
                **minimum_detectable_cmi(
                    curve, target_power=float(cfg["target_power"])
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["scale", "sample_length_periods"]
    ).reset_index(drop=True)
    if len(result) != 12:
        raise AssertionError("Stage 5 must produce 12 scale-sample MDCMI rows.")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_dir = root / "empirical" / "runs" / f"{stamp}_E19_MVP_stage5_mdcmi"
    run_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(run_dir / "minimum_detectable_cmi.csv", index=False)
    power.to_csv(run_dir / "source_power_curve.csv", index=False)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    status = {
        "experiment_id": "E19",
        "stage": "MVP_stage_5",
        "status": "complete",
        "scale_sample_cells": int(len(result)),
        "target_power": float(cfg["target_power"]),
        "thresholds_reached_within_grid": int(
            result["threshold_status"].eq("reached_within_tested_grid").sum()
        ),
        "monotonicity_adjustments": int(result["monotonicity_adjusted"].sum()),
        "additional_simulations_run": 0,
        "manuscript_modified": False,
    }
    (run_dir / "stage5_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def run_micro_validation(config: dict[str, object]) -> pd.DataFrame:
    validate_config(config)
    cfg = config["stage_1_micro_validation"]
    rows: list[dict[str, object]] = []
    for scale in cfg["synthetic_scale_keys"]:
        for periods in cfg["sample_lengths"]:
            baseline, bicycle, base_mean = synthetic_fixed_design(scale, int(periods))
            for target_bits in cfg["target_cmi_bits"]:
                beta, truth = calibrate_beta(
                    float(target_bits),
                    baseline,
                    bicycle,
                    base_mean,
                    float(cfg["dispersion"]),
                )
                if abs(truth["cmi_true_bits"] - float(target_bits)) > float(
                    cfg["calibration_tolerance_bits"]
                ):
                    raise AssertionError("Population-CMI calibration missed tolerance.")
                centered = bicycle - float(np.mean(bicycle))
                means = np.clip(
                    base_mean * np.exp(beta * centered),
                    float(cfg["outcome_mean_floor"]),
                    float(cfg["outcome_mean_ceiling"]),
                )
                probability = float(cfg["dispersion"]) / (
                    float(cfg["dispersion"]) + means
                )
                for replicate in range(1, int(cfg["replicates"]) + 1):
                    label = f"{scale}|{periods}|{target_bits}|{replicate}"
                    rng = np.random.default_rng(_seed(int(config["random_seed"]), label))
                    y = rng.negative_binomial(
                        float(cfg["dispersion"]), probability
                    ).astype(int)
                    for estimate in estimate_all(
                        y,
                        baseline,
                        bicycle,
                        jeffreys_alpha=float(cfg["jeffreys_alpha"]),
                    ):
                        rows.append(
                            {
                                "scale": scale,
                                "sample_length_periods": int(periods),
                                "observations": len(y),
                                "target_cmi_bits": float(target_bits),
                                **truth,
                                "beta": beta,
                                "replicate": replicate,
                                **estimate,
                            }
                        )
    return pd.DataFrame(rows)


def run(config_path: Path, *, development: bool = False) -> Path:
    if not development:
        raise RuntimeError(
            "E19 MVP stage 1 blocks formal execution. Use --development only."
        )
    config = load_config(config_path)
    root = config_path.resolve().parents[2]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_dir = root / "empirical" / "runs" / f"{stamp}_E19_MVP_stage1_micro"
    run_dir.mkdir(parents=True, exist_ok=False)
    results = run_micro_validation(config)
    results.to_csv(run_dir / "micro_validation.csv", index=False)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    status = {
        "experiment_id": "E19",
        "stage": "MVP_stage_1",
        "status": "complete",
        "formal_execution": False,
        "rows": int(len(results)),
        "micro_replicates": int(config["stage_1_micro_validation"]["replicates"]),
        "randomization_surrogates_run": 0,
        "manuscript_modified": False,
    }
    (run_dir / "stage1_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--stage3", action="store_true")
    parser.add_argument("--stage4", action="store_true")
    parser.add_argument("--stage5", action="store_true")
    args = parser.parse_args()
    selected = (
        int(args.development)
        + int(args.stage2)
        + int(args.stage3)
        + int(args.stage4)
        + int(args.stage5)
    )
    if selected > 1:
        parser.error("--development, --stage2, and --stage3 are mutually exclusive")
    if args.stage2:
        print(run_stage2(args.config))
    elif args.stage3:
        print(run_stage3(args.config))
    elif args.stage4:
        print(run_stage4(args.config))
    elif args.stage5:
        print(run_stage5(args.config))
    else:
        print(run(args.config, development=args.development))


if __name__ == "__main__":
    main()
