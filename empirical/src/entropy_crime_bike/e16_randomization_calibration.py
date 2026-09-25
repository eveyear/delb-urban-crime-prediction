from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from .figure_typography import normalize_chart_typography
import numpy as np
import pandas as pd
import pyarrow
import yaml

from entropy_crime_bike.e10_placebo import _state_codes, randomization_p_value
from entropy_crime_bike.e16_delb_estimation import (
    CITY_ORDER,
    RESOLUTIONS,
    _load_panel,
    _prepare_fixed_bins,
    baseline_state_fields,
)
from entropy_crime_bike.spatial_information import BoundInterpolator, benjamini_hochberg


DESIGNS = ["temporal_block", "spatial_series", "circular_shift"]
FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
LOG_TWO = math.log(2.0)


def _load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(master: int, label: str) -> int:
    return int((master + zlib.crc32(label.encode("utf-8"))) % 2**32)


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e16_randomization_calibration")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in [
        logging.FileHandler(run_dir / "logs" / "e16_s16_6.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _environment(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"timestamp={datetime.now().isoformat()}",
                f"python={sys.version.replace(os.linesep, ' ')}",
                f"executable={sys.executable}",
                f"platform={platform.platform()}",
                f"pandas={pd.__version__}",
                f"numpy={np.__version__}",
                f"pyarrow={pyarrow.__version__}",
                f"matplotlib={plt.matplotlib.__version__}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def temporal_block_randomization(
    values: np.ndarray, rng: np.random.Generator, block_periods: int
) -> np.ndarray:
    matrix = np.asarray(values)
    grids, periods = matrix.shape
    complete = periods - periods % int(block_periods)
    result = matrix.copy()
    if complete == 0:
        return result
    blocks = matrix[:, :complete].reshape(grids, complete // block_periods, block_periods)
    order = np.argsort(rng.random((grids, blocks.shape[1])), axis=1)
    permuted = np.take_along_axis(blocks, order[:, :, None], axis=1)
    result[:, :complete] = permuted.reshape(grids, complete)
    return result


def spatial_series_randomization(
    values: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    matrix = np.asarray(values)
    return matrix[rng.permutation(matrix.shape[0])]


def circular_shift_randomization(
    values: np.ndarray, rng: np.random.Generator, minimum_periods: int
) -> np.ndarray:
    matrix = np.asarray(values)
    grids, periods = matrix.shape
    if periods <= 2 * minimum_periods:
        raise ValueError("Series is too short for the frozen circular separation.")
    shifts = rng.integers(minimum_periods, periods - minimum_periods + 1, size=grids)
    indices = (np.arange(periods)[None, :] + shifts[:, None]) % periods
    return np.take_along_axis(matrix, indices, axis=1)


def _entropy_from_counts(counts: np.ndarray, observations: int) -> float:
    positive = counts[counts > 0].astype(np.float64, copy=False)
    return float(
        (math.log(observations) - np.dot(positive, np.log(positive)) / observations)
        / LOG_TWO
    )


def conditional_entropy_and_support(
    z: np.ndarray,
    zy: np.ndarray,
    zy_to_z: np.ndarray,
    bicycle: np.ndarray,
    mask: np.ndarray,
    *,
    z_count: int,
) -> dict[str, float | int]:
    selected = np.asarray(mask, dtype=bool)
    observations = int(selected.sum())
    if observations <= 0:
        raise ValueError("Randomization sample cannot be empty.")
    z_selected = z[selected]
    b_selected = bicycle[selected].astype(np.int64, copy=False)
    state_codes = z_selected * 4 + b_selected
    joint_codes = zy[selected] * 4 + b_selected
    state_counts = np.bincount(state_codes, minlength=z_count * 4)
    joint_counts = np.bincount(joint_codes)
    state_atoms = int(np.count_nonzero(state_counts))
    joint_atoms = int(np.count_nonzero(joint_counts))
    plugin = _entropy_from_counts(joint_counts, observations) - _entropy_from_counts(
        state_counts, observations
    )
    miller_madow = plugin + (joint_atoms - state_atoms) / (
        2.0 * observations * LOG_TWO
    )
    r_z = (state_counts.reshape(z_count, 4) > 0).sum(axis=1)
    zy_present = np.flatnonzero(np.bincount(zy[selected], minlength=len(zy_to_z)) > 0)
    c_z = np.bincount(zy_to_z[zy_present], minlength=z_count)
    effective_df = int(
        np.dot(np.maximum(r_z - 1, 0), np.maximum(c_z - 1, 0))
    )
    singleton_atoms = int(np.count_nonzero(joint_counts == 1))
    return {
        "hb_miller_madow_bits": float(miller_madow),
        "support_atoms": joint_atoms,
        "support_atoms_per_observation": joint_atoms / observations,
        "singleton_atom_count": singleton_atoms,
        "singleton_observation_share": singleton_atoms / observations,
        "effective_df": effective_df,
        "effective_df_per_observation": effective_df / observations,
        "observations": observations,
    }


def _randomized_matrix(
    design: str,
    observed: np.ndarray,
    rng: np.random.Generator,
    *,
    temporal: str,
    randomization: dict[str, object],
) -> np.ndarray:
    if design == "temporal_block":
        return temporal_block_randomization(
            observed,
            rng,
            int(randomization["temporal_block_periods"][temporal]),
        )
    if design == "spatial_series":
        return spatial_series_randomization(observed, rng)
    if design == "circular_shift":
        return circular_shift_randomization(
            observed,
            rng,
            int(randomization["circular_minimum_separation_periods"][temporal]),
        )
    raise ValueError(design)


def _task(payload: dict[str, object]) -> str:
    city = str(payload["city"])
    resolution = int(payload["resolution"])
    temporal = str(payload["temporal"])
    output = Path(str(payload["output"]))
    if output.exists():
        return str(output)
    data_dir = Path(str(payload["data_dir"]))
    s16_3_run = Path(str(payload["s16_3_run"]))
    source_points = Path(str(payload["source_points"]))
    config = dict(payload["config"])
    randomization = dict(config["randomization_budget_reserved_for_S16_6"])
    quantiles = [
        float(value)
        for value in config["information_sets"]["common"]["bicycle_positive_quantiles"]
    ]

    panel = _load_panel(data_dir, city, resolution, temporal)
    prepared, _ = _prepare_fixed_bins(panel, temporal, quantiles)
    period_map = pd.read_parquet(s16_3_run / "tables/period_to_primary_block.parquet")
    mapping = period_map.loc[
        period_map["city"].eq(city)
        & period_map["temporal_resolution"].eq(temporal)
        & period_map["included_in_primary_universe"].astype(bool),
        ["period_start", "block_id"],
    ].copy()
    mapping["period_start"] = pd.to_datetime(mapping["period_start"])
    frame = prepared.merge(
        mapping, left_on="date", right_on="period_start", how="inner", validate="many_to_one"
    ).sort_values(["grid_id", "date"]).reset_index(drop=True)
    grid_counts = frame.groupby("grid_id", observed=True).size()
    if grid_counts.nunique() != 1:
        raise AssertionError("S16.6 requires a balanced grid-period matrix.")
    grids = int(frame["grid_id"].nunique())
    periods = int(grid_counts.iloc[0])
    observed_matrix = frame["bicycle_lag_bin"].cat.codes.to_numpy(np.int8).reshape(grids, periods)
    z = _state_codes(frame, baseline_state_fields(temporal))
    z_count = int(z.max()) + 1
    y = frame["crime_count_all"].to_numpy(np.int64)
    zy_index = pd.MultiIndex.from_arrays([z, y])
    zy, zy_uniques = pd.factorize(zy_index, sort=False)
    zy = zy.astype(np.int64)
    zy_to_z = zy_uniques.get_level_values(0).to_numpy(np.int64)

    membership = pd.read_parquet(s16_3_run / "tables/primary_block_membership.parquet")
    canonical = int(randomization["canonical_thinning_replicate"])
    membership = membership.loc[
        membership["city"].eq(city)
        & membership["temporal_resolution"].eq(temporal)
        & membership["replicate"].eq(canonical)
    ]
    block_ids = frame["block_id"].astype(str).to_numpy()
    masks: dict[float, np.ndarray] = {}
    for fraction in FRACTIONS[:-1]:
        selected_blocks = set(
            membership.loc[np.isclose(membership["fraction"], fraction), "block_id"].astype(str)
        )
        masks[fraction] = np.isin(block_ids, list(selected_blocks))
    masks[1.0] = np.ones(len(frame), dtype=bool)

    points = pd.read_parquet(source_points)
    points = points.loc[
        points["city"].eq(city)
        & points["resolution_m"].eq(resolution)
        & points["temporal_resolution"].eq(temporal)
        & points["estimator"].eq("miller_madow")
        & (
            (points["fraction"].lt(1) & points["replicate"].eq(canonical))
            | points["fraction"].eq(1)
        )
    ].copy()
    if len(points) != 5:
        raise AssertionError("Expected five canonical observed estimates.")
    source = {float(row.fraction): row for row in points.itertuples(index=False)}

    observed_flat = observed_matrix.ravel()
    reconciliation = []
    for fraction, mask in masks.items():
        metric = conditional_entropy_and_support(
            z, zy, zy_to_z, observed_flat, mask, z_count=z_count
        )
        row = source[fraction]
        reconciliation.append(
            max(
                abs(float(row.hb_raw_bits) - float(metric["hb_miller_madow_bits"])),
                abs(int(row.observations) - int(metric["observations"])),
            )
        )
    if max(reconciliation) > 1e-10:
        raise AssertionError(f"Observed S16.4 reconciliation failed: {max(reconciliation)}")

    key_scales = {
        (int(item["spatial_resolution_m"]), str(item["temporal_resolution"]))
        for item in randomization["key_scales"]
    }
    key_scale = (resolution, temporal) in key_scales
    repetitions = int(
        randomization["key_scale_repetitions"]
        if key_scale
        else randomization["all_90_cells_screening_repetitions"]
    )
    records = []
    master = int(config["randomness"]["master_seed"])
    for design in DESIGNS:
        rng = np.random.default_rng(
            _seed(master, f"S16.6|{city}|{resolution}|{temporal}|{design}")
        )
        for replicate in range(1, repetitions + 1):
            randomized = _randomized_matrix(
                design,
                observed_matrix,
                rng,
                temporal=temporal,
                randomization=randomization,
            ).ravel()
            for fraction, mask in masks.items():
                metric = conditional_entropy_and_support(
                    z, zy, zy_to_z, randomized, mask, z_count=z_count
                )
                observed = source[fraction]
                records.append(
                    {
                        "city": city,
                        "resolution_m": resolution,
                        "temporal_resolution": temporal,
                        "fraction": fraction,
                        "canonical_sample_replicate": 0 if fraction == 1 else canonical,
                        "null_design": design,
                        "null_replicate": replicate,
                        "randomization_repetitions": repetitions,
                        "key_scale": key_scale,
                        "observed_h0_bits": float(observed.h0_raw_bits),
                        "observed_hb_bits": float(observed.hb_raw_bits),
                        "observed_delta_h_bits": float(observed.delta_h_raw_bits),
                        "observed_delta_l_exact_mse": float(observed.delta_l_exact_mse),
                        "null_delta_h_bits": float(observed.h0_raw_bits)
                        - float(metric["hb_miller_madow_bits"]),
                        **metric,
                    }
                )
    result = pd.DataFrame(records)
    temporary = output.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    return str(output)


def summarize_randomization(nulls: pd.DataFrame) -> pd.DataFrame:
    keys = ["city", "resolution_m", "temporal_resolution", "fraction", "null_design"]
    records = []
    for key, part in nulls.groupby(keys, sort=True, observed=True):
        observed = float(part["observed_delta_h_bits"].iloc[0])
        values = part["null_delta_h_bits"].to_numpy(float)
        records.append(
            {
                **dict(zip(keys, key)),
                "canonical_sample_replicate": int(part["canonical_sample_replicate"].iloc[0]),
                "key_scale": bool(part["key_scale"].iloc[0]),
                "randomization_repetitions": len(values),
                "observed_delta_h_bits": observed,
                "observed_delta_l_exact_mse": float(part["observed_delta_l_exact_mse"].iloc[0]),
                "null_mean_delta_h_bits": float(values.mean()),
                "null_sd_delta_h_bits": float(values.std(ddof=1)),
                "null_q025_delta_h_bits": float(np.quantile(values, 0.025)),
                "null_q975_delta_h_bits": float(np.quantile(values, 0.975)),
                "observed_minus_null_mean_bits": observed - float(values.mean()),
                "randomization_p_value": randomization_p_value(observed, values),
                "mean_null_support_atoms_per_observation": float(part["support_atoms_per_observation"].mean()),
                "mean_null_singleton_observation_share": float(part["singleton_observation_share"].mean()),
                "mean_null_effective_df_per_observation": float(part["effective_df_per_observation"].mean()),
            }
        )
    result = pd.DataFrame(records)
    result["q_value_global_bh"] = benjamini_hochberg(result["randomization_p_value"])
    result["q_value_city_design_bh"] = np.nan
    for _, indices in result.groupby(["city", "null_design"], observed=True).groups.items():
        result.loc[indices, "q_value_city_design_bh"] = benjamini_hochberg(
            result.loc[indices, "randomization_p_value"]
        )
    result["separates_global_fdr"] = result["q_value_global_bh"].le(0.05)
    result["separates_city_design_fdr"] = result["q_value_city_design_bh"].le(0.05)
    return result


def _add_null_bound_metrics(nulls: pd.DataFrame) -> pd.DataFrame:
    result = nulls.copy()
    maximum = max(4.0, float(result[["observed_h0_bits", "hb_miller_madow_bits"]].max().max()) + 0.1)
    interpolator = BoundInterpolator.build(
        maximum, tail_tolerance=1e-15, root_tolerance=1e-12
    )
    h0 = np.maximum(result["observed_h0_bits"].to_numpy(float), 0)
    hb = np.maximum(result["hb_miller_madow_bits"].to_numpy(float), 0)
    hb = np.minimum(hb, h0)
    result["null_delta_h_projected_bits"] = h0 - hb
    result["null_delta_l_exact_mse"] = interpolator.transform(h0) - interpolator.transform(hb)
    area = (result["resolution_m"].to_numpy(float) / 1000.0) ** 2
    duration = np.where(result["temporal_resolution"].eq("day"), 1.0, 7.0)
    scale = area * duration
    result["null_delta_l_exact_intensity_mse"] = result["null_delta_l_exact_mse"] / scale**2
    return result


def _make_figures(summary: pd.DataFrame, run_dir: Path, dpi: int) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {0.2: "#440154", 0.4: "#3B528B", 0.6: "#21918C", 0.8: "#5EC962", 1.0: "#FDE725"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    limits = [float(summary[["observed_delta_h_bits", "null_mean_delta_h_bits"]].min().min()), float(summary[["observed_delta_h_bits", "null_mean_delta_h_bits"]].max().max())]
    for axis, design in zip(axes, DESIGNS):
        part = summary.loc[summary["null_design"].eq(design)]
        for fraction in FRACTIONS:
            selected = part.loc[np.isclose(part["fraction"], fraction)]
            axis.scatter(selected["observed_delta_h_bits"], selected["null_mean_delta_h_bits"], s=22, alpha=0.75, color=colors[fraction], label=f"{int(fraction*100)}%")
        axis.plot(limits, limits, "--", color="#555555", lw=1)
        axis.set_title(design.replace("_", " ").title())
        axis.set_xlabel("Observed CMI (bits)")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
    axes[0].set_ylabel("Randomization-null mean (bits)")
    axes[-1].legend(title="Sample", frameon=False, fontsize=7)
    normalize_chart_typography(fig)
    for suffix in ["pdf", "svg", "png"]:
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        fig.savefig(run_dir / "figures" / f"e16_s16_6_observed_vs_null.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    full = summary.loc[summary["fraction"].eq(1) & summary["key_scale"]].copy()
    full["scale"] = full["resolution_m"].map({500: "500 m day", 1000: "1 km day", 2000: "2 km week"})
    columns = [f"{scale}\n{design.replace('_', ' ')}" for scale in ["500 m day", "1 km day", "2 km week"] for design in DESIGNS]
    matrix = np.full((3, 9), np.nan)
    labels = [["" for _ in range(9)] for _ in range(3)]
    for row, city in enumerate(CITY_ORDER):
        for column, (scale, design) in enumerate((s, d) for s in ["500 m day", "1 km day", "2 km week"] for d in DESIGNS):
            value = full.loc[full["city"].eq(city) & full["scale"].eq(scale) & full["null_design"].eq(design)].iloc[0]
            matrix[row, column] = float(value.randomization_p_value)
            labels[row][column] = f"{value.randomization_p_value:.3f}"
    fig, axis = plt.subplots(figsize=(12.0, 3.4), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(9), columns, rotation=35, ha="right")
    axis.set_yticks(range(3), CITY_ORDER)
    for row in range(3):
        for column in range(9):
            axis.text(column, row, labels[row][column], ha="center", va="center", fontsize=7, color="white" if matrix[row, column] > 0.55 else "black")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Upper-tail p-value")
    axis.set_title("Full-sample calibration at prespecified key scales (999 randomizations)")
    normalize_chart_typography(fig)
    for suffix in ["pdf", "svg", "png"]:
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        fig.savefig(run_dir / "figures" / f"e16_s16_6_key_scale_pvalues.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    config_path = Path(args.config).resolve()
    empirical_root = config_path.parent.parent
    config = _load_yaml(config_path)
    s16_2_run = Path((empirical_root / "runs/latest_e16_s16_2_run.txt").read_text().strip())
    s16_3_run = Path((empirical_root / "runs/latest_e16_s16_3_run.txt").read_text().strip())
    s16_4_run = Path((empirical_root / "runs/latest_e16_s16_4_run.txt").read_text().strip())
    s16_5_run = Path((empirical_root / "runs/latest_e16_s16_5_run.txt").read_text().strip())
    for stage, path in [("S16.2", s16_2_run), ("S16.3", s16_3_run), ("S16.4", s16_4_run), ("S16.5", s16_5_run)]:
        status = json.loads((path / "run_status.json").read_text())
        if status.get("status") != "complete" or status.get("stage") != stage:
            raise ValueError(f"S16.6 requires accepted {stage}.")
    data_dir = Path(json.loads((s16_2_run / "run_status.json").read_text())["data_directory"])

    timezone = ZoneInfo("Asia/Shanghai")
    run_id = args.resume_run or datetime.now(timezone).strftime("%Y%m%d_%H%M%S") + "_E16_S16_6_randomization"
    run_dir = empirical_root / "runs" / run_id
    for directory in [run_dir / "logs", run_dir / "tables", run_dir / "figures", run_dir / "state"]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    command_name = "resume_command.txt" if args.resume_run else "command.txt"
    (run_dir / command_name).write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    if not args.resume_run:
        _environment(run_dir / "environment.txt")
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    payloads = []
    for city in CITY_ORDER:
        for resolution in RESOLUTIONS:
            for temporal in ["day", "week"]:
                payloads.append(
                    {
                        "city": city,
                        "resolution": resolution,
                        "temporal": temporal,
                        "output": str(run_dir / "state" / f"{city}_{resolution}m_{temporal}_nulls.parquet"),
                        "data_dir": str(data_dir),
                        "s16_3_run": str(s16_3_run),
                        "source_points": str(s16_4_run / "tables/primary_delb_estimates.parquet"),
                        "config": config,
                    }
                )
    workers = min(int(config["randomization_budget_reserved_for_S16_6"]["worker_processes"]), len(payloads))
    logger.info("Starting %d scale cells with %d workers", len(payloads), workers)
    # Thread workers avoid platform semaphore requirements while NumPy/Arrow
    # release the GIL for the dominant array and I/O operations.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_task, payload): payload for payload in payloads}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            path = future.result()
            completed += 1
            logger.info("Checkpoint %d/%d: %s", completed, len(payloads), Path(path).name)

    state_paths = sorted((run_dir / "state").glob("*_nulls.parquet"))
    nulls = pd.concat([pd.read_parquet(path) for path in state_paths], ignore_index=True)
    nulls = _add_null_bound_metrics(nulls)
    summary = summarize_randomization(nulls)
    tables = run_dir / "tables"
    nulls.to_parquet(tables / "randomization_replicates.parquet", index=False, compression="zstd")
    summary.to_csv(tables / "randomization_summary.csv", index=False)

    key = summary.loc[summary["key_scale"]].copy()
    key.to_csv(tables / "key_scale_randomization_summary.csv", index=False)
    screen = (
        summary.groupby(["city", "resolution_m", "temporal_resolution", "fraction"], as_index=False)
        .agg(
            designs=("null_design", "nunique"),
            minimum_p=("randomization_p_value", "min"),
            maximum_p=("randomization_p_value", "max"),
            passes_all_global_fdr=("separates_global_fdr", "all"),
            passes_any_global_fdr=("separates_global_fdr", "any"),
        )
    )
    screen.to_csv(tables / "cell_screening_summary.csv", index=False)

    randomization = config["randomization_budget_reserved_for_S16_6"]
    expected_rows = 9 * 3 * int(randomization["key_scale_repetitions"]) * 5 + 9 * 3 * int(randomization["all_90_cells_screening_repetitions"]) * 5
    min_screen = 1 / (int(randomization["all_90_cells_screening_repetitions"]) + 1)
    min_key = 1 / (int(randomization["key_scale_repetitions"]) + 1)
    checks = [
        ("scale checkpoints", len(state_paths), 18),
        ("randomization replicate rows", len(nulls), expected_rows),
        ("summary tests", len(summary), 270),
        ("screening cells", len(screen), 90),
        ("key summary tests", len(key), 135),
        ("finite null CMI", int(np.isfinite(nulls["null_delta_h_bits"]).all()), 1),
        ("finite null DELB", int(np.isfinite(nulls["null_delta_l_exact_mse"]).all()), 1),
        ("screening p-value floor", int((summary.loc[~summary["key_scale"], "randomization_p_value"] >= min_screen - 1e-15).all()), 1),
        ("key p-value floor", int((summary.loc[summary["key_scale"], "randomization_p_value"] >= min_key - 1e-15).all()), 1),
        ("support shares valid", int(nulls["singleton_observation_share"].between(0, 1).all()), 1),
        ("three null designs", summary["null_design"].nunique(), 3),
        ("no DELB divided by N", int(not any(column.lower() in {"l_over_n", "delb_per_observation"} for column in nulls.columns)), 1),
    ]
    acceptance = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    acceptance["status"] = np.where(acceptance["observed"].eq(acceptance["expected"]), "PASS", "FAIL")
    acceptance.to_csv(tables / "acceptance_checklist.csv", index=False)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("S16.6 acceptance failed: " + "; ".join(acceptance.loc[acceptance["status"].ne("PASS"), "check"]))

    _make_figures(summary, run_dir, int(config["reporting"]["figure_dpi"]))
    pd.DataFrame(
        [{"file": path.name, "sha256": _sha256(path), "generator": "empirical/run_e16_randomization_calibration.py"} for path in sorted((run_dir / "figures").iterdir())]
    ).to_csv(tables / "figure_manifest.csv", index=False)

    elapsed = time.monotonic() - started
    status = {
        "experiment": "E16",
        "stage": "S16.6",
        "status": "complete",
        "run_id": run_id,
        "source_s16_2_run": str(s16_2_run),
        "source_s16_3_run": str(s16_3_run),
        "source_s16_4_run": str(s16_4_run),
        "source_s16_5_run": str(s16_5_run),
        "randomization_rows": len(nulls),
        "summary_tests": len(summary),
        "screening_cells": len(screen),
        "global_fdr_passes": int(summary["separates_global_fdr"].sum()),
        "acceptance_passed": int(acceptance["status"].eq("PASS").sum()),
        "acceptance_total": len(acceptance),
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(timezone).isoformat(),
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (run_dir / "interpretation.md").write_text(
        "# S16.6 randomization calibration\n\n"
        "The frozen 199/999-replicate temporal-block, spatial-series, and circular-shift references calibrate Miller--Madow CMI over all 90 city--space--time--sample cells. Fractions below 100% use prespecified S16.3 replicate 1; sample-size stability was separately characterized in S16.5. Upper-tail plus-one p-values and BH adjustments are design-conditioned diagnostics, not exact conditional-randomization tests or causal evidence. A nonzero null mean is permitted and does not estimate population CMI. Count-level DELBs remain primary, and no bound is divided by sample size. The manuscript is not updated in this stage.\n",
        encoding="utf-8",
    )
    (empirical_root / "runs/latest_e16_s16_6_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    logger.info("S16.6 complete in %.1f seconds", elapsed)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate E16 multiscale CMI with frozen randomization references")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-run")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
