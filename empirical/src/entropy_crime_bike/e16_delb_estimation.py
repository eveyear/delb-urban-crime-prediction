from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import yaml
from scipy import sparse

from entropy_crime_bike.conditional_information import (
    ConditionalCodebook,
    apply_bicycle_bins,
    crime_lag_bins,
    fit_positive_bicycle_quantiles,
)
from entropy_crime_bike.discrete_bound import (
    closed_form_delb,
    inverse_entropy_envelope,
)


CITY_ORDER = ["DC", "NY", "VAN"]
RESOLUTIONS = [500, 1000, 2000]
ESTIMATORS = ["plugin", "miller_madow"]


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
    logger = logging.getLogger("e16_delb_estimation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in [
        logging.FileHandler(run_dir / "logs" / "e16_s16_4.log", encoding="utf-8"),
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


def baseline_state_fields(temporal: str) -> list[str]:
    if temporal == "day":
        return ["grid_id", "crime_lag_bin", "day_of_week", "season", "is_holiday"]
    if temporal == "week":
        return ["grid_id", "crime_lag_bin", "season", "contains_holiday"]
    raise ValueError(temporal)


@dataclass(frozen=True)
class SupportCodebook:
    block_count: int
    observation_count: int
    joint_by_block: sparse.csr_matrix
    zb_by_block: sparse.csr_matrix
    zy_by_block: sparse.csr_matrix
    zb_to_z: sparse.csr_matrix
    zy_to_z: sparse.csr_matrix

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        target: str,
        baseline_state: list[str],
        bicycle_state: str,
        block_field: str,
    ) -> "SupportCodebook":
        required = [target, *baseline_state, bicycle_state, block_field]
        if frame[required].isna().any().any():
            raise ValueError("Support codebook inputs must be complete.")
        z_index = pd.MultiIndex.from_frame(frame[baseline_state].astype(object))
        z, z_unique = pd.factorize(z_index, sort=False)
        b = frame[bicycle_state].astype(object).to_numpy()
        y = pd.to_numeric(frame[target], errors="raise").to_numpy(np.int64)
        block, block_unique = pd.factorize(frame[block_field], sort=True)
        zb_index = pd.MultiIndex.from_arrays([z, b])
        zb, zb_unique = pd.factorize(zb_index, sort=False)
        zy_index = pd.MultiIndex.from_arrays([z, y])
        zy, zy_unique = pd.factorize(zy_index, sort=False)
        joint_index = pd.MultiIndex.from_arrays([z, b, y])
        joint, joint_unique = pd.factorize(joint_index, sort=False)
        ones = np.ones(len(frame), dtype=np.float64)

        def by_block(codes: np.ndarray, count: int) -> sparse.csr_matrix:
            return sparse.coo_matrix(
                (ones, (block, codes)), shape=(len(block_unique), count)
            ).tocsr()

        zb_z = zb_unique.get_level_values(0).to_numpy(np.int64)
        zy_z = zy_unique.get_level_values(0).to_numpy(np.int64)
        zb_to_z = sparse.coo_matrix(
            (np.ones(len(zb_unique)), (np.arange(len(zb_unique)), zb_z)),
            shape=(len(zb_unique), len(z_unique)),
        ).tocsr()
        zy_to_z = sparse.coo_matrix(
            (np.ones(len(zy_unique)), (np.arange(len(zy_unique)), zy_z)),
            shape=(len(zy_unique), len(z_unique)),
        ).tocsr()
        return cls(
            block_count=len(block_unique),
            observation_count=len(frame),
            joint_by_block=by_block(joint, len(joint_unique)),
            zb_by_block=by_block(zb, len(zb_unique)),
            zy_by_block=by_block(zy, len(zy_unique)),
            zb_to_z=zb_to_z,
            zy_to_z=zy_to_z,
        )

    def diagnose(self, block_weights: np.ndarray) -> pd.DataFrame:
        weights = np.asarray(block_weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if weights.shape[1] != self.block_count:
            raise ValueError("Support weights do not match block count.")
        joint = np.asarray(weights @ self.joint_by_block)
        zb = np.asarray(weights @ self.zb_by_block)
        zy = np.asarray(weights @ self.zy_by_block)
        observations = joint.sum(axis=1)
        support_atoms = (joint > 0).sum(axis=1)
        singleton_atoms = (joint == 1).sum(axis=1)
        r = np.asarray((zb > 0).astype(float) @ self.zb_to_z)
        c = np.asarray((zy > 0).astype(float) @ self.zy_to_z)
        effective_df = (np.maximum(r - 1, 0) * np.maximum(c - 1, 0)).sum(axis=1)
        return pd.DataFrame(
            {
                "support_atoms": support_atoms,
                "support_atoms_per_observation": support_atoms / observations,
                "singleton_atom_count": singleton_atoms,
                "singleton_observation_share": singleton_atoms / observations,
                "effective_df": effective_df.astype(np.int64),
                "effective_df_per_observation": effective_df / observations,
            }
        )


def block_weight_matrix(
    block_order: list[str],
    sample_summary: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    index = {value: position for position, value in enumerate(block_order)}
    metadata = sample_summary.sort_values(["replicate", "fraction"]).reset_index(drop=True)
    weights = np.zeros((len(metadata), len(block_order)), dtype=np.float64)
    groups = {
        (int(rep), float(frac)): set(part["block_id"].astype(str))
        for (rep, frac), part in membership.groupby(["replicate", "fraction"])
    }
    for row_index, row in metadata.iterrows():
        key = (int(row["replicate"]), float(row["fraction"]))
        selected = groups[key]
        missing = selected - set(index)
        if missing:
            raise KeyError(f"Unknown selected blocks: {sorted(missing)[:3]}")
        weights[row_index, [index[value] for value in selected]] = 1.0
    return weights, metadata


def bound_metrics(h0: float, hb: float, *, tail: float, root: float) -> dict[str, object]:
    h0_bound = max(0.0, float(h0))
    hb_nonnegative = max(0.0, float(hb))
    hb_bound = min(hb_nonnegative, h0_bound)
    l0 = inverse_entropy_envelope(h0_bound, tail_tolerance=tail, root_tolerance=root)
    lb = inverse_entropy_envelope(hb_bound, tail_tolerance=tail, root_tolerance=root)
    c0 = closed_form_delb(h0_bound)
    cb = closed_form_delb(hb_bound)
    return {
        "h0_for_bound_bits": h0_bound,
        "hb_for_bound_bits": hb_bound,
        "delta_h_projected_bits": h0_bound - hb_bound,
        "projection_applied": not math.isclose(hb_bound, hb, rel_tol=0, abs_tol=1e-14),
        "l0_exact_mse": l0,
        "lb_exact_mse": lb,
        "delta_l_exact_mse": l0 - lb,
        "l0_closed_mse": c0,
        "lb_closed_mse": cb,
        "delta_l_closed_mse": c0 - cb,
    }


def estimates_to_long(
    estimate: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    metadata: pd.DataFrame,
    *,
    config: dict[str, object],
) -> pd.DataFrame:
    numerics = config["numerics"] if "numerics" in config else {
        "lattice_tail_tolerance": 1e-15,
        "root_tolerance": 1e-12,
    }
    records = []
    for row_index in range(len(estimate)):
        meta = metadata.iloc[row_index].to_dict()
        diagnostic = {} if diagnostics is None else diagnostics.iloc[row_index].to_dict()
        for estimator in ESTIMATORS:
            prefix = "plugin" if estimator == "plugin" else "miller_madow"
            h0 = float(estimate.iloc[row_index][f"h0_{prefix}_bits"])
            hb = float(estimate.iloc[row_index][f"hb_{prefix}_bits"])
            records.append(
                {
                    **meta,
                    "estimator": estimator,
                    "observations": int(round(estimate.iloc[row_index]["sample_size"])),
                    "h0_raw_bits": h0,
                    "hb_raw_bits": hb,
                    "delta_h_raw_bits": h0 - hb,
                    **bound_metrics(
                        h0,
                        hb,
                        tail=float(numerics["lattice_tail_tolerance"]),
                        root=float(numerics["root_tolerance"]),
                    ),
                    **diagnostic,
                }
            )
    return pd.DataFrame(records)


def _load_panel(data_dir: Path, city: str, resolution: int, temporal: str) -> pd.DataFrame:
    files = sorted(
        (
            data_dir
            / "common_core"
            / f"resolution={resolution}m"
            / f"temporal={temporal}"
            / f"city={city}"
        ).rglob("*.parquet")
    )
    if not files:
        raise FileNotFoundError((data_dir, city, resolution, temporal))
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "grid_id"]).reset_index(drop=True)


def _prepare_fixed_bins(
    frame: pd.DataFrame, temporal: str, quantiles: list[float]
) -> tuple[pd.DataFrame, tuple[float, ...]]:
    result = frame.loc[
        frame["bike_coverage_training"].astype(bool)
        & frame["crime_count_lag1"].notna()
        & frame["bike_total_flow_lag1"].notna()
    ].copy()
    result["crime_lag_bin"] = crime_lag_bins(result["crime_count_lag1"])
    training = result.loc[result["split"].eq("train")]
    cut_points = fit_positive_bicycle_quantiles(
        training["bike_total_flow_lag1"], quantiles
    )
    result["bicycle_lag_bin"] = apply_bicycle_bins(
        result["bike_total_flow_lag1"], cut_points
    )
    required = ["crime_count_all", *baseline_state_fields(temporal), "bicycle_lag_bin"]
    if result[required].isna().any().any():
        raise AssertionError("Missing fixed-bin information state.")
    return result, cut_points


def _superblock_mapping(universe: pd.DataFrame, temporal: str) -> dict[str, str]:
    part = universe.loc[universe["temporal_resolution"].eq(temporal)].sort_values("block_index")
    if temporal == "day":
        return {
            str(row.block_id): f"B28_{(int(row.block_index) - 1) // 4 + 1:03d}"
            for row in part.itertuples(index=False)
        }
    return {str(row.block_id): f"B28_{int(row.block_index):03d}" for row in part.itertuples(index=False)}


def _primary_specification(
    prepared: pd.DataFrame,
    *,
    city: str,
    resolution: int,
    temporal: str,
    summary: pd.DataFrame,
    membership: pd.DataFrame,
    period_map: pd.DataFrame,
    universe: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = period_map.loc[
        period_map["city"].eq(city)
        & period_map["temporal_resolution"].eq(temporal)
        & period_map["included_in_primary_universe"].astype(bool),
        ["period_start", "block_id"],
    ].copy()
    mapping["period_start"] = pd.to_datetime(mapping["period_start"])
    subset = prepared.merge(mapping, left_on="date", right_on="period_start", how="inner", validate="many_to_one")
    subset.drop(columns="period_start", inplace=True)
    states = baseline_state_fields(temporal)
    codebook = ConditionalCodebook.from_frame(
        subset,
        target="crime_count_all",
        baseline_state=states,
        bicycle_state_field="bicycle_lag_bin",
        block_field="block_id",
    )
    support = SupportCodebook.from_frame(
        subset,
        target="crime_count_all",
        baseline_state=states,
        bicycle_state="bicycle_lag_bin",
        block_field="block_id",
    )
    block_order = sorted(subset["block_id"].astype(str).unique())
    weights, metadata = block_weight_matrix(block_order, summary, membership)
    estimate = codebook.estimate(weights)
    diagnostics = support.diagnose(weights)
    metadata = metadata.copy()
    metadata.insert(0, "design", "fixed_coverage_block_thinning")
    points = estimates_to_long(estimate, diagnostics, metadata, config=config)

    superblock = _superblock_mapping(
        universe.loc[universe["city"].eq(city)], temporal
    )
    subset["superblock_id"] = subset["block_id"].astype(str).map(superblock)
    if subset["superblock_id"].isna().any():
        raise AssertionError("Missing 28-day bootstrap superblock.")
    bootstrap_codebook = ConditionalCodebook.from_frame(
        subset,
        target="crime_count_all",
        baseline_state=states,
        bicycle_state_field="bicycle_lag_bin",
        block_field="superblock_id",
    )
    repetitions = int(config["finite_sample_diagnostics"]["block_bootstrap"]["full_sample_repetitions"])
    rng = np.random.default_rng(
        _seed(int(config["randomness"]["master_seed"]), f"bootstrap|{city}|{resolution}|{temporal}")
    )
    weights_boot = rng.multinomial(
        bootstrap_codebook.block_count,
        np.full(bootstrap_codebook.block_count, 1 / bootstrap_codebook.block_count),
        size=repetitions,
    )
    estimate_boot = bootstrap_codebook.estimate(weights_boot)
    metadata_boot = pd.DataFrame(
        {
            "design": "full_sample_28day_block_bootstrap",
            "city": city,
            "resolution_m": resolution,
            "temporal_resolution": temporal,
            "replicate": np.arange(1, repetitions + 1),
            "fraction": 1.0,
            "bootstrap_blocks": bootstrap_codebook.block_count,
        }
    )
    bootstrap = estimates_to_long(estimate_boot, None, metadata_boot, config=config)
    return points, bootstrap


def _single_estimate(
    subset: pd.DataFrame,
    temporal: str,
    *,
    metadata: dict[str, object],
    config: dict[str, object],
) -> pd.DataFrame:
    states = baseline_state_fields(temporal)
    subset = subset.copy()
    subset["one_block"] = 0
    codebook = ConditionalCodebook.from_frame(
        subset,
        target="crime_count_all",
        baseline_state=states,
        bicycle_state_field="bicycle_lag_bin",
        block_field="one_block",
    )
    support = SupportCodebook.from_frame(
        subset,
        target="crime_count_all",
        baseline_state=states,
        bicycle_state="bicycle_lag_bin",
        block_field="one_block",
    )
    estimate = codebook.estimate(np.ones(1))
    diagnostics = support.diagnose(np.ones(1))
    return estimates_to_long(
        estimate, diagnostics, pd.DataFrame([metadata]), config=config
    )


def _expanding_specifications(
    fixed_prepared: pd.DataFrame,
    *,
    city: str,
    resolution: int,
    temporal: str,
    windows: pd.DataFrame,
    quantiles: list[float],
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    threshold_records = []
    base = fixed_prepared.copy()
    for row in windows.sort_values("fraction").itertuples(index=False):
        selected = base.loc[
            base["date"].between(pd.Timestamp(row.window_start), pd.Timestamp(row.window_end_period_start))
        ].copy()
        cut_points = fit_positive_bicycle_quantiles(
            selected["bike_total_flow_lag1"], quantiles
        )
        selected["bicycle_lag_bin"] = apply_bicycle_bins(
            selected["bike_total_flow_lag1"], cut_points
        )
        metadata = {
            "design": "expanding_historical_window",
            "city": city,
            "resolution_m": resolution,
            "temporal_resolution": temporal,
            "replicate": 0,
            "fraction": float(row.fraction),
            "selected_periods": int(row.selected_periods),
            "grids": int(row.grids),
            "window_start": str(pd.Timestamp(row.window_start).date()),
            "window_end": str(pd.Timestamp(row.window_end).date()),
        }
        parts.append(_single_estimate(selected, temporal, metadata=metadata, config=config))
        threshold_records.append(
            {
                **metadata,
                "lower_positive_tertile": cut_points[0],
                "upper_positive_tertile": cut_points[1],
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(threshold_records)


def _plot_qc(points: pd.DataFrame, output: Path, dpi: int) -> None:
    mm = points.loc[points["estimator"].eq("miller_madow")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    for temporal, marker in [("day", "o"), ("week", "s")]:
        part = mm.loc[
            mm["temporal_resolution"].eq(temporal)
            & mm["resolution_m"].eq(1000)
        ]
        grouped = part.groupby("fraction", as_index=False).agg(
            cmi=("delta_h_raw_bits", "median"),
            singleton=("singleton_observation_share", "median"),
        )
        axes[0].plot(grouped["fraction"], grouped["cmi"], marker=marker, label=temporal)
        axes[1].plot(grouped["fraction"], grouped["singleton"], marker=marker, label=temporal)
    axes[0].set(xlabel="Target sample fraction", ylabel="Median MM CMI (bits)", title="Finite-sample CMI path")
    axes[1].set(xlabel="Target sample fraction", ylabel="Median singleton-observation share", title="Sparse-support path")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "e16_s16_4_estimation_qc.pdf", bbox_inches="tight")
    fig.savefig(output / "e16_s16_4_estimation_qc.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    config_path = Path(args.config).resolve()
    empirical_root = config_path.parent.parent
    config = _load_yaml(config_path)
    s16_2_run = Path((empirical_root / "runs/latest_e16_s16_2_run.txt").read_text().strip())
    s16_3_run = Path((empirical_root / "runs/latest_e16_s16_3_run.txt").read_text().strip())
    for source, stage in [(s16_2_run, "S16.2"), (s16_3_run, "S16.3")]:
        status = json.loads((source / "run_status.json").read_text())
        if status.get("status") != "complete" or status.get("stage") != stage:
            raise ValueError(f"S16.4 requires accepted {stage}.")
    data_dir = Path(json.loads((s16_2_run / "run_status.json").read_text())["data_directory"])
    timezone = ZoneInfo("Asia/Shanghai")
    run_id = args.resume_run or datetime.now(timezone).strftime("%Y%m%d_%H%M%S") + "_E16_S16_4_delb_estimation"
    run_dir = empirical_root / "runs" / run_id
    for directory in [run_dir / "logs", run_dir / "tables", run_dir / "figures", run_dir / "state"]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    command_name = "resume_command.txt" if args.resume_run else "command.txt"
    (run_dir / command_name).write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    if not args.resume_run:
        _environment(run_dir / "environment.txt")
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    membership = pd.read_parquet(s16_3_run / "tables/primary_block_membership.parquet")
    sample_summary = pd.read_csv(s16_3_run / "tables/primary_sample_summary.csv")
    period_map = pd.read_parquet(s16_3_run / "tables/period_to_primary_block.parquet")
    universe = pd.read_csv(s16_3_run / "tables/primary_block_universe.csv")
    windows = pd.read_csv(s16_3_run / "tables/expanding_window_index.csv")
    quantiles = [float(value) for value in config["information_sets"]["common"]["bicycle_positive_quantiles"]]

    point_paths = []
    bootstrap_paths = []
    expanding_paths = []
    threshold_records = []
    for city in CITY_ORDER:
        for resolution in RESOLUTIONS:
            for temporal in ["day", "week"]:
                key = f"{city}_{resolution}m_{temporal}"
                point_path = run_dir / "state" / f"{key}_points.parquet"
                boot_path = run_dir / "state" / f"{key}_bootstrap.parquet"
                expanding_path = run_dir / "state" / f"{key}_expanding.parquet"
                threshold_path = run_dir / "state" / f"{key}_thresholds.json"
                if all(path.exists() for path in [point_path, boot_path, expanding_path, threshold_path]):
                    logger.info("Loaded checkpoint %s", key)
                else:
                    panel = _load_panel(data_dir, city, resolution, temporal)
                    prepared, cut_points = _prepare_fixed_bins(panel, temporal, quantiles)
                    local_summary = sample_summary.loc[
                        sample_summary["city"].eq(city)
                        & sample_summary["resolution_m"].eq(resolution)
                        & sample_summary["temporal_resolution"].eq(temporal)
                    ].copy()
                    local_membership = membership.loc[
                        membership["city"].eq(city)
                        & membership["temporal_resolution"].eq(temporal)
                    ].copy()
                    points, bootstrap = _primary_specification(
                        prepared,
                        city=city,
                        resolution=resolution,
                        temporal=temporal,
                        summary=local_summary,
                        membership=local_membership,
                        period_map=period_map,
                        universe=universe,
                        config=config,
                    )
                    points.to_parquet(point_path, index=False, compression="zstd")
                    bootstrap.to_parquet(boot_path, index=False, compression="zstd")
                    local_windows = windows.loc[
                        windows["city"].eq(city)
                        & windows["resolution_m"].eq(resolution)
                        & windows["temporal_resolution"].eq(temporal)
                    ].copy()
                    expanding, expanding_thresholds = _expanding_specifications(
                        prepared,
                        city=city,
                        resolution=resolution,
                        temporal=temporal,
                        windows=local_windows,
                        quantiles=quantiles,
                        config=config,
                    )
                    expanding.to_parquet(expanding_path, index=False, compression="zstd")
                    threshold_payload = {
                        "city": city,
                        "resolution_m": resolution,
                        "temporal_resolution": temporal,
                        "fixed_training_cut_points": cut_points,
                        "expanding_thresholds": expanding_thresholds.to_dict(orient="records"),
                    }
                    threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
                    del panel, prepared, points, bootstrap, expanding
                    logger.info("Completed %s", key)
                point_paths.append(point_path)
                bootstrap_paths.append(boot_path)
                expanding_paths.append(expanding_path)
                threshold_records.append(json.loads(threshold_path.read_text()))

    points = pd.concat([pd.read_parquet(path) for path in point_paths], ignore_index=True)
    bootstrap = pd.concat([pd.read_parquet(path) for path in bootstrap_paths], ignore_index=True)
    expanding = pd.concat([pd.read_parquet(path) for path in expanding_paths], ignore_index=True)
    tables = run_dir / "tables"
    points.to_parquet(tables / "primary_delb_estimates.parquet", index=False, compression="zstd")
    points.to_csv(tables / "primary_delb_estimates.csv", index=False)
    bootstrap.to_parquet(tables / "full_sample_block_bootstrap.parquet", index=False, compression="zstd")
    expanding.to_csv(tables / "expanding_window_delb_estimates.csv", index=False)
    threshold_rows = []
    for payload in threshold_records:
        fixed = payload["fixed_training_cut_points"]
        threshold_rows.append(
            {
                "design": "fixed_training_reference",
                "city": payload["city"],
                "resolution_m": payload["resolution_m"],
                "temporal_resolution": payload["temporal_resolution"],
                "fraction": 1.0,
                "lower_positive_tertile": fixed[0],
                "upper_positive_tertile": fixed[1],
            }
        )
        threshold_rows.extend(payload["expanding_thresholds"])
    pd.DataFrame(threshold_rows).to_csv(tables / "bicycle_bin_thresholds.csv", index=False)

    primary_expected = 3 * 3 * 2 * (50 * 4 + 1) * 2
    bootstrap_expected = 3 * 3 * 2 * 1000 * 2
    expanding_expected = 3 * 3 * 2 * 5 * 2
    nested_obs = (
        points.loc[points["replicate"].gt(0) & points["estimator"].eq("miller_madow")]
        .sort_values("fraction")
        .groupby(["city", "resolution_m", "temporal_resolution", "replicate"])["observations"]
        .apply(lambda x: x.is_monotonic_increasing)
    )
    finite_columns = [
        "h0_raw_bits", "hb_raw_bits", "delta_h_raw_bits", "l0_exact_mse",
        "lb_exact_mse", "delta_l_exact_mse", "l0_closed_mse", "lb_closed_mse",
        "singleton_observation_share", "effective_df_per_observation",
    ]
    checks = [
        ("primary estimate rows", len(points), primary_expected),
        ("full-sample bootstrap rows", len(bootstrap), bootstrap_expected),
        ("expanding-window estimate rows", len(expanding), expanding_expected),
        ("primary values finite", int(np.isfinite(points[finite_columns].to_numpy(float)).all()), 1),
        ("expanding values finite", int(np.isfinite(expanding[finite_columns].to_numpy(float)).all()), 1),
        ("exact bound ordering", int((points["lb_exact_mse"] <= points["l0_exact_mse"] + 1e-12).all()), 1),
        ("closed bound ordering", int((points["lb_closed_mse"] <= points["l0_closed_mse"] + 1e-12).all()), 1),
        ("closed form no stronger than exact", int((points["l0_closed_mse"] <= points["l0_exact_mse"] + 1e-10).all() and (points["lb_closed_mse"] <= points["lb_exact_mse"] + 1e-10).all()), 1),
        ("sample observations nested", int(nested_obs.sum()), len(nested_obs)),
        ("singleton shares in unit interval", int(points["singleton_observation_share"].between(0, 1).all()), 1),
        ("effective degrees nonnegative", int(points["effective_df"].ge(0).all()), 1),
        ("two entropy estimators retained", points["estimator"].nunique(), 2),
    ]
    acceptance = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    acceptance["status"] = np.where(acceptance["observed"].eq(acceptance["expected"]), "PASS", "FAIL")
    acceptance.to_csv(tables / "acceptance_checklist.csv", index=False)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("S16.4 acceptance failed: " + "; ".join(acceptance.loc[acceptance["status"].ne("PASS"), "check"]))

    summary = (
        points.groupby(["city", "resolution_m", "temporal_resolution", "fraction", "estimator"], as_index=False)
        .agg(
            repetitions=("replicate", "nunique"),
            median_observations=("observations", "median"),
            median_cmi_bits=("delta_h_raw_bits", "median"),
            median_delta_l_exact_mse=("delta_l_exact_mse", "median"),
            median_singleton_share=("singleton_observation_share", "median"),
            median_effective_df_per_observation=("effective_df_per_observation", "median"),
        )
    )
    summary.to_csv(tables / "primary_estimate_summary.csv", index=False)
    _plot_qc(points, run_dir / "figures", int(config["reporting"]["figure_dpi"]))
    pd.DataFrame(
        [
            {"file": path.name, "sha256": _sha256(path), "generator": "empirical/run_e16_delb_estimation.py"}
            for path in sorted((run_dir / "figures").iterdir())
        ]
    ).to_csv(tables / "figure_manifest.csv", index=False)

    elapsed = time.monotonic() - started
    status = {
        "experiment": "E16",
        "stage": "S16.4",
        "status": "complete",
        "run_id": run_id,
        "source_s16_2_run": str(s16_2_run),
        "source_s16_3_run": str(s16_3_run),
        "primary_rows": len(points),
        "bootstrap_rows": len(bootstrap),
        "expanding_rows": len(expanding),
        "acceptance_passed": int(acceptance["status"].eq("PASS").sum()),
        "acceptance_total": len(acceptance),
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(timezone).isoformat(),
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (run_dir / "interpretation.md").write_text(
        "# S16.4 DELB estimation\n\n"
        "This stage estimates plugin and Miller--Madow conditional entropies, raw and projected CMI, exact integer-lattice DELBs, conservative closed-form bounds, and sparse-support diagnostics for the frozen multiscale sample-size design. The primary design uses fixed 2020--2021 training thresholds; expanding windows refit thresholds only within the available window. Full-sample uncertainty uses paired 28-day block bootstrap weights. No randomization-null calibration, intensity normalization, prediction model, or manuscript update is performed here.\n",
        encoding="utf-8",
    )
    (empirical_root / "runs/latest_e16_s16_4_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    logger.info("S16.4 complete in %.1f seconds", elapsed)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate E16 S16.4 entropy and DELB across scales")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-run")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

