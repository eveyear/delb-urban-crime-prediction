from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import yaml


CITY_ORDER = ["DC", "NY", "VAN"]
RESOLUTIONS = [500, 1000, 2000]
TEMPORAL_ORDER = ["day", "week"]


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


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e16_scale_normalization")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in [
        logging.FileHandler(run_dir / "logs" / "e16_s16_5.log", encoding="utf-8"),
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


def add_scale_normalization(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["area_km2"] = (pd.to_numeric(result["resolution_m"]) / 1000.0) ** 2
    result["duration_days"] = np.where(
        result["temporal_resolution"].eq("day"), 1.0, 7.0
    )
    result["area_time_km2_days"] = result["area_km2"] * result["duration_days"]
    scale = result["area_time_km2_days"]
    scale_squared = scale**2
    for suffix in ["exact", "closed"]:
        l0 = f"l0_{suffix}_mse"
        lb = f"lb_{suffix}_mse"
        delta = f"delta_l_{suffix}_mse"
        result[f"l0_{suffix}_intensity_mse"] = result[l0] / scale_squared
        result[f"lb_{suffix}_intensity_mse"] = result[lb] / scale_squared
        result[f"delta_l_{suffix}_intensity_mse"] = result[delta] / scale_squared
        result[f"l0_{suffix}_rmse"] = np.sqrt(np.maximum(result[l0], 0))
        result[f"lb_{suffix}_rmse"] = np.sqrt(np.maximum(result[lb], 0))
        result[f"l0_{suffix}_intensity_rmse"] = result[f"l0_{suffix}_rmse"] / scale
        result[f"lb_{suffix}_intensity_rmse"] = result[f"lb_{suffix}_rmse"] / scale
        result[f"delta_{suffix}_intensity_rmse"] = (
            result[f"l0_{suffix}_intensity_rmse"]
            - result[f"lb_{suffix}_intensity_rmse"]
        )
        result[f"relative_reduction_{suffix}"] = np.where(
            result[l0] > 0, result[delta] / result[l0], np.nan
        )
        result[f"relative_reduction_{suffix}_from_intensity"] = np.where(
            result[f"l0_{suffix}_intensity_mse"] > 0,
            result[f"delta_l_{suffix}_intensity_mse"]
            / result[f"l0_{suffix}_intensity_mse"],
            np.nan,
        )
    result["discrete_entropy_power_ratio"] = 2.0 ** (
        -2.0 * result["delta_h_projected_bits"]
    )
    return result


def add_stability(
    frame: pd.DataFrame,
    *,
    epsilon: float,
    reference_fraction: float = 1.0,
) -> pd.DataFrame:
    keys = ["city", "resolution_m", "temporal_resolution", "estimator"]
    references = frame.loc[frame["fraction"].eq(reference_fraction)].copy()
    if references.duplicated(keys).any():
        raise ValueError("Stability reference must be unique within scale.")
    metrics = [
        "h0_raw_bits",
        "hb_raw_bits",
        "delta_h_raw_bits",
        "l0_exact_mse",
        "lb_exact_mse",
        "delta_l_exact_mse",
        "l0_exact_intensity_mse",
        "lb_exact_intensity_mse",
        "delta_l_exact_intensity_mse",
        "l0_exact_intensity_rmse",
        "lb_exact_intensity_rmse",
        "delta_exact_intensity_rmse",
    ]
    references = references[keys + metrics].rename(
        columns={value: f"reference_{value}" for value in metrics}
    )
    result = frame.merge(references, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        reference = result[f"reference_{metric}"]
        result[f"stability_{metric}"] = (
            (result[metric] - reference).abs()
            / np.maximum(reference.abs(), epsilon)
        )
    return result


def summarize_repeated_thinning(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["city", "resolution_m", "temporal_resolution", "fraction", "estimator"]
    metrics = [
        "observations",
        "l0_exact_intensity_mse",
        "lb_exact_intensity_mse",
        "delta_l_exact_intensity_mse",
        "l0_exact_intensity_rmse",
        "lb_exact_intensity_rmse",
        "delta_exact_intensity_rmse",
        "relative_reduction_exact",
        "delta_h_raw_bits",
        "singleton_observation_share",
        "effective_df_per_observation",
        "stability_l0_exact_mse",
        "stability_lb_exact_mse",
        "stability_delta_l_exact_mse",
        "stability_delta_h_raw_bits",
    ]
    records = []
    for key, part in frame.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record["repetitions"] = int(part["replicate"].nunique())
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            record[f"{metric}_median"] = float(values.median())
            record[f"{metric}_q025"] = float(values.quantile(0.025))
            record[f"{metric}_q975"] = float(values.quantile(0.975))
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_intervals(
    bootstrap: pd.DataFrame, point: pd.DataFrame
) -> pd.DataFrame:
    keys = ["city", "resolution_m", "temporal_resolution", "estimator"]
    metrics = [
        "l0_exact_mse",
        "lb_exact_mse",
        "delta_l_exact_mse",
        "l0_exact_intensity_mse",
        "lb_exact_intensity_mse",
        "delta_l_exact_intensity_mse",
        "l0_exact_intensity_rmse",
        "lb_exact_intensity_rmse",
        "delta_exact_intensity_rmse",
        "relative_reduction_exact",
        "delta_h_raw_bits",
    ]
    point = point.loc[point["fraction"].eq(1), keys + metrics]
    records = []
    for key, part in bootstrap.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        selected_point = point.copy()
        for field, value in zip(keys, key):
            selected_point = selected_point.loc[selected_point[field].eq(value)]
        if len(selected_point) != 1:
            raise AssertionError(f"Missing unique full-sample point: {key}")
        record["bootstrap_repetitions"] = len(part)
        record["interval_type"] = "paired_28day_block_percentile_95"
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="raise")
            record[f"{metric}_point"] = float(selected_point.iloc[0][metric])
            record[f"{metric}_bootstrap_mean"] = float(values.mean())
            record[f"{metric}_low"] = float(values.quantile(0.025))
            record[f"{metric}_high"] = float(values.quantile(0.975))
        records.append(record)
    return pd.DataFrame(records)


def _plot_full_sample_heatmaps(
    full: pd.DataFrame, output: Path, dpi: int
) -> None:
    mm = full.loc[full["estimator"].eq("miller_madow")].copy()
    values = mm["delta_exact_intensity_rmse"]
    vmin, vmax = float(values.min()), float(values.max())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    image = None
    for ax, city in zip(axes, CITY_ORDER):
        part = mm.loc[mm["city"].eq(city)]
        matrix = (
            part.pivot(index="temporal_resolution", columns="resolution_m", values="delta_exact_intensity_rmse")
            .reindex(index=TEMPORAL_ORDER, columns=RESOLUTIONS)
        )
        image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(3), ["500 m", "1 km", "2 km"])
        ax.set_yticks(range(2), ["day", "week"])
        ax.set_title(city)
        for row in range(2):
            for col in range(3):
                value = matrix.iloc[row, col]
                ax.text(col, row, f"{value:.3g}", ha="center", va="center", color="white" if value > (vmin + vmax) / 2 else "black", fontsize=8)
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Reduction in RMSE floor (crimes km$^{-2}$ day$^{-1}$)")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        fig.savefig(output / f"e16_s16_5_full_sample_intensity_heatmap.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _plot_stability(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    mm = summary.loc[summary["estimator"].eq("miller_madow")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for temporal, marker in [("day", "o"), ("week", "s")]:
        part = mm.loc[mm["temporal_resolution"].eq(temporal)]
        grouped = part.groupby("fraction", as_index=False).agg(
            bound=("stability_l0_exact_mse_median", "median"),
            cmi=("stability_delta_h_raw_bits_median", "median"),
        )
        axes[0].plot(grouped["fraction"], grouped["bound"], marker=marker, label=temporal)
        axes[1].plot(grouped["fraction"], grouped["cmi"], marker=marker, label=temporal)
    axes[0].set(title="Baseline DELB stability", xlabel="Target sample fraction", ylabel="Relative deviation from 100% estimate")
    axes[1].set(title="CMI-estimate stability", xlabel="Target sample fraction", ylabel="Relative deviation from 100% estimate")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for suffix in ["pdf", "svg", "png"]:
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        fig.savefig(output / f"e16_s16_5_sample_stability.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    config_path = Path(args.config).resolve()
    empirical_root = config_path.parent.parent
    config = _load_yaml(config_path)
    s16_4_run = Path((empirical_root / "runs/latest_e16_s16_4_run.txt").read_text().strip())
    source_status = json.loads((s16_4_run / "run_status.json").read_text())
    if source_status.get("status") != "complete" or source_status.get("stage") != "S16.4":
        raise ValueError("S16.5 requires accepted S16.4.")
    timezone = ZoneInfo("Asia/Shanghai")
    run_id = datetime.now(timezone).strftime("%Y%m%d_%H%M%S") + "_E16_S16_5_scale_normalization"
    run_dir = empirical_root / "runs" / run_id
    for directory in [run_dir / "logs", run_dir / "tables", run_dir / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    (run_dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    _environment(run_dir / "environment.txt")
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    primary = pd.read_parquet(s16_4_run / "tables/primary_delb_estimates.parquet")
    expanding = pd.read_csv(s16_4_run / "tables/expanding_window_delb_estimates.csv")
    bootstrap = pd.read_parquet(s16_4_run / "tables/full_sample_block_bootstrap.parquet")
    epsilon = float(config["finite_sample_diagnostics"]["stability_epsilon"])

    primary = add_scale_normalization(primary)
    primary = add_stability(primary, epsilon=epsilon)
    expanding = add_scale_normalization(expanding)
    expanding = add_stability(expanding, epsilon=epsilon)
    bootstrap = add_scale_normalization(bootstrap)

    summary = summarize_repeated_thinning(primary)
    expanding_summary = summarize_repeated_thinning(expanding)
    intervals = bootstrap_intervals(bootstrap, primary)
    full_sample = primary.loc[primary["fraction"].eq(1)].copy()

    tables = run_dir / "tables"
    primary.to_parquet(tables / "primary_normalized_estimates.parquet", index=False, compression="zstd")
    primary.to_csv(tables / "primary_normalized_estimates.csv", index=False)
    expanding.to_csv(tables / "expanding_window_normalized_estimates.csv", index=False)
    summary.to_csv(tables / "sample_stability_summary.csv", index=False)
    expanding_summary.to_csv(tables / "expanding_window_stability_summary.csv", index=False)
    intervals.to_csv(tables / "full_sample_bootstrap_intervals.csv", index=False)
    full_sample.to_csv(tables / "full_sample_scale_comparison.csv", index=False)

    exact_identity = (
        primary["delta_l_exact_intensity_mse"]
        - primary["delta_l_exact_mse"] / primary["area_time_km2_days"] ** 2
    ).abs().max()
    rmse_identity = (
        primary["l0_exact_intensity_rmse"]
        - np.sqrt(primary["l0_exact_mse"]) / primary["area_time_km2_days"]
    ).abs().max()
    relative_identity = (
        primary["relative_reduction_exact"]
        - primary["relative_reduction_exact_from_intensity"]
    ).abs().max()
    full_stability_columns = [
        "stability_l0_exact_mse",
        "stability_lb_exact_mse",
        "stability_delta_l_exact_mse",
        "stability_delta_h_raw_bits",
    ]
    prohibited = [column for column in primary.columns if column.lower() in {"l_over_n", "delb_per_observation", "error_floor_per_observation"}]
    checks = [
        ("normalized primary rows", len(primary), source_status["primary_rows"]),
        ("normalized expanding rows", len(expanding), source_status["expanding_rows"]),
        ("bootstrap interval groups", len(intervals), 36),
        ("MSE normalization identity", int(exact_identity <= 1e-12), 1),
        ("RMSE normalization identity", int(rmse_identity <= 1e-12), 1),
        ("relative reduction invariant", int(relative_identity <= 1e-12), 1),
        ("full-sample stability equals zero", int((full_sample[full_stability_columns].abs().to_numpy() <= 1e-12).all()), 1),
        ("normalized bound ordering", int((primary["lb_exact_intensity_mse"] <= primary["l0_exact_intensity_mse"] + 1e-12).all()), 1),
        ("bootstrap intervals ordered", int(all((intervals[f"{metric}_low"] <= intervals[f"{metric}_high"]).all() for metric in ["l0_exact_mse", "lb_exact_mse", "delta_l_exact_mse", "delta_l_exact_intensity_mse", "delta_h_raw_bits"])), 1),
        ("no DELB divided by N metric", len(prohibited), 0),
        ("intensity values finite", int(np.isfinite(primary[["l0_exact_intensity_mse", "lb_exact_intensity_mse", "l0_exact_intensity_rmse", "lb_exact_intensity_rmse"]].to_numpy()).all()), 1),
    ]
    acceptance = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    acceptance["status"] = np.where(acceptance["observed"].eq(acceptance["expected"]), "PASS", "FAIL")
    acceptance.to_csv(tables / "acceptance_checklist.csv", index=False)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("S16.5 acceptance failed: " + "; ".join(acceptance.loc[acceptance["status"].ne("PASS"), "check"]))

    _plot_full_sample_heatmaps(full_sample, run_dir / "figures", int(config["reporting"]["figure_dpi"]))
    _plot_stability(summary, run_dir / "figures", int(config["reporting"]["figure_dpi"]))
    pd.DataFrame(
        [
            {"file": path.name, "sha256": _sha256(path), "generator": "empirical/run_e16_scale_normalization.py"}
            for path in sorted((run_dir / "figures").iterdir())
        ]
    ).to_csv(tables / "figure_manifest.csv", index=False)

    elapsed = time.monotonic() - started
    status = {
        "experiment": "E16",
        "stage": "S16.5",
        "status": "complete",
        "run_id": run_id,
        "source_s16_4_run": str(s16_4_run),
        "normalized_primary_rows": len(primary),
        "normalized_expanding_rows": len(expanding),
        "bootstrap_interval_groups": len(intervals),
        "acceptance_passed": int(acceptance["status"].eq("PASS").sum()),
        "acceptance_total": len(acceptance),
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(timezone).isoformat(),
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (run_dir / "interpretation.md").write_text(
        "# S16.5 scale normalization and stability\n\n"
        "Count-level DELBs remain primary. MSE floors for the deterministically scaled crime-intensity target are divided by (area times duration)^2; RMSE floors are divided by area times duration. Sample-size stability is relative deviation from the 100% estimate within a fixed city, spatial scale, temporal scale, and estimator. Repeated-thinning quantiles describe design variability and are not population confidence intervals. Full-sample intervals are paired 28-day block-bootstrap percentile intervals. No DELB-over-N metric, randomization null, prediction model, or manuscript update is produced.\n",
        encoding="utf-8",
    )
    (empirical_root / "runs/latest_e16_s16_5_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    logger.info("S16.5 complete in %.2f seconds", elapsed)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize E16 DELBs and assess sample-size stability")
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

