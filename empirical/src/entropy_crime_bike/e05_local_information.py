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
from statistics import NormalDist
from zoneinfo import ZoneInfo
import zlib

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import pyarrow
from scipy import stats
import scipy
import yaml

from entropy_crime_bike.conditional_information import (
    ConditionalCodebook,
    apply_bicycle_bins,
    crime_lag_bins,
)
from entropy_crime_bike.discrete_bound import (
    closed_form_delb,
    inverse_entropy_envelope,
)
from entropy_crime_bike.spatial_information import (
    BoundInterpolator,
    benjamini_hochberg,
    global_moran,
    local_moran,
    maximum_true_run,
    queen_neighbor_indices,
)


CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {
    "DC": "Washington, DC",
    "NY": "New York City",
    "VAN": "Vancouver",
}
CITY_COLORS = {
    "DC": "#4472C4",
    "NY": "#ED7D31",
    "VAN": "#70AD47",
}
BOOTSTRAP_METRICS = [
    "h0_miller_madow_bits",
    "hb_miller_madow_bits",
    "delta_h_miller_madow_raw_bits",
    "delta_h_projected_bits",
    "l0_exact_mse",
    "lb_exact_mse",
    "delta_l_exact_mse",
    "relative_l_reduction",
]
MORAN_METRICS = [
    "delta_h_miller_madow_raw_bits",
    "delta_l_exact_mse",
    "relative_l_reduction",
]


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return loaded


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e05_local_information")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e05_local_information.log", encoding="utf-8"
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


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(
        path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def _load_city_panel(data_dir: Path, city: str, target: str) -> pd.DataFrame:
    columns = [
        "city",
        "grid_id",
        "x_index",
        "y_index",
        "centroid_longitude",
        "centroid_latitude",
        "date",
        "year",
        "day_of_week",
        "season",
        "is_holiday",
        "split",
        "bike_coverage_training",
        target,
        "crime_count_lag1",
        "bike_total_flow_lag1",
    ]
    files = sorted(
        (data_dir / "grid_day_panel" / f"city={city}").rglob("*.parquet")
    )
    if not files:
        raise FileNotFoundError(f"No accepted E02 panel files for {city}.")
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files],
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _load_e04_thresholds(
    empirical_root: Path, config: dict[str, object]
) -> tuple[Path, pd.DataFrame]:
    pointer = empirical_root / str(config["source_e04_run_pointer"])
    e04_run = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    threshold_path = e04_run / "tables" / "bicycle_bin_thresholds.csv"
    thresholds = pd.read_csv(threshold_path)
    required = {
        "city",
        "source_split",
        "positive_tertile_1_upper",
        "positive_tertile_2_upper",
    }
    if not required.issubset(thresholds.columns):
        raise AssertionError("Accepted E04 threshold table is incomplete.")
    if set(thresholds["city"]) != set(CITY_ORDER):
        raise AssertionError("Accepted E04 threshold table does not cover all cities.")
    if not thresholds["source_split"].eq("train").all():
        raise AssertionError("E05 requires E04 training-only bicycle thresholds.")
    return e04_run, thresholds


def _prepare_city(
    frame: pd.DataFrame,
    cut_points: tuple[float, float],
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = str(config["target"])
    crime_field = str(config["information_sets"]["crime_lag_field"])
    bicycle_field = str(config["information_sets"]["bicycle_lag_field"])
    catalog = (
        frame[
            [
                "city",
                "grid_id",
                "x_index",
                "y_index",
                "centroid_longitude",
                "centroid_latitude",
                "bike_coverage_training",
            ]
        ]
        .drop_duplicates()
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    missing_lag = frame[crime_field].isna() | frame[bicycle_field].isna()
    prepared = frame.loc[~missing_lag].copy()
    prepared["crime_lag_bin"] = crime_lag_bins(prepared[crime_field])
    prepared["bicycle_lag_bin"] = apply_bicycle_bins(
        prepared[bicycle_field], cut_points
    )
    if prepared[["crime_lag_bin", "bicycle_lag_bin"]].isna().any().any():
        raise AssertionError("E05 state mapping produced missing values.")
    target_values = pd.to_numeric(prepared[target], errors="raise")
    if (
        target_values.lt(0).any()
        or not np.equal(target_values, np.floor(target_values)).all()
    ):
        raise AssertionError("E05 target is not a nonnegative integer count.")
    return prepared, catalog


def _eligibility_table(
    prepared: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    target = str(config["target"])
    bicycle_field = str(config["information_sets"]["bicycle_lag_field"])
    rules = config["eligibility"]
    rows = []
    covered = prepared.loc[prepared["bike_coverage_training"]].copy()
    for grid_id, subset in covered.groupby("grid_id", sort=True):
        subset = subset.sort_values("date")
        record = {
            "city": str(subset["city"].iloc[0]),
            "city_label": CITY_LABELS[str(subset["city"].iloc[0])],
            "grid_id": str(grid_id),
            "x_index": int(subset["x_index"].iloc[0]),
            "y_index": int(subset["y_index"].iloc[0]),
            "centroid_longitude": float(subset["centroid_longitude"].iloc[0]),
            "centroid_latitude": float(subset["centroid_latitude"].iloc[0]),
            "valid_days": len(subset),
            "period_start": str(subset["date"].min().date()),
            "period_end": str(subset["date"].max().date()),
            "total_crime_events": int(subset[target].sum()),
            "crime_nonzero_days": int(subset[target].gt(0).sum()),
            "positive_bicycle_days": int(subset[bicycle_field].gt(0).sum()),
            "positive_bicycle_day_share": float(
                subset[bicycle_field].gt(0).mean()
            ),
            "maximum_consecutive_zero_bicycle_days": maximum_true_run(
                subset[bicycle_field].eq(0).to_numpy()
            ),
            "target_states": int(subset[target].nunique()),
            "bicycle_states": int(subset["bicycle_lag_bin"].nunique()),
        }
        checks = {
            "insufficient_valid_days": (
                record["valid_days"] < int(rules["minimum_valid_days"])
            ),
            "insufficient_crime_events": (
                record["total_crime_events"]
                < int(rules["minimum_total_crime_events"])
            ),
            "insufficient_positive_bicycle_days": (
                record["positive_bicycle_days"]
                < int(rules["minimum_positive_bicycle_days"])
            ),
            "severe_bicycle_coverage_discontinuity": (
                record["maximum_consecutive_zero_bicycle_days"]
                > int(rules["maximum_consecutive_zero_bicycle_days"])
            ),
            "insufficient_target_states": (
                record["target_states"]
                < int(rules["minimum_target_states"])
            ),
            "insufficient_bicycle_states": (
                record["bicycle_states"]
                < int(rules["minimum_bicycle_states"])
            ),
        }
        reasons = [name for name, failed in checks.items() if failed]
        record.update({f"fails_{name}": failed for name, failed in checks.items()})
        record["eligible"] = not reasons
        record["exclusion_reasons"] = ";".join(reasons)
        rows.append(record)
    return pd.DataFrame(rows)


def _seed(base_seed: int, label: str) -> int:
    return int((base_seed + zlib.crc32(label.encode("utf-8"))) % 2**32)


def _shared_block_weights(
    block_count: int, repetitions: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        block_count,
        np.full(block_count, 1.0 / block_count),
        size=repetitions,
    )


def _add_point_bound_metrics(
    point: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    result = point.copy()
    h0 = max(
        float(result["h0_miller_madow_bits"].iloc[0]),
        float(config["estimation"]["entropy_floor_bits"]),
    )
    hb_raw = max(
        float(result["hb_miller_madow_bits"].iloc[0]),
        float(config["estimation"]["entropy_floor_bits"]),
    )
    hb = min(hb_raw, h0)
    tail = float(config["numerics"]["lattice_tail_tolerance"])
    root = float(config["numerics"]["root_tolerance"])
    result["h0_for_bound_bits"] = h0
    result["hb_for_bound_bits"] = hb
    result["delta_h_projected_bits"] = h0 - hb
    result["l0_exact_mse"] = inverse_entropy_envelope(
        h0, tail_tolerance=tail, root_tolerance=root
    )
    result["lb_exact_mse"] = inverse_entropy_envelope(
        hb, tail_tolerance=tail, root_tolerance=root
    )
    result["delta_l_exact_mse"] = (
        result["l0_exact_mse"] - result["lb_exact_mse"]
    )
    result["relative_l_reduction"] = np.where(
        result["l0_exact_mse"].gt(0),
        result["delta_l_exact_mse"] / result["l0_exact_mse"],
        np.nan,
    )
    result["l0_closed_mse"] = closed_form_delb(h0)
    result["lb_closed_mse"] = closed_form_delb(hb)
    result["delta_l_closed_mse"] = (
        result["l0_closed_mse"] - result["lb_closed_mse"]
    )
    result["projection_applied"] = hb_raw > h0
    return result


def _add_bootstrap_bound_metrics(
    bootstrap: pd.DataFrame,
    interpolator: BoundInterpolator,
    config: dict[str, object],
) -> pd.DataFrame:
    result = bootstrap.copy()
    floor = float(config["estimation"]["entropy_floor_bits"])
    h0 = np.maximum(result["h0_miller_madow_bits"].to_numpy(float), floor)
    hb_raw = np.maximum(result["hb_miller_madow_bits"].to_numpy(float), floor)
    hb = np.minimum(hb_raw, h0)
    result["h0_for_bound_bits"] = h0
    result["hb_for_bound_bits"] = hb
    result["delta_h_projected_bits"] = h0 - hb
    result["l0_exact_mse"] = interpolator.transform(h0)
    result["lb_exact_mse"] = interpolator.transform(hb)
    result["delta_l_exact_mse"] = (
        result["l0_exact_mse"] - result["lb_exact_mse"]
    )
    result["relative_l_reduction"] = np.where(
        result["l0_exact_mse"].gt(0),
        result["delta_l_exact_mse"] / result["l0_exact_mse"],
        np.nan,
    )
    result["l0_closed_mse"] = [closed_form_delb(value) for value in h0]
    result["lb_closed_mse"] = [closed_form_delb(value) for value in hb]
    result["delta_l_closed_mse"] = (
        result["l0_closed_mse"] - result["lb_closed_mse"]
    )
    result["projection_applied"] = hb_raw > h0
    return result


def _estimate_grid(
    subset: pd.DataFrame,
    *,
    block_weights: np.ndarray,
    interpolator: BoundInterpolator,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    codebook = ConditionalCodebook.from_frame(
        subset,
        target=str(config["target"]),
        baseline_state=list(config["information_sets"]["baseline_state"]),
        bicycle_state_field=str(
            config["information_sets"]["bicycle_state_field"]
        ),
        block_field="block_id",
    )
    if codebook.block_count != block_weights.shape[1]:
        raise AssertionError("Local grid does not contain the shared city blocks.")
    point = _add_point_bound_metrics(
        codebook.estimate(np.ones(codebook.block_count)), config
    )
    chunk_size = int(config["bootstrap"]["matrix_chunk_repetitions"])
    parts = []
    for start in range(0, len(block_weights), chunk_size):
        stop = min(len(block_weights), start + chunk_size)
        part = codebook.estimate(block_weights[start:stop])
        part = _add_bootstrap_bound_metrics(part, interpolator, config)
        part.insert(0, "replicate", np.arange(start + 1, stop + 1))
        parts.append(part)
    bootstrap = pd.concat(parts, ignore_index=True)
    return point, bootstrap, codebook.sparsity_diagnostics()


def _run_city(
    prepared: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    city: str,
    interpolator: BoundInterpolator,
    config: dict[str, object],
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point_path = run_dir / "state" / f"{city}_local_points.parquet"
    bootstrap_path = run_dir / "state" / f"{city}_local_bootstrap.parquet"
    sparsity_path = run_dir / "state" / f"{city}_local_sparsity.parquet"
    if point_path.exists() and bootstrap_path.exists() and sparsity_path.exists():
        logger.info("Loading completed E05 city checkpoint: %s", city)
        return (
            pd.read_parquet(point_path),
            pd.read_parquet(bootstrap_path),
            pd.read_parquet(sparsity_path),
        )
    eligible_ids = set(
        eligibility.loc[eligibility["eligible"], "grid_id"].astype(str)
    )
    subset_city = prepared.loc[
        prepared["grid_id"].astype(str).isin(eligible_ids)
    ].copy()
    block_days = int(config["bootstrap"]["block_days"])
    subset_city["block_id"] = (
        (subset_city["date"] - subset_city["date"].min()).dt.days
        // block_days
    ).astype(np.int16)
    block_count = int(subset_city["block_id"].max()) + 1
    repetitions = int(config["bootstrap"]["repetitions"])
    seed = _seed(int(config["random_seed"]), f"{city}__shared_time_blocks")
    weights = _shared_block_weights(block_count, repetitions, seed)
    np.savez_compressed(
        run_dir / "state" / f"{city}_shared_block_weights.npz",
        weights=weights,
        bootstrap_seed=np.asarray([seed], dtype=np.uint32),
    )
    point_frames = []
    bootstrap_frames = []
    sparsity_rows = []
    for index, grid_id in enumerate(sorted(eligible_ids), start=1):
        subset = subset_city.loc[
            subset_city["grid_id"].astype(str).eq(grid_id)
        ].sort_values("date")
        point, bootstrap, sparsity = _estimate_grid(
            subset,
            block_weights=weights,
            interpolator=interpolator,
            config=config,
        )
        meta = eligibility.loc[eligibility["grid_id"].astype(str).eq(grid_id)].iloc[0]
        for frame in [point, bootstrap]:
            frame.insert(0, "city", city)
            frame.insert(1, "city_label", CITY_LABELS[city])
            frame.insert(2, "grid_id", grid_id)
            frame.insert(3, "x_index", int(meta["x_index"]))
            frame.insert(4, "y_index", int(meta["y_index"]))
            frame.insert(5, "centroid_longitude", float(meta["centroid_longitude"]))
            frame.insert(6, "centroid_latitude", float(meta["centroid_latitude"]))
        point["total_crime_events"] = int(meta["total_crime_events"])
        point["positive_bicycle_days"] = int(meta["positive_bicycle_days"])
        point["positive_bicycle_day_share"] = float(
            meta["positive_bicycle_day_share"]
        )
        point["maximum_consecutive_zero_bicycle_days"] = int(
            meta["maximum_consecutive_zero_bicycle_days"]
        )
        point_frames.append(point)
        bootstrap_frames.append(bootstrap)
        sparsity_rows.append(
            {
                "city": city,
                "grid_id": grid_id,
                **sparsity,
            }
        )
        if index % 25 == 0 or index == len(eligible_ids):
            logger.info(
                "%s local grids %d/%d complete", city, index, len(eligible_ids)
            )
    points = pd.concat(point_frames, ignore_index=True)
    bootstraps = pd.concat(bootstrap_frames, ignore_index=True)
    sparsity = pd.DataFrame(sparsity_rows)
    points.to_parquet(point_path, index=False, compression="zstd")
    bootstraps.to_parquet(bootstrap_path, index=False, compression="zstd")
    sparsity.to_parquet(sparsity_path, index=False, compression="zstd")
    return points, bootstraps, sparsity


def _reconcile_local_to_pooled(
    prepared: pd.DataFrame,
    eligibility: pd.DataFrame,
    local_points: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, float | int | str]:
    eligible_ids = set(
        eligibility.loc[eligibility["eligible"], "grid_id"].astype(str)
    )
    subset = prepared.loc[
        prepared["grid_id"].astype(str).isin(eligible_ids)
    ].copy()
    subset["single_block"] = 0
    codebook = ConditionalCodebook.from_frame(
        subset,
        target=str(config["target"]),
        baseline_state=[
            "grid_id",
            *list(config["information_sets"]["baseline_state"]),
        ],
        bicycle_state_field=str(
            config["information_sets"]["bicycle_state_field"]
        ),
        block_field="single_block",
    )
    pooled = codebook.estimate(np.ones(1)).iloc[0]
    sample_weights = (
        local_points["sample_size"] / local_points["sample_size"].sum()
    )
    record: dict[str, float | int | str] = {
        "city": str(subset["city"].iloc[0]),
        "city_label": CITY_LABELS[str(subset["city"].iloc[0])],
        "eligible_grids": len(local_points),
        "observations": len(subset),
    }
    for metric in [
        "h0_plugin_bits",
        "hb_plugin_bits",
        "delta_h_plugin_raw_bits",
        "h0_miller_madow_bits",
        "hb_miller_madow_bits",
        "delta_h_miller_madow_raw_bits",
    ]:
        weighted = float(np.dot(local_points[metric], sample_weights))
        record[f"pooled_{metric}"] = float(pooled[metric])
        record[f"weighted_local_{metric}"] = weighted
        record[f"absolute_error_{metric}"] = abs(
            float(pooled[metric]) - weighted
        )
    return record


def _summarize_uncertainty(
    points: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    confidence = float(config["bootstrap"]["confidence_level"])
    alpha = 1.0 - confidence
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rows = []
    for _, point in points.iterrows():
        subset = bootstrap.loc[
            bootstrap["city"].eq(point["city"])
            & bootstrap["grid_id"].eq(point["grid_id"])
        ]
        record = point.to_dict()
        for metric in BOOTSTRAP_METRICS:
            values = pd.to_numeric(subset[metric], errors="raise")
            standard_error = float(values.std(ddof=1))
            lower = float(point[metric]) - z_value * standard_error
            upper = float(point[metric]) + z_value * standard_error
            if metric != "delta_h_miller_madow_raw_bits":
                lower = max(0.0, lower)
            if metric == "relative_l_reduction":
                upper = min(1.0, upper)
            record[f"{metric}_bootstrap_mean"] = float(values.mean())
            record[f"{metric}_bootstrap_bias"] = float(
                values.mean() - point[metric]
            )
            record[f"{metric}_se"] = standard_error
            record[f"{metric}_ci_low"] = lower
            record[f"{metric}_ci_high"] = upper
            record[f"{metric}_percentile_low_diagnostic"] = float(
                values.quantile(alpha / 2.0)
            )
            record[f"{metric}_percentile_high_diagnostic"] = float(
                values.quantile(1.0 - alpha / 2.0)
            )
        cmi_se = record["delta_h_miller_madow_raw_bits_se"]
        if cmi_se > 0:
            record["raw_cmi_positive_p_value_one_sided"] = float(
                stats.norm.sf(
                    point["delta_h_miller_madow_raw_bits"] / cmi_se
                )
            )
        else:
            record["raw_cmi_positive_p_value_one_sided"] = float(
                point["delta_h_miller_madow_raw_bits"] <= 0
            )
        record["raw_cmi_bootstrap_nonpositive_share"] = float(
            subset["delta_h_miller_madow_raw_bits"].le(0).mean()
        )
        rows.append(record)
    estimates = pd.DataFrame(rows)
    estimates["raw_cmi_positive_q_value_bh"] = np.nan
    for city, city_index in estimates.groupby("city").groups.items():
        estimates.loc[
            city_index, "raw_cmi_positive_q_value_bh"
        ] = benjamini_hochberg(
            estimates.loc[
                city_index, "raw_cmi_positive_p_value_one_sided"
            ]
        )
    alpha_fdr = float(config["inference"]["false_discovery_rate_alpha"])
    estimates["cmi_two_sided_ci_above_zero"] = estimates[
        "delta_h_miller_madow_raw_bits_ci_low"
    ].gt(0)
    estimates["delta_l_two_sided_ci_above_zero"] = estimates[
        "delta_l_exact_mse_ci_low"
    ].gt(0)
    estimates["significant_positive_bound_reduction"] = (
        estimates["raw_cmi_positive_q_value_bh"].le(alpha_fdr)
        & estimates["cmi_two_sided_ci_above_zero"]
        & estimates["delta_l_two_sided_ci_above_zero"]
    )
    estimates["information_evidence"] = np.select(
        [
            estimates["significant_positive_bound_reduction"],
            estimates["delta_h_miller_madow_raw_bits"].le(0),
            estimates["cmi_two_sided_ci_above_zero"],
        ],
        [
            "positive_after_fdr_and_interval_checks",
            "nonpositive_point_estimate",
            "positive_interval_but_not_fdr_classified",
        ],
        default="positive_but_interval_includes_zero",
    )
    return estimates


def _support_associations(
    estimates: pd.DataFrame, sparsity: pd.DataFrame
) -> pd.DataFrame:
    merged = estimates.merge(
        sparsity[
            [
                "city",
                "grid_id",
                "bicycle_observation_share_in_singleton_states",
                "bicycle_median_observations_per_state",
            ]
        ],
        on=["city", "grid_id"],
        validate="one_to_one",
    )
    drivers = [
        "total_crime_events",
        "positive_bicycle_days",
        "h0_miller_madow_bits",
        "bicycle_observation_share_in_singleton_states",
    ]
    rows = []
    for city in CITY_ORDER:
        city_data = merged.loc[merged["city"].eq(city)]
        for driver in drivers:
            coefficient, p_value = stats.spearmanr(
                city_data["delta_h_miller_madow_raw_bits"],
                city_data[driver],
            )
            rows.append(
                {
                    "city": city,
                    "city_label": CITY_LABELS[city],
                    "outcome": "delta_h_miller_madow_raw_bits",
                    "support_driver": driver,
                    "spearman_rho": float(coefficient),
                    "spearman_p_value_two_sided": float(p_value),
                    "grids": len(city_data),
                }
            )
    return pd.DataFrame(rows)


def _spatial_analysis(
    estimates: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repetitions = int(config["spatial"]["permutation_repetitions"])
    global_rows = []
    local_frames = []
    for city in CITY_ORDER:
        city_data = (
            estimates.loc[estimates["city"].eq(city)]
            .sort_values("grid_id")
            .reset_index(drop=True)
        )
        neighbors = queen_neighbor_indices(
            city_data["x_index"], city_data["y_index"]
        )
        for metric in MORAN_METRICS:
            label = f"{city}__{metric}"
            seed = _seed(int(config["random_seed"]), label)
            global_rows.append(
                {
                    "city": city,
                    "city_label": CITY_LABELS[city],
                    "metric": metric,
                    "weights": str(config["spatial"]["weights"]),
                    **global_moran(
                        city_data[metric],
                        neighbors,
                        permutations=repetitions,
                        seed=seed,
                    ),
                }
            )
            local = local_moran(
                city_data[metric],
                neighbors,
                permutations=repetitions,
                seed=seed + 1,
            )
            local.insert(0, "city", city)
            local.insert(1, "city_label", CITY_LABELS[city])
            local.insert(2, "grid_id", city_data["grid_id"].to_numpy())
            local.insert(3, "x_index", city_data["x_index"].to_numpy())
            local.insert(4, "y_index", city_data["y_index"].to_numpy())
            local.insert(5, "metric", metric)
            local["local_moran_q_value_bh"] = benjamini_hochberg(
                local["permutation_p_value_two_sided"]
            )
            local["local_moran_significant"] = local[
                "local_moran_q_value_bh"
            ].le(float(config["inference"]["false_discovery_rate_alpha"]))
            local["cluster_significant"] = np.where(
                local["local_moran_significant"],
                local["cluster"],
                "not_significant",
            )
            local_frames.append(local)
    return pd.DataFrame(global_rows), pd.concat(local_frames, ignore_index=True)


def _city_summary(
    eligibility: pd.DataFrame, estimates: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for city in CITY_ORDER:
        covered = eligibility.loc[eligibility["city"].eq(city)]
        local = estimates.loc[estimates["city"].eq(city)]
        significant = local["significant_positive_bound_reduction"]
        rows.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "covered_grids": len(covered),
                "eligible_grids": len(local),
                "eligible_share_of_covered": len(local) / len(covered),
                "positive_raw_cmi_grids": int(
                    local["delta_h_miller_madow_raw_bits"].gt(0).sum()
                ),
                "significant_positive_bound_reduction_grids": int(
                    significant.sum()
                ),
                "significant_share_of_eligible": float(significant.mean()),
                "median_raw_cmi_bits": float(
                    local["delta_h_miller_madow_raw_bits"].median()
                ),
                "mean_raw_cmi_bits": float(
                    local["delta_h_miller_madow_raw_bits"].mean()
                ),
                "median_exact_delb_reduction_mse": float(
                    local["delta_l_exact_mse"].median()
                ),
                "mean_exact_delb_reduction_mse": float(
                    local["delta_l_exact_mse"].mean()
                ),
                "median_relative_bound_reduction": float(
                    local["relative_l_reduction"].median()
                ),
                "mean_relative_bound_reduction": float(
                    local["relative_l_reduction"].mean()
                ),
                "projection_applied_grids": int(
                    local["projection_applied"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _rectangles(frame: pd.DataFrame) -> list[Rectangle]:
    return [
        Rectangle((float(row.x_index) - 0.5, float(row.y_index) - 0.5), 1, 1)
        for row in frame.itertuples()
    ]


def _map_metric(
    catalog: pd.DataFrame,
    eligibility: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    metric: str,
    title: str,
    colorbar_label: str,
    path: Path,
    dpi: int,
    cmap: str,
    diverging: bool = False,
    upper_limit: float | None = None,
) -> None:
    values = estimates[metric].to_numpy(float)
    if diverging:
        extent = float(np.quantile(np.abs(values), 0.98))
        extent = max(extent, np.finfo(float).eps)
        normalization = colors.TwoSlopeNorm(
            vmin=-extent, vcenter=0.0, vmax=extent
        )
    else:
        maximum = (
            float(upper_limit)
            if upper_limit is not None
            else float(np.quantile(values, 0.98))
        )
        maximum = max(maximum, np.finfo(float).eps)
        normalization = colors.Normalize(vmin=0.0, vmax=maximum)
    figure, axes = plt.subplots(
        1, 3, figsize=(10.5, 4.2), constrained_layout=True
    )
    last_collection = None
    eligible_ids = set(estimates["grid_id"].astype(str))
    covered_ids = set(eligibility["grid_id"].astype(str))
    for axis, city in zip(axes, CITY_ORDER, strict=True):
        city_catalog = catalog.loc[catalog["city"].eq(city)]
        outside = city_catalog.loc[
            ~city_catalog["grid_id"].astype(str).isin(covered_ids)
        ]
        excluded = city_catalog.loc[
            city_catalog["grid_id"].astype(str).isin(covered_ids)
            & ~city_catalog["grid_id"].astype(str).isin(eligible_ids)
        ]
        city_values = estimates.loc[estimates["city"].eq(city)]
        if len(outside):
            axis.add_collection(
                PatchCollection(
                    _rectangles(outside),
                    facecolor="#F2F2F2",
                    edgecolor="#FFFFFF",
                    linewidth=0.15,
                )
            )
        if len(excluded):
            axis.add_collection(
                PatchCollection(
                    _rectangles(excluded),
                    facecolor="#C9C9C9",
                    edgecolor="#FFFFFF",
                    linewidth=0.15,
                )
            )
        collection = PatchCollection(
            _rectangles(city_values),
            cmap=cmap,
            norm=normalization,
            edgecolor="#FFFFFF",
            linewidth=0.15,
        )
        collection.set_array(city_values[metric].to_numpy(float))
        axis.add_collection(collection)
        last_collection = collection
        axis.autoscale_view()
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{CITY_LABELS[city]}\n$n={len(city_values)}$ eligible grids",
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(title, fontsize=13, fontweight="bold")
    colorbar = figure.colorbar(
        last_collection, ax=axes, shrink=0.78, pad=0.02
    )
    colorbar.set_label(colorbar_label)
    figure.legend(
        handles=[
            Patch(facecolor="#F2F2F2", label="Outside bicycle footprint"),
            Patch(facecolor="#C9C9C9", label="Covered but ineligible"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    _save_figure(figure, path, dpi)


def _map_significance(
    catalog: pd.DataFrame,
    eligibility: pd.DataFrame,
    estimates: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(10.5, 4.2), constrained_layout=True
    )
    covered_ids = set(eligibility["grid_id"].astype(str))
    eligible_ids = set(estimates["grid_id"].astype(str))
    for axis, city in zip(axes, CITY_ORDER, strict=True):
        city_catalog = catalog.loc[catalog["city"].eq(city)]
        groups = [
            (
                city_catalog.loc[
                    ~city_catalog["grid_id"].astype(str).isin(covered_ids)
                ],
                "#F2F2F2",
            ),
            (
                city_catalog.loc[
                    city_catalog["grid_id"].astype(str).isin(covered_ids)
                    & ~city_catalog["grid_id"].astype(str).isin(eligible_ids)
                ],
                "#C9C9C9",
            ),
            (
                estimates.loc[
                    estimates["city"].eq(city)
                    & ~estimates["significant_positive_bound_reduction"]
                ],
                "#9ECAE1",
            ),
            (
                estimates.loc[
                    estimates["city"].eq(city)
                    & estimates["significant_positive_bound_reduction"]
                ],
                "#238B45",
            ),
        ]
        for frame, color in groups:
            if len(frame):
                axis.add_collection(
                    PatchCollection(
                        _rectangles(frame),
                        facecolor=color,
                        edgecolor="#FFFFFF",
                        linewidth=0.15,
                    )
                )
        axis.autoscale_view()
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        city_estimates = estimates.loc[estimates["city"].eq(city)]
        significant = int(
            city_estimates["significant_positive_bound_reduction"].sum()
        )
        axis.set_title(
            f"{CITY_LABELS[city]}\n{significant}/{len(city_estimates)} "
            "pass sampling screen",
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(
        "Local bicycle-information sampling-uncertainty screen",
        fontsize=13,
        fontweight="bold",
    )
    figure.legend(
        handles=[
            Patch(facecolor="#F2F2F2", label="Outside bicycle footprint"),
            Patch(facecolor="#C9C9C9", label="Covered but ineligible"),
            Patch(facecolor="#9ECAE1", label="Eligible, does not pass screen"),
            Patch(facecolor="#238B45", label="Passes FDR + interval screen"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    _save_figure(figure, path, dpi)


def _map_lisa(
    catalog: pd.DataFrame,
    eligibility: pd.DataFrame,
    local_moran_results: pd.DataFrame,
    config: dict[str, object],
    path: Path,
    dpi: int,
) -> None:
    metric = str(config["spatial"]["local_moran_metric_for_primary_map"])
    local = local_moran_results.loc[
        local_moran_results["metric"].eq(metric)
    ]
    palette = {
        "not_significant": "#D9D9D9",
        "high_high": "#B2182B",
        "low_low": "#2166AC",
        "high_low": "#EF8A62",
        "low_high": "#67A9CF",
        "island": "#969696",
    }
    figure, axes = plt.subplots(
        1, 3, figsize=(10.5, 4.2), constrained_layout=True
    )
    covered_ids = set(eligibility["grid_id"].astype(str))
    for axis, city in zip(axes, CITY_ORDER, strict=True):
        city_catalog = catalog.loc[catalog["city"].eq(city)]
        outside = city_catalog.loc[
            ~city_catalog["grid_id"].astype(str).isin(covered_ids)
        ]
        if len(outside):
            axis.add_collection(
                PatchCollection(
                    _rectangles(outside),
                    facecolor="#F7F7F7",
                    edgecolor="#FFFFFF",
                    linewidth=0.15,
                )
            )
        city_local = local.loc[local["city"].eq(city)]
        for cluster, color in palette.items():
            frame = city_local.loc[
                city_local["cluster_significant"].eq(cluster)
            ]
            if len(frame):
                axis.add_collection(
                    PatchCollection(
                        _rectangles(frame),
                        facecolor=color,
                        edgecolor="#FFFFFF",
                        linewidth=0.15,
                    )
                )
        axis.autoscale_view()
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(CITY_LABELS[city], fontweight="bold")
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(
        "Local Moran clusters of exact DELB reduction",
        fontsize=13,
        fontweight="bold",
    )
    figure.legend(
        handles=[
            Patch(facecolor=palette[key], label=key.replace("_", " ").title())
            for key in [
                "high_high",
                "low_low",
                "high_low",
                "low_high",
                "not_significant",
                "island",
            ]
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    _save_figure(figure, path, dpi)


def _plot_distributions(
    estimates: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(10.3, 3.8), constrained_layout=True
    )
    definitions = [
        (
            "delta_h_miller_madow_raw_bits",
            "Local CMI (bits)",
            "Conditional information",
        ),
        (
            "delta_l_exact_mse",
            "Exact DELB reduction (MSE)",
            "Absolute bound reduction",
        ),
        (
            "relative_l_reduction",
            "Relative reduction",
            "Relative bound reduction",
        ),
    ]
    rng = np.random.default_rng(20260718)
    for axis, (metric, ylabel, title) in zip(
        axes, definitions, strict=True
    ):
        data = [
            estimates.loc[estimates["city"].eq(city), metric].to_numpy(float)
            for city in CITY_ORDER
        ]
        box = axis.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "#1F1F1F", "linewidth": 1.2},
        )
        for patch, city in zip(box["boxes"], CITY_ORDER, strict=True):
            patch.set_facecolor(CITY_COLORS[city])
            patch.set_alpha(0.55)
        for index, (city, values) in enumerate(
            zip(CITY_ORDER, data, strict=True), start=1
        ):
            jitter = rng.normal(index, 0.045, size=len(values))
            axis.scatter(
                jitter,
                values,
                s=7,
                color=CITY_COLORS[city],
                alpha=0.45,
                linewidths=0,
            )
        axis.axhline(0, color="#7F7F7F", linewidth=0.8)
        axis.set_xticks(
            [1, 2, 3], ["DC", "NYC", "VAN"]
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", color="#E7E6E6", linewidth=0.6)
        if metric == "relative_l_reduction":
            axis.yaxis.set_major_formatter(
                plt.matplotlib.ticker.PercentFormatter(1.0)
            )
    figure.suptitle(
        "Distribution of local bicycle information value",
        fontsize=13,
        fontweight="bold",
    )
    _save_figure(figure, path, dpi)


def _plot_support(
    eligibility: pd.DataFrame, path: Path, dpi: int
) -> None:
    rows = []
    for city in CITY_ORDER:
        subset = eligibility.loc[eligibility["city"].eq(city)]
        rows.append(
            {
                "city": city,
                "eligible": int(subset["eligible"].sum()),
                "excluded": int((~subset["eligible"]).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    figure, axis = plt.subplots(
        figsize=(6.7, 4.0), constrained_layout=True
    )
    x_values = np.arange(len(summary))
    axis.bar(
        x_values,
        summary["eligible"],
        color="#4472C4",
        label="Eligible",
    )
    axis.bar(
        x_values,
        summary["excluded"],
        bottom=summary["eligible"],
        color="#BFBFBF",
        label="Excluded by prespecified support rules",
    )
    axis.set_xticks(
        x_values, [CITY_LABELS[city] for city in CITY_ORDER]
    )
    axis.set_ylabel("Training bicycle-covered 1 km grids")
    axis.set_title(
        "Local-analysis eligibility",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#E7E6E6", linewidth=0.6)
    _save_figure(figure, path, dpi)


def _plot_support_sensitivity(
    estimates: pd.DataFrame,
    sparsity: pd.DataFrame,
    support_associations: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    merged = estimates.merge(
        sparsity[
            [
                "city",
                "grid_id",
                "bicycle_observation_share_in_singleton_states",
            ]
        ],
        on=["city", "grid_id"],
        validate="one_to_one",
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(10.3, 3.7), constrained_layout=True
    )
    for axis, city in zip(axes, CITY_ORDER, strict=True):
        city_data = merged.loc[merged["city"].eq(city)]
        diagnostic = support_associations.loc[
            support_associations["city"].eq(city)
            & support_associations["support_driver"].eq(
                "bicycle_observation_share_in_singleton_states"
            )
        ].iloc[0]
        axis.scatter(
            city_data["bicycle_observation_share_in_singleton_states"],
            city_data["delta_h_miller_madow_raw_bits"],
            s=15,
            color=CITY_COLORS[city],
            alpha=0.65,
            linewidths=0,
        )
        axis.text(
            0.04,
            0.94,
            f"Spearman $\\rho={diagnostic['spearman_rho']:.2f}$",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
        )
        axis.set_title(CITY_LABELS[city], fontweight="bold")
        axis.set_xlabel("Observations in singleton augmented states")
        axis.set_ylabel("Local CMI (bits)")
        axis.xaxis.set_major_formatter(
            plt.matplotlib.ticker.PercentFormatter(1.0)
        )
        axis.grid(color="#E7E6E6", linewidth=0.6)
    figure.suptitle(
        "Finite-sample support diagnostic for local information estimates",
        fontsize=13,
        fontweight="bold",
    )
    _save_figure(figure, path, dpi)


def _acceptance_checks(
    eligibility: pd.DataFrame,
    estimates: pd.DataFrame,
    bootstrap: pd.DataFrame,
    interpolation_validation: pd.DataFrame,
    global_moran_results: pd.DataFrame,
    local_moran_results: pd.DataFrame,
    thresholds: pd.DataFrame,
    reconciliation: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    expected_by_city = {
        str(key): int(value)
        for key, value in config["acceptance"][
            "expected_eligible_by_city"
        ].items()
    }
    observed_by_city = (
        estimates.groupby("city")["grid_id"].nunique().to_dict()
    )
    cmi_identity_error = (
        estimates["h0_miller_madow_bits"]
        - estimates["hb_miller_madow_bits"]
        - estimates["delta_h_miller_madow_raw_bits"]
    ).abs()
    repetitions = bootstrap.groupby(["city", "grid_id"])[
        "replicate"
    ].nunique()
    reconciliation_errors = reconciliation.filter(
        regex=r"^absolute_error_"
    ).to_numpy(float)
    checks = [
        {
            "check": "Expected bicycle-covered grids",
            "criterion": f"equals {config['acceptance']['expected_covered_grids']}",
            "observed": len(eligibility),
            "pass": len(eligibility)
            == int(config["acceptance"]["expected_covered_grids"]),
        },
        {
            "check": "Expected eligible local grids",
            "criterion": f"equals {config['acceptance']['expected_eligible_grids']}",
            "observed": len(estimates),
            "pass": len(estimates)
            == int(config["acceptance"]["expected_eligible_grids"]),
        },
        {
            "check": "Expected eligible grids by city",
            "criterion": str(expected_by_city),
            "observed": str(observed_by_city),
            "pass": observed_by_city == expected_by_city,
        },
        {
            "check": "Expected local bootstrap rows",
            "criterion": f"equals {config['acceptance']['expected_bootstrap_replicates']}",
            "observed": len(bootstrap),
            "pass": len(bootstrap)
            == int(config["acceptance"]["expected_bootstrap_replicates"]),
        },
        {
            "check": "Bootstrap repetitions per local grid",
            "criterion": f"all equal {config['bootstrap']['repetitions']}",
            "observed": int(repetitions.min()),
            "pass": bool(
                repetitions.eq(int(config["bootstrap"]["repetitions"])).all()
            ),
        },
        {
            "check": "Conditional mutual-information identity",
            "criterion": (
                "maximum error <= "
                f"{config['acceptance']['cmi_identity_tolerance_bits']} bits"
            ),
            "observed": float(cmi_identity_error.max()),
            "pass": bool(
                cmi_identity_error.max()
                <= float(
                    config["acceptance"]["cmi_identity_tolerance_bits"]
                )
            ),
        },
        {
            "check": "Projected local information is nonnegative",
            "criterion": "minimum >= 0",
            "observed": float(estimates["delta_h_projected_bits"].min()),
            "pass": bool(estimates["delta_h_projected_bits"].ge(0).all()),
        },
        {
            "check": "Bicycle-aware local exact DELB ordering",
            "criterion": "LB <= L0 for every eligible grid",
            "observed": float(
                (estimates["lb_exact_mse"] - estimates["l0_exact_mse"]).max()
            ),
            "pass": bool(
                (
                    estimates["lb_exact_mse"]
                    <= estimates["l0_exact_mse"]
                    + float(config["acceptance"]["bound_order_tolerance"])
                ).all()
            ),
        },
        {
            "check": "Local exact bound reductions are nonnegative",
            "criterion": "minimum >= 0",
            "observed": float(estimates["delta_l_exact_mse"].min()),
            "pass": bool(estimates["delta_l_exact_mse"].ge(-1e-12).all()),
        },
        {
            "check": "Bootstrap DELB interpolation validation",
            "criterion": (
                "maximum absolute error <= "
                f"{config['numerics']['bootstrap_bound_interpolation_validation_tolerance_mse']}"
            ),
            "observed": float(
                interpolation_validation["absolute_error_mse"].max()
            ),
            "pass": bool(
                interpolation_validation["absolute_error_mse"].max()
                <= float(
                    config["numerics"][
                        "bootstrap_bound_interpolation_validation_tolerance_mse"
                    ]
                )
            ),
        },
        {
            "check": "Expected global Moran results",
            "criterion": f"equals {config['acceptance']['expected_global_moran_rows']}",
            "observed": len(global_moran_results),
            "pass": len(global_moran_results)
            == int(config["acceptance"]["expected_global_moran_rows"]),
        },
        {
            "check": "Expected local Moran results",
            "criterion": f"equals {config['acceptance']['expected_local_moran_rows']}",
            "observed": len(local_moran_results),
            "pass": len(local_moran_results)
            == int(config["acceptance"]["expected_local_moran_rows"]),
        },
        {
            "check": "FDR p and q values are valid",
            "criterion": "all finite values within [0, 1]",
            "observed": int(
                estimates[
                    [
                        "raw_cmi_positive_p_value_one_sided",
                        "raw_cmi_positive_q_value_bh",
                    ]
                ]
                .apply(lambda column: column.between(0, 1).all())
                .sum()
            ),
            "pass": bool(
                estimates[
                    [
                        "raw_cmi_positive_p_value_one_sided",
                        "raw_cmi_positive_q_value_bh",
                    ]
                ]
                .apply(lambda column: column.between(0, 1).all())
                .all()
            ),
        },
        {
            "check": "All E05 primary numerical estimates are finite",
            "criterion": "zero non-finite values",
            "observed": int(
                (~np.isfinite(
                    estimates[
                        [
                            "h0_miller_madow_bits",
                            "hb_miller_madow_bits",
                            "delta_h_miller_madow_raw_bits",
                            "l0_exact_mse",
                            "lb_exact_mse",
                            "delta_l_exact_mse",
                            "relative_l_reduction",
                        ]
                    ].to_numpy(float)
                )).sum()
            ),
            "pass": bool(
                np.isfinite(
                    estimates[
                        [
                            "h0_miller_madow_bits",
                            "hb_miller_madow_bits",
                            "delta_h_miller_madow_raw_bits",
                            "l0_exact_mse",
                            "lb_exact_mse",
                            "delta_l_exact_mse",
                            "relative_l_reduction",
                        ]
                    ].to_numpy(float)
                ).all()
            ),
        },
        {
            "check": "Frozen E04 training thresholds reused",
            "criterion": "three cities and source_split=train",
            "observed": int(thresholds["source_split"].eq("train").sum()),
            "pass": bool(
                len(thresholds) == 3
                and thresholds["source_split"].eq("train").all()
            ),
        },
        {
            "check": "Local-to-pooled conditional entropy reconciliation",
            "criterion": (
                "maximum error <= "
                f"{config['acceptance']['local_to_pooled_reconciliation_tolerance_bits']} bits"
            ),
            "observed": float(reconciliation_errors.max()),
            "pass": bool(
                reconciliation_errors.max()
                <= float(
                    config["acceptance"][
                        "local_to_pooled_reconciliation_tolerance_bits"
                    ]
                )
            ),
        },
        {
            "check": "Every eligible grid has all strict-lag days",
            "criterion": f"minimum equals {config['eligibility']['minimum_valid_days']}",
            "observed": int(
                eligibility.loc[eligibility["eligible"], "valid_days"].min()
            ),
            "pass": bool(
                eligibility.loc[eligibility["eligible"], "valid_days"].ge(
                    int(config["eligibility"]["minimum_valid_days"])
                ).all()
            ),
        },
    ]
    frame = pd.DataFrame(checks)
    frame["status"] = np.where(frame["pass"], "PASS", "FAIL")
    return frame.drop(columns="pass")


def _write_interpretation(
    run_dir: Path,
    summary: pd.DataFrame,
    global_moran_results: pd.DataFrame,
    estimates: pd.DataFrame,
    support_associations: pd.DataFrame,
) -> None:
    lines = [
        "# E05 Spatially Localized Bicycle Information Results",
        "",
        "## Prespecified estimand and eligibility",
        "",
        (
            "Local conditional entropy is estimated separately within each "
            "eligible 1 km grid. The baseline state contains lagged crime bin, "
            "day of week, season, and holiday status; the augmented state adds "
            "the frozen E04 city-specific lagged bicycle-flow bin."
        ),
        "",
        "## City summaries",
        "",
    ]
    for row in summary.itertuples():
        lines.extend(
            [
                f"### {row.city_label}",
                "",
                (
                    f"- Eligible grids: {row.eligible_grids} of "
                    f"{row.covered_grids} training bicycle-covered grids."
                ),
                (
                    f"- Median raw local CMI: "
                    f"{row.median_raw_cmi_bits:.4f} bits."
                ),
                (
                    f"- Median exact DELB reduction: "
                    f"{row.median_exact_delb_reduction_mse:.5f} MSE units."
                ),
                (
                    f"- Median relative bound reduction: "
                    f"{100 * row.median_relative_bound_reduction:.2f}%."
                ),
                (
                    f"- Positive local reductions after FDR and both interval "
                    f"checks: "
                    f"{row.significant_positive_bound_reduction_grids} "
                    f"({100 * row.significant_share_of_eligible:.1f}%)."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Spatial autocorrelation",
            "",
        ]
    )
    for row in global_moran_results.loc[
        global_moran_results["metric"].eq("delta_l_exact_mse")
    ].itertuples():
        lines.append(
            f"- {row.city_label}: global Moran's I for exact DELB reduction "
            f"was {row.moran_i:.3f} (spatial-randomization "
            f"$p={row.permutation_p_value_two_sided:.3f}$)."
        )
    lines.extend(
        [
            "",
            "## Finite-sample support diagnostic",
            "",
        ]
    )
    for city in CITY_ORDER:
        row = support_associations.loc[
            support_associations["city"].eq(city)
            & support_associations["support_driver"].eq(
                "bicycle_observation_share_in_singleton_states"
            )
        ].iloc[0]
        lines.append(
            f"- {CITY_LABELS[city]}: Spearman correlation between raw local "
            f"CMI and the observation share in singleton augmented states was "
            f"{row['spearman_rho']:.3f} "
            f"($p={row['spearman_p_value_two_sided']:.3g}$)."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            (
                "- Local CMI and bound reductions are predictive-information "
                "measures, not causal bicycle effects."
            ),
            (
                "- A theoretical lower-bound reduction is not a guaranteed "
                "realized error reduction for a fitted model."
            ),
            (
                "- The local multiple-testing classification is deliberately "
                "conservative: it combines a one-sided FDR-adjusted CMI test "
                "with two-sided CMI and DELB interval checks."
            ),
            (
                "- Nevertheless, the block bootstrap quantifies sampling "
                "variability around the observed joint distribution and is not "
                "a null-independence randomization. The high screen-pass rates "
                "and strong association with singleton-state support mean that "
                "E10 permutation placebos are required before calling local "
                "effects statistically significant."
            ),
            (
                "- Spatial clusters describe geographic concentration among "
                "eligible bicycle-covered grids. They do not generalize to "
                "areas outside the observed system footprint."
            ),
            (
                f"- {int(estimates['projection_applied'].sum())} grids required "
                "nonnegative projection for theoretical-bound insertion; raw "
                "CMI remains unchanged in all tables."
            ),
            (
                "- E09 estimator/scale robustness and E10 temporal/spatial "
                "placebos are required before final manuscript claims."
            ),
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
) -> None:
    registry = empirical_root / "docs" / "experiment_registry.csv"
    existing = pd.read_csv(registry)
    if run_id in set(existing["run_id"].astype(str)):
        return
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "experiment": "E05",
                "status": "complete",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "run_directory": str(run_dir),
                "notes": (
                    "Grid-level conditional information, exact DELB reduction, "
                    "FDR classification, and Moran spatial diagnostics."
                ),
            }
        ]
    ).to_csv(registry, mode="a", header=False, index=False)


def run(config_path: Path, resume_run: Path | None = None) -> Path:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    empirical_root = Path(__file__).resolve().parents[2]
    timezone = ZoneInfo("Asia/Shanghai")
    started_at = datetime.now(timezone)
    if resume_run is None:
        run_id = (
            f"{started_at.strftime('%Y%m%d_%H%M%S')}_"
            "E05_local_spatial_information"
        )
        run_dir = empirical_root / "runs" / run_id
    else:
        run_dir = resume_run.resolve()
        run_id = run_dir.name
    for directory in ["logs", "tables", "figures", "state"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    command = (
        f"{sys.executable} empirical/run_e05_local_information.py "
        f"--config {config_path}"
    )
    if resume_run is not None:
        command += f" --resume-run {run_dir}"
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    if not (run_dir / "environment.txt").exists():
        _write_environment(run_dir / "environment.txt")
    started_clock = time.monotonic()
    logger.info("Starting E05 local spatial information experiment")
    try:
        e02_pointer = empirical_root / str(
            config["source_e02_data_pointer"]
        )
        data_dir = Path(
            e02_pointer.read_text(encoding="utf-8").strip()
        ).resolve()
        e04_run, thresholds = _load_e04_thresholds(
            empirical_root, config
        )
        tail = float(config["numerics"]["lattice_tail_tolerance"])
        root = float(config["numerics"]["root_tolerance"])
        interpolator = BoundInterpolator.build(
            float(
                config["numerics"][
                    "bootstrap_bound_interpolation_max_entropy_bits"
                ]
            ),
            tail_tolerance=tail,
            root_tolerance=root,
        )
        interpolation_validation = interpolator.midpoint_validation(
            tail_tolerance=tail, root_tolerance=root
        )
        interpolation_grid = pd.DataFrame(
            {
                "entropy_bits": interpolator.entropy_grid,
                "exact_delb_mse": interpolator.exact_bound_grid,
                "log1p_exact_delb": np.log1p(
                    interpolator.exact_bound_grid
                ),
            }
        )
        all_eligibility = []
        all_catalog = []
        all_points = []
        all_bootstraps = []
        all_sparsity = []
        reconciliation_records = []
        for city in CITY_ORDER:
            threshold = thresholds.loc[thresholds["city"].eq(city)].iloc[0]
            cut_points = (
                float(threshold["positive_tertile_1_upper"]),
                float(threshold["positive_tertile_2_upper"]),
            )
            raw = _load_city_panel(data_dir, city, str(config["target"]))
            prepared, catalog = _prepare_city(raw, cut_points, config)
            eligibility = _eligibility_table(prepared, config)
            all_eligibility.append(eligibility)
            all_catalog.append(catalog)
            logger.info(
                "%s eligibility fixed: %d/%d covered grids",
                city,
                int(eligibility["eligible"].sum()),
                len(eligibility),
            )
            points, bootstraps, sparsity = _run_city(
                prepared,
                eligibility,
                city=city,
                interpolator=interpolator,
                config=config,
                run_dir=run_dir,
                logger=logger,
            )
            all_points.append(points)
            all_bootstraps.append(bootstraps)
            all_sparsity.append(sparsity)
            reconciliation_records.append(
                _reconcile_local_to_pooled(
                    prepared, eligibility, points, config
                )
            )
            del raw, prepared
        eligibility = pd.concat(all_eligibility, ignore_index=True)
        catalog = pd.concat(all_catalog, ignore_index=True)
        points = pd.concat(all_points, ignore_index=True)
        bootstrap = pd.concat(all_bootstraps, ignore_index=True)
        sparsity = pd.concat(all_sparsity, ignore_index=True)
        reconciliation = pd.DataFrame(reconciliation_records)
        estimates = _summarize_uncertainty(points, bootstrap, config)
        estimates.sort_values(["city", "grid_id"], inplace=True)
        support_associations = _support_associations(estimates, sparsity)
        global_moran_results, local_moran_results = _spatial_analysis(
            estimates, config
        )
        summary = _city_summary(eligibility, estimates)
        acceptance = _acceptance_checks(
            eligibility,
            estimates,
            bootstrap,
            interpolation_validation,
            global_moran_results,
            local_moran_results,
            thresholds,
            reconciliation,
            config,
        )
        table_dir = run_dir / "tables"
        eligibility.to_csv(table_dir / "grid_eligibility.csv", index=False)
        estimates.to_csv(
            table_dir / "local_information_estimates.csv", index=False
        )
        estimates.to_parquet(
            table_dir / "local_information_estimates.parquet",
            index=False,
            compression="zstd",
        )
        bootstrap.to_parquet(
            table_dir / "local_bootstrap_replicates.parquet",
            index=False,
            compression="zstd",
        )
        sparsity.to_csv(
            table_dir / "local_state_sparsity.csv", index=False
        )
        reconciliation.to_csv(
            table_dir / "city_local_to_pooled_reconciliation.csv",
            index=False,
        )
        support_associations.to_csv(
            table_dir / "support_association_diagnostics.csv",
            index=False,
        )
        summary.to_csv(
            table_dir / "city_local_summary.csv", index=False
        )
        global_moran_results.to_csv(
            table_dir / "global_moran_results.csv", index=False
        )
        local_moran_results.to_csv(
            table_dir / "local_moran_results.csv", index=False
        )
        thresholds.to_csv(
            table_dir / "e04_bicycle_thresholds_reused.csv", index=False
        )
        interpolation_grid.to_csv(
            table_dir / "bootstrap_delb_interpolation_grid.csv", index=False
        )
        interpolation_validation.to_csv(
            table_dir / "bootstrap_delb_interpolation_validation.csv",
            index=False,
        )
        acceptance.to_csv(
            table_dir / "acceptance_checklist.csv", index=False
        )
        failed = acceptance.loc[acceptance["status"].ne("PASS")]
        if len(failed):
            raise AssertionError(
                "E05 acceptance gate failed: "
                + ", ".join(failed["check"].astype(str))
            )
        _publication_style()
        dpi = int(config["reporting"]["figure_dpi"])
        figure_dir = run_dir / "figures"
        _plot_support(
            eligibility, figure_dir / "e05_eligibility_support.png", dpi
        )
        _map_metric(
            catalog,
            eligibility,
            estimates,
            metric="delta_h_miller_madow_raw_bits",
            title="Local conditional information from lagged bicycle mobility",
            colorbar_label="Raw Miller–Madow CMI (bits)",
            path=figure_dir / "e05_local_cmi_maps.png",
            dpi=dpi,
            cmap="RdBu_r",
            diverging=True,
        )
        _map_metric(
            catalog,
            eligibility,
            estimates,
            metric="delta_l_exact_mse",
            title="Local reduction in the exact discrete prediction bound",
            colorbar_label="Exact DELB reduction (MSE units)",
            path=figure_dir / "e05_local_delb_reduction_maps.png",
            dpi=dpi,
            cmap="viridis",
        )
        _map_metric(
            catalog,
            eligibility,
            estimates,
            metric="relative_l_reduction",
            title="Relative local reduction in the exact discrete bound",
            colorbar_label="Relative exact DELB reduction",
            path=figure_dir / "e05_relative_bound_reduction_maps.png",
            dpi=dpi,
            cmap="magma",
            upper_limit=1.0,
        )
        _map_significance(
            catalog,
            eligibility,
            estimates,
            figure_dir / "e05_significance_maps.png",
            dpi,
        )
        _map_lisa(
            catalog,
            eligibility,
            local_moran_results,
            config,
            figure_dir / "e05_local_moran_clusters.png",
            dpi,
        )
        _plot_distributions(
            estimates,
            figure_dir / "e05_local_information_distributions.png",
            dpi,
        )
        _plot_support_sensitivity(
            estimates,
            sparsity,
            support_associations,
            figure_dir / "e05_support_sensitivity.png",
            dpi,
        )
        _write_interpretation(
            run_dir,
            summary,
            global_moran_results,
            estimates,
            support_associations,
        )
        elapsed = time.monotonic() - started_clock
        completed_at = datetime.now(timezone)
        status = {
            "run_id": run_id,
            "status": "complete",
            "acceptance_status": "PASS",
            "acceptance_checks": len(acceptance),
            "source_e02_data_directory": str(data_dir),
            "source_e04_run_directory": str(e04_run),
            "covered_grids": len(eligibility),
            "eligible_grids": len(estimates),
            "bootstrap_replicates": len(bootstrap),
            "global_moran_results": len(global_moran_results),
            "local_moran_results": len(local_moran_results),
            "python_article_figure_groups": 7,
            "elapsed_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "article_figures_generated_by": "Python matplotlib",
        }
        (run_dir / "run_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        (empirical_root / "runs" / "latest_e05_run.txt").write_text(
            str(run_dir) + "\n", encoding="utf-8"
        )
        _append_registry(
            empirical_root,
            run_id,
            run_dir,
            started_at,
            completed_at,
            elapsed,
        )
        logger.info(
            "E05 complete: %d eligible grids, %d bootstrap rows, %.1f seconds",
            len(estimates),
            len(bootstrap),
            elapsed,
        )
        return run_dir
    except Exception:
        logger.exception("E05 failed; resumable checkpoints remain available.")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E05 local spatial information-value experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("empirical/config/e05.yaml"),
    )
    parser.add_argument("--resume-run", type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.resume_run)


if __name__ == "__main__":
    main()
