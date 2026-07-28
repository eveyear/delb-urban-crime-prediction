from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml

from entropy_crime_bike.e06_predictive_models import (
    CITY_COLORS,
    CITY_LABELS,
    CITY_ORDER,
)
from entropy_crime_bike.e09_robustness import (
    _load_panel,
    _prepare_spec,
    apply_flow_bins,
    crime_bins,
)
from entropy_crime_bike.spatial_information import (
    BoundInterpolator,
    benjamini_hochberg,
)


NULL_ORDER = [
    "temporal_block_permutation",
    "spatial_series_permutation",
    "circular_shift_surrogate",
]
NULL_LABELS = {
    "temporal_block_permutation": "Temporal blocks",
    "spatial_series_permutation": "Spatial series",
    "circular_shift_surrogate": "Circular surrogate",
}
LOG_TWO = math.log(2.0)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def _read_pointer(root: Path, relative: str) -> Path:
    result = Path((root / relative).read_text(encoding="utf-8").strip()).resolve()
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e10_placebo")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in [
        logging.FileHandler(
            run_dir / "logs" / "e10_placebo.log", encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _seed(base: int, label: str) -> int:
    return int((base + zlib.crc32(label.encode("utf-8"))) % 2**32)


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


def _entropy_from_counts(counts: np.ndarray) -> float:
    positive = np.asarray(counts, dtype=float)
    positive = positive[positive > 0]
    total = float(positive.sum())
    if total <= 0:
        raise ValueError("Entropy counts must have positive mass.")
    probabilities = positive / total
    return float(-np.dot(probabilities, np.log2(probabilities)))


def miller_madow_conditional_entropy(
    target_codes: np.ndarray,
    state_codes: np.ndarray,
    *,
    target_alphabet: int | None = None,
) -> float:
    y = np.asarray(target_codes, dtype=np.int64)
    state = np.asarray(state_codes, dtype=np.int64)
    if y.shape != state.shape or y.ndim != 1:
        raise ValueError("Target and state codes must be aligned vectors.")
    if np.any(y < 0) or np.any(state < 0):
        raise ValueError("Entropy codes must be nonnegative.")
    alphabet = int(y.max()) + 1 if target_alphabet is None else int(target_alphabet)
    joint = state * alphabet + y
    state_counts = np.bincount(state)
    joint_counts = np.bincount(joint)
    plugin = _entropy_from_counts(joint_counts) - _entropy_from_counts(
        state_counts
    )
    state_atoms = int(np.count_nonzero(state_counts))
    joint_atoms = int(np.count_nonzero(joint_counts))
    return float(
        plugin + (joint_atoms - state_atoms) / (2.0 * len(y) * LOG_TWO)
    )


def conditional_information(
    target_codes: np.ndarray,
    baseline_codes: np.ndarray,
    bicycle_codes: np.ndarray,
) -> tuple[float, float, float]:
    y = np.asarray(target_codes, dtype=np.int64)
    s = np.asarray(baseline_codes, dtype=np.int64)
    b = np.asarray(bicycle_codes, dtype=np.int64)
    if not (y.shape == s.shape == b.shape):
        raise ValueError("Conditional-information vectors must align.")
    bicycle_alphabet = int(b.max()) + 1
    h0 = miller_madow_conditional_entropy(y, s)
    hb = miller_madow_conditional_entropy(
        y, s * bicycle_alphabet + b
    )
    return h0, hb, h0 - hb


def _state_codes(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    index = pd.MultiIndex.from_frame(frame[columns].astype(object))
    codes, _ = pd.factorize(index, sort=False)
    if np.any(codes < 0):
        raise ValueError("State factorization produced missing codes.")
    return codes.astype(np.int64)


def temporal_block_permutation(
    bicycle_matrix: np.ndarray,
    rng: np.random.Generator,
    block_days: int,
) -> np.ndarray:
    values = np.asarray(bicycle_matrix)
    grids, days = values.shape
    complete_days = days - days % int(block_days)
    blocks = values[:, :complete_days].reshape(
        grids, complete_days // int(block_days), int(block_days)
    )
    permuted = np.empty_like(blocks)
    for grid in range(grids):
        permuted[grid] = blocks[grid, rng.permutation(blocks.shape[1])]
    result = values.copy()
    result[:, :complete_days] = permuted.reshape(grids, complete_days)
    return result


def spatial_series_permutation(
    bicycle_matrix: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return np.asarray(bicycle_matrix)[rng.permutation(len(bicycle_matrix))]


def circular_shift_surrogate(
    bicycle_matrix: np.ndarray,
    rng: np.random.Generator,
    minimum_shift: int,
) -> np.ndarray:
    values = np.asarray(bicycle_matrix)
    grids, days = values.shape
    if days <= 2 * minimum_shift:
        raise ValueError("Series is too short for the requested surrogate shift.")
    shifts = rng.integers(minimum_shift, days - minimum_shift + 1, size=grids)
    indices = (
        np.arange(days, dtype=np.int64)[None, :] + shifts[:, None]
    ) % days
    return np.take_along_axis(values, indices, axis=1)


def randomization_p_value(observed: float, null_values: np.ndarray) -> float:
    null = np.asarray(null_values, dtype=float)
    if not np.isfinite(null).all():
        raise ValueError("Null values must be finite.")
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def _matrix_view(
    prepared: pd.DataFrame,
    *,
    baseline_state: list[str],
) -> dict[str, object]:
    ordered = prepared.sort_values(["grid_id", "date"]).reset_index(drop=True)
    grids = ordered["grid_id"].drop_duplicates().astype(str).tolist()
    day_counts = ordered.groupby("grid_id", observed=True).size()
    if day_counts.nunique() != 1:
        raise AssertionError("Every grid must have the same placebo sequence length.")
    days = int(day_counts.iloc[0])
    dates = ordered.loc[
        ordered["grid_id"].astype(str).eq(grids[0]), "date"
    ].to_numpy()
    return {
        "frame": ordered,
        "grids": grids,
        "grid_count": len(grids),
        "days": days,
        "dates": dates,
        "target": ordered["crime_count_all"].to_numpy(np.int64),
        "baseline": _state_codes(ordered, baseline_state),
        "bicycle_matrix": ordered["bicycle_lag_bin"]
        .cat.codes.to_numpy(np.int16)
        .reshape(len(grids), days),
    }


def _null_bicycle_matrix(
    design: str,
    observed: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, object],
) -> np.ndarray:
    if design == "temporal_block_permutation":
        return temporal_block_permutation(
            observed,
            rng,
            int(
                config["frozen_null_matrix"]["temporal_block_permutation"][
                    "block_days"
                ]
            ),
        )
    if design == "spatial_series_permutation":
        return spatial_series_permutation(observed, rng)
    if design == "circular_shift_surrogate":
        return circular_shift_surrogate(
            observed,
            rng,
            int(
                config["frozen_null_matrix"]["circular_shift_surrogate"][
                    "minimum_absolute_shift_days"
                ]
            ),
        )
    raise ValueError(design)


def _bound_reduction(
    h0: np.ndarray | float,
    hb: np.ndarray | float,
    interpolator: BoundInterpolator,
) -> np.ndarray:
    h0_values = np.maximum(np.asarray(h0, dtype=float), 0.0)
    hb_values = np.maximum(np.asarray(hb, dtype=float), 0.0)
    hb_values = np.minimum(hb_values, h0_values)
    return interpolator.transform(h0_values) - interpolator.transform(hb_values)


def _city_nulls(
    view: dict[str, object],
    *,
    city: str,
    design: str,
    config: dict[str, object],
    interpolator: BoundInterpolator,
) -> tuple[pd.DataFrame, dict[str, float]]:
    y = np.asarray(view["target"], dtype=np.int64)
    s = np.asarray(view["baseline"], dtype=np.int64)
    observed_matrix = np.asarray(view["bicycle_matrix"], dtype=np.int16)
    observed_h0, observed_hb, observed_cmi = conditional_information(
        y, s, observed_matrix.ravel()
    )
    repetitions = int(config["frozen_null_matrix"]["repetitions"])
    rng = np.random.default_rng(
        _seed(int(config["random_seed"]), f"E10|city|{city}|{design}")
    )
    records = []
    for repetition in range(1, repetitions + 1):
        null_matrix = _null_bicycle_matrix(
            design, observed_matrix, rng, config
        )
        h0, hb, cmi = conditional_information(y, s, null_matrix.ravel())
        records.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "null_design": design,
                "replicate": repetition,
                "h0_miller_madow_bits": h0,
                "hb_miller_madow_bits": hb,
                "delta_h_raw_bits": cmi,
                "delta_l_exact_mse": float(
                    _bound_reduction(h0, hb, interpolator)
                ),
            }
        )
    observed = {
        "h0_miller_madow_bits": observed_h0,
        "hb_miller_madow_bits": observed_hb,
        "delta_h_raw_bits": observed_cmi,
        "delta_l_exact_mse": float(
            _bound_reduction(observed_h0, observed_hb, interpolator)
        ),
    }
    return pd.DataFrame(records), observed


def _local_components(
    prepared: pd.DataFrame,
    eligible_ids: list[str],
    baseline_state: list[str],
) -> dict[str, object]:
    subset = prepared.loc[
        prepared["grid_id"].astype(str).isin(eligible_ids)
    ].sort_values(["grid_id", "date"]).copy()
    groups = []
    for grid_id, grid in subset.groupby("grid_id", observed=True, sort=True):
        groups.append(
            {
                "grid_id": str(grid_id),
                "target": grid["crime_count_all"].to_numpy(np.int64),
                "baseline": _state_codes(grid, baseline_state),
                "bicycle": grid["bicycle_lag_bin"].cat.codes.to_numpy(np.int16),
            }
        )
    if [group["grid_id"] for group in groups] != sorted(eligible_ids):
        raise AssertionError("Local eligible grid alignment failed.")
    lengths = {len(group["target"]) for group in groups}
    if len(lengths) != 1:
        raise AssertionError("Local grids must share a common sequence length.")
    bicycle_matrix = np.vstack([group["bicycle"] for group in groups])
    return {"groups": groups, "bicycle_matrix": bicycle_matrix}


def _local_cmi(
    components: dict[str, object], bicycle_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = components["groups"]
    h0 = np.empty(len(groups), dtype=float)
    hb = np.empty(len(groups), dtype=float)
    cmi = np.empty(len(groups), dtype=float)
    for index, group in enumerate(groups):
        h0[index], hb[index], cmi[index] = conditional_information(
            group["target"], group["baseline"], bicycle_matrix[index]
        )
    return h0, hb, cmi


def _local_nulls(
    components: dict[str, object],
    *,
    city: str,
    design: str,
    config: dict[str, object],
    interpolator: BoundInterpolator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_matrix = np.asarray(components["bicycle_matrix"], dtype=np.int16)
    observed_h0, observed_hb, observed_cmi = _local_cmi(
        components, observed_matrix
    )
    observed_delta_l = _bound_reduction(
        observed_h0, observed_hb, interpolator
    )
    repetitions = int(config["frozen_null_matrix"]["repetitions"])
    rng = np.random.default_rng(
        _seed(int(config["random_seed"]), f"E10|local|{city}|{design}")
    )
    parts = []
    grid_ids = [group["grid_id"] for group in components["groups"]]
    for repetition in range(1, repetitions + 1):
        null_matrix = _null_bicycle_matrix(
            design, observed_matrix, rng, config
        )
        h0, hb, cmi = _local_cmi(components, null_matrix)
        parts.append(
            pd.DataFrame(
                {
                    "city": city,
                    "grid_id": grid_ids,
                    "null_design": design,
                    "replicate": repetition,
                    "delta_h_raw_bits": cmi,
                    "delta_l_exact_mse": _bound_reduction(
                        h0, hb, interpolator
                    ),
                }
            )
        )
    observed = pd.DataFrame(
        {
            "city": city,
            "grid_id": grid_ids,
            "null_design": design,
            "observed_h0_bits": observed_h0,
            "observed_hb_bits": observed_hb,
            "observed_delta_h_bits": observed_cmi,
            "observed_delta_l_mse": observed_delta_l,
        }
    )
    return pd.concat(parts, ignore_index=True), observed


def _summarize_city_nulls(
    nulls: pd.DataFrame, observed: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for row in observed.itertuples(index=False):
        values = nulls.loc[
            nulls["city"].eq(row.city)
            & nulls["null_design"].eq(row.null_design),
            "delta_h_raw_bits",
        ].to_numpy(float)
        rows.append(
            {
                **row._asdict(),
                "null_mean_delta_h_bits": float(values.mean()),
                "null_sd_delta_h_bits": float(values.std(ddof=1)),
                "null_q025_delta_h_bits": float(np.quantile(values, 0.025)),
                "null_q975_delta_h_bits": float(np.quantile(values, 0.975)),
                "observed_minus_null_mean_bits": float(
                    row.observed_delta_h_bits - values.mean()
                ),
                "standardized_separation": float(
                    (row.observed_delta_h_bits - values.mean())
                    / values.std(ddof=1)
                ),
                "randomization_p_value": randomization_p_value(
                    row.observed_delta_h_bits, values
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(
        result["randomization_p_value"]
    )
    result["separates_from_null"] = result["q_value_bh"].le(0.05)
    return result


def _summarize_local_nulls(
    nulls: pd.DataFrame, observed: pd.DataFrame
) -> pd.DataFrame:
    grouped = nulls.groupby(
        ["city", "grid_id", "null_design"], observed=True
    )["delta_h_raw_bits"]
    null_summary = grouped.agg(
        null_mean_delta_h_bits="mean",
        null_sd_delta_h_bits="std",
        null_q025_delta_h_bits=lambda value: value.quantile(0.025),
        null_q975_delta_h_bits=lambda value: value.quantile(0.975),
    ).reset_index()
    exceedances = (
        nulls.merge(
            observed[
                ["city", "grid_id", "null_design", "observed_delta_h_bits"]
            ],
            on=["city", "grid_id", "null_design"],
            how="left",
        )
        .assign(exceeds=lambda value: value["delta_h_raw_bits"].ge(
            value["observed_delta_h_bits"]
        ))
        .groupby(
            ["city", "grid_id", "null_design"], observed=True
        )["exceeds"]
        .sum()
        .rename("null_exceedances")
        .reset_index()
    )
    result = observed.merge(
        null_summary,
        on=["city", "grid_id", "null_design"],
        how="left",
    ).merge(
        exceedances,
        on=["city", "grid_id", "null_design"],
        how="left",
    )
    repetitions = nulls["replicate"].nunique()
    result["randomization_p_value"] = (
        1 + result["null_exceedances"]
    ) / (repetitions + 1)
    result["q_value_bh"] = np.nan
    for (_, _), indices in result.groupby(
        ["city", "null_design"], observed=True
    ).groups.items():
        result.loc[indices, "q_value_bh"] = benjamini_hochberg(
            result.loc[indices, "randomization_p_value"]
        )
    result["separates_from_null"] = result["q_value_bh"].le(0.05)
    result["observed_minus_null_mean_bits"] = (
        result["observed_delta_h_bits"] - result["null_mean_delta_h_bits"]
    )
    result["standardized_separation"] = (
        result["observed_minus_null_mean_bits"]
        / result["null_sd_delta_h_bits"]
    )
    return result


def _distant_lags(
    frame: pd.DataFrame,
    *,
    city: str,
    cut_points: tuple[float, float],
    eligible_ids: list[str],
    config: dict[str, object],
    interpolator: BoundInterpolator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.sort_values(["grid_id", "date"]).copy()
    grouped = working.groupby("grid_id", sort=False)
    working["crime_count_lag1"] = grouped["crime_count_all"].shift(1)
    working["bike_total_flow_lag1"] = grouped["bike_total_flow"].shift(1)
    for lag in config["frozen_null_matrix"]["distant_lag_placebos_days"]:
        working[f"bike_total_flow_lag{lag}"] = grouped[
            "bike_total_flow"
        ].shift(int(lag))
    city_rows = []
    local_rows = []
    baseline_city = list(config["information_sets"]["baseline_state"])
    baseline_local = list(config["information_sets"]["local_baseline_state"])
    for lag in config["frozen_null_matrix"]["distant_lag_placebos_days"]:
        distant_field = f"bike_total_flow_lag{lag}"
        subset = working.loc[
            working["bike_coverage_training"].astype(bool)
            & working["crime_count_lag1"].notna()
            & working["bike_total_flow_lag1"].notna()
            & working[distant_field].notna()
        ].copy()
        subset["crime_lag_bin"] = crime_bins(
            subset["crime_count_lag1"], "four_state"
        )
        subset["recent_bicycle_bin"] = apply_flow_bins(
            subset["bike_total_flow_lag1"], cut_points
        )
        subset["distant_bicycle_bin"] = apply_flow_bins(
            subset[distant_field], cut_points
        )
        y = subset["crime_count_all"].to_numpy(np.int64)
        s = _state_codes(subset, baseline_city)
        _, hb_recent, recent_cmi = conditional_information(
            y, s, subset["recent_bicycle_bin"].cat.codes.to_numpy(np.int16)
        )
        h0, hb_distant, distant_cmi = conditional_information(
            y, s, subset["distant_bicycle_bin"].cat.codes.to_numpy(np.int16)
        )
        city_rows.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "distant_lag_days": int(lag),
                "observations": len(subset),
                "matched_recent_delta_h_bits": recent_cmi,
                "distant_delta_h_bits": distant_cmi,
                "recent_minus_distant_bits": recent_cmi - distant_cmi,
                "matched_recent_delta_l_mse": float(
                    _bound_reduction(h0, hb_recent, interpolator)
                ),
                "distant_delta_l_mse": float(
                    _bound_reduction(h0, hb_distant, interpolator)
                ),
            }
        )
        for grid_id, grid in subset.loc[
            subset["grid_id"].astype(str).isin(eligible_ids)
        ].groupby("grid_id", observed=True, sort=True):
            local_y = grid["crime_count_all"].to_numpy(np.int64)
            local_s = _state_codes(grid, baseline_local)
            local_h0, local_hb_recent, local_recent = conditional_information(
                local_y,
                local_s,
                grid["recent_bicycle_bin"].cat.codes.to_numpy(np.int16),
            )
            _, local_hb_distant, local_distant = conditional_information(
                local_y,
                local_s,
                grid["distant_bicycle_bin"].cat.codes.to_numpy(np.int16),
            )
            local_rows.append(
                {
                    "city": city,
                    "grid_id": str(grid_id),
                    "distant_lag_days": int(lag),
                    "observations": len(grid),
                    "matched_recent_delta_h_bits": local_recent,
                    "distant_delta_h_bits": local_distant,
                    "recent_minus_distant_bits": local_recent - local_distant,
                    "matched_recent_delta_l_mse": float(
                        _bound_reduction(
                            local_h0, local_hb_recent, interpolator
                        )
                    ),
                    "distant_delta_l_mse": float(
                        _bound_reduction(
                            local_h0, local_hb_distant, interpolator
                        )
                    ),
                }
            )
    return pd.DataFrame(city_rows), pd.DataFrame(local_rows)


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
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_figures(
    city_nulls: pd.DataFrame,
    city_summary: pd.DataFrame,
    local_summary: pd.DataFrame,
    distant_city: pd.DataFrame,
    run_dir: Path,
    dpi: int,
) -> None:
    _publication_style()
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.5), sharex=False)
    for row, design in enumerate(NULL_ORDER):
        for column, city in enumerate(CITY_ORDER):
            axis = axes[row, column]
            values = city_nulls.loc[
                city_nulls["city"].eq(city)
                & city_nulls["null_design"].eq(design),
                "delta_h_raw_bits",
            ]
            summary = city_summary.loc[
                city_summary["city"].eq(city)
                & city_summary["null_design"].eq(design)
            ].iloc[0]
            axis.hist(values, bins=35, color="#B0BEC5", edgecolor="white")
            axis.axvline(
                summary["observed_delta_h_bits"],
                color=CITY_COLORS[city],
                lw=2,
                label="Observed",
            )
            if row == 0:
                axis.set_title(CITY_LABELS[city])
            if column == 0:
                axis.set_ylabel(NULL_LABELS[design] + "\nFrequency")
            if row == 2:
                axis.set_xlabel("Conditional information (bits)")
            axis.text(
                0.98,
                0.92,
                f"p={summary['randomization_p_value']:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
            )
    fig.suptitle("Observed conditional information versus E10 null distributions")
    _save_figure(fig, run_dir / "figures" / "e10_city_null_distributions.png", dpi)

    local_counts = (
        local_summary.groupby(["city", "null_design"], as_index=False)
        .agg(
            pass_count=("separates_from_null", "sum"),
            grids=("grid_id", "nunique"),
        )
    )
    matrix = np.zeros((len(NULL_ORDER), len(CITY_ORDER)), dtype=float)
    annotations: list[list[str]] = []
    for row, design in enumerate(NULL_ORDER):
        row_labels = []
        for column, city in enumerate(CITY_ORDER):
            value = local_counts.loc[
                local_counts["city"].eq(city)
                & local_counts["null_design"].eq(design)
            ].iloc[0]
            matrix[row, column] = value["pass_count"] / value["grids"]
            row_labels.append(
                f"{int(value['pass_count'])}/{int(value['grids'])}"
            )
        annotations.append(row_labels)
    fig, axis = plt.subplots(figsize=(7.7, 4.0))
    image = axis.imshow(
        matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto"
    )
    axis.set_xticks(
        np.arange(len(CITY_ORDER)),
        [CITY_LABELS[city] for city in CITY_ORDER],
    )
    axis.set_yticks(
        np.arange(len(NULL_ORDER)),
        [NULL_LABELS[design] for design in NULL_ORDER],
    )
    for row in range(len(NULL_ORDER)):
        for column in range(len(CITY_ORDER)):
            axis.text(
                column,
                row,
                annotations[row][column],
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("FDR-screen pass share")
    axis.set_title(
        "Local null-calibrated screens (passing grids / eligible grids)"
    )
    _save_figure(fig, run_dir / "figures" / "e10_local_null_screens.png", dpi)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=False)
    for axis, city in zip(axes, CITY_ORDER):
        subset = distant_city.loc[distant_city["city"].eq(city)]
        axis.plot(
            subset["distant_lag_days"],
            subset["matched_recent_delta_h_bits"],
            "o-",
            label="Matched recent lag",
            color=CITY_COLORS[city],
        )
        axis.plot(
            subset["distant_lag_days"],
            subset["distant_delta_h_bits"],
            "s--",
            label="Distant lag",
            color="#616161",
        )
        axis.set_title(CITY_LABELS[city])
        axis.set_xlabel("Distant bicycle lag (days)")
        axis.set_xticks([14, 30])
    axes[0].set_ylabel("Conditional information (bits)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Recent versus distant-lag information on matched samples")
    _save_figure(fig, run_dir / "figures" / "e10_distant_lags.png", dpi)


def _acceptance(
    city_nulls: pd.DataFrame,
    city_summary: pd.DataFrame,
    local_nulls: pd.DataFrame,
    local_summary: pd.DataFrame,
    distant_city: pd.DataFrame,
    distant_local: pd.DataFrame,
    source_reconciliation: pd.DataFrame,
    local_reconciliation: pd.DataFrame,
    run_dir: Path,
    config: dict[str, object],
) -> pd.DataFrame:
    checks = [
        (
            "city_null_rows",
            len(city_nulls)
            == int(config["acceptance"]["expected_city_null_rows"]),
        ),
        (
            "local_null_rows",
            len(local_nulls)
            == int(config["acceptance"]["expected_local_null_rows"]),
        ),
        (
            "city_summary_rows",
            len(city_summary) == 3 * len(NULL_ORDER),
        ),
        (
            "local_summary_rows",
            len(local_summary)
            == int(config["acceptance"]["expected_eligible_local_grids"])
            * len(NULL_ORDER),
        ),
        (
            "distant_city_rows",
            len(distant_city)
            == int(config["acceptance"]["expected_distant_city_rows"]),
        ),
        (
            "distant_local_rows",
            len(distant_local)
            == int(config["acceptance"]["expected_distant_local_rows"]),
        ),
        ("finite_city_nulls", np.isfinite(city_nulls.select_dtypes("number")).all().all()),
        ("finite_local_nulls", np.isfinite(local_nulls.select_dtypes("number")).all().all()),
        (
            "source_city_reconciliation",
            source_reconciliation["absolute_difference_bits"].max()
            <= float(config["numerics"]["source_reconciliation_tolerance_bits"]),
        ),
        (
            "source_local_reconciliation",
            local_reconciliation["maximum_absolute_difference_bits"].max()
            <= float(config["numerics"]["source_reconciliation_tolerance_bits"]),
        ),
        (
            "p_value_floor",
            city_summary["randomization_p_value"].min()
            >= float(config["acceptance"]["minimum_randomization_p_value"]),
        ),
        (
            "python_figures",
            all(
                (run_dir / "figures" / name).exists()
                for name in [
                    "e10_city_null_distributions.png",
                    "e10_local_null_screens.png",
                    "e10_distant_lags.png",
                ]
            ),
        ),
    ]
    result = pd.DataFrame(
        [{"check": name, "passed": bool(value), "detail": ""} for name, value in checks]
    )
    result.to_csv(run_dir / "tables" / "acceptance_checklist.csv", index=False)
    if not result["passed"].all():
        raise AssertionError(
            "E10 acceptance failed: "
            + ", ".join(result.loc[~result["passed"], "check"])
        )
    return result


def _write_interpretation(
    city_summary: pd.DataFrame,
    local_summary: pd.DataFrame,
    distant_city: pd.DataFrame,
    run_dir: Path,
) -> None:
    lines = [
        "# E10 Placebo and Null-Calibration Interpretation",
        "",
        "E10 was frozen before its null outputs were inspected. Randomization "
        "tests quantify separation from specified predictive null mechanisms; "
        "they do not identify a causal effect.",
        "",
        "## City-level null separation",
        "",
    ]
    for city in CITY_ORDER:
        subset = city_summary.loc[city_summary["city"].eq(city)]
        details = ", ".join(
            f"{NULL_LABELS[row.null_design]} p={row.randomization_p_value:.3f}"
            for row in subset.itertuples(index=False)
        )
        lines.append(f"- **{CITY_LABELS[city]}**: {details}.")
    lines.extend(["", "## Local null calibration", ""])
    for city in CITY_ORDER:
        subset = local_summary.loc[local_summary["city"].eq(city)]
        counts = (
            subset.groupby("null_design")["separates_from_null"].sum().to_dict()
        )
        total = subset["grid_id"].nunique()
        lines.append(
            f"- **{CITY_LABELS[city]}** ({total} grids): "
            + ", ".join(
                f"{NULL_LABELS[design]} {int(counts.get(design, 0))}/{total}"
                for design in NULL_ORDER
            )
            + "."
        )
    lines.extend(["", "## Distant-lag diagnostics", ""])
    for row in distant_city.itertuples(index=False):
        lines.append(
            f"- {CITY_LABELS[row.city]}, lag {row.distant_lag_days}: recent "
            f"{row.matched_recent_delta_h_bits:.6f} bits versus distant "
            f"{row.distant_delta_h_bits:.6f} bits."
        )
    lines.extend(
        [
            "",
            "None of the nine pooled city-by-null comparisons rejected in the "
            "prespecified upper tail, and no eligible local grid survived FDR "
            "calibration under any of the three null designs. The observed CMI "
            "was generally below the placebo mean. Together with the near-equality "
            "of recent and distant lags, this means the positive E04/E05 point "
            "estimates do not isolate recent, locally matched bicycle-specific "
            "information under these tests.",
            "",
            "The elevated placebo CMI is itself diagnostic. Permuting mobility "
            "changes its relationship with conditioning-state support, and the "
            "high-dimensional Miller--Madow estimator retains finite-sample "
            "support bias. Slow shared seasonality and omitted urban activity "
            "structure are also plausible. These mechanisms do not invalidate "
            "the mathematical DELB, but they materially limit the empirical "
            "mobility-specific claim.",
            "",
            "A temporal, spatial, or surrogate result is conditional on that "
            "particular randomization design; these are not exact conditional "
            "randomization tests preserving the full distribution of mobility "
            "given every baseline state. None of the analyses identifies a "
            "causal bicycle effect.",
            "",
            "E11 has not been started or updated. The current manuscript remains "
            "the pre-E09/E10 draft until the user authorizes a new synthesis.",
        ]
    )
    (run_dir / "interpretation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_workbook(
    acceptance: pd.DataFrame,
    city_summary: pd.DataFrame,
    local_summary: pd.DataFrame,
    distant_city: pd.DataFrame,
    distant_local: pd.DataFrame,
    reconciliation: pd.DataFrame,
    local_reconciliation: pd.DataFrame,
    run_dir: Path,
) -> None:
    with pd.ExcelWriter(
        run_dir / "tables" / "E10_placebo_QC.xlsx", engine="openpyxl"
    ) as writer:
        acceptance.to_excel(writer, sheet_name="Acceptance", index=False)
        city_summary.to_excel(writer, sheet_name="City Null Summary", index=False)
        local_summary.to_excel(writer, sheet_name="Local Null Summary", index=False)
        distant_city.to_excel(writer, sheet_name="Distant City", index=False)
        distant_local.to_excel(writer, sheet_name="Distant Local", index=False)
        reconciliation.to_excel(writer, sheet_name="Reconciliation", index=False)
        local_reconciliation.to_excel(
            writer, sheet_name="Local Reconciliation", index=False
        )


def run(config_path: Path, resume_run: Path | None = None) -> Path:
    started_clock = time.monotonic()
    started_at = datetime.now().astimezone()
    previous_elapsed = 0.0
    config = _load_yaml(config_path)
    project = config_path.resolve().parents[2]
    empirical_root = project / "empirical"
    paths = _load_yaml(empirical_root / "config" / "paths.yaml")
    if resume_run:
        run_dir = resume_run.resolve()
        run_id = run_dir.name
        previous_status_path = run_dir / "run_status.json"
        if previous_status_path.exists():
            previous_status = json.loads(
                previous_status_path.read_text(encoding="utf-8")
            )
            started_at = datetime.fromisoformat(previous_status["started_at"])
            previous_elapsed = float(previous_status.get("elapsed_seconds", 0.0))
    else:
        timezone = ZoneInfo(str(paths.get("timezone", "Asia/Shanghai")))
        run_id = (
            datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
            + "_E10_placebo_null"
        )
        run_dir = empirical_root / "runs" / run_id
    for path in [
        run_dir / "logs",
        run_dir / "tables",
        run_dir / "figures",
        run_dir / "state",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    e02_data = _read_pointer(
        empirical_root, str(config["source_e02_data_pointer"])
    )
    e04_run = _read_pointer(
        empirical_root, str(config["source_e04_run_pointer"])
    )
    e05_run = _read_pointer(
        empirical_root, str(config["source_e05_run_pointer"])
    )
    e09_run = _read_pointer(
        empirical_root, str(config["source_e09_run_pointer"])
    )
    if not resume_run:
        (run_dir / "command.txt").write_text(
            " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
        )
        _write_environment(run_dir / "environment.txt")
        (run_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "e10": config,
                    "source_e02_data": str(e02_data),
                    "source_e04_run": str(e04_run),
                    "source_e05_run": str(e05_run),
                    "source_e09_run": str(e09_run),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "frozen_specification.md").write_text(
            "# Frozen E10 Specification\n\n"
            "Before inspecting E10 outputs, the analysis fixed 999 temporal "
            "block permutations, 999 within-city spatial series permutations, "
            "999 independent circular-shift surrogates, and 14/30-day distant "
            "lags. City and E05-eligible local tests use Miller--Madow CMI. "
            "Randomization results are predictive null tests, not causal tests.\n",
            encoding="utf-8",
        )
    logger.info("Starting E10 run %s", run_id)

    interpolator = BoundInterpolator.build(
        float(config["numerics"]["bound_interpolation_max_entropy_bits"]),
        tail_tolerance=float(config["numerics"]["lattice_tail_tolerance"]),
        root_tolerance=float(config["numerics"]["root_tolerance"]),
    )
    thresholds = pd.read_csv(
        e04_run / "tables" / "bicycle_bin_thresholds.csv"
    ).set_index("city")
    eligibility = pd.read_csv(
        e05_run / "tables" / "grid_eligibility.csv"
    )
    e05_points = pd.read_csv(
        e05_run / "tables" / "local_information_estimates.csv"
    )
    e09_theory = pd.read_csv(
        e09_run / "tables" / "theory_robustness_matrix.csv"
    )

    city_null_parts = []
    city_observed_rows = []
    local_null_parts = []
    local_observed_parts = []
    distant_city_parts = []
    distant_local_parts = []
    reconciliation_rows = []
    for city in CITY_ORDER:
        panel = _load_panel(e02_data, city)
        prepared, _ = _prepare_spec(panel, "primary")
        city_view = _matrix_view(
            prepared,
            baseline_state=list(config["information_sets"]["baseline_state"]),
        )
        city_observed_for_design = {}
        for design in NULL_ORDER:
            city_checkpoint = (
                run_dir / "state" / f"{city}_{design}_city_null.parquet"
            )
            local_checkpoint = (
                run_dir / "state" / f"{city}_{design}_local_null.parquet"
            )
            local_observed_checkpoint = (
                run_dir / "state" / f"{city}_{design}_local_observed.parquet"
            )
            if city_checkpoint.exists():
                city_null = pd.read_parquet(city_checkpoint)
                _, observed = _city_nulls(
                    city_view,
                    city=city,
                    design=design,
                    config={**config, "frozen_null_matrix": {
                        **config["frozen_null_matrix"], "repetitions": 0
                    }},
                    interpolator=interpolator,
                )
            else:
                city_null, observed = _city_nulls(
                    city_view,
                    city=city,
                    design=design,
                    config=config,
                    interpolator=interpolator,
                )
                city_null.to_parquet(
                    city_checkpoint, index=False, compression="zstd"
                )
            city_observed_for_design[design] = observed
            city_null_parts.append(city_null)
            city_observed_rows.append(
                {
                    "city": city,
                    "city_label": CITY_LABELS[city],
                    "null_design": design,
                    **observed,
                    "observed_delta_h_bits": observed["delta_h_raw_bits"],
                    "observed_delta_l_mse": observed["delta_l_exact_mse"],
                }
            )

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
            if local_checkpoint.exists() and local_observed_checkpoint.exists():
                local_null = pd.read_parquet(local_checkpoint)
                local_observed = pd.read_parquet(local_observed_checkpoint)
            else:
                local_null, local_observed = _local_nulls(
                    components,
                    city=city,
                    design=design,
                    config=config,
                    interpolator=interpolator,
                )
                local_null.to_parquet(
                    local_checkpoint, index=False, compression="zstd"
                )
                local_observed.to_parquet(
                    local_observed_checkpoint, index=False, compression="zstd"
                )
            local_null_parts.append(local_null)
            local_observed_parts.append(local_observed)
            logger.info("%s %s null complete", city, design)

        accepted = e09_theory.loc[
            e09_theory["city"].eq(city)
            & e09_theory["spec_id"].eq("primary")
            & e09_theory["estimator"].eq("miller_madow")
        ].iloc[0]
        observed = city_observed_for_design[NULL_ORDER[0]]
        reconciliation_rows.append(
            {
                "city": city,
                "e10_observed_delta_h_bits": observed["delta_h_raw_bits"],
                "e09_primary_delta_h_bits": accepted["delta_h_raw_bits"],
                "absolute_difference_bits": abs(
                    observed["delta_h_raw_bits"] - accepted["delta_h_raw_bits"]
                ),
            }
        )
        cut_points = (
            float(thresholds.loc[city, "positive_tertile_1_upper"]),
            float(thresholds.loc[city, "positive_tertile_2_upper"]),
        )
        eligible_ids = sorted(
            eligibility.loc[
                eligibility["city"].eq(city) & eligibility["eligible"],
                "grid_id",
            ].astype(str)
        )
        distant_city, distant_local = _distant_lags(
            panel,
            city=city,
            cut_points=cut_points,
            eligible_ids=eligible_ids,
            config=config,
            interpolator=interpolator,
        )
        distant_city_parts.append(distant_city)
        distant_local_parts.append(distant_local)

    city_nulls = pd.concat(city_null_parts, ignore_index=True)
    city_observed = pd.DataFrame(city_observed_rows).drop(
        columns=["delta_h_raw_bits", "delta_l_exact_mse"]
    )
    city_summary = _summarize_city_nulls(city_nulls, city_observed)
    local_nulls = pd.concat(local_null_parts, ignore_index=True)
    local_observed = pd.concat(local_observed_parts, ignore_index=True)
    local_summary = _summarize_local_nulls(local_nulls, local_observed)
    distant_city = pd.concat(distant_city_parts, ignore_index=True)
    distant_local = pd.concat(distant_local_parts, ignore_index=True)
    reconciliation = pd.DataFrame(reconciliation_rows)
    local_reference = local_summary.loc[
        local_summary["null_design"].eq(NULL_ORDER[0]),
        ["city", "grid_id", "observed_delta_h_bits"],
    ].merge(
        e05_points[
            ["city", "grid_id", "delta_h_miller_madow_raw_bits"]
        ],
        on=["city", "grid_id"],
        how="left",
        validate="one_to_one",
    )
    local_reference["absolute_difference_bits"] = (
        local_reference["observed_delta_h_bits"]
        - local_reference["delta_h_miller_madow_raw_bits"]
    ).abs()
    local_reconciliation = (
        local_reference.groupby("city", as_index=False)
        .agg(
            grids=("grid_id", "nunique"),
            maximum_absolute_difference_bits=(
                "absolute_difference_bits", "max"
            ),
            mean_absolute_difference_bits=(
                "absolute_difference_bits", "mean"
            ),
        )
    )

    joint = (
        local_summary.pivot_table(
            index=["city", "grid_id"],
            columns="null_design",
            values="separates_from_null",
            aggfunc="first",
        )
        .reset_index()
    )
    joint["separates_all_three_nulls"] = joint[NULL_ORDER].all(axis=1)
    e05_screen = e05_points[
        ["city", "grid_id", "significant_positive_bound_reduction"]
    ]
    joint = joint.merge(e05_screen, on=["city", "grid_id"], how="left")
    joint["e05_screen_and_all_three_nulls"] = (
        joint["significant_positive_bound_reduction"]
        & joint["separates_all_three_nulls"]
    )

    city_nulls.to_parquet(
        run_dir / "tables" / "city_null_replicates.parquet",
        index=False,
        compression="zstd",
    )
    local_nulls.to_parquet(
        run_dir / "tables" / "local_null_replicates.parquet",
        index=False,
        compression="zstd",
    )
    for name, frame in [
        ("city_null_summary.csv", city_summary),
        ("local_null_summary.csv", local_summary),
        ("local_joint_null_screen.csv", joint),
        ("distant_lag_city.csv", distant_city),
        ("distant_lag_local.csv", distant_local),
        ("source_reconciliation.csv", reconciliation),
        ("local_source_reconciliation.csv", local_reconciliation),
    ]:
        frame.to_csv(run_dir / "tables" / name, index=False)

    _make_figures(
        city_nulls,
        city_summary,
        local_summary,
        distant_city,
        run_dir,
        int(config["reporting"]["figure_dpi"]),
    )
    acceptance = _acceptance(
        city_nulls,
        city_summary,
        local_nulls,
        local_summary,
        distant_city,
        distant_local,
        reconciliation,
        local_reconciliation,
        run_dir,
        config,
    )
    _write_interpretation(
        city_summary, local_summary, distant_city, run_dir
    )
    _write_workbook(
        acceptance,
        city_summary,
        local_summary,
        distant_city,
        distant_local,
        reconciliation,
        local_reconciliation,
        run_dir,
    )
    completed_at = datetime.now().astimezone()
    elapsed = time.monotonic() - started_clock
    status = {
        "experiment": "E10",
        "status": "complete",
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": previous_elapsed + elapsed,
        "latest_pass_seconds": elapsed,
        "city_null_rows": len(city_nulls),
        "local_null_rows": len(local_nulls),
        "local_grids": local_summary["grid_id"].nunique(),
        "checks_passed": int(acceptance["passed"].sum()),
        "checks_total": len(acceptance),
        "e11_started": False,
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (empirical_root / "runs" / "latest_e10_run.txt").write_text(
        str(run_dir) + "\n", encoding="utf-8"
    )
    if not resume_run:
        registry = empirical_root / "docs" / "experiment_registry.csv"
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{run_id},E10,complete,{started_at.isoformat()},"
                f"{completed_at.isoformat()},{elapsed:.3f},{run_dir},"
                '"Temporal-block; spatial-series; circular-surrogate nulls and '
                '14/30-day distant-lag diagnostics."\n'
            )
    logger.info("E10 complete in %.1f seconds; E11 was not started", elapsed)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E10 placebo analysis.")
    parser.add_argument(
        "--config", type=Path, default=Path("empirical/config/e10.yaml")
    )
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    run(args.config, args.resume_run)


if __name__ == "__main__":
    main()
