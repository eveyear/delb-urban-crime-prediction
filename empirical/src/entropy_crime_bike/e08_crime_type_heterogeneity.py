from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime
from itertools import combinations
from zoneinfo import ZoneInfo
import zlib

import joblib
import matplotlib.pyplot as plt
from .figure_typography import normalize_chart_typography
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml

from entropy_crime_bike.conditional_information import ConditionalCodebook
from entropy_crime_bike.discrete_bound import inverse_entropy_envelope
from entropy_crime_bike.e06_predictive_models import (
    CITY_COLORS,
    CITY_LABELS,
    CITY_ORDER,
    INFO_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    _load_e04_thresholds,
    _prepare_city,
    _run_city,
)
from entropy_crime_bike.heterogeneity import (
    category_lag,
    normal_interval,
    paired_category_difference,
)
from entropy_crime_bike.spatial_information import (
    BoundInterpolator,
    benjamini_hochberg,
)
from entropy_crime_bike.theory_practice import calculate_gap_metrics


OUTCOME_ORDER = ["PROPERTY_THEFT", "VEHICLE_THEFT", "BURGLARY"]
OUTCOME_COLORS = {
    "PROPERTY_THEFT": "#4472C4",
    "VEHICLE_THEFT": "#ED7D31",
    "BURGLARY": "#70AD47",
}
PERIOD_ORDER = ["pooled_2020_2022", "test"]
PERIOD_LABELS = {
    "pooled_2020_2022": "Pooled 2020–2022",
    "test": "Held-out test period",
}


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return loaded


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e08_crime_type_heterogeneity")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e08_crime_type_heterogeneity.log",
        encoding="utf-8",
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
        f"scikit_learn={sklearn.__version__}",
        f"matplotlib={plt.matplotlib.__version__}",
        f"pyarrow={pyarrow.__version__}",
        f"joblib={joblib.__version__}",
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


def _seed(base_seed: int, name: str) -> int:
    return int((base_seed + zlib.crc32(name.encode("utf-8"))) % 2**32)


def _read_pointer(empirical_root: Path, relative: str) -> Path:
    pointer = empirical_root / relative
    result = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    if not result.exists():
        raise FileNotFoundError(f"Pointer target does not exist: {result}")
    return result


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    normalize_chart_typography(figure)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _load_city_panel(data_dir: Path, city: str) -> pd.DataFrame:
    targets = [
        "crime_count_all",
        "crime_count_property_theft",
        "crime_count_vehicle_theft",
        "crime_count_burglary",
    ]
    columns = [
        "city",
        "grid_id",
        "x_index",
        "y_index",
        "centroid_longitude",
        "centroid_latitude",
        "date",
        "year",
        "month",
        "day_of_week",
        "season",
        "is_holiday",
        "split",
        "bike_coverage_training",
        *targets,
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
    identity = (
        frame[
            [
                "crime_count_property_theft",
                "crime_count_vehicle_theft",
                "crime_count_burglary",
            ]
        ].sum(axis=1)
        - frame["crime_count_all"]
    )
    if not identity.eq(0).all():
        raise AssertionError(f"E02 category-count identity failed for {city}.")
    return frame


def _category_config(
    config: dict[str, object], category: str
) -> dict[str, object]:
    result = deepcopy(config)
    outcome = result["outcomes"][category]
    result["target"] = str(outcome["target"])
    return result


def _prepare_category(
    panel: pd.DataFrame,
    *,
    city: str,
    category: str,
    cut_points: tuple[float, float],
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    category_config = _category_config(config, category)
    target = str(category_config["target"])
    working = panel.copy()
    working["category_crime_count_lag1"] = category_lag(
        working, target=target
    )
    prepared, diagnostic = _prepare_city(
        working,
        cut_points=cut_points,
        config=category_config,
    )
    diagnostic["outcome"] = category
    diagnostic["outcome_label"] = str(
        category_config["outcomes"][category]["label"]
    )
    expected_missing = int(working["grid_id"].nunique())
    actual_missing = int(working["category_crime_count_lag1"].isna().sum())
    if actual_missing != expected_missing:
        raise AssertionError(
            f"{city}/{category} strict category lag has {actual_missing} "
            f"missing values; expected {expected_missing}."
        )
    return prepared, diagnostic


def _period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    if period == "pooled_2020_2022":
        return pd.Series(True, index=frame.index)
    if period == "test":
        return frame["split"].eq("test")
    raise ValueError(f"Unknown E08 period: {period}")


def _shared_weights(
    *,
    city: str,
    period: str,
    block_count: int,
    config: dict[str, object],
) -> np.ndarray:
    repetitions = int(config["uncertainty"]["repetitions"])
    rng = np.random.default_rng(
        _seed(int(config["random_seed"]), f"E08|blocks|{city}|{period}")
    )
    return rng.multinomial(
        block_count,
        np.full(block_count, 1.0 / block_count),
        size=repetitions,
    )


def _exact_bound_pair(
    h0_raw: float,
    hb_raw: float,
    *,
    config: dict[str, object],
) -> dict[str, float | bool]:
    h0 = max(float(h0_raw), 0.0)
    hb_nonnegative = max(float(hb_raw), 0.0)
    hb = min(hb_nonnegative, h0)
    tail = float(config["numerics"]["lattice_tail_tolerance"])
    root = float(config["numerics"]["root_tolerance"])
    l0 = inverse_entropy_envelope(
        h0, tail_tolerance=tail, root_tolerance=root
    )
    lb = inverse_entropy_envelope(
        hb, tail_tolerance=tail, root_tolerance=root
    )
    return {
        "h0_for_bound_bits": h0,
        "hb_for_bound_bits": hb,
        "delta_h_projected_bits": h0 - hb,
        "l0_exact_mse": l0,
        "lb_exact_mse": lb,
        "delta_l_exact_mse": l0 - lb,
        "relative_l_reduction": (l0 - lb) / l0 if l0 > 0 else np.nan,
        "projection_applied": hb_nonnegative > h0,
    }


def _estimate_theory(
    prepared: pd.DataFrame,
    *,
    city: str,
    category: str,
    period: str,
    weights: np.ndarray | None,
    interpolator: BoundInterpolator,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    subset = prepared.loc[_period_mask(prepared, period)].copy()
    block_days = int(config["uncertainty"]["block_days"])
    subset["block_id"] = (
        (subset["date"] - subset["date"].min()).dt.days // block_days
    ).astype(np.int16)
    codebook = ConditionalCodebook.from_frame(
        subset,
        target=str(config["outcomes"][category]["target"]),
        baseline_state=list(config["information_sets"]["baseline_state"]),
        bicycle_state_field=str(
            config["information_sets"]["bicycle_state_field"]
        ),
        block_field="block_id",
    )
    if weights is None:
        weights = _shared_weights(
            city=city,
            period=period,
            block_count=codebook.block_count,
            config=config,
        )
    if weights.shape[1] != codebook.block_count:
        raise AssertionError(
            f"Shared block count mismatch for {city}/{category}/{period}."
        )
    point_entropy = codebook.estimate(np.ones(codebook.block_count)).iloc[0]
    bound = _exact_bound_pair(
        point_entropy["h0_miller_madow_bits"],
        point_entropy["hb_miller_madow_bits"],
        config=config,
    )
    bootstrap_entropy = codebook.estimate(weights)
    h0 = np.maximum(
        bootstrap_entropy["h0_miller_madow_bits"].to_numpy(float), 0.0
    )
    hb = np.minimum(
        np.maximum(
            bootstrap_entropy["hb_miller_madow_bits"].to_numpy(float), 0.0
        ),
        h0,
    )
    bootstrap = pd.DataFrame(
        {
            "city": city,
            "city_label": CITY_LABELS[city],
            "outcome": category,
            "outcome_label": str(config["outcomes"][category]["label"]),
            "period": period,
            "period_label": PERIOD_LABELS[period],
            "replicate": np.arange(len(weights)),
            "h0_miller_madow_bits": bootstrap_entropy[
                "h0_miller_madow_bits"
            ].to_numpy(float),
            "hb_miller_madow_bits": bootstrap_entropy[
                "hb_miller_madow_bits"
            ].to_numpy(float),
            "delta_h_miller_madow_raw_bits": bootstrap_entropy[
                "delta_h_miller_madow_raw_bits"
            ].to_numpy(float),
            "h0_for_bound_bits": h0,
            "hb_for_bound_bits": hb,
            "delta_h_projected_bits": h0 - hb,
            "l0_exact_mse": interpolator.transform(h0),
            "lb_exact_mse": interpolator.transform(hb),
        }
    )
    bootstrap["delta_l_exact_mse"] = (
        bootstrap["l0_exact_mse"] - bootstrap["lb_exact_mse"]
    )
    point: dict[str, object] = {
        "city": city,
        "city_label": CITY_LABELS[city],
        "outcome": category,
        "outcome_label": str(config["outcomes"][category]["label"]),
        "target_field": str(config["outcomes"][category]["target"]),
        "period": period,
        "period_label": PERIOD_LABELS[period],
        "period_start": str(subset["date"].min().date()),
        "period_end": str(subset["date"].max().date()),
        "observations": len(subset),
        "grids": int(subset["grid_id"].nunique()),
        "days": int(subset["date"].nunique()),
        "blocks": codebook.block_count,
        "crime_events": int(
            subset[str(config["outcomes"][category]["target"])].sum()
        ),
        **point_entropy.to_dict(),
        **bound,
        **codebook.sparsity_diagnostics(),
    }
    confidence = float(config["uncertainty"]["confidence_level"])
    raw_inference = normal_interval(
        float(point_entropy["delta_h_miller_madow_raw_bits"]),
        bootstrap["delta_h_miller_madow_raw_bits"],
        confidence_level=confidence,
        alternative="greater",
    )
    bound_inference = normal_interval(
        float(bound["delta_l_exact_mse"]),
        bootstrap["delta_l_exact_mse"],
        confidence_level=confidence,
        alternative="greater",
    )
    point.update(
        {
            f"delta_h_raw_{key}": value
            for key, value in raw_inference.items()
        }
    )
    point.update(
        {
            f"delta_l_{key}": value
            for key, value in bound_inference.items()
        }
    )
    return pd.DataFrame([point]), bootstrap, weights


def _theory_pairwise(
    theory: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    config: dict[str, object],
) -> pd.DataFrame:
    confidence = float(config["uncertainty"]["confidence_level"])
    records = []
    point = theory.loc[theory["period"].eq("pooled_2020_2022")]
    boot = bootstrap.loc[bootstrap["period"].eq("pooled_2020_2022")]
    for city in CITY_ORDER:
        city_point = point.loc[point["city"].eq(city)].set_index("outcome")
        city_boot = boot.loc[boot["city"].eq(city)]
        for outcome_a, outcome_b in combinations(OUTCOME_ORDER, 2):
            a_boot = city_boot.loc[
                city_boot["outcome"].eq(outcome_a)
            ].sort_values("replicate")
            b_boot = city_boot.loc[
                city_boot["outcome"].eq(outcome_b)
            ].sort_values("replicate")
            raw = paired_category_difference(
                city_point.loc[
                    outcome_a, "delta_h_miller_madow_raw_bits"
                ],
                city_point.loc[
                    outcome_b, "delta_h_miller_madow_raw_bits"
                ],
                a_boot["delta_h_miller_madow_raw_bits"],
                b_boot["delta_h_miller_madow_raw_bits"],
                confidence_level=confidence,
            )
            bound = paired_category_difference(
                city_point.loc[outcome_a, "delta_l_exact_mse"],
                city_point.loc[outcome_b, "delta_l_exact_mse"],
                a_boot["delta_l_exact_mse"],
                b_boot["delta_l_exact_mse"],
                confidence_level=confidence,
            )
            records.append(
                {
                    "city": city,
                    "city_label": CITY_LABELS[city],
                    "outcome_a": outcome_a,
                    "outcome_a_label": str(
                        config["outcomes"][outcome_a]["label"]
                    ),
                    "outcome_b": outcome_b,
                    "outcome_b_label": str(
                        config["outcomes"][outcome_b]["label"]
                    ),
                    **{f"delta_h_{key}": value for key, value in raw.items()},
                    **{f"delta_l_{key}": value for key, value in bound.items()},
                }
            )
    result = pd.DataFrame(records)
    result["delta_h_bh_adjusted_p_value"] = benjamini_hochberg(
        result["delta_h_p_value"]
    )
    result["delta_l_bh_adjusted_p_value"] = benjamini_hochberg(
        result["delta_l_p_value"]
    )
    result["delta_l_heterogeneity_screen"] = (
        result["delta_l_bh_adjusted_p_value"].le(
            float(config["inference"]["false_discovery_rate_alpha"])
        )
        & (
            result["delta_l_normal_ci_low"].gt(0)
            | result["delta_l_normal_ci_high"].lt(0)
        )
    )
    return result


def _run_category_models(
    prepared: pd.DataFrame,
    *,
    city: str,
    category: str,
    config: dict[str, object],
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    category_config = _category_config(config, category)
    category_dir = run_dir / "category_models" / category
    for directory in ["state", "models"]:
        (category_dir / directory).mkdir(parents=True, exist_ok=True)
    predictions, tuning, metadata, metrics = _run_city(
        prepared,
        city=city,
        config=category_config,
        run_dir=category_dir,
        logger=logger,
    )
    for frame in [predictions, tuning, metrics]:
        frame.insert(2, "outcome", category)
        frame.insert(
            3, "outcome_label", str(config["outcomes"][category]["label"])
        )
    for record in metadata:
        record["outcome"] = category
        record["outcome_label"] = str(
            config["outcomes"][category]["label"]
        )
        record["target"] = str(config["outcomes"][category]["target"])
    return predictions, tuning, metadata, metrics


def _prediction_contrasts(
    predictions: pd.DataFrame,
    theory: pd.DataFrame,
    theory_bootstrap: pd.DataFrame,
    weights_by_city: dict[str, np.ndarray],
    *,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    confidence = float(config["uncertainty"]["confidence_level"])
    records = []
    bootstrap_records = []
    test_theory = theory.loc[theory["period"].eq("test")].set_index(
        ["city", "outcome"]
    )
    test_boot = theory_bootstrap.loc[
        theory_bootstrap["period"].eq("test")
    ]
    block_days = int(config["uncertainty"]["block_days"])
    for city in CITY_ORDER:
        weights = weights_by_city[city]
        for category in OUTCOME_ORDER:
            bound_point = test_theory.loc[(city, category)]
            bound_boot = test_boot.loc[
                test_boot["city"].eq(city)
                & test_boot["outcome"].eq(category)
            ].sort_values("replicate")
            for family in MODEL_ORDER:
                subset = predictions.loc[
                    predictions["city"].eq(city)
                    & predictions["outcome"].eq(category)
                    & predictions["model_family"].eq(family)
                ]
                keys = ["city", "grid_id", "date", "observed_count"]
                baseline = subset.loc[
                    subset["information_set"].eq("baseline"),
                    keys + ["squared_error_integer"],
                ]
                bicycle = subset.loc[
                    subset["information_set"].eq("bicycle_aware"),
                    keys + ["squared_error_integer"],
                ]
                paired = baseline.merge(
                    bicycle,
                    on=keys,
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_baseline", "_bicycle"),
                )
                if len(paired) != len(baseline) or len(paired) != len(bicycle):
                    raise AssertionError(
                        f"Unmatched E08 pair for {city}/{category}/{family}."
                    )
                paired["block_id"] = (
                    (paired["date"] - paired["date"].min()).dt.days
                    // block_days
                ).astype(int)
                block = (
                    paired.groupby("block_id", observed=True)
                    .agg(
                        baseline_sse=(
                            "squared_error_integer_baseline",
                            "sum",
                        ),
                        bicycle_sse=(
                            "squared_error_integer_bicycle",
                            "sum",
                        ),
                        observations=("observed_count", "size"),
                    )
                    .reindex(range(weights.shape[1]), fill_value=0)
                )
                counts = block["observations"].to_numpy(float)
                mse0_boot = (
                    weights @ block["baseline_sse"].to_numpy(float)
                ) / (weights @ counts)
                mseb_boot = (
                    weights @ block["bicycle_sse"].to_numpy(float)
                ) / (weights @ counts)
                point_metrics = calculate_gap_metrics(
                    paired["squared_error_integer_baseline"].mean(),
                    paired["squared_error_integer_bicycle"].mean(),
                    float(bound_point["l0_exact_mse"]),
                    float(bound_point["lb_exact_mse"]),
                )
                boot_metrics = calculate_gap_metrics(
                    mse0_boot,
                    mseb_boot,
                    bound_boot["l0_exact_mse"].to_numpy(float),
                    bound_boot["lb_exact_mse"].to_numpy(float),
                )
                boot_frame = pd.DataFrame(
                    {
                        "city": city,
                        "outcome": category,
                        "model_family": family,
                        "replicate": np.arange(len(weights)),
                        "mse_baseline_integer": mse0_boot,
                        "mse_bicycle_integer": mseb_boot,
                        **boot_metrics,
                    }
                )
                bootstrap_records.append(boot_frame)
                delta_mse_inference = normal_interval(
                    float(point_metrics["delta_mse_integer"]),
                    boot_frame["delta_mse_integer"],
                    confidence_level=confidence,
                    alternative="greater",
                )
                gap_inference = normal_interval(
                    float(point_metrics["gap_narrowing"]),
                    boot_frame["gap_narrowing"],
                    confidence_level=confidence,
                    alternative="two-sided",
                )
                records.append(
                    {
                        "city": city,
                        "city_label": CITY_LABELS[city],
                        "outcome": category,
                        "outcome_label": str(
                            config["outcomes"][category]["label"]
                        ),
                        "model_family": family,
                        "model_label": MODEL_LABELS[family],
                        "observations": len(paired),
                        "test_days": int(paired["date"].nunique()),
                        "test_grids": int(paired["grid_id"].nunique()),
                        "mse_baseline_integer": float(
                            paired[
                                "squared_error_integer_baseline"
                            ].mean()
                        ),
                        "mse_bicycle_integer": float(
                            paired[
                                "squared_error_integer_bicycle"
                            ].mean()
                        ),
                        "l0_exact_mse": float(bound_point["l0_exact_mse"]),
                        "lb_exact_mse": float(bound_point["lb_exact_mse"]),
                        **point_metrics,
                        **{
                            f"delta_mse_{key}": value
                            for key, value in delta_mse_inference.items()
                        },
                        **{
                            f"gap_narrowing_{key}": value
                            for key, value in gap_inference.items()
                        },
                    }
                )
    result = pd.DataFrame(records)
    result["delta_mse_bh_adjusted_p_value"] = benjamini_hochberg(
        result["delta_mse_p_value"]
    )
    alpha = float(config["inference"]["false_discovery_rate_alpha"])
    result["positive_prediction_screen"] = (
        result["delta_mse_bh_adjusted_p_value"].le(alpha)
        & result["delta_mse_normal_ci_low"].gt(0)
    )
    return result, pd.concat(bootstrap_records, ignore_index=True)


def _prediction_pairwise(
    contrasts: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    config: dict[str, object],
) -> pd.DataFrame:
    confidence = float(config["uncertainty"]["confidence_level"])
    records = []
    for city in CITY_ORDER:
        for family in MODEL_ORDER:
            point = contrasts.loc[
                contrasts["city"].eq(city)
                & contrasts["model_family"].eq(family)
            ].set_index("outcome")
            boot = bootstrap.loc[
                bootstrap["city"].eq(city)
                & bootstrap["model_family"].eq(family)
            ]
            for outcome_a, outcome_b in combinations(OUTCOME_ORDER, 2):
                first = boot.loc[boot["outcome"].eq(outcome_a)].sort_values(
                    "replicate"
                )
                second = boot.loc[boot["outcome"].eq(outcome_b)].sort_values(
                    "replicate"
                )
                delta_mse = paired_category_difference(
                    point.loc[outcome_a, "delta_mse_integer"],
                    point.loc[outcome_b, "delta_mse_integer"],
                    first["delta_mse_integer"],
                    second["delta_mse_integer"],
                    confidence_level=confidence,
                )
                gap = paired_category_difference(
                    point.loc[outcome_a, "gap_narrowing"],
                    point.loc[outcome_b, "gap_narrowing"],
                    first["gap_narrowing"],
                    second["gap_narrowing"],
                    confidence_level=confidence,
                )
                records.append(
                    {
                        "city": city,
                        "city_label": CITY_LABELS[city],
                        "model_family": family,
                        "model_label": MODEL_LABELS[family],
                        "outcome_a": outcome_a,
                        "outcome_a_label": str(
                            config["outcomes"][outcome_a]["label"]
                        ),
                        "outcome_b": outcome_b,
                        "outcome_b_label": str(
                            config["outcomes"][outcome_b]["label"]
                        ),
                        **{
                            f"delta_mse_{key}": value
                            for key, value in delta_mse.items()
                        },
                        **{
                            f"gap_narrowing_{key}": value
                            for key, value in gap.items()
                        },
                    }
                )
    result = pd.DataFrame(records)
    result["delta_mse_bh_adjusted_p_value"] = benjamini_hochberg(
        result["delta_mse_p_value"]
    )
    result["gap_narrowing_bh_adjusted_p_value"] = benjamini_hochberg(
        result["gap_narrowing_p_value"]
    )
    return result


def _descriptive_composition(
    panel_by_city: dict[str, pd.DataFrame],
    *,
    config: dict[str, object],
) -> pd.DataFrame:
    records = []
    for city in CITY_ORDER:
        panel = panel_by_city[city]
        for domain, mask in [
            ("complete_crime_domain", pd.Series(True, index=panel.index)),
            (
                "training_bicycle_covered",
                panel["bike_coverage_training"].astype(bool),
            ),
        ]:
            subset = panel.loc[mask]
            total = float(subset["crime_count_all"].sum())
            for category in OUTCOME_ORDER:
                target = str(config["outcomes"][category]["target"])
                events = int(subset[target].sum())
                records.append(
                    {
                        "city": city,
                        "city_label": CITY_LABELS[city],
                        "domain": domain,
                        "outcome": category,
                        "outcome_label": str(
                            config["outcomes"][category]["label"]
                        ),
                        "grid_days": len(subset),
                        "grids": int(subset["grid_id"].nunique()),
                        "days": int(subset["date"].nunique()),
                        "crime_events": events,
                        "crime_event_share": events / total if total > 0 else np.nan,
                        "mean_grid_day_count": float(subset[target].mean()),
                        "zero_grid_day_share": float(subset[target].eq(0).mean()),
                    }
                )
    return pd.DataFrame(records)


def _acceptance_checks(
    *,
    panel_by_city: dict[str, pd.DataFrame],
    theory: pd.DataFrame,
    theory_bootstrap: pd.DataFrame,
    theory_pairwise: pd.DataFrame,
    predictions: pd.DataFrame,
    tuning: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: list[dict[str, object]],
    contrasts: pd.DataFrame,
    prediction_bootstrap: pd.DataFrame,
    prediction_pairwise: pd.DataFrame,
    model_count: int,
    config: dict[str, object],
) -> pd.DataFrame:
    expected = config["acceptance"]
    tolerance = float(config["numerics"]["identity_tolerance"])
    rows = []

    def add(
        criterion: str,
        expected_value: object,
        observed: object,
        passed: bool,
        detail: str = "",
    ) -> None:
        rows.append(
            {
                "criterion": criterion,
                "expected": expected_value,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    count_checks = [
        (
            "Theory rows",
            "expected_theory_rows",
            len(theory),
        ),
        (
            "Theory bootstrap rows",
            "expected_theory_bootstrap_rows",
            len(theory_bootstrap),
        ),
        (
            "Theory pairwise rows",
            "expected_theory_pairwise_rows",
            len(theory_pairwise),
        ),
        ("Metric rows", "expected_metric_rows", len(metrics)),
        (
            "Matched prediction contrasts",
            "expected_matched_contrasts",
            len(contrasts),
        ),
        (
            "Prediction bootstrap rows",
            "expected_prediction_bootstrap_rows",
            len(prediction_bootstrap),
        ),
        (
            "Prediction pairwise rows",
            "expected_prediction_pairwise_rows",
            len(prediction_pairwise),
        ),
        ("Prediction rows", "expected_prediction_rows", len(predictions)),
        ("Serialized matched models", "expected_serialized_models", model_count),
    ]
    for label, key, observed in count_checks:
        add(
            label,
            expected[key],
            observed,
            observed == int(expected[key]),
        )
    add(
        "Authoritative outcome labels",
        OUTCOME_ORDER,
        sorted(theory["outcome"].unique()),
        set(theory["outcome"]) == set(OUTCOME_ORDER),
    )
    identity_max = max(
        float(
            (
                frame[
                    [
                        "crime_count_property_theft",
                        "crime_count_vehicle_theft",
                        "crime_count_burglary",
                    ]
                ].sum(axis=1)
                - frame["crime_count_all"]
            )
            .abs()
            .max()
        )
        for frame in panel_by_city.values()
    )
    add(
        "Category counts sum to crime_count_all",
        0,
        identity_max,
        identity_max == 0,
    )
    test_counts = (
        predictions[
            ["city", "outcome", "grid_id", "date", "observed_count"]
        ]
        .drop_duplicates()
        .groupby("outcome", observed=True)
        .size()
    )
    add(
        "Equal matched test observations per outcome",
        expected["expected_test_observations_per_outcome"],
        test_counts.to_dict(),
        len(test_counts) == int(expected["expected_outcomes"])
        and test_counts.eq(
            int(expected["expected_test_observations_per_outcome"])
        ).all(),
    )
    bound_order = (
        theory["lb_exact_mse"].ge(-tolerance)
        & theory["lb_exact_mse"].le(theory["l0_exact_mse"] + tolerance)
    )
    add(
        "Exact DELB ordering",
        len(theory),
        int(bound_order.sum()),
        bound_order.all(),
    )
    empirical_order = (
        contrasts["mse_baseline_integer"]
        + tolerance
        >= contrasts["l0_exact_mse"]
    ) & (
        contrasts["mse_bicycle_integer"]
        + tolerance
        >= contrasts["lb_exact_mse"]
    )
    add(
        "Held-out MSE respects matched category DELB",
        len(contrasts),
        int(empirical_order.sum()),
        empirical_order.all(),
    )
    gap_identity = float(
        (
            contrasts["gap_narrowing"]
            - (
                contrasts["delta_mse_integer"]
                - contrasts["delta_l_exact_mse"]
            )
        )
        .abs()
        .max()
    )
    add(
        "Gap-narrowing identity",
        f"<= {tolerance}",
        gap_identity,
        gap_identity <= tolerance,
    )
    add(
        "No test observations used for hyperparameter selection",
        0,
        int(
            sum(bool(record["test_used_for_selection"]) for record in metadata)
        ),
        not any(bool(record["test_used_for_selection"]) for record in metadata),
    )
    selected_counts = (
        tuning.loc[tuning["selected"].astype(bool)]
        .groupby(
            ["city", "outcome", "model_family", "information_set"],
            observed=True,
        )
        .size()
    )
    add(
        "Exactly one validation-selected candidate per matched specification",
        int(expected["expected_serialized_models"]),
        int(selected_counts.eq(1).sum()),
        len(selected_counts) == int(expected["expected_serialized_models"])
        and selected_counts.eq(1).all(),
    )
    prediction_integer = predictions["prediction_integer"].to_numpy(float)
    add(
        "Primary predictions are nonnegative integers",
        "all rows",
        int(
            (
                (prediction_integer >= 0)
                & np.equal(prediction_integer, np.floor(prediction_integer))
            ).sum()
        ),
        (
            (prediction_integer >= 0)
            & np.equal(prediction_integer, np.floor(prediction_integer))
        ).all(),
    )
    shared_replicates = (
        theory_bootstrap.groupby(
            ["city", "period", "outcome"], observed=True
        )["replicate"]
        .nunique()
        .eq(int(config["uncertainty"]["repetitions"]))
        .all()
    )
    add(
        "Every theory specification has complete shared bootstrap replicates",
        int(config["uncertainty"]["repetitions"]),
        shared_replicates,
        bool(shared_replicates),
    )
    add(
        "Seasonal benchmark excluded from matched contrasts",
        MODEL_ORDER,
        sorted(contrasts["model_family"].unique()),
        set(contrasts["model_family"]) == set(MODEL_ORDER),
    )
    add(
        "Finite primary estimands",
        "all",
        int(
            np.isfinite(
                theory[
                    [
                        "delta_h_miller_madow_raw_bits",
                        "delta_l_exact_mse",
                    ]
                ].to_numpy(float)
            ).sum()
        ),
        np.isfinite(
            theory[
                [
                    "delta_h_miller_madow_raw_bits",
                    "delta_l_exact_mse",
                ]
            ].to_numpy(float)
        ).all()
        and np.isfinite(
            contrasts[
                ["delta_mse_integer", "gap_narrowing"]
            ].to_numpy(float)
        ).all(),
    )
    return pd.DataFrame(rows)


def _plot_composition(
    composition: pd.DataFrame, path: Path, dpi: int
) -> None:
    subset = composition.loc[
        composition["domain"].eq("training_bicycle_covered")
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.7))
    bottom = np.zeros(len(CITY_ORDER))
    x = np.arange(len(CITY_ORDER))
    for category in OUTCOME_ORDER:
        values = (
            subset.loc[subset["outcome"].eq(category)]
            .set_index("city")
            .loc[CITY_ORDER, "crime_event_share"]
            .to_numpy(float)
        )
        axis.bar(
            x,
            values,
            bottom=bottom,
            color=OUTCOME_COLORS[category],
            label=subset.loc[
                subset["outcome"].eq(category), "outcome_label"
            ].iloc[0],
        )
        bottom += values
    axis.set_xticks(x)
    axis.set_xticklabels([CITY_LABELS[city] for city in CITY_ORDER])
    axis.set_ylabel("Share of harmonized crime events")
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=1.0)
    )
    figure.suptitle(
        "Crime-type composition in the training bicycle-covered domain",
        x=0.125,
        y=0.98,
        ha="left",
        fontsize=10.5,
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
    )
    axis.grid(axis="y", alpha=0.2)
    figure.subplots_adjust(top=0.80)
    _save_figure(figure, path, dpi)


def _plot_theory_metric(
    theory: pd.DataFrame,
    *,
    metric: str,
    low: str,
    high: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    subset = theory.loc[theory["period"].eq("pooled_2020_2022")]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharey=False)
    x = np.arange(len(OUTCOME_ORDER))
    for axis, city in zip(axes, CITY_ORDER):
        city_data = subset.loc[subset["city"].eq(city)].set_index(
            "outcome"
        ).loc[OUTCOME_ORDER]
        values = city_data[metric].to_numpy(float)
        lower = city_data[low].to_numpy(float)
        upper = city_data[high].to_numpy(float)
        axis.vlines(x, lower, upper, color="#555555", linewidth=1.1)
        axis.scatter(
            x,
            values,
            c=[OUTCOME_COLORS[item] for item in OUTCOME_ORDER],
            s=42,
            zorder=3,
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xticks(x)
        axis.set_xticklabels(["Property", "Vehicle", "Burglary"])
        axis.set_title(CITY_LABELS[city])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(title, y=1.02, fontsize=12)
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _plot_test_bounds(theory: pd.DataFrame, path: Path, dpi: int) -> None:
    subset = theory.loc[theory["period"].eq("test")]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=False)
    x = np.arange(len(OUTCOME_ORDER))
    width = 0.35
    for axis, city in zip(axes, CITY_ORDER):
        city_data = subset.loc[subset["city"].eq(city)].set_index(
            "outcome"
        ).loc[OUTCOME_ORDER]
        axis.bar(
            x - width / 2,
            city_data["l0_exact_mse"],
            width,
            color="#9EADBA",
            label="Baseline DELB",
        )
        axis.bar(
            x + width / 2,
            city_data["lb_exact_mse"],
            width,
            color=CITY_COLORS[city],
            label="Bicycle-aware DELB",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(["Property", "Vehicle", "Burglary"])
        axis.set_title(CITY_LABELS[city])
        axis.set_ylabel("Exact test-period DELB (MSE)")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.suptitle(
        "Category-specific matched test-period theoretical floors",
        y=1.01,
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    _save_figure(figure, path, dpi)


def _heatmap(
    contrasts: pd.DataFrame,
    *,
    metric: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))
    maximum = float(np.nanmax(np.abs(contrasts[metric])))
    maximum = max(maximum, 1.0e-9)
    image = None
    for axis, city in zip(axes, CITY_ORDER):
        city_data = (
            contrasts.loc[contrasts["city"].eq(city)]
            .pivot(index="outcome", columns="model_family", values=metric)
            .loc[OUTCOME_ORDER, MODEL_ORDER]
        )
        image = axis.imshow(
            city_data.to_numpy(float),
            cmap="RdBu_r",
            vmin=-maximum,
            vmax=maximum,
            aspect="auto",
        )
        axis.set_xticks(range(len(MODEL_ORDER)))
        axis.set_xticklabels(
            ["State", "Poisson", "NB", "Boosting"], rotation=25, ha="right"
        )
        axis.set_yticks(range(len(OUTCOME_ORDER)))
        axis.set_yticklabels(["Property", "Vehicle", "Burglary"])
        axis.set_title(CITY_LABELS[city])
        for row in range(len(OUTCOME_ORDER)):
            for column in range(len(MODEL_ORDER)):
                value = city_data.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.55 * maximum else "black",
                    fontsize=7.5,
                )
    if image is not None:
        colorbar = figure.colorbar(
            image, ax=axes, orientation="horizontal", fraction=0.08, pad=0.20
        )
        colorbar.set_label(metric.replace("_", " "))
    figure.suptitle(title, y=0.99, fontsize=12)
    figure.subplots_adjust(bottom=0.28, top=0.82, wspace=0.28)
    _save_figure(figure, path, dpi)


def _plot_display_model(
    contrasts: pd.DataFrame,
    *,
    family: str,
    path: Path,
    dpi: int,
) -> None:
    subset = contrasts.loc[contrasts["model_family"].eq(family)]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), sharey=False)
    x = np.arange(len(OUTCOME_ORDER))
    width = 0.35
    for axis, city in zip(axes, CITY_ORDER):
        city_data = subset.loc[subset["city"].eq(city)].set_index(
            "outcome"
        ).loc[OUTCOME_ORDER]
        axis.bar(
            x - width / 2,
            city_data["mse_baseline_integer"],
            width,
            color="#9EADBA",
            label="Baseline",
        )
        axis.bar(
            x + width / 2,
            city_data["mse_bicycle_integer"],
            width,
            color=CITY_COLORS[city],
            label="Bicycle-aware",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(["Property", "Vehicle", "Burglary"])
        axis.set_title(CITY_LABELS[city])
        axis.set_ylabel("Held-out integer MSE")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.suptitle(
        f"Category-specific held-out performance: {MODEL_LABELS[family]}",
        y=1.01,
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    _save_figure(figure, path, dpi)


def _plot_pairwise_theory(
    pairwise: pd.DataFrame, path: Path, dpi: int
) -> None:
    labels = (
        pairwise["outcome_a_label"].str.replace(" theft", "", regex=False)
        + " − "
        + pairwise["outcome_b_label"].str.replace(" theft", "", regex=False)
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), sharey=True)
    for axis, city in zip(axes, CITY_ORDER):
        subset = pairwise.loc[pairwise["city"].eq(city)].copy()
        values = subset["delta_l_point_difference_a_minus_b"].to_numpy(float)
        lower = subset["delta_l_normal_ci_low"].to_numpy(float)
        upper = subset["delta_l_normal_ci_high"].to_numpy(float)
        y = np.arange(len(subset))
        axis.hlines(y, lower, upper, color="#555555")
        axis.scatter(values, y, color=CITY_COLORS[city], zorder=3)
        axis.axvline(0.0, color="black", linewidth=0.7)
        axis.set_yticks(y)
        axis.set_yticklabels(labels.loc[subset.index])
        axis.set_title(CITY_LABELS[city])
        axis.set_xlabel("Difference in exact DELB reduction")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Within-city paired crime-type heterogeneity in theoretical information value",
        y=1.02,
        fontsize=12,
    )
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _write_interpretation(
    run_dir: Path,
    composition: pd.DataFrame,
    theory: pd.DataFrame,
    pairwise: pd.DataFrame,
    contrasts: pd.DataFrame,
    prediction_pairwise: pd.DataFrame,
) -> None:
    pooled = theory.loc[theory["period"].eq("pooled_2020_2022")]
    lines = [
        "# E08 Crime-Type Heterogeneity Results",
        "",
        "## Design",
        "",
        "- The three authoritative crime_type_unified categories were used directly; no new crosswalk was fitted.",
        "- Each outcome used its own strict lag-1 category count in the baseline information state.",
        "- E04 training-only bicycle thresholds and the training bicycle-covered 1 km domain were frozen before E08.",
        "- City and category comparisons used shared seven-day calendar-block weights, so pairwise heterogeneity contrasts are aligned by replicate.",
        "- Prediction hyperparameters were selected on the validation period and evaluated once on July--December 2022.",
        "",
        "## Pooled theoretical information value",
        "",
    ]
    for city in CITY_ORDER:
        lines.extend([f"### {CITY_LABELS[city]}", ""])
        subset = pooled.loc[pooled["city"].eq(city)].set_index("outcome")
        for category in OUTCOME_ORDER:
            row = subset.loc[category]
            lines.append(
                f"- {row['outcome_label']}: raw CMI "
                f"{row['delta_h_miller_madow_raw_bits']:.5f} bits "
                f"(95% centered interval "
                f"[{row['delta_h_raw_normal_ci_low']:.5f}, "
                f"{row['delta_h_raw_normal_ci_high']:.5f}]); exact delta L "
                f"{row['delta_l_exact_mse']:.5f}."
            )
        lines.append("")
    significant_pairwise = pairwise.loc[
        pairwise["delta_l_heterogeneity_screen"].astype(bool)
    ]
    lines.extend(
        [
            "## Paired theoretical heterogeneity",
            "",
            f"- {len(significant_pairwise)} of {len(pairwise)} within-city "
            "crime-type comparisons passed the paired DELB heterogeneity screen.",
            "",
            "## Held-out prediction",
            "",
        ]
    )
    positive = contrasts.loc[
        contrasts["positive_prediction_screen"].astype(bool)
    ]
    lines.append(
        f"- {len(positive)} of {len(contrasts)} category-specific matched "
        "model comparisons passed the positive-improvement screen."
    )
    for _, row in positive.iterrows():
        lines.append(
            f"- {row['city_label']}, {row['outcome_label']}, "
            f"{row['model_label']}: delta MSE "
            f"{row['delta_mse_integer']:.5f} "
            f"(95% centered interval "
            f"[{row['delta_mse_normal_ci_low']:.5f}, "
            f"{row['delta_mse_normal_ci_high']:.5f}])."
        )
    significant_prediction = prediction_pairwise.loc[
        prediction_pairwise["delta_mse_bh_adjusted_p_value"].le(0.05)
        & (
            prediction_pairwise["delta_mse_normal_ci_low"].gt(0)
            | prediction_pairwise["delta_mse_normal_ci_high"].lt(0)
        )
    ]
    lines.extend(
        [
            "",
            f"- {len(significant_prediction)} of "
            f"{len(prediction_pairwise)} paired category differences in "
            "empirical MSE improvement passed the adjusted screen.",
            "",
            "## Interpretation boundaries",
            "",
            "- Crime-type differences are secondary heterogeneity findings; the combined harmonized count remains the primary outcome.",
            "- A larger category-specific DELB reduction means the added mobility state changes the estimated theoretical floor more strongly for that outcome. It does not guarantee a commensurate model improvement.",
            "- Cross-city differences may reflect offense composition, reporting systems, bicycle coverage, and sparse discrete states.",
            "- These information and prediction contrasts are not causal effects of bicycle activity on any crime category.",
            "- E09 robustness and E10 placebo experiments remain required before final manuscript claims.",
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
    row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "experiment": "E08",
                "status": "complete",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "run_directory": str(run_dir),
                "notes": (
                    "Category-specific conditional information, exact DELBs, "
                    "matched prediction models, and paired heterogeneity tests."
                ),
            }
        ]
    )
    if registry.exists():
        existing = pd.read_csv(registry)
        existing = existing.loc[~existing["run_id"].eq(run_id)]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(registry, index=False)


def run(config_path: Path, resume_run: Path | None = None) -> Path:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    empirical_root = Path(__file__).resolve().parents[2]
    timezone = ZoneInfo("Asia/Shanghai")
    started_at = datetime.now(timezone)
    if resume_run is None:
        run_id = started_at.strftime(
            "%Y%m%d_%H%M%S_E08_crime_type_heterogeneity"
        )
        run_dir = empirical_root / "runs" / run_id
    else:
        run_dir = resume_run.resolve()
        run_id = run_dir.name
    for directory in [
        "logs",
        "tables",
        "figures",
        "state",
        "category_models",
    ]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    command = (
        f"{sys.executable} empirical/run_e08_crime_type_heterogeneity.py "
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
    logger.info("Starting E08 crime-type heterogeneity experiment")
    try:
        if list(config["outcomes"].keys()) != OUTCOME_ORDER:
            raise AssertionError(
                "E08 outcome order must preserve the authoritative three categories."
            )
        data_dir = _read_pointer(
            empirical_root, str(config["source_e02_data_pointer"])
        )
        e04_run, thresholds = _load_e04_thresholds(empirical_root, config)
        interpolator = BoundInterpolator.build(
            float(
                config["numerics"][
                    "bootstrap_bound_interpolation_max_entropy_bits"
                ]
            ),
            tail_tolerance=float(config["numerics"]["lattice_tail_tolerance"]),
            root_tolerance=float(config["numerics"]["root_tolerance"]),
        )
        panel_by_city: dict[str, pd.DataFrame] = {}
        theory_frames = []
        theory_bootstrap_frames = []
        diagnostics = []
        predictions_frames = []
        tuning_frames = []
        metrics_frames = []
        metadata_records: list[dict[str, object]] = []
        weights_by_city_period: dict[tuple[str, str], np.ndarray] = {}
        for city in CITY_ORDER:
            panel = _load_city_panel(data_dir, city)
            panel_by_city[city] = panel
            threshold = thresholds.loc[thresholds["city"].eq(city)].iloc[0]
            cut_points = (
                float(threshold["positive_tertile_1_upper"]),
                float(threshold["positive_tertile_2_upper"]),
            )
            for category in OUTCOME_ORDER:
                prepared, diagnostic = _prepare_category(
                    panel,
                    city=city,
                    category=category,
                    cut_points=cut_points,
                    config=config,
                )
                diagnostics.append(diagnostic)
                for period in PERIOD_ORDER:
                    key = (city, period)
                    point, bootstrap, weights = _estimate_theory(
                        prepared,
                        city=city,
                        category=category,
                        period=period,
                        weights=weights_by_city_period.get(key),
                        interpolator=interpolator,
                        config=config,
                    )
                    weights_by_city_period[key] = weights
                    theory_frames.append(point)
                    theory_bootstrap_frames.append(bootstrap)
                (
                    predictions,
                    tuning,
                    metadata,
                    metrics,
                ) = _run_category_models(
                    prepared,
                    city=city,
                    category=category,
                    config=config,
                    run_dir=run_dir,
                    logger=logger,
                )
                predictions_frames.append(predictions)
                tuning_frames.append(tuning)
                metadata_records.extend(metadata)
                metrics_frames.append(metrics)
                logger.info(
                    "%s %s complete: %d test observations",
                    city,
                    category,
                    diagnostic["test_rows"],
                )
                del prepared
        theory = pd.concat(theory_frames, ignore_index=True)
        theory_bootstrap = pd.concat(
            theory_bootstrap_frames, ignore_index=True
        )
        diagnostics_frame = pd.DataFrame(diagnostics)
        predictions = pd.concat(predictions_frames, ignore_index=True)
        predictions["date"] = pd.to_datetime(predictions["date"])
        tuning = pd.concat(tuning_frames, ignore_index=True)
        metrics = pd.concat(metrics_frames, ignore_index=True)
        theory["delta_h_raw_bh_adjusted_p_value"] = benjamini_hochberg(
            theory["delta_h_raw_p_value"]
        )
        theory["delta_l_bh_adjusted_p_value"] = benjamini_hochberg(
            theory["delta_l_p_value"]
        )
        alpha = float(config["inference"]["false_discovery_rate_alpha"])
        theory["positive_information_screen"] = (
            theory["delta_h_raw_bh_adjusted_p_value"].le(alpha)
            & theory["delta_h_raw_normal_ci_low"].gt(0)
            & theory["delta_l_normal_ci_low"].gt(0)
        )
        theory_pairwise = _theory_pairwise(
            theory, theory_bootstrap, config=config
        )
        weights_test = {
            city: weights_by_city_period[(city, "test")]
            for city in CITY_ORDER
        }
        contrasts, prediction_bootstrap = _prediction_contrasts(
            predictions,
            theory,
            theory_bootstrap,
            weights_test,
            config=config,
        )
        prediction_pairwise = _prediction_pairwise(
            contrasts, prediction_bootstrap, config=config
        )
        composition = _descriptive_composition(
            panel_by_city, config=config
        )
        model_count = len(
            list((run_dir / "category_models").rglob("*.joblib"))
        )
        acceptance = _acceptance_checks(
            panel_by_city=panel_by_city,
            theory=theory,
            theory_bootstrap=theory_bootstrap,
            theory_pairwise=theory_pairwise,
            predictions=predictions,
            tuning=tuning,
            metrics=metrics,
            metadata=metadata_records,
            contrasts=contrasts,
            prediction_bootstrap=prediction_bootstrap,
            prediction_pairwise=prediction_pairwise,
            model_count=model_count,
            config=config,
        )
        table_dir = run_dir / "tables"
        composition.to_csv(
            table_dir / "crime_type_composition.csv", index=False
        )
        theory.to_csv(
            table_dir / "crime_type_information_bounds.csv", index=False
        )
        theory_bootstrap.to_parquet(
            table_dir / "crime_type_theory_bootstrap.parquet",
            index=False,
            compression="zstd",
        )
        theory_pairwise.to_csv(
            table_dir / "paired_theory_heterogeneity.csv", index=False
        )
        predictions.sort_values(
            [
                "city",
                "outcome",
                "model_family",
                "information_set",
                "date",
                "grid_id",
            ],
            inplace=True,
        )
        predictions.to_parquet(
            table_dir / "crime_type_test_predictions.parquet",
            index=False,
            compression="zstd",
        )
        metrics.to_csv(
            table_dir / "crime_type_test_metrics.csv", index=False
        )
        contrasts.to_csv(
            table_dir / "crime_type_prediction_contrasts.csv", index=False
        )
        prediction_bootstrap.to_parquet(
            table_dir / "crime_type_prediction_bootstrap.parquet",
            index=False,
            compression="zstd",
        )
        prediction_pairwise.to_csv(
            table_dir / "paired_prediction_heterogeneity.csv", index=False
        )
        tuning.to_csv(
            table_dir / "crime_type_validation_tuning.csv", index=False
        )
        pd.DataFrame(metadata_records).assign(
            selected_parameters_json=lambda frame: frame[
                "selected_parameters"
            ].map(lambda value: json.dumps(value, sort_keys=True)),
            features_json=lambda frame: frame["features"].map(json.dumps),
            refit_splits_json=lambda frame: frame["refit_splits"].map(
                json.dumps
            ),
        ).drop(
            columns=["selected_parameters", "features", "refit_splits"]
        ).to_csv(
            table_dir / "crime_type_model_metadata.csv", index=False
        )
        diagnostics_frame.to_csv(
            table_dir / "crime_type_input_diagnostics.csv", index=False
        )
        thresholds.to_csv(
            table_dir / "e04_bicycle_thresholds_reused.csv", index=False
        )
        acceptance.to_csv(
            table_dir / "acceptance_checklist.csv", index=False
        )
        failed = acceptance.loc[acceptance["status"].ne("PASS")]
        if len(failed):
            raise AssertionError(
                "E08 acceptance gate failed: "
                + ", ".join(failed["criterion"].astype(str))
            )
        _publication_style()
        dpi = int(config["reporting"]["figure_dpi"])
        display_model = str(config["reporting"]["primary_display_model"])
        figure_dir = run_dir / "figures"
        _plot_composition(
            composition, figure_dir / "e08_crime_type_composition.png", dpi
        )
        _plot_theory_metric(
            theory,
            metric="delta_h_miller_madow_raw_bits",
            low="delta_h_raw_normal_ci_low",
            high="delta_h_raw_normal_ci_high",
            ylabel="Conditional information gain (bits)",
            title="Crime-type heterogeneity in bicycle conditional information",
            path=figure_dir / "e08_crime_type_cmi.png",
            dpi=dpi,
        )
        _plot_theory_metric(
            theory,
            metric="delta_l_exact_mse",
            low="delta_l_normal_ci_low",
            high="delta_l_normal_ci_high",
            ylabel="Exact DELB reduction (MSE)",
            title="Crime-type heterogeneity in theoretical error-floor reduction",
            path=figure_dir / "e08_crime_type_delb_reduction.png",
            dpi=dpi,
        )
        _plot_test_bounds(
            theory, figure_dir / "e08_test_category_bounds.png", dpi
        )
        _heatmap(
            contrasts,
            metric="delta_mse_integer",
            title="Held-out empirical MSE improvement by crime type and model",
            path=figure_dir / "e08_prediction_improvement_heatmap.png",
            dpi=dpi,
        )
        _heatmap(
            contrasts,
            metric="gap_narrowing",
            title="Theory–practice gap narrowing by crime type and model",
            path=figure_dir / "e08_gap_narrowing_heatmap.png",
            dpi=dpi,
        )
        _plot_display_model(
            contrasts,
            family=display_model,
            path=figure_dir / "e08_display_model_category_mse.png",
            dpi=dpi,
        )
        _plot_pairwise_theory(
            theory_pairwise,
            figure_dir / "e08_paired_theory_heterogeneity.png",
            dpi,
        )
        _write_interpretation(
            run_dir,
            composition,
            theory,
            theory_pairwise,
            contrasts,
            prediction_pairwise,
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
            "outcomes": OUTCOME_ORDER,
            "theory_rows": len(theory),
            "theory_bootstrap_rows": len(theory_bootstrap),
            "matched_prediction_contrasts": len(contrasts),
            "prediction_bootstrap_rows": len(prediction_bootstrap),
            "test_prediction_rows": len(predictions),
            "serialized_models": model_count,
            "python_article_figure_groups": 8,
            "elapsed_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "article_figures_generated_by": "Python matplotlib",
        }
        (run_dir / "run_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        (empirical_root / "runs" / "latest_e08_run.txt").write_text(
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
            "E08 passed all %d checks in %.1f seconds",
            len(acceptance),
            elapsed,
        )
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
        logger.exception("E08 failed; valid model checkpoints were retained.")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E08 crime-type heterogeneity analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("empirical/config/e08.yaml"),
    )
    parser.add_argument("--resume-run", type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.resume_run)
