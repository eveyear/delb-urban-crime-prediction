from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml

from entropy_crime_bike.conditional_information import ConditionalCodebook
from entropy_crime_bike.discrete_bound import inverse_entropy_envelope
from entropy_crime_bike.e06_predictive_models import (
    CITY_COLORS,
    CITY_LABELS,
    CITY_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    _load_city_panel,
    _load_e04_thresholds,
    _prepare_city,
)
from entropy_crime_bike.spatial_information import (
    BoundInterpolator,
    benjamini_hochberg,
)
from entropy_crime_bike.theory_practice import (
    calculate_gap_metrics,
    ols_city_fixed_effect_hc3,
    paired_grid_classification,
    spearman_with_spatial_bootstrap,
)


CATEGORY_ORDER = [
    "high_theory__high_empirical",
    "high_theory__low_empirical",
    "low_theory__high_empirical",
    "low_theory__low_empirical",
]
CATEGORY_LABELS = {
    "high_theory__high_empirical": "High theory / high empirical",
    "high_theory__low_empirical": "High theory / low empirical",
    "low_theory__high_empirical": "Low theory / high empirical",
    "low_theory__low_empirical": "Low theory / low empirical",
}
CATEGORY_COLORS = {
    "high_theory__high_empirical": "#2E7D32",
    "high_theory__low_empirical": "#F9A825",
    "low_theory__high_empirical": "#1565C0",
    "low_theory__low_empirical": "#B0BEC5",
}


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return loaded


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e07_theory_practice")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e07_theory_practice.log", encoding="utf-8"
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


def _seed(base_seed: int, name: str) -> int:
    return int((base_seed + zlib.crc32(name.encode("utf-8"))) % 2**32)


def _read_pointer(empirical_root: Path, relative: str) -> Path:
    pointer = empirical_root / relative
    result = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    if not result.exists():
        raise FileNotFoundError(f"Run pointer target does not exist: {result}")
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
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _bound_pair(
    h0_raw: float,
    hb_raw: float,
    *,
    tail_tolerance: float,
    root_tolerance: float,
) -> dict[str, float | bool]:
    h0 = max(float(h0_raw), 0.0)
    hb_nonnegative = max(float(hb_raw), 0.0)
    hb = min(hb_nonnegative, h0)
    l0 = inverse_entropy_envelope(
        h0,
        tail_tolerance=tail_tolerance,
        root_tolerance=root_tolerance,
    )
    lb = inverse_entropy_envelope(
        hb,
        tail_tolerance=tail_tolerance,
        root_tolerance=root_tolerance,
    )
    return {
        "h0_for_bound_bits": h0,
        "hb_for_bound_bits": hb,
        "delta_h_projected_bits": h0 - hb,
        "l0_exact_mse": l0,
        "lb_exact_mse": lb,
        "delta_l_exact_mse": l0 - lb,
        "projection_applied": hb_nonnegative > h0,
    }


def _city_test_bound(
    prepared: pd.DataFrame,
    *,
    city: str,
    config: dict[str, object],
    interpolator: BoundInterpolator,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    subset = prepared.loc[prepared["split"].eq("test")].copy()
    block_days = int(config["uncertainty"]["block_days"])
    subset["block_id"] = (
        (subset["date"] - subset["date"].min()).dt.days // block_days
    ).astype(np.int16)
    codebook = ConditionalCodebook.from_frame(
        subset,
        target=str(config["target"]),
        baseline_state=list(config["information_sets"]["baseline_state"]),
        bicycle_state_field=str(
            config["information_sets"]["bicycle_state_field"]
        ),
        block_field="block_id",
    )
    point_entropy = codebook.estimate(np.ones(codebook.block_count)).iloc[0]
    point_bound = _bound_pair(
        point_entropy["h0_miller_madow_bits"],
        point_entropy["hb_miller_madow_bits"],
        tail_tolerance=float(config["numerics"]["lattice_tail_tolerance"]),
        root_tolerance=float(config["numerics"]["root_tolerance"]),
    )
    point = {
        "city": city,
        "city_label": CITY_LABELS[city],
        "period": "test",
        "period_start": str(subset["date"].min().date()),
        "period_end": str(subset["date"].max().date()),
        "observations": len(subset),
        "grids": int(subset["grid_id"].nunique()),
        "days": int(subset["date"].nunique()),
        "blocks": codebook.block_count,
        **point_entropy.to_dict(),
        **point_bound,
        **codebook.sparsity_diagnostics(),
    }
    repetitions = int(config["uncertainty"]["city_repetitions"])
    rng = np.random.default_rng(
        _seed(int(config["random_seed"]), f"E07|city-joint|{city}")
    )
    weights = rng.multinomial(
        codebook.block_count,
        np.full(codebook.block_count, 1.0 / codebook.block_count),
        size=repetitions,
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
    bootstrap_bound = pd.DataFrame(
        {
            "replicate": np.arange(repetitions),
            "h0_for_bound_bits": h0,
            "hb_for_bound_bits": hb,
            "delta_h_projected_bits": h0 - hb,
            "l0_exact_mse": interpolator.transform(h0),
            "lb_exact_mse": interpolator.transform(hb),
        }
    )
    bootstrap_bound["delta_l_exact_mse"] = (
        bootstrap_bound["l0_exact_mse"]
        - bootstrap_bound["lb_exact_mse"]
    )
    return pd.DataFrame([point]), weights, bootstrap_bound


def _paired_city_model(
    predictions: pd.DataFrame,
    *,
    city: str,
    family: str,
    weights: np.ndarray,
    bound_point: pd.Series,
    bound_bootstrap: pd.DataFrame,
    config: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    subset = predictions.loc[
        predictions["city"].eq(city)
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
        raise AssertionError(f"Unmatched E07 prediction pairs for {city}/{family}.")
    block_days = int(config["uncertainty"]["block_days"])
    paired["block_id"] = (
        (paired["date"] - paired["date"].min()).dt.days // block_days
    ).astype(int)
    block_count = weights.shape[1]
    block = (
        paired.groupby("block_id", observed=True)
        .agg(
            baseline_sse=("squared_error_integer_baseline", "sum"),
            bicycle_sse=("squared_error_integer_bicycle", "sum"),
            observations=("observed_count", "size"),
        )
        .reindex(range(block_count), fill_value=0)
    )
    counts = block["observations"].to_numpy(float)
    mse0_bootstrap = (
        weights @ block["baseline_sse"].to_numpy(float)
    ) / (weights @ counts)
    mseb_bootstrap = (
        weights @ block["bicycle_sse"].to_numpy(float)
    ) / (weights @ counts)
    point_metrics = calculate_gap_metrics(
        paired["squared_error_integer_baseline"].mean(),
        paired["squared_error_integer_bicycle"].mean(),
        float(bound_point["l0_exact_mse"]),
        float(bound_point["lb_exact_mse"]),
    )
    bootstrap_metrics = calculate_gap_metrics(
        mse0_bootstrap,
        mseb_bootstrap,
        bound_bootstrap["l0_exact_mse"].to_numpy(float),
        bound_bootstrap["lb_exact_mse"].to_numpy(float),
    )
    bootstrap = pd.DataFrame(
        {
            "city": city,
            "model_family": family,
            "replicate": np.arange(len(weights)),
            "mse_baseline_integer": mse0_bootstrap,
            "mse_bicycle_integer": mseb_bootstrap,
            **bootstrap_metrics,
        }
    )
    alpha = 1.0 - float(config["uncertainty"]["confidence_level"])
    record: dict[str, object] = {
        "city": city,
        "city_label": CITY_LABELS[city],
        "model_family": family,
        "model_label": MODEL_LABELS[family],
        "observations": len(paired),
        "test_days": int(paired["date"].nunique()),
        "test_grids": int(paired["grid_id"].nunique()),
        "mse_baseline_integer": float(
            paired["squared_error_integer_baseline"].mean()
        ),
        "mse_bicycle_integer": float(
            paired["squared_error_integer_bicycle"].mean()
        ),
        "l0_exact_mse": float(bound_point["l0_exact_mse"]),
        "lb_exact_mse": float(bound_point["lb_exact_mse"]),
        **point_metrics,
    }
    interval_fields = [
        "gap_baseline",
        "gap_bicycle",
        "information_use_efficiency_baseline",
        "information_use_efficiency_bicycle",
        "delta_mse_integer",
        "delta_l_exact_mse",
        "gap_narrowing",
        "realization_ratio_diagnostic",
    ]
    for field in interval_fields:
        values = bootstrap[field].to_numpy(float)
        finite = values[np.isfinite(values)]
        record[f"{field}_ci_low"] = float(np.quantile(finite, alpha / 2.0))
        record[f"{field}_ci_high"] = float(
            np.quantile(finite, 1.0 - alpha / 2.0)
        )
        record[f"{field}_bootstrap_se"] = float(np.std(finite, ddof=1))
    return record, bootstrap


def _local_pretest_theory(
    prepared: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    city: str,
    config: dict[str, object],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_city = eligibility.loc[
        eligibility["city"].eq(city) & eligibility["eligible"].astype(bool)
    ].copy()
    eligible_ids = set(eligible_city["grid_id"].astype(str))
    periods = set(str(value) for value in config["estimands"]["local_theory_periods"])
    subset_city = prepared.loc[
        prepared["split"].astype(str).isin(periods)
        & prepared["grid_id"].astype(str).isin(eligible_ids)
    ].copy()
    block_days = int(config["uncertainty"]["block_days"])
    subset_city["block_id"] = (
        (subset_city["date"] - subset_city["date"].min()).dt.days // block_days
    ).astype(np.int16)
    metadata = eligible_city.set_index(eligible_city["grid_id"].astype(str))
    records = []
    exclusions = []
    for index, grid_id in enumerate(sorted(eligible_ids), start=1):
        grid = subset_city.loc[
            subset_city["grid_id"].astype(str).eq(grid_id)
        ].sort_values("date")
        if grid.empty:
            exclusions.append(
                {
                    "city": city,
                    "grid_id": grid_id,
                    "reason": "no_pretest_observations",
                }
            )
            continue
        codebook = ConditionalCodebook.from_frame(
            grid,
            target=str(config["target"]),
            baseline_state=list(
                config["information_sets"]["local_baseline_state"]
            ),
            bicycle_state_field=str(
                config["information_sets"]["bicycle_state_field"]
            ),
            block_field="block_id",
        )
        entropy = codebook.estimate(np.ones(codebook.block_count)).iloc[0]
        bound = _bound_pair(
            entropy["h0_miller_madow_bits"],
            entropy["hb_miller_madow_bits"],
            tail_tolerance=float(config["numerics"]["lattice_tail_tolerance"]),
            root_tolerance=float(config["numerics"]["root_tolerance"]),
        )
        meta = metadata.loc[grid_id]
        records.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "grid_id": grid_id,
                "x_index": int(meta["x_index"]),
                "y_index": int(meta["y_index"]),
                "centroid_longitude": float(meta["centroid_longitude"]),
                "centroid_latitude": float(meta["centroid_latitude"]),
                "theory_period_start": str(grid["date"].min().date()),
                "theory_period_end": str(grid["date"].max().date()),
                "theory_observations": len(grid),
                "theory_blocks": codebook.block_count,
                **entropy.to_dict(),
                **bound,
                **codebook.sparsity_diagnostics(),
            }
        )
        if index % 50 == 0 or index == len(eligible_ids):
            logger.info(
                "%s local pretest theory: %d/%d eligible grids",
                city,
                index,
                len(eligible_ids),
            )
    return pd.DataFrame(records), pd.DataFrame(exclusions)


def _local_comparison(
    local_theory: pd.DataFrame,
    local_test: pd.DataFrame,
    *,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = local_theory.merge(
        local_test,
        on=[
            "city",
            "grid_id",
            "x_index",
            "y_index",
            "centroid_longitude",
            "centroid_latitude",
        ],
        how="inner",
        validate="one_to_many",
        suffixes=("_theory", "_empirical"),
    )
    comparison["gap_narrowing_local"] = (
        comparison["delta_mse_integer"]
        - comparison["delta_l_exact_mse"]
    )
    comparison["realization_ratio_diagnostic"] = np.divide(
        comparison["delta_mse_integer"],
        comparison["delta_l_exact_mse"],
        out=np.full(len(comparison), np.nan),
        where=comparison["delta_l_exact_mse"].abs().to_numpy()
        > np.finfo(float).eps,
    )
    classifications = []
    thresholds = []
    for (city, family), subset in comparison.groupby(
        ["city", "model_family"], observed=True, sort=True
    ):
        labels, theory_threshold, empirical_threshold = paired_grid_classification(
            subset["delta_l_exact_mse"], subset["delta_mse_integer"]
        )
        part = subset.copy()
        part["theory_practice_category"] = labels
        part["theory_high"] = (
            part["delta_l_exact_mse"] > theory_threshold
        )
        part["empirical_high"] = (
            part["delta_mse_integer"] > empirical_threshold
        )
        part["theory_threshold_city_median"] = theory_threshold
        part["empirical_threshold_city_model_median"] = empirical_threshold
        classifications.append(part)
        counts = pd.Series(labels).value_counts()
        thresholds.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "model_family": family,
                "model_label": MODEL_LABELS[family],
                "grids": len(part),
                "theory_threshold_city_median": theory_threshold,
                "empirical_threshold_city_model_median": empirical_threshold,
                **{
                    f"n_{category}": int(counts.get(category, 0))
                    for category in CATEGORY_ORDER
                },
            }
        )
    classified = pd.concat(classifications, ignore_index=True)
    threshold_frame = pd.DataFrame(thresholds)
    association_records = []
    repetitions = int(
        config["uncertainty"]["local_correlation_repetitions"]
    )
    for (city, family), subset in classified.groupby(
        ["city", "model_family"], observed=True, sort=True
    ):
        result = spearman_with_spatial_bootstrap(
            subset["delta_l_exact_mse"],
            subset["delta_mse_integer"],
            repetitions=repetitions,
            seed=_seed(
                int(config["random_seed"]),
                f"E07|local-correlation|{city}|{family}",
            ),
            confidence_level=float(config["uncertainty"]["confidence_level"]),
        )
        association_records.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "model_family": family,
                "model_label": MODEL_LABELS[family],
                **result,
            }
        )
    associations = pd.DataFrame(association_records)
    associations["bh_adjusted_p_value"] = benjamini_hochberg(
        associations["asymptotic_two_sided_p_value"]
    )
    regressions = []
    for family, subset in classified.groupby(
        "model_family", observed=True, sort=True
    ):
        result = ols_city_fixed_effect_hc3(
            subset,
            outcome="delta_mse_integer",
            exposure="delta_l_exact_mse",
            confidence_level=float(config["uncertainty"]["confidence_level"]),
        )
        regressions.append(
            {
                "model_family": family,
                "model_label": MODEL_LABELS[family],
                **result,
            }
        )
    return (
        classified,
        associations,
        pd.DataFrame(regressions),
        threshold_frame,
    )


def _acceptance_checks(
    *,
    city_bounds: pd.DataFrame,
    city_models: pd.DataFrame,
    city_bootstrap: pd.DataFrame,
    local_theory: pd.DataFrame,
    local_comparison: pd.DataFrame,
    local_associations: pd.DataFrame,
    regressions: pd.DataFrame,
    local_exclusions: pd.DataFrame,
    e06_contrasts: pd.DataFrame,
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

    add(
        "Expected city bound rows",
        expected["expected_city_bound_rows"],
        len(city_bounds),
        len(city_bounds) == int(expected["expected_city_bound_rows"]),
    )
    add(
        "Expected city-model rows",
        expected["expected_city_model_rows"],
        len(city_models),
        len(city_models) == int(expected["expected_city_model_rows"]),
    )
    add(
        "Expected joint city bootstrap rows",
        expected["expected_city_bootstrap_rows"],
        len(city_bootstrap),
        len(city_bootstrap) == int(expected["expected_city_bootstrap_rows"]),
    )
    test_observations = int(city_bounds["observations"].sum())
    add(
        "Matched test observations",
        expected["expected_test_observations"],
        test_observations,
        test_observations == int(expected["expected_test_observations"]),
    )
    add(
        "All accepted E05 grids retained for pretest theory",
        expected["expected_e05_eligible_grids"],
        len(local_theory),
        len(local_theory) == int(expected["expected_e05_eligible_grids"])
        and local_exclusions.empty,
    )
    add(
        "Expected local joined rows",
        expected["expected_local_join_rows"],
        len(local_comparison),
        len(local_comparison) == int(expected["expected_local_join_rows"]),
    )
    add(
        "Expected local association rows",
        expected["expected_local_association_rows"],
        len(local_associations),
        len(local_associations)
        == int(expected["expected_local_association_rows"]),
    )
    add(
        "Expected city-fixed-effect regressions",
        expected["expected_regression_rows"],
        len(regressions),
        len(regressions) == int(expected["expected_regression_rows"]),
    )
    maximum_gap_identity = float(
        np.max(
            np.abs(
                city_models["gap_narrowing"]
                - (
                    city_models["delta_mse_integer"]
                    - city_models["delta_l_exact_mse"]
                )
            )
        )
    )
    add(
        "Gap-narrowing identity",
        f"<= {tolerance}",
        maximum_gap_identity,
        maximum_gap_identity <= tolerance,
    )
    add(
        "Exact bound ordering",
        "0 <= LB <= L0",
        int(
            (
                (city_bounds["lb_exact_mse"] >= -tolerance)
                & (
                    city_bounds["lb_exact_mse"]
                    <= city_bounds["l0_exact_mse"] + tolerance
                )
            ).sum()
        ),
        (
            (city_bounds["lb_exact_mse"] >= -tolerance)
            & (
                city_bounds["lb_exact_mse"]
                <= city_bounds["l0_exact_mse"] + tolerance
            )
        ).all(),
    )
    add(
        "Empirical MSE respects matched exact DELB",
        "all 24 information-set comparisons",
        int(
            (
                (
                    city_models["mse_baseline_integer"]
                    + tolerance
                    >= city_models["l0_exact_mse"]
                )
                & (
                    city_models["mse_bicycle_integer"]
                    + tolerance
                    >= city_models["lb_exact_mse"]
                )
            ).sum()
        ),
        (
            (
                city_models["mse_baseline_integer"]
                + tolerance
                >= city_models["l0_exact_mse"]
            )
            & (
                city_models["mse_bicycle_integer"]
                + tolerance
                >= city_models["lb_exact_mse"]
            )
        ).all(),
    )
    add(
        "Local theory ends before held-out prediction period",
        "< 2022-07-01",
        local_theory["theory_period_end"].max(),
        pd.to_datetime(local_theory["theory_period_end"]).max()
        < pd.Timestamp("2022-07-01"),
    )
    merged = city_models.merge(
        e06_contrasts[
            [
                "city",
                "model_family",
                "mse_baseline_integer",
                "mse_bicycle_integer",
            ]
        ],
        on=["city", "model_family"],
        suffixes=("_e07", "_e06"),
        validate="one_to_one",
    )
    maximum_mse_difference = float(
        np.max(
            np.abs(
                np.concatenate(
                    [
                        merged["mse_baseline_integer_e07"]
                        - merged["mse_baseline_integer_e06"],
                        merged["mse_bicycle_integer_e07"]
                        - merged["mse_bicycle_integer_e06"],
                    ]
                )
            )
        )
    )
    add(
        "E07 point MSE exactly reuses E06 held-out predictions",
        f"<= {tolerance}",
        maximum_mse_difference,
        maximum_mse_difference <= tolerance,
    )
    add(
        "No seasonal-naive model in DELB-matched comparison",
        "four matched families only",
        sorted(city_models["model_family"].unique()),
        set(city_models["model_family"]) == set(MODEL_ORDER),
    )
    add(
        "Information-use efficiencies lie in [0,1]",
        "all 24 ratios",
        int(
            (
                city_models[
                    [
                        "information_use_efficiency_baseline",
                        "information_use_efficiency_bicycle",
                    ]
                ]
                .stack()
                .between(-tolerance, 1.0 + tolerance)
            ).sum()
        ),
        city_models[
            [
                "information_use_efficiency_baseline",
                "information_use_efficiency_bicycle",
            ]
        ]
        .stack()
        .between(-tolerance, 1.0 + tolerance)
        .all(),
    )
    add(
        "All local classifications assigned",
        len(local_comparison),
        int(local_comparison["theory_practice_category"].notna().sum()),
        local_comparison["theory_practice_category"].notna().all(),
    )
    return pd.DataFrame(rows)


def _plot_city_gaps(city_models: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 4.0), sharey=False)
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for axis, city in zip(axes, CITY_ORDER):
        subset = city_models.loc[city_models["city"].eq(city)].set_index(
            "model_family"
        )
        axis.bar(
            x - width / 2,
            subset.loc[MODEL_ORDER, "gap_baseline"],
            width,
            color="#9EADBA",
            label="Baseline",
        )
        axis.bar(
            x + width / 2,
            subset.loc[MODEL_ORDER, "gap_bicycle"],
            width,
            color=CITY_COLORS[city],
            label="Bicycle-aware",
        )
        axis.set_title(CITY_LABELS[city])
        axis.set_xticks(x)
        axis.set_xticklabels(["State", "Poisson", "NB", "Boosting"])
        axis.set_ylabel("Predictability gap (MSE)")
        axis.grid(axis="y", alpha=0.25)
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
        "Held-out empirical error above the matched discrete entropy lower bound",
        y=1.01,
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    _save_figure(figure, path, dpi)


def _plot_efficiency(city_models: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 4.0), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    for axis, city in zip(axes, CITY_ORDER):
        subset = city_models.loc[city_models["city"].eq(city)].set_index(
            "model_family"
        )
        axis.plot(
            x,
            subset.loc[
                MODEL_ORDER, "information_use_efficiency_baseline"
            ],
            marker="o",
            color="#6D7B86",
            label="Baseline",
        )
        axis.plot(
            x,
            subset.loc[
                MODEL_ORDER, "information_use_efficiency_bicycle"
            ],
            marker="s",
            color=CITY_COLORS[city],
            label="Bicycle-aware",
        )
        axis.set_title(CITY_LABELS[city])
        axis.set_xticks(x)
        axis.set_xticklabels(["State", "Poisson", "NB", "Boosting"])
        axis.set_ylim(bottom=0)
        axis.set_ylabel("Matched DELB / held-out MSE")
        axis.grid(alpha=0.25)
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
        "Information-use efficiency as proximity to the matched theoretical floor",
        y=1.01,
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    _save_figure(figure, path, dpi)


def _plot_gap_narrowing(city_models: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 4.2), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    for axis, city in zip(axes, CITY_ORDER):
        subset = city_models.loc[city_models["city"].eq(city)].set_index(
            "model_family"
        ).loc[MODEL_ORDER]
        values = subset["gap_narrowing"].to_numpy(float)
        lower = subset["gap_narrowing_ci_low"].to_numpy(float)
        upper = subset["gap_narrowing_ci_high"].to_numpy(float)
        axis.vlines(x, lower, upper, color="#555555", linewidth=1.1)
        axis.hlines(
            np.concatenate([lower, upper]),
            np.concatenate([x - 0.05, x - 0.05]),
            np.concatenate([x + 0.05, x + 0.05]),
            color="#555555",
            linewidth=1.1,
        )
        axis.scatter(x, values, color=CITY_COLORS[city], zorder=3)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(CITY_LABELS[city])
        axis.set_xticks(x)
        axis.set_xticklabels(["State", "Poisson", "NB", "Boosting"])
        axis.set_ylabel(r"Gap narrowing: $\Delta$MSE $-$ $\Delta L$")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Joint block-bootstrap comparison of empirical and theoretical reductions",
        y=1.02,
        fontsize=12,
    )
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _plot_local_scatter(
    comparison: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        3, 4, figsize=(13.2, 10.0), sharex=False, sharey=False
    )
    for row, city in enumerate(CITY_ORDER):
        for column, family in enumerate(MODEL_ORDER):
            axis = axes[row, column]
            subset = comparison.loc[
                comparison["city"].eq(city)
                & comparison["model_family"].eq(family)
            ]
            axis.scatter(
                subset["delta_l_exact_mse"],
                subset["delta_mse_integer"],
                s=12,
                alpha=0.62,
                color=CITY_COLORS[city],
                linewidths=0,
            )
            rho = scipy.stats.spearmanr(
                subset["delta_l_exact_mse"], subset["delta_mse_integer"]
            ).statistic
            axis.axhline(0.0, color="#777777", linewidth=0.6)
            axis.text(
                0.04,
                0.94,
                rf"$\rho_s={rho:.2f}$",
                transform=axis.transAxes,
                va="top",
            )
            if row == 0:
                axis.set_title(MODEL_LABELS[family])
            if column == 0:
                axis.set_ylabel(
                    f"{CITY_LABELS[city]}\nLocal held-out " + r"$\Delta$MSE"
                )
            if row == len(CITY_ORDER) - 1:
                axis.set_xlabel(r"Pretest local $\Delta L$")
            axis.grid(alpha=0.18)
    figure.suptitle(
        "Local theoretical lower-bound reduction versus held-out MSE improvement",
        y=0.995,
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    _save_figure(figure, path, dpi)


def _plot_category_maps(
    comparison: pd.DataFrame,
    *,
    family: str,
    path: Path,
    dpi: int,
) -> None:
    subset = comparison.loc[comparison["model_family"].eq(family)]
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    for axis, city in zip(axes, CITY_ORDER):
        city_data = subset.loc[subset["city"].eq(city)]
        for category in CATEGORY_ORDER:
            part = city_data.loc[
                city_data["theory_practice_category"].eq(category)
            ]
            axis.scatter(
                part["x_index"],
                part["y_index"],
                s=36,
                marker="s",
                color=CATEGORY_COLORS[category],
                label=CATEGORY_LABELS[category],
                linewidths=0,
            )
        axis.set_title(CITY_LABELS[city])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        f"Local theory--practice classification: {MODEL_LABELS[family]}",
        y=0.98,
        fontsize=12,
    )
    figure.subplots_adjust(bottom=0.20, top=0.84, wspace=0.10)
    _save_figure(figure, path, dpi)


def _plot_associations(
    associations: pd.DataFrame, path: Path, dpi: int
) -> None:
    display = associations.copy()
    display["position"] = (
        display["model_family"].map(
            {model: index for index, model in enumerate(MODEL_ORDER)}
        )
        + display["city"].map({"DC": -0.2, "NY": 0.0, "VAN": 0.2})
    )
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    for city in CITY_ORDER:
        subset = display.loc[display["city"].eq(city)].sort_values("position")
        values = subset["spearman_rho"].to_numpy(float)
        positions = subset["position"].to_numpy(float)
        lower = subset["bootstrap_ci_low"].to_numpy(float)
        upper = subset["bootstrap_ci_high"].to_numpy(float)
        axis.vlines(
            positions,
            lower,
            upper,
            color=CITY_COLORS[city],
            linewidth=1.1,
        )
        axis.scatter(
            positions,
            values,
            color=CITY_COLORS[city],
            label=CITY_LABELS[city],
            zorder=3,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(MODEL_ORDER)))
    axis.set_xticklabels(["State mean", "Poisson", "NB", "Boosting"])
    axis.set_ylabel("Spearman correlation")
    axis.set_title(
        "Spatial association between pretest theoretical and held-out empirical gains",
        loc="left",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    _save_figure(figure, path, dpi)


def _plot_theory_empirical_shifts(
    city_models: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 4.1), sharey=False)
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for axis, city in zip(axes, CITY_ORDER):
        subset = city_models.loc[city_models["city"].eq(city)].set_index(
            "model_family"
        ).loc[MODEL_ORDER]
        axis.bar(
            x - width / 2,
            subset["delta_l_exact_mse"],
            width,
            color="#7A5195",
            label=r"Theoretical $\Delta L$",
        )
        axis.bar(
            x + width / 2,
            subset["delta_mse_integer"],
            width,
            color=CITY_COLORS[city],
            label=r"Empirical $\Delta$MSE",
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(CITY_LABELS[city])
        axis.set_xticks(x)
        axis.set_xticklabels(["State", "Poisson", "NB", "Boosting"])
        axis.set_ylabel("Reduction in MSE units")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Matched theoretical opportunity and realized held-out improvement",
        y=1.02,
        fontsize=12,
    )
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _write_interpretation(
    run_dir: Path,
    city_bounds: pd.DataFrame,
    city_models: pd.DataFrame,
    associations: pd.DataFrame,
    regressions: pd.DataFrame,
) -> None:
    lines = [
        "# E07 Theory--Practice Predictability-Gap Results",
        "",
        "## Estimand alignment",
        "",
        "- City-level DELBs were re-estimated on the exact July--December 2022 E06 test domain; pooled 2020--2022 E04 bounds were not subtracted from held-out E06 MSE.",
        "- The same nonoverlapping seven-day city blocks jointly resampled the conditional-entropy bound and both members of each E06 prediction pair.",
        "- Local DELB reductions were estimated from training plus validation data ending 30 June 2022, whereas local empirical MSE improvements came only from held-out test predictions.",
        "- Information-use efficiency is DELB/MSE (proximity to the floor). The separate delta-MSE/delta-L realization ratio is unbounded and retained only as a diagnostic.",
        "",
        "## Matched test-period theoretical opportunity",
        "",
    ]
    for _, row in city_bounds.iterrows():
        lines.append(
            f"- {row['city_label']}: H0={row['h0_for_bound_bits']:.5f} bits, "
            f"HB={row['hb_for_bound_bits']:.5f} bits, "
            f"L0={row['l0_exact_mse']:.5f}, LB={row['lb_exact_mse']:.5f}, "
            f"delta L={row['delta_l_exact_mse']:.5f}."
        )
    lines.extend(["", "## City-model gap comparison", ""])
    for city in CITY_ORDER:
        lines.extend([f"### {CITY_LABELS[city]}", ""])
        subset = city_models.loc[city_models["city"].eq(city)].set_index(
            "model_family"
        )
        for family in MODEL_ORDER:
            row = subset.loc[family]
            lines.append(
                f"- {MODEL_LABELS[family]}: delta MSE "
                f"{row['delta_mse_integer']:.5f}; delta L "
                f"{row['delta_l_exact_mse']:.5f}; gap narrowing "
                f"{row['gap_narrowing']:.5f} "
                f"(95% joint block interval "
                f"[{row['gap_narrowing_ci_low']:.5f}, "
                f"{row['gap_narrowing_ci_high']:.5f}])."
            )
        lines.append("")
    lines.extend(["## Local theory--practice association", ""])
    for _, row in associations.iterrows():
        lines.append(
            f"- {row['city_label']}, {row['model_label']}: Spearman rho "
            f"{row['spearman_rho']:.3f} (spatial-bootstrap 95% interval "
            f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]; "
            f"BH-adjusted asymptotic p={row['bh_adjusted_p_value']:.4g})."
        )
    lines.extend(["", "## Pooled city-fixed-effect slopes", ""])
    for _, row in regressions.iterrows():
        lines.append(
            f"- {row['model_label']}: slope={row['slope_delta_l']:.3f}, "
            f"HC3 95% interval [{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
            f"p={row['two_sided_p_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- DELB reduction is a change in irreducible-error opportunity, not a guarantee that a fitted model will improve by the same amount.",
            "- Local correlations and four-category maps are descriptive spatial comparisons; paired-grid resampling does not model cross-grid dependence.",
            "- Negative empirical improvements are valid evidence that a fitted model did not exploit the added state in this held-out period.",
            "- None of the information-value, predictability-gap, or association quantities identifies a causal effect of bicycle flows on crime.",
            "- Placebo, alternative-lag, scale, and rolling-origin analyses remain for later experiments.",
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
                "experiment": "E07",
                "status": "complete",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "run_directory": str(run_dir),
                "notes": (
                    "Test-window matched DELB gaps, pretest local theory, "
                    "joint uncertainty, and theory-practice spatial comparison."
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
        run_id = started_at.strftime("%Y%m%d_%H%M%S_E07_theory_practice")
        run_dir = empirical_root / "runs" / run_id
    else:
        run_dir = resume_run.resolve()
        run_id = run_dir.name
    for directory in ["logs", "tables", "figures", "state"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    command = (
        f"{sys.executable} empirical/run_e07_theory_practice.py "
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
    logger.info("Starting E07 matched theory--practice experiment")
    try:
        data_dir = _read_pointer(
            empirical_root, str(config["source_e02_data_pointer"])
        )
        e04_run, thresholds = _load_e04_thresholds(empirical_root, config)
        e05_run = _read_pointer(
            empirical_root, str(config["source_e05_run_pointer"])
        )
        e06_run = _read_pointer(
            empirical_root, str(config["source_e06_run_pointer"])
        )
        e05_status = json.loads(
            (e05_run / "run_status.json").read_text(encoding="utf-8")
        )
        e06_status = json.loads(
            (e06_run / "run_status.json").read_text(encoding="utf-8")
        )
        if (
            e05_status.get("acceptance_status") != "PASS"
            or e06_status.get("acceptance_status") != "PASS"
        ):
            raise AssertionError("E07 requires accepted E05 and E06 runs.")
        predictions = pd.read_parquet(
            e06_run / "tables" / "test_predictions.parquet"
        )
        predictions["date"] = pd.to_datetime(predictions["date"])
        predictions = predictions.loc[
            predictions["model_family"].isin(MODEL_ORDER)
        ].copy()
        e06_contrasts = pd.read_csv(
            e06_run / "tables" / "paired_mse_contrasts.csv"
        )
        local_test = pd.read_csv(
            e06_run / "tables" / "local_test_metrics.csv",
            dtype={"grid_id": str},
        )
        eligibility = pd.read_csv(
            e05_run / "tables" / "grid_eligibility.csv",
            dtype={"grid_id": str},
        )
        interpolator = BoundInterpolator.build(
            float(
                config["numerics"][
                    "bootstrap_bound_interpolation_max_entropy_bits"
                ]
            ),
            tail_tolerance=float(config["numerics"]["lattice_tail_tolerance"]),
            root_tolerance=float(config["numerics"]["root_tolerance"]),
        )
        city_bound_frames = []
        city_model_records = []
        city_bootstrap_frames = []
        local_theory_frames = []
        local_exclusion_frames = []
        for city in CITY_ORDER:
            threshold = thresholds.loc[thresholds["city"].eq(city)].iloc[0]
            cut_points = (
                float(threshold["positive_tertile_1_upper"]),
                float(threshold["positive_tertile_2_upper"]),
            )
            raw = _load_city_panel(data_dir, city, str(config["target"]))
            prepared, diagnostic = _prepare_city(
                raw, cut_points=cut_points, config=config
            )
            logger.info(
                "%s prepared: %d train, %d validation, %d test observations",
                city,
                diagnostic["training_rows"],
                diagnostic["validation_rows"],
                diagnostic["test_rows"],
            )
            city_bound, weights, bound_bootstrap = _city_test_bound(
                prepared,
                city=city,
                config=config,
                interpolator=interpolator,
            )
            city_bound_frames.append(city_bound)
            for family in MODEL_ORDER:
                record, bootstrap = _paired_city_model(
                    predictions,
                    city=city,
                    family=family,
                    weights=weights,
                    bound_point=city_bound.iloc[0],
                    bound_bootstrap=bound_bootstrap,
                    config=config,
                )
                city_model_records.append(record)
                city_bootstrap_frames.append(bootstrap)
            local_theory, local_exclusions = _local_pretest_theory(
                prepared,
                eligibility,
                city=city,
                config=config,
                logger=logger,
            )
            local_theory_frames.append(local_theory)
            if not local_exclusions.empty:
                local_exclusion_frames.append(local_exclusions)
            logger.info(
                "%s test DELB: L0 %.6f, LB %.6f, delta L %.6f",
                city,
                city_bound.iloc[0]["l0_exact_mse"],
                city_bound.iloc[0]["lb_exact_mse"],
                city_bound.iloc[0]["delta_l_exact_mse"],
            )
            del raw, prepared
        city_bounds = pd.concat(city_bound_frames, ignore_index=True)
        city_models = pd.DataFrame(city_model_records)
        city_bootstrap = pd.concat(city_bootstrap_frames, ignore_index=True)
        local_theory = pd.concat(local_theory_frames, ignore_index=True)
        local_exclusions = (
            pd.concat(local_exclusion_frames, ignore_index=True)
            if local_exclusion_frames
            else pd.DataFrame(columns=["city", "grid_id", "reason"])
        )
        (
            local_comparison,
            associations,
            regressions,
            classification_thresholds,
        ) = _local_comparison(
            local_theory,
            local_test,
            config=config,
        )
        acceptance = _acceptance_checks(
            city_bounds=city_bounds,
            city_models=city_models,
            city_bootstrap=city_bootstrap,
            local_theory=local_theory,
            local_comparison=local_comparison,
            local_associations=associations,
            regressions=regressions,
            local_exclusions=local_exclusions,
            e06_contrasts=e06_contrasts,
            config=config,
        )
        table_dir = run_dir / "tables"
        city_bounds.to_csv(table_dir / "test_window_city_delb.csv", index=False)
        city_models.to_csv(
            table_dir / "city_model_predictability_gaps.csv", index=False
        )
        city_bootstrap.to_parquet(
            table_dir / "joint_city_block_bootstrap.parquet",
            index=False,
            compression="zstd",
        )
        local_theory.to_csv(
            table_dir / "pretest_local_theory.csv", index=False
        )
        local_theory.to_parquet(
            table_dir / "pretest_local_theory.parquet",
            index=False,
            compression="zstd",
        )
        local_exclusions.to_csv(
            table_dir / "pretest_local_theory_exclusions.csv", index=False
        )
        local_comparison.to_csv(
            table_dir / "local_theory_practice_comparison.csv", index=False
        )
        local_comparison.to_parquet(
            table_dir / "local_theory_practice_comparison.parquet",
            index=False,
            compression="zstd",
        )
        associations.to_csv(
            table_dir / "local_spearman_associations.csv", index=False
        )
        regressions.to_csv(
            table_dir / "city_fixed_effect_regressions.csv", index=False
        )
        classification_thresholds.to_csv(
            table_dir / "theory_practice_classification_summary.csv",
            index=False,
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
                "E07 acceptance gate failed: "
                + ", ".join(failed["criterion"].astype(str))
            )
        _publication_style()
        dpi = int(config["reporting"]["figure_dpi"])
        display_model = str(config["reporting"]["primary_display_model"])
        figure_dir = run_dir / "figures"
        _plot_city_gaps(
            city_models, figure_dir / "e07_predictability_gaps.png", dpi
        )
        _plot_efficiency(
            city_models, figure_dir / "e07_information_use_efficiency.png", dpi
        )
        _plot_gap_narrowing(
            city_models, figure_dir / "e07_gap_narrowing.png", dpi
        )
        _plot_theory_empirical_shifts(
            city_models, figure_dir / "e07_theory_empirical_shifts.png", dpi
        )
        _plot_local_scatter(
            local_comparison, figure_dir / "e07_local_theory_practice.png", dpi
        )
        _plot_category_maps(
            local_comparison,
            family=display_model,
            path=figure_dir / "e07_theory_practice_categories.png",
            dpi=dpi,
        )
        _plot_associations(
            associations, figure_dir / "e07_local_correlations.png", dpi
        )
        _write_interpretation(
            run_dir, city_bounds, city_models, associations, regressions
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
            "source_e05_run_directory": str(e05_run),
            "source_e06_run_directory": str(e06_run),
            "test_observations": int(city_bounds["observations"].sum()),
            "city_model_comparisons": len(city_models),
            "joint_bootstrap_rows": len(city_bootstrap),
            "local_theory_grids": len(local_theory),
            "local_theory_practice_rows": len(local_comparison),
            "local_associations": len(associations),
            "city_fixed_effect_regressions": len(regressions),
            "python_article_figure_groups": 7,
            "elapsed_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "article_figures_generated_by": "Python matplotlib",
        }
        (run_dir / "run_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        (empirical_root / "runs" / "latest_e07_run.txt").write_text(
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
            "E07 passed all %d checks in %.1f seconds",
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
        logger.exception("E07 failed; completed outputs were retained.")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E07 matched theory--practice predictability-gap analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("empirical/config/e07.yaml"),
    )
    parser.add_argument("--resume-run", type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.resume_run)
