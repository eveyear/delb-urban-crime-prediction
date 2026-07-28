from __future__ import annotations

import argparse
import copy
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
from scipy.special import digamma

from entropy_crime_bike.conditional_information import ConditionalCodebook
from entropy_crime_bike.discrete_bound import inverse_entropy_envelope
from entropy_crime_bike.e06_predictive_models import (
    CITY_COLORS,
    CITY_LABELS,
    CITY_ORDER,
    MODEL_LABELS,
    _fit_model,
    _predict_model,
)
from entropy_crime_bike.panel import (
    _aggregate_bicycle_city,
    _aggregate_crime,
    _build_city_panel,
    _grid_reference,
    assign_projected_grid,
)
from entropy_crime_bike.predictive_models import nonnegative_half_up


MODEL_ORDER = [
    "state_mean",
    "poisson_glm",
    "negative_binomial_glm",
    "histogram_gradient_boosting_poisson",
]
SPEC_ORDER = [
    "primary",
    "spatial_500m",
    "spatial_2000m",
    "weekly",
    "crime_bins_coarse",
    "bicycle_bins_median",
    "bicycle_bins_quintiles",
    "crime_lag2",
    "crime_lag3",
    "crime_lag7",
    "bicycle_lag2",
    "bicycle_lag3",
    "bicycle_lag7",
    "bicycle_outflow",
    "bicycle_inflow",
    "bicycle_absolute_net",
]


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_pointer(root: Path, relative: str) -> Path:
    value = Path((root / relative).read_text(encoding="utf-8").strip()).resolve()
    if not value.exists():
        raise FileNotFoundError(value)
    return value


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e09_robustness")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in [
        logging.FileHandler(
            run_dir / "logs" / "e09_robustness.log", encoding="utf-8"
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
        frozen = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        lines.extend(["", "[pip-freeze]", frozen])
    except subprocess.SubprocessError as exc:
        lines.append(f"pip_freeze_error={exc}")
    path.write_text("\n".join(lines), encoding="utf-8")


def crime_bins(values: pd.Series, mapping: str) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="raise")
    if mapping == "four_state":
        labels = np.select(
            [numeric.eq(0), numeric.eq(1), numeric.eq(2), numeric.ge(3)],
            ["zero", "one", "two", "three_plus"],
            default=None,
        )
        categories = ["zero", "one", "two", "three_plus"]
    elif mapping == "zero_one_two_plus":
        labels = np.select(
            [numeric.eq(0), numeric.eq(1), numeric.ge(2)],
            ["zero", "one", "two_plus"],
            default=None,
        )
        categories = ["zero", "one", "two_plus"]
    else:
        raise ValueError(f"Unknown crime mapping: {mapping}")
    return pd.Categorical(labels, categories=categories, ordered=True)


def fit_flow_cut_points(
    values: pd.Series, mapping: str
) -> tuple[float, ...]:
    positive = pd.to_numeric(values, errors="raise")
    positive = positive.loc[positive.gt(0)]
    if positive.empty:
        raise ValueError("Positive bicycle flow is required.")
    probabilities = {
        "zero_plus_positive_median": [0.5],
        "zero_plus_positive_tertiles": [1 / 3, 2 / 3],
        "zero_plus_positive_quintiles": [0.2, 0.4, 0.6, 0.8],
    }[mapping]
    return tuple(float(value) for value in positive.quantile(probabilities))


def apply_flow_bins(
    values: pd.Series, cut_points: tuple[float, ...]
) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="raise")
    labels = np.full(len(numeric), None, dtype=object)
    labels[numeric.eq(0).to_numpy()] = "zero"
    positive = numeric.gt(0).to_numpy()
    edges = np.asarray(cut_points, dtype=float)
    indices = np.searchsorted(edges, numeric.to_numpy(float), side="left")
    for index in range(len(edges) + 1):
        labels[positive & (indices == index)] = f"positive_{index + 1}"
    categories = ["zero"] + [
        f"positive_{index + 1}" for index in range(len(edges) + 1)
    ]
    return pd.Categorical(labels, categories=categories, ordered=True)


def _dirichlet_expected_entropy(parameters: np.ndarray) -> float:
    values = np.asarray(parameters, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(values <= 0):
        raise ValueError("Dirichlet parameters must be a positive vector.")
    total = float(values.sum())
    return float(
        (
            digamma(total + 1.0)
            - np.dot(values / total, digamma(values + 1.0))
        )
        / math.log(2.0)
    )


def jeffreys_conditional_pair(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline_state: list[str],
    bicycle_state_field: str,
    alpha: float = 0.5,
) -> tuple[float, float]:
    """Coherent posterior means of H(Y|S) and H(Y|S,B).

    A single symmetric Dirichlet posterior is placed on the observed atoms of
    the full joint distribution (Y,S,B). Dirichlet aggregation then supplies
    every marginal entropy. Consequently, the difference between the returned
    conditional entropies is the posterior mean of conditional mutual
    information and cannot be made incoherent by fitting separate priors to
    baseline and augmented state spaces.
    """
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    required = [target, *baseline_state, bicycle_state_field]
    if frame[required].isna().any().any():
        raise ValueError("Jeffreys estimator inputs must be complete.")
    working = frame[required].copy()
    working[target] = pd.to_numeric(
        working[target], errors="raise"
    ).astype(int)
    if working[target].lt(0).any():
        raise ValueError("Target must be nonnegative.")
    full = (
        working.groupby(
            [*baseline_state, bicycle_state_field, target],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    full["parameter"] = full["count"].astype(float) + float(alpha)
    sb = (
        full.groupby(
            [*baseline_state, bicycle_state_field],
            observed=True,
            dropna=False,
        )["parameter"]
        .sum()
        .to_numpy(float)
    )
    sy = (
        full.groupby(
            [*baseline_state, target],
            observed=True,
            dropna=False,
        )["parameter"]
        .sum()
        .to_numpy(float)
    )
    s = (
        full.groupby(
            baseline_state, observed=True, dropna=False
        )["parameter"]
        .sum()
        .to_numpy(float)
    )
    full_parameters = full["parameter"].to_numpy(float)
    h0 = _dirichlet_expected_entropy(sy) - _dirichlet_expected_entropy(s)
    hb = _dirichlet_expected_entropy(
        full_parameters
    ) - _dirichlet_expected_entropy(sb)
    return float(h0), float(hb)


def _bound_metrics(h0: float, hb: float, config: dict[str, object]) -> dict:
    h0_bound = max(float(h0), 0.0)
    hb_unordered = max(float(hb), 0.0)
    hb_bound = min(hb_unordered, h0_bound)
    numerics = config["numerics"]
    l0 = inverse_entropy_envelope(
        h0_bound,
        tail_tolerance=float(numerics["lattice_tail_tolerance"]),
        root_tolerance=float(numerics["root_tolerance"]),
    )
    lb = inverse_entropy_envelope(
        hb_bound,
        tail_tolerance=float(numerics["lattice_tail_tolerance"]),
        root_tolerance=float(numerics["root_tolerance"]),
    )
    return {
        "h0_for_bound_bits": h0_bound,
        "hb_for_bound_bits": hb_bound,
        "delta_h_projected_bits": h0_bound - hb_bound,
        "l0_exact_mse": l0,
        "lb_exact_mse": lb,
        "delta_l_exact_mse": l0 - lb,
        "projection_applied": hb_unordered > h0_bound,
    }


def _load_panel(data_dir: Path, city: str) -> pd.DataFrame:
    files = sorted(
        (data_dir / "grid_day_panel" / f"city={city}").rglob("*.parquet")
    )
    if not files:
        raise FileNotFoundError(f"No panel files for {city} in {data_dir}")
    columns = [
        "city",
        "grid_id",
        "x_index",
        "y_index",
        "date",
        "year",
        "month",
        "day_of_week",
        "season",
        "is_holiday",
        "split",
        "bike_coverage_training",
        "crime_count_all",
        "bike_outflow",
        "bike_inflow",
        "bike_total_flow",
        "bike_net_flow",
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files],
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["grid_id", "date"]).reset_index(drop=True)


def _add_daily_lags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["grid_id", "date"]).copy()
    grouped = result.groupby("grid_id", sort=False)
    for lag in [1, 2, 3, 7]:
        result[f"crime_count_lag{lag}"] = grouped["crime_count_all"].shift(lag)
        for measure in [
            "bike_total_flow",
            "bike_outflow",
            "bike_inflow",
            "bike_absolute_net_flow",
        ]:
            source = (
                result["bike_net_flow"].abs()
                if measure == "bike_absolute_net_flow"
                else result[measure]
            )
            result[f"{measure}_lag{lag}"] = source.groupby(
                result["grid_id"], sort=False
            ).shift(lag)
    return result


def _weekly_panel(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["week_start"] = (
        working["date"] - pd.to_timedelta(working["date"].dt.dayofweek, unit="D")
    )
    grouped = (
        working.groupby(
            ["city", "grid_id", "x_index", "y_index", "week_start"],
            observed=True,
            as_index=False,
        )
        .agg(
            crime_count_all=("crime_count_all", "sum"),
            bike_outflow=("bike_outflow", "sum"),
            bike_inflow=("bike_inflow", "sum"),
            bike_total_flow=("bike_total_flow", "sum"),
            bike_net_flow=("bike_net_flow", "sum"),
            bike_coverage_training=("bike_coverage_training", "first"),
            is_holiday=("is_holiday", "max"),
        )
        .rename(columns={"week_start": "date"})
    )
    grouped = grouped.loc[
        grouped["date"].between("2020-01-06", "2022-12-26")
    ].copy()
    grouped["year"] = grouped["date"].dt.year
    grouped["month"] = grouped["date"].dt.month
    grouped["day_of_week"] = 0
    grouped["season"] = grouped["month"].map(
        {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn",
        }
    )
    grouped["split"] = np.select(
        [
            grouped["date"].le("2021-12-31"),
            grouped["date"].le("2022-06-30"),
        ],
        ["train", "validation"],
        default="test",
    )
    return grouped


def _specification(spec_id: str) -> dict[str, object]:
    spec = {
        "resolution": 1000,
        "temporal": "day",
        "crime_lag": 1,
        "bicycle_lag": 1,
        "crime_bins": "four_state",
        "bicycle_bins": "zero_plus_positive_tertiles",
        "bicycle_measure": "bike_total_flow",
    }
    if spec_id.startswith("spatial_"):
        spec["resolution"] = int(spec_id.split("_", 1)[1][:-1])
    elif spec_id == "weekly":
        spec["temporal"] = "week"
    elif spec_id == "crime_bins_coarse":
        spec["crime_bins"] = "zero_one_two_plus"
    elif spec_id == "bicycle_bins_median":
        spec["bicycle_bins"] = "zero_plus_positive_median"
    elif spec_id == "bicycle_bins_quintiles":
        spec["bicycle_bins"] = "zero_plus_positive_quintiles"
    elif spec_id.startswith("crime_lag"):
        spec["crime_lag"] = int(spec_id.replace("crime_lag", ""))
    elif spec_id.startswith("bicycle_lag"):
        spec["bicycle_lag"] = int(spec_id.replace("bicycle_lag", ""))
    elif spec_id == "bicycle_outflow":
        spec["bicycle_measure"] = "bike_outflow"
    elif spec_id == "bicycle_inflow":
        spec["bicycle_measure"] = "bike_inflow"
    elif spec_id == "bicycle_absolute_net":
        spec["bicycle_measure"] = "bike_absolute_net_flow"
    elif spec_id != "primary":
        raise ValueError(spec_id)
    return spec


def _prepare_spec(
    frame: pd.DataFrame, spec_id: str
) -> tuple[pd.DataFrame, tuple[float, ...]]:
    settings = _specification(spec_id)
    working = _weekly_panel(frame) if settings["temporal"] == "week" else frame
    working = _add_daily_lags(working)
    crime_field = f"crime_count_lag{settings['crime_lag']}"
    bicycle_field = (
        f"{settings['bicycle_measure']}_lag{settings['bicycle_lag']}"
    )
    eligible = (
        working["bike_coverage_training"].astype(bool)
        & working[crime_field].notna()
        & working[bicycle_field].notna()
    )
    prepared = working.loc[eligible].copy()
    prepared["crime_lag_bin"] = crime_bins(
        prepared[crime_field], str(settings["crime_bins"])
    )
    training = prepared.loc[prepared["split"].eq("train")]
    cut_points = fit_flow_cut_points(
        training[bicycle_field], str(settings["bicycle_bins"])
    )
    prepared["bicycle_lag_bin"] = apply_flow_bins(
        prepared[bicycle_field], cut_points
    )
    if prepared[["crime_lag_bin", "bicycle_lag_bin"]].isna().any().any():
        raise AssertionError(f"Missing mapped states in {spec_id}")
    return prepared, cut_points


def estimate_specification(
    frame: pd.DataFrame,
    *,
    city: str,
    spec_id: str,
    config: dict[str, object],
) -> pd.DataFrame:
    prepared, cut_points = _prepare_spec(frame, spec_id)
    prepared["one_block"] = 0
    baseline = list(config["information_sets"]["baseline_state"])
    codebook = ConditionalCodebook.from_frame(
        prepared,
        target=str(config["target"]),
        baseline_state=baseline,
        bicycle_state_field="bicycle_lag_bin",
        block_field="one_block",
    )
    conventional = codebook.estimate(np.ones(1)).iloc[0]
    values = {
        "plugin": (
            conventional["h0_plugin_bits"],
            conventional["hb_plugin_bits"],
        ),
        "miller_madow": (
            conventional["h0_miller_madow_bits"],
            conventional["hb_miller_madow_bits"],
        ),
        "jeffreys_dirichlet": jeffreys_conditional_pair(
            prepared,
            target=str(config["target"]),
            baseline_state=baseline,
            bicycle_state_field="bicycle_lag_bin",
            alpha=float(config["estimation"]["jeffreys_alpha"]),
        ),
    }
    diagnostics = codebook.sparsity_diagnostics()
    records = []
    for estimator, (h0, hb) in values.items():
        records.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "spec_id": spec_id,
                "estimator": estimator,
                "observations": len(prepared),
                "grids": int(prepared["grid_id"].nunique()),
                "days_or_weeks": int(prepared["date"].nunique()),
                "h0_bits": float(h0),
                "hb_bits": float(hb),
                "delta_h_raw_bits": float(h0 - hb),
                "bicycle_cut_points": json.dumps(cut_points),
                **_bound_metrics(float(h0), float(hb), config),
                **diagnostics,
            }
        )
    return pd.DataFrame(records)


def _build_alternative_panel(
    *,
    resolution: int,
    e01_data: Path,
    e02_config: dict[str, object],
    cities_config: dict[str, object],
    run_dir: Path,
    data_root: Path,
    logger: logging.Logger,
) -> Path:
    data_dir = data_root / f"resolution_{resolution}m"
    ready = data_dir / "_COMPLETE.json"
    if ready.exists():
        logger.info("Loading completed %d m panel checkpoint", resolution)
        return data_dir
    prep_run = run_dir / "state" / f"resolution_{resolution}m"
    for path in [prep_run / "tables", prep_run / "state", data_dir / "checkpoints"]:
        path.mkdir(parents=True, exist_ok=True)
    local = copy.deepcopy(e02_config)
    local["spatial"]["resolution_m"] = int(resolution)
    assignments, crime, active = _aggregate_crime(
        e01_data,
        list(CITY_ORDER),
        cities_config,
        local,
        prep_run,
        data_dir,
        logger,
    )
    for city in CITY_ORDER:
        flow, trips, _, _ = _aggregate_bicycle_city(
            city,
            e01_data,
            active[city],
            cities_config,
            local,
            prep_run,
            data_dir,
            logger,
        )
        reference = _grid_reference(
            city,
            active[city],
            assignments,
            flow,
            cities_config,
            local,
        )
        reference_path = (
            data_dir
            / "grid_reference"
            / f"city={city}"
            / "grid_reference.parquet"
        )
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference.to_parquet(reference_path, index=False, compression="zstd")
        _build_city_panel(
            city, reference, crime, flow, trips, local, data_dir, logger
        )
    ready.write_text(
        json.dumps(
            {
                "resolution_m": resolution,
                "completed_at": datetime.now().isoformat(),
                "source_e01": str(e01_data),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return data_dir


def _diagnose_250m(
    e01_data: Path,
    cities_config: dict[str, object],
    run_dir: Path,
) -> pd.DataFrame:
    from pyproj import Transformer

    records = []
    for city in CITY_ORDER:
        transformer = Transformer.from_crs(
            "EPSG:4326",
            cities_config["cities"][city]["projected_crs"],
            always_xy=True,
        )
        files = sorted(
            (e01_data / "crime_events" / f"city={city}").rglob("*.parquet")
        )
        counts: dict[tuple[int, int], int] = {}
        total = 0
        for path in files:
            frame = pd.read_parquet(
                path, columns=["event_date", "longitude", "latitude"]
            )
            frame["event_date"] = pd.to_datetime(frame["event_date"])
            frame = frame.loc[frame["event_date"].le("2021-12-31")]
            x, y, valid = assign_projected_grid(
                frame["longitude"], frame["latitude"], transformer, 250
            )
            if not valid.all():
                raise AssertionError("Invalid 250 m crime projection.")
            for key, count in (
                pd.DataFrame({"x": x, "y": y})
                .value_counts()
                .items()
            ):
                counts[(int(key[0]), int(key[1]))] = (
                    counts.get((int(key[0]), int(key[1])), 0) + int(count)
                )
            total += len(frame)
        mean_events = total / len(counts)
        records.append(
            {
                "city": city,
                "training_events": total,
                "active_250m_grids": len(counts),
                "mean_training_events_per_active_grid": mean_events,
                "threshold": 20.0,
                "promoted": False,
                "reason": "prespecified_diagnostic_only_and_not_promoted",
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(run_dir / "tables" / "diagnostic_250m.csv", index=False)
    return result


def _block_sensitivity(
    frame: pd.DataFrame,
    *,
    city: str,
    config: dict[str, object],
) -> pd.DataFrame:
    prepared, _ = _prepare_spec(frame, "primary")
    baseline = list(config["information_sets"]["baseline_state"])
    records = []
    repetitions = int(config["estimation"]["block_bootstrap_repetitions"])
    for block_days in config["frozen_matrix"]["bootstrap_block_days"]:
        prepared["block_id"] = (
            (prepared["date"] - prepared["date"].min()).dt.days
            // int(block_days)
        )
        codebook = ConditionalCodebook.from_frame(
            prepared,
            target=str(config["target"]),
            baseline_state=baseline,
            bicycle_state_field="bicycle_lag_bin",
            block_field="block_id",
        )
        point = codebook.estimate(np.ones(codebook.block_count)).iloc[0]
        rng = np.random.default_rng(
            _seed(
                int(config["random_seed"]),
                f"E09|block|{city}|{block_days}",
            )
        )
        weights = rng.multinomial(
            codebook.block_count,
            np.full(codebook.block_count, 1 / codebook.block_count),
            size=repetitions,
        )
        boot = codebook.estimate(weights)
        values = boot["delta_h_miller_madow_raw_bits"].to_numpy(float)
        se = float(values.std(ddof=1))
        estimate = float(point["delta_h_miller_madow_raw_bits"])
        records.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "block_days": int(block_days),
                "blocks": codebook.block_count,
                "repetitions": repetitions,
                "delta_h_bits": estimate,
                "bootstrap_se": se,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
            }
        )
    return pd.DataFrame(records)


def _rolling_predictions(
    frame: pd.DataFrame,
    *,
    city: str,
    selected: pd.DataFrame,
    e06_config: dict[str, object],
    config: dict[str, object],
    logger: logging.Logger,
) -> pd.DataFrame:
    prepared, _ = _prepare_spec(frame, "primary")
    origins = pd.date_range("2022-01-01", "2022-12-01", freq="MS")
    records = []
    for origin in origins:
        end = origin + pd.offsets.MonthEnd(1)
        training = prepared.loc[prepared["date"].lt(origin)]
        testing = prepared.loc[prepared["date"].between(origin, end)]
        if training.empty or testing.empty:
            raise AssertionError(f"Empty rolling split: {city} {origin}")
        y_train = training[str(config["target"])].to_numpy(float)
        y_test = testing[str(config["target"])].to_numpy(float)
        for family in MODEL_ORDER:
            predictions: dict[str, np.ndarray] = {}
            for information_set in ["baseline", "bicycle_aware"]:
                row = selected.loc[
                    selected["city"].eq(city)
                    & selected["model_family"].eq(family)
                    & selected["information_set"].eq(information_set)
                ].iloc[0]
                parameters = json.loads(row["selected_parameters_json"])
                features = json.loads(row["features_json"])
                model = _fit_model(
                    family,
                    training,
                    y_train,
                    features=features,
                    parameters=parameters,
                    config=e06_config,
                    seed=_seed(
                        int(config["random_seed"]),
                        f"E09|roll|{city}|{origin.date()}|{family}|{information_set}",
                    ),
                )
                continuous = _predict_model(
                    model, family, testing, features
                )
                predictions[information_set] = nonnegative_half_up(continuous)
            mse0 = float(
                np.mean(np.square(y_test - predictions["baseline"]))
            )
            mseb = float(
                np.mean(np.square(y_test - predictions["bicycle_aware"]))
            )
            records.append(
                {
                    "city": city,
                    "city_label": CITY_LABELS[city],
                    "origin": str(origin.date()),
                    "forecast_end": str(end.date()),
                    "model_family": family,
                    "model_label": MODEL_LABELS[family],
                    "training_end": str((origin - pd.Timedelta(days=1)).date()),
                    "training_observations": len(training),
                    "test_observations": len(testing),
                    "mse_baseline": mse0,
                    "mse_bicycle": mseb,
                    "delta_mse": mse0 - mseb,
                    "relative_delta_mse": (
                        (mse0 - mseb) / mse0 if mse0 > 0 else np.nan
                    ),
                }
            )
            logger.info(
                "Rolling %s %s %s: delta MSE %.5f",
                city,
                origin.strftime("%Y-%m"),
                family,
                mse0 - mseb,
            )
    return pd.DataFrame(records)


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
    theory: pd.DataFrame,
    blocks: pd.DataFrame,
    rolling: pd.DataFrame,
    run_dir: Path,
    dpi: int,
) -> None:
    _publication_style()
    mm = theory.loc[theory["estimator"].eq("miller_madow")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.8), sharey=True)
    for axis, city in zip(axes, CITY_ORDER):
        subset = mm.loc[mm["city"].eq(city)].set_index("spec_id").reindex(SPEC_ORDER)
        axis.axvline(0, color="#777777", lw=0.8)
        axis.scatter(
            subset["delta_l_exact_mse"],
            np.arange(len(subset)),
            color=CITY_COLORS[city],
            s=24,
        )
        axis.set_title(CITY_LABELS[city])
        axis.set_xlabel("Exact DELB reduction (MSE)")
        axis.set_yticks(np.arange(len(subset)))
        axis.set_yticklabels([value.replace("_", " ") for value in SPEC_ORDER])
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("E09 alternative-specification robustness")
    _save_figure(fig, run_dir / "figures" / "e09_specification_matrix.png", dpi)

    primary = theory.loc[theory["spec_id"].eq("primary")].copy()
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(3)
    width = 0.24
    for index, estimator in enumerate(
        ["plugin", "miller_madow", "jeffreys_dirichlet"]
    ):
        subset = primary.loc[primary["estimator"].eq(estimator)].set_index("city")
        axis.bar(
            positions + (index - 1) * width,
            [subset.loc[city, "delta_h_raw_bits"] for city in CITY_ORDER],
            width,
            label=estimator.replace("_", " ").title(),
        )
    axis.axhline(0, color="#444444", lw=0.8)
    axis.set_xticks(positions, [CITY_LABELS[city] for city in CITY_ORDER])
    axis.set_ylabel("Estimated conditional information (bits)")
    axis.set_title("Entropy-estimator sensitivity")
    axis.legend(frameon=False)
    _save_figure(fig, run_dir / "figures" / "e09_estimator_sensitivity.png", dpi)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    for axis, city in zip(axes, CITY_ORDER):
        subset = blocks.loc[blocks["city"].eq(city)]
        axis.errorbar(
            subset["block_days"],
            subset["delta_h_bits"],
            yerr=[
                subset["delta_h_bits"] - subset["ci_low"],
                subset["ci_high"] - subset["delta_h_bits"],
            ],
            fmt="o-",
            color=CITY_COLORS[city],
            capsize=3,
        )
        axis.axhline(0, color="#777777", lw=0.8)
        axis.set_title(CITY_LABELS[city])
        axis.set_xlabel("Calendar block length (days)")
    axes[0].set_ylabel("Conditional information (bits)")
    fig.suptitle("Block-length sensitivity of uncertainty intervals")
    _save_figure(fig, run_dir / "figures" / "e09_block_sensitivity.png", dpi)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True)
    for axis, city in zip(axes, CITY_ORDER):
        subset = rolling.loc[rolling["city"].eq(city)].copy()
        subset["origin"] = pd.to_datetime(subset["origin"])
        for family in MODEL_ORDER:
            model = subset.loc[subset["model_family"].eq(family)]
            axis.plot(
                model["origin"],
                model["delta_mse"],
                marker="o",
                ms=2.8,
                lw=1,
                label=MODEL_LABELS[family],
            )
        axis.axhline(0, color="#444444", lw=0.8)
        axis.set_title(CITY_LABELS[city])
        axis.set_xlabel("2022 forecast origin")
        axis.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel(r"$\Delta$MSE (baseline minus bicycle)")
    axes[-1].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Monthly expanding-window prediction robustness")
    _save_figure(fig, run_dir / "figures" / "e09_rolling_origins.png", dpi)


def _acceptance(
    theory: pd.DataFrame,
    blocks: pd.DataFrame,
    rolling: pd.DataFrame,
    diagnostic: pd.DataFrame,
    run_dir: Path,
    config: dict[str, object],
) -> pd.DataFrame:
    required = set(config["acceptance"]["required_theory_specifications"])
    checks = [
        ("three_cities", set(theory["city"]) == set(CITY_ORDER)),
        ("all_theory_specs", set(theory["spec_id"]) == required),
        (
            "three_estimators",
            set(theory["estimator"])
            == {"plugin", "miller_madow", "jeffreys_dirichlet"},
        ),
        ("finite_theory", np.isfinite(theory.select_dtypes("number")).all().all()),
        (
            "bound_order",
            bool(
                (
                    theory["lb_exact_mse"]
                    <= theory["l0_exact_mse"]
                    + float(config["acceptance"]["bound_order_tolerance"])
                ).all()
            ),
        ),
        (
            "rolling_rows",
            len(rolling)
            == int(config["acceptance"]["expected_rolling_model_contrasts"]),
        ),
        ("rolling_origins", rolling["origin"].nunique() == 12),
        ("rolling_models", set(rolling["model_family"]) == set(MODEL_ORDER)),
        ("block_lengths", set(blocks["block_days"]) == {7, 14, 28}),
        ("bootstrap_repetitions", blocks["repetitions"].eq(1000).all()),
        ("diagnostic_250_not_promoted", (~diagnostic["promoted"]).all()),
        (
            "python_figures",
            all(
                (run_dir / "figures" / name).exists()
                for name in [
                    "e09_specification_matrix.png",
                    "e09_estimator_sensitivity.png",
                    "e09_block_sensitivity.png",
                    "e09_rolling_origins.png",
                ]
            ),
        ),
    ]
    result = pd.DataFrame(
        [
            {"check": name, "passed": bool(passed), "detail": ""}
            for name, passed in checks
        ]
    )
    result.to_csv(run_dir / "tables" / "acceptance_checklist.csv", index=False)
    if not result["passed"].all():
        raise AssertionError(
            "E09 acceptance failed: "
            + ", ".join(result.loc[~result["passed"], "check"])
        )
    return result


def _write_interpretation(
    theory: pd.DataFrame,
    blocks: pd.DataFrame,
    rolling: pd.DataFrame,
    run_dir: Path,
) -> None:
    mm = theory.loc[theory["estimator"].eq("miller_madow")]
    lines = [
        "# E09 Robustness Interpretation",
        "",
        "E09 was frozen before inspecting its outputs. It evaluates alternative "
        "spatial and temporal scales, state mappings, estimators, lags, mobility "
        "measures, bootstrap block lengths, and monthly expanding-window forecasts.",
        "",
        "## Theory-side robustness",
        "",
    ]
    for city in CITY_ORDER:
        city_rows = mm.loc[mm["city"].eq(city)]
        positive = int(city_rows["delta_l_exact_mse"].gt(0).sum())
        primary = city_rows.loc[city_rows["spec_id"].eq("primary")].iloc[0]
        lines.append(
            f"- **{CITY_LABELS[city]}**: {positive}/{len(city_rows)} "
            f"Miller--Madow specifications produced a positive exact DELB "
            f"reduction; the primary estimate was "
            f"{primary['delta_l_exact_mse']:.6f} MSE units."
        )
    lines.extend(["", "## Rolling prediction robustness", ""])
    for city in CITY_ORDER:
        city_rows = rolling.loc[rolling["city"].eq(city)]
        positive = int(city_rows["delta_mse"].gt(0).sum())
        lines.append(
            f"- **{CITY_LABELS[city]}**: bicycle-aware forecasts improved "
            f"integer MSE in {positive}/{len(city_rows)} monthly model--origin "
            "comparisons."
        )
    lines.extend(
        [
            "",
            "The alternative estimators and discretizations are finite-sample "
            "sensitivity analyses, not interchangeable estimands. Positive "
            "conditional information describes predictive association, not a "
            "causal effect. Rolling-origin changes remain model-specific and "
            "do not validate the mathematical lower-bound theorem.",
            "",
            "E10 has not been started. Distant-lag, temporal-permutation, "
            "spatial-permutation, and surrogate null experiments remain reserved "
            "for E10.",
        ]
    )
    (run_dir / "interpretation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_workbook(
    theory: pd.DataFrame,
    blocks: pd.DataFrame,
    rolling: pd.DataFrame,
    diagnostic: pd.DataFrame,
    acceptance: pd.DataFrame,
    run_dir: Path,
) -> None:
    path = run_dir / "tables" / "E09_robustness_QC.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        acceptance.to_excel(writer, sheet_name="Acceptance", index=False)
        theory.to_excel(writer, sheet_name="Theory Matrix", index=False)
        blocks.to_excel(writer, sheet_name="Block Sensitivity", index=False)
        rolling.to_excel(writer, sheet_name="Rolling Origins", index=False)
        diagnostic.to_excel(writer, sheet_name="250m Diagnostic", index=False)


def run(config_path: Path, resume_run: Path | None = None) -> Path:
    started_clock = time.monotonic()
    started_at = datetime.now().astimezone()
    previous_elapsed = 0.0
    config = _load_yaml(config_path)
    project = config_path.resolve().parents[2]
    empirical_root = project / "empirical"
    paths = _load_yaml(empirical_root / "config" / "paths.yaml")
    cities_config = _load_yaml(empirical_root / "config" / "cities.yaml")
    e02_config = _load_yaml(empirical_root / "config" / "e02.yaml")
    e06_config = _load_yaml(empirical_root / "config" / "e06.yaml")
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
            + "_E09_robustness"
        )
        run_dir = empirical_root / "runs" / run_id
    data_root = empirical_root / "processed_data" / "e09" / run_id
    for path in [
        run_dir / "logs",
        run_dir / "tables",
        run_dir / "figures",
        run_dir / "state",
        data_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    e01_data = _read_pointer(
        empirical_root, str(config["source_e01_data_pointer"])
    )
    e02_data = _read_pointer(
        empirical_root, str(config["source_e02_data_pointer"])
    )
    e04_run = _read_pointer(
        empirical_root, str(config["source_e04_run_pointer"])
    )
    e06_run = _read_pointer(
        empirical_root, str(config["source_e06_run_pointer"])
    )
    if not resume_run:
        (run_dir / "command.txt").write_text(
            " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
        )
        _write_environment(run_dir / "environment.txt")
        (run_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "e09": config,
                    "source_e01_data": str(e01_data),
                    "source_e02_data": str(e02_data),
                    "source_e04_run": str(e04_run),
                    "source_e06_run": str(e06_run),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "frozen_specification.md").write_text(
            "# Frozen E09 Specification\n\n"
            "This matrix was written before E09 outputs were inspected. E09 "
            "contains alternative scales, aggregation, discretization, entropy "
            "estimators, lags, mobility measures, bootstrap block lengths, and "
            "monthly rolling origins. Distant-lag and permutation null tests "
            "are reserved for E10.\n",
            encoding="utf-8",
        )
    logger.info("Starting E09 robustness run %s", run_id)

    panel_dirs = {1000: e02_data}
    for resolution in [500, 2000]:
        panel_dirs[resolution] = _build_alternative_panel(
            resolution=resolution,
            e01_data=e01_data,
            e02_config=e02_config,
            cities_config=cities_config,
            run_dir=run_dir,
            data_root=data_root,
            logger=logger,
        )
    diagnostic = _diagnose_250m(e01_data, cities_config, run_dir)

    panels: dict[tuple[int, str], pd.DataFrame] = {}
    theory_parts = []
    block_parts = []
    for city in CITY_ORDER:
        for resolution, directory in panel_dirs.items():
            panels[(resolution, city)] = _load_panel(directory, city)
        for spec_id in SPEC_ORDER:
            resolution = int(_specification(spec_id)["resolution"])
            theory_parts.append(
                estimate_specification(
                    panels[(resolution, city)],
                    city=city,
                    spec_id=spec_id,
                    config=config,
                )
            )
            logger.info("Theory specification complete: %s %s", city, spec_id)
        block_parts.append(
            _block_sensitivity(panels[(1000, city)], city=city, config=config)
        )
    theory = pd.concat(theory_parts, ignore_index=True)
    blocks = pd.concat(block_parts, ignore_index=True)
    theory.to_csv(run_dir / "tables" / "theory_robustness_matrix.csv", index=False)
    theory.to_parquet(
        run_dir / "tables" / "theory_robustness_matrix.parquet",
        index=False,
        compression="zstd",
    )
    blocks.to_csv(run_dir / "tables" / "block_length_sensitivity.csv", index=False)

    selected = pd.read_csv(
        e06_run / "tables" / "selected_model_metadata.csv"
    )
    rolling_parts = []
    rolling_checkpoint = run_dir / "state" / "rolling_origins.parquet"
    if rolling_checkpoint.exists():
        rolling = pd.read_parquet(rolling_checkpoint)
        logger.info("Loaded rolling-origin checkpoint with %d rows", len(rolling))
    else:
        for city in CITY_ORDER:
            part = _rolling_predictions(
                panels[(1000, city)],
                city=city,
                selected=selected,
                e06_config=e06_config,
                config=config,
                logger=logger,
            )
            rolling_parts.append(part)
            pd.concat(rolling_parts, ignore_index=True).to_parquet(
                rolling_checkpoint, index=False, compression="zstd"
            )
        rolling = pd.concat(rolling_parts, ignore_index=True)
    rolling.to_csv(run_dir / "tables" / "rolling_origin_results.csv", index=False)
    summary = (
        rolling.groupby(["city", "model_family"], as_index=False)
        .agg(
            origins=("origin", "nunique"),
            mean_delta_mse=("delta_mse", "mean"),
            median_delta_mse=("delta_mse", "median"),
            positive_origin_share=("delta_mse", lambda value: value.gt(0).mean()),
        )
    )
    summary.to_csv(
        run_dir / "tables" / "rolling_origin_summary.csv", index=False
    )

    dpi = int(config["reporting"]["figure_dpi"])
    _make_figures(theory, blocks, rolling, run_dir, dpi)
    acceptance = _acceptance(
        theory, blocks, rolling, diagnostic, run_dir, config
    )
    _write_interpretation(theory, blocks, rolling, run_dir)
    _write_workbook(
        theory, blocks, rolling, diagnostic, acceptance, run_dir
    )
    completed_at = datetime.now().astimezone()
    elapsed = time.monotonic() - started_clock
    status = {
        "experiment": "E09",
        "status": "complete",
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": previous_elapsed + elapsed,
        "latest_pass_seconds": elapsed,
        "source_e01_data": str(e01_data),
        "source_e02_data": str(e02_data),
        "source_e04_run": str(e04_run),
        "source_e06_run": str(e06_run),
        "theory_rows": len(theory),
        "rolling_rows": len(rolling),
        "checks_passed": int(acceptance["passed"].sum()),
        "checks_total": len(acceptance),
        "e10_started": False,
    }
    (run_dir / "run_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (empirical_root / "runs" / "latest_e09_run.txt").write_text(
        str(run_dir) + "\n", encoding="utf-8"
    )
    if not resume_run:
        registry = empirical_root / "docs" / "experiment_registry.csv"
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{run_id},E09,complete,{started_at.isoformat()},"
                f"{completed_at.isoformat()},{elapsed:.3f},{run_dir},"
                '"Prespecified alternative scales; aggregation; discretization; '
                'estimators; lags; block lengths; and rolling origins."\n'
            )
    logger.info("E09 complete in %.1f seconds; E10 was not started", elapsed)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E09 robustness analysis.")
    parser.add_argument(
        "--config", type=Path, default=Path("empirical/config/e09.yaml")
    )
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    run(args.config, args.resume_run)


if __name__ == "__main__":
    main()
