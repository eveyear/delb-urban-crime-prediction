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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
from sklearn.metrics import mean_poisson_deviance
import yaml

from entropy_crime_bike.conditional_information import apply_bicycle_bins
from entropy_crime_bike.predictive_models import (
    StateMeanRegressor,
    build_histogram_poisson_pipeline,
    build_poisson_pipeline,
    crime_lag_state,
    fit_negative_binomial,
    model_input,
    nonnegative_half_up,
)
from entropy_crime_bike.spatial_information import benjamini_hochberg


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
MODEL_ORDER = [
    "state_mean",
    "poisson_glm",
    "negative_binomial_glm",
    "histogram_gradient_boosting_poisson",
]
MODEL_LABELS = {
    "state_mean": "State mean",
    "poisson_glm": "Poisson GLM",
    "negative_binomial_glm": "Negative-binomial GLM",
    "histogram_gradient_boosting_poisson": "Poisson gradient boosting",
    "seasonal_naive_lag7": "Seasonal naive (lag 7)",
}
INFO_ORDER = ["baseline", "bicycle_aware"]
INFO_LABELS = {
    "baseline": "Baseline",
    "bicycle_aware": "Bicycle-aware",
    "external_nonmatched": "External benchmark",
}


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return loaded


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e06_predictive_models")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e06_predictive_models.log", encoding="utf-8"
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


def _publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
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


def _spec_seed(base_seed: int, spec_id: str) -> int:
    return int((base_seed + zlib.crc32(spec_id.encode("utf-8"))) % 2**32)


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
        raise AssertionError("E06 requires one accepted E04 threshold row per city.")
    if not thresholds["source_split"].eq("train").all():
        raise AssertionError("E06 requires E04 training-only bicycle thresholds.")
    return e04_run, thresholds


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
        "month",
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


def _prepare_city(
    frame: pd.DataFrame,
    *,
    cut_points: tuple[float, float],
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    target = str(config["target"])
    crime_lag = str(config["information_sets"]["crime_lag_field"])
    bicycle_lag = str(config["information_sets"]["bicycle_lag_field"])
    working = frame.sort_values(["grid_id", "date"]).copy()
    working["crime_count_lag7"] = working.groupby(
        "grid_id", observed=True
    )[target].shift(7)
    missing_strict_lag = working[crime_lag].isna() | working[bicycle_lag].isna()
    outside_domain = ~working["bike_coverage_training"].astype(bool)
    prepared = working.loc[~missing_strict_lag & ~outside_domain].copy()
    prepared["crime_lag_bin"] = crime_lag_state(prepared[crime_lag])
    prepared["bicycle_lag_bin"] = apply_bicycle_bins(
        prepared[bicycle_lag], cut_points
    )
    if prepared[["crime_lag_bin", "bicycle_lag_bin"]].isna().any().any():
        raise AssertionError("E06 state mapping produced missing values.")
    target_values = pd.to_numeric(prepared[target], errors="raise")
    if (
        target_values.lt(0).any()
        or not np.equal(target_values, np.floor(target_values)).all()
    ):
        raise AssertionError("E06 target is not a nonnegative integer count.")
    split_counts = prepared.groupby("split", observed=True).size().to_dict()
    diagnostic = {
        "city": str(frame["city"].iloc[0]),
        "city_label": CITY_LABELS[str(frame["city"].iloc[0])],
        "e02_panel_rows": len(frame),
        "rows_excluded_missing_strict_lag": int(missing_strict_lag.sum()),
        "rows_excluded_outside_training_bicycle_coverage": int(
            (~missing_strict_lag & outside_domain).sum()
        ),
        "analysis_rows": len(prepared),
        "training_rows": int(split_counts.get("train", 0)),
        "validation_rows": int(split_counts.get("validation", 0)),
        "test_rows": int(split_counts.get("test", 0)),
        "analysis_grids": int(prepared["grid_id"].nunique()),
        "test_start": str(
            prepared.loc[prepared["split"].eq("test"), "date"].min().date()
        ),
        "test_end": str(
            prepared.loc[prepared["split"].eq("test"), "date"].max().date()
        ),
        "test_days": int(
            prepared.loc[prepared["split"].eq("test"), "date"].nunique()
        ),
        "target_mean": float(target_values.mean()),
        "target_zero_share": float(target_values.eq(0).mean()),
        "target_maximum": int(target_values.max()),
    }
    return prepared, diagnostic


def _prediction_metrics(
    observed: np.ndarray,
    continuous: np.ndarray,
    integer: np.ndarray,
    prediction_floor: float,
) -> dict[str, float | int]:
    y = np.asarray(observed, dtype=float)
    mu = np.asarray(continuous, dtype=float)
    yhat = np.asarray(integer, dtype=float)
    integer_error = y - yhat
    continuous_error = y - mu
    observed_total = float(y.sum())
    predicted_total = float(mu.sum())
    mse_integer = float(np.mean(np.square(integer_error)))
    return {
        "observations": len(y),
        "observed_total": observed_total,
        "observed_mean": float(y.mean()),
        "predicted_continuous_total": predicted_total,
        "predicted_continuous_mean": float(mu.mean()),
        "predicted_integer_total": float(yhat.sum()),
        "predicted_integer_mean": float(yhat.mean()),
        "mse_integer": mse_integer,
        "rmse_integer": math.sqrt(mse_integer),
        "mae_integer": float(np.mean(np.abs(integer_error))),
        "mse_continuous": float(np.mean(np.square(continuous_error))),
        "mean_poisson_deviance_continuous": float(
            mean_poisson_deviance(y, np.maximum(mu, prediction_floor))
        ),
        "exact_count_accuracy": float(np.mean(y == yhat)),
        "predicted_to_observed_total_ratio": (
            predicted_total / observed_total if observed_total > 0 else np.nan
        ),
    }


def _fit_model(
    family: str,
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    features: list[str],
    parameters: dict[str, object],
    config: dict[str, object],
    seed: int,
):
    if family == "state_mean":
        model = StateMeanRegressor(
            features=list(features),
            smoothing=float(parameters["smoothing"]),
        )
        return model.fit(frame, target)
    if family == "poisson_glm":
        settings = config["models"]["poisson_glm"]
        model = build_poisson_pipeline(
            features,
            alpha=float(parameters["alpha"]),
            max_iter=int(settings["max_iter"]),
            tolerance=float(settings["tolerance"]),
        )
        return model.fit(model_input(frame, features), target)
    if family == "negative_binomial_glm":
        settings = config["models"]["negative_binomial_glm"]
        return fit_negative_binomial(
            frame,
            target,
            features=features,
            dispersion=float(parameters["dispersion"]),
            l2_penalty=float(settings["l2_penalty"]),
            max_iter=int(settings["max_iter"]),
            tolerance=float(settings["tolerance"]),
        )
    if family == "histogram_gradient_boosting_poisson":
        settings = config["models"]["histogram_gradient_boosting_poisson"]
        model = build_histogram_poisson_pipeline(
            features,
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2_regularization"]),
            max_iter=int(settings["max_iter"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            random_seed=seed,
        )
        return model.fit(model_input(frame, features), target)
    raise ValueError(f"Unknown E06 matched model family: {family}")


def _predict_model(model, family: str, frame: pd.DataFrame, features: list[str]):
    if family in {
        "poisson_glm",
        "histogram_gradient_boosting_poisson",
    }:
        predicted = model.predict(model_input(frame, features))
    else:
        predicted = model.predict(frame)
    return np.maximum(np.asarray(predicted, dtype=float), 0.0)


def _candidate_parameters(
    family: str, config: dict[str, object]
) -> list[dict[str, object]]:
    settings = config["models"][family]
    if family == "state_mean":
        return [
            {"smoothing": float(value)}
            for value in settings["smoothing_candidates"]
        ]
    if family == "poisson_glm":
        return [
            {"alpha": float(value)} for value in settings["alpha_candidates"]
        ]
    if family == "negative_binomial_glm":
        return [
            {"dispersion": float(value)}
            for value in settings["dispersion_candidates"]
        ]
    if family == "histogram_gradient_boosting_poisson":
        return [dict(value) for value in settings["candidates"]]
    raise ValueError(f"Unknown E06 family: {family}")


def _tune_and_refit(
    prepared: pd.DataFrame,
    *,
    city: str,
    family: str,
    information_set: str,
    features: list[str],
    config: dict[str, object],
    logger: logging.Logger,
):
    target = str(config["target"])
    split_names = config["splits"]
    training = prepared.loc[
        prepared["split"].eq(str(split_names["training"]))
    ].copy()
    validation = prepared.loc[
        prepared["split"].eq(str(split_names["validation"]))
    ].copy()
    refit = prepared.loc[
        prepared["split"].isin(
            [str(split_names["training"]), str(split_names["validation"])]
        )
    ].copy()
    y_train = training[target].to_numpy(float)
    y_validation = validation[target].to_numpy(float)
    prediction_floor = float(config["prediction"]["continuous_prediction_floor"])
    records: list[dict[str, object]] = []
    models: list[object] = []
    parameters_list = _candidate_parameters(family, config)
    for candidate_index, parameters in enumerate(parameters_list):
        seed = _spec_seed(
            int(config["random_seed"]),
            f"{city}|{family}|{information_set}|{candidate_index}",
        )
        started = time.monotonic()
        model = _fit_model(
            family,
            training,
            y_train,
            features=features,
            parameters=parameters,
            config=config,
            seed=seed,
        )
        predicted = _predict_model(model, family, validation, features)
        integer = nonnegative_half_up(predicted)
        metrics = _prediction_metrics(
            y_validation, predicted, integer, prediction_floor
        )
        records.append(
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "model_family": family,
                "information_set": information_set,
                "candidate_index": candidate_index,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "training_rows": len(training),
                "validation_rows": len(validation),
                "validation_start": str(validation["date"].min().date()),
                "validation_end": str(validation["date"].max().date()),
                "validation_mse_integer": metrics["mse_integer"],
                "validation_mse_continuous": metrics["mse_continuous"],
                "validation_mae_integer": metrics["mae_integer"],
                "fit_seconds": time.monotonic() - started,
                "optimizer_converged": bool(
                    getattr(model, "converged", True)
                ),
                "optimizer_iterations": int(
                    getattr(model, "iterations", 0)
                ),
                "selected": False,
            }
        )
        models.append(model)
    tuning = pd.DataFrame(records)
    best_index = int(
        tuning.sort_values(
            [
                "validation_mse_integer",
                "validation_mse_continuous",
                "candidate_index",
            ],
            kind="mergesort",
        ).iloc[0]["candidate_index"]
    )
    tuning.loc[tuning["candidate_index"].eq(best_index), "selected"] = True
    selected_parameters = parameters_list[best_index]
    refit_seed = _spec_seed(
        int(config["random_seed"]),
        f"{city}|{family}|{information_set}|refit",
    )
    final_model = _fit_model(
        family,
        refit,
        refit[target].to_numpy(float),
        features=features,
        parameters=selected_parameters,
        config=config,
        seed=refit_seed,
    )
    logger.info(
        "%s %s %s selected %s (validation integer MSE %.6f)",
        city,
        family,
        information_set,
        json.dumps(selected_parameters, sort_keys=True),
        float(
            tuning.loc[
                tuning["candidate_index"].eq(best_index),
                "validation_mse_integer",
            ].iloc[0]
        ),
    )
    metadata = {
        "city": city,
        "model_family": family,
        "information_set": information_set,
        "features": list(features),
        "selected_parameters": selected_parameters,
        "selection_metric": "validation_mse_integer",
        "training_split": str(split_names["training"]),
        "validation_split": str(split_names["validation"]),
        "test_used_for_selection": False,
        "refit_splits": [
            str(split_names["training"]),
            str(split_names["validation"]),
        ],
        "refit_rows": len(refit),
        "optimizer_converged": bool(
            getattr(final_model, "converged", True)
        ),
        "optimizer_iterations": int(
            getattr(final_model, "iterations", 0)
        ),
    }
    return final_model, tuning, metadata


def _checkpoint_paths(
    run_dir: Path, city: str, family: str, information_set: str
) -> dict[str, Path]:
    stem = f"{city}_{family}_{information_set}"
    return {
        "predictions": run_dir / "state" / f"{stem}_predictions.parquet",
        "metric": run_dir / "state" / f"{stem}_metric.csv",
        "tuning": run_dir / "state" / f"{stem}_tuning.csv",
        "metadata": run_dir / "state" / f"{stem}_metadata.json",
        "model": run_dir / "models" / f"{stem}.joblib",
    }


def _prediction_frame(
    test: pd.DataFrame,
    *,
    family: str,
    information_set: str,
    continuous: np.ndarray,
    target: str,
) -> pd.DataFrame:
    integer = nonnegative_half_up(continuous)
    result = test[
        [
            "city",
            "grid_id",
            "x_index",
            "y_index",
            "centroid_longitude",
            "centroid_latitude",
            "date",
            target,
            "crime_lag_bin",
            "bicycle_lag_bin",
        ]
    ].copy()
    result.rename(columns={target: "observed_count"}, inplace=True)
    result["model_family"] = family
    result["information_set"] = information_set
    result["prediction_continuous"] = continuous
    result["prediction_integer"] = integer
    result["squared_error_continuous"] = np.square(
        result["observed_count"] - result["prediction_continuous"]
    )
    result["squared_error_integer"] = np.square(
        result["observed_count"] - result["prediction_integer"]
    )
    result["absolute_error_integer"] = np.abs(
        result["observed_count"] - result["prediction_integer"]
    )
    return result


def _run_matched_specification(
    prepared: pd.DataFrame,
    *,
    city: str,
    family: str,
    information_set: str,
    features: list[str],
    config: dict[str, object],
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    paths = _checkpoint_paths(run_dir, city, family, information_set)
    if all(path.exists() for path in paths.values()):
        logger.info(
            "Reusing completed checkpoint: %s %s %s",
            city,
            family,
            information_set,
        )
        return (
            pd.read_parquet(paths["predictions"]),
            pd.read_csv(paths["tuning"]),
            json.loads(paths["metadata"].read_text(encoding="utf-8")),
            pd.read_csv(paths["metric"]),
        )
    model, tuning, metadata = _tune_and_refit(
        prepared,
        city=city,
        family=family,
        information_set=information_set,
        features=features,
        config=config,
        logger=logger,
    )
    test = prepared.loc[
        prepared["split"].eq(str(config["splits"]["test"]))
    ].copy()
    continuous = _predict_model(model, family, test, features)
    prediction = _prediction_frame(
        test,
        family=family,
        information_set=information_set,
        continuous=continuous,
        target=str(config["target"]),
    )
    metrics = _prediction_metrics(
        prediction["observed_count"].to_numpy(float),
        prediction["prediction_continuous"].to_numpy(float),
        prediction["prediction_integer"].to_numpy(float),
        float(config["prediction"]["continuous_prediction_floor"]),
    )
    metric = pd.DataFrame(
        [
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "model_family": family,
                "model_label": MODEL_LABELS[family],
                "information_set": information_set,
                "information_set_label": INFO_LABELS[information_set],
                "matched_to_delb": True,
                "test_start": str(test["date"].min().date()),
                "test_end": str(test["date"].max().date()),
                "test_days": int(test["date"].nunique()),
                "test_grids": int(test["grid_id"].nunique()),
                **metrics,
            }
        ]
    )
    temporary_model = paths["model"].with_suffix(".joblib.tmp")
    joblib.dump(model, temporary_model, compress=3)
    temporary_model.replace(paths["model"])
    prediction.to_parquet(
        paths["predictions"], index=False, compression="zstd"
    )
    tuning.to_csv(paths["tuning"], index=False)
    metric.to_csv(paths["metric"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return prediction, tuning, metadata, metric


def _seasonal_benchmark(
    prepared: pd.DataFrame,
    *,
    city: str,
    config: dict[str, object],
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path = run_dir / "state" / f"{city}_seasonal_naive_predictions.parquet"
    metric_path = run_dir / "state" / f"{city}_seasonal_naive_metric.csv"
    if prediction_path.exists() and metric_path.exists():
        return pd.read_parquet(prediction_path), pd.read_csv(metric_path)
    test = prepared.loc[
        prepared["split"].eq(str(config["splits"]["test"]))
    ].copy()
    if test["crime_count_lag7"].isna().any():
        raise AssertionError("Test seasonal-naive lag is unexpectedly missing.")
    continuous = test["crime_count_lag7"].to_numpy(float)
    prediction = _prediction_frame(
        test,
        family="seasonal_naive_lag7",
        information_set="external_nonmatched",
        continuous=continuous,
        target=str(config["target"]),
    )
    metrics = _prediction_metrics(
        prediction["observed_count"].to_numpy(float),
        continuous,
        nonnegative_half_up(continuous),
        float(config["prediction"]["continuous_prediction_floor"]),
    )
    metric = pd.DataFrame(
        [
            {
                "city": city,
                "city_label": CITY_LABELS[city],
                "model_family": "seasonal_naive_lag7",
                "model_label": MODEL_LABELS["seasonal_naive_lag7"],
                "information_set": "external_nonmatched",
                "information_set_label": INFO_LABELS["external_nonmatched"],
                "matched_to_delb": False,
                "test_start": str(test["date"].min().date()),
                "test_end": str(test["date"].max().date()),
                "test_days": int(test["date"].nunique()),
                "test_grids": int(test["grid_id"].nunique()),
                **metrics,
            }
        ]
    )
    prediction.to_parquet(prediction_path, index=False, compression="zstd")
    metric.to_csv(metric_path, index=False)
    return prediction, metric


def _run_city(
    prepared: pd.DataFrame,
    *,
    city: str,
    config: dict[str, object],
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    predictions = []
    tuning_records = []
    metadata_records: list[dict[str, object]] = []
    metrics = []
    for family in MODEL_ORDER:
        for information_set in INFO_ORDER:
            features = [
                str(value)
                for value in config["information_sets"][information_set]
            ]
            prediction, tuning, metadata, metric = _run_matched_specification(
                prepared,
                city=city,
                family=family,
                information_set=information_set,
                features=features,
                config=config,
                run_dir=run_dir,
                logger=logger,
            )
            predictions.append(prediction)
            tuning_records.append(tuning)
            metadata_records.append(metadata)
            metrics.append(metric)
    benchmark_predictions, benchmark_metric = _seasonal_benchmark(
        prepared, city=city, config=config, run_dir=run_dir
    )
    predictions.append(benchmark_predictions)
    metrics.append(benchmark_metric)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(tuning_records, ignore_index=True),
        metadata_records,
        pd.concat(metrics, ignore_index=True),
    )


def _paired_contrasts(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repetitions = int(config["uncertainty"]["repetitions"])
    block_days = int(config["uncertainty"]["block_days"])
    confidence = float(config["uncertainty"]["confidence_level"])
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2)
    records = []
    bootstrap_records = []
    local_records = []
    for city in CITY_ORDER:
        for family in MODEL_ORDER:
            subset = predictions.loc[
                predictions["city"].eq(city)
                & predictions["model_family"].eq(family)
            ]
            baseline = subset.loc[
                subset["information_set"].eq("baseline")
            ].copy()
            bicycle = subset.loc[
                subset["information_set"].eq("bicycle_aware")
            ].copy()
            keys = ["city", "grid_id", "date", "observed_count"]
            paired = baseline[
                keys
                + [
                    "x_index",
                    "y_index",
                    "centroid_longitude",
                    "centroid_latitude",
                    "squared_error_integer",
                ]
            ].merge(
                bicycle[keys + ["squared_error_integer"]],
                on=keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_baseline", "_bicycle"),
            )
            if len(paired) != len(baseline) or len(paired) != len(bicycle):
                raise AssertionError(f"Unmatched E06 prediction keys for {city} {family}.")
            paired["loss_difference"] = (
                paired["squared_error_integer_baseline"]
                - paired["squared_error_integer_bicycle"]
            )
            point = float(paired["loss_difference"].mean())
            start = paired["date"].min()
            paired["block_id"] = (
                (paired["date"] - start).dt.days // block_days
            ).astype(int)
            block = (
                paired.groupby("block_id", observed=True)["loss_difference"]
                .agg(["sum", "count"])
                .reset_index()
            )
            rng = np.random.default_rng(
                _spec_seed(
                    int(config["random_seed"]),
                    f"E06|bootstrap|{city}|{family}",
                )
            )
            values = np.empty(repetitions, dtype=float)
            for replicate in range(repetitions):
                indices = rng.integers(0, len(block), size=len(block))
                sampled = block.iloc[indices]
                values[replicate] = float(sampled["sum"].sum()) / float(
                    sampled["count"].sum()
                )
                bootstrap_records.append(
                    {
                        "city": city,
                        "model_family": family,
                        "replicate": replicate,
                        "delta_mse_integer": values[replicate],
                    }
                )
            standard_error = float(np.std(values, ddof=1))
            normal_low = point - z_value * standard_error
            normal_high = point + z_value * standard_error
            if standard_error > 0:
                one_sided_p = 1.0 - NormalDist().cdf(point / standard_error)
            else:
                one_sided_p = 0.0 if point > 0 else 1.0
            baseline_metric = metrics.loc[
                metrics["city"].eq(city)
                & metrics["model_family"].eq(family)
                & metrics["information_set"].eq("baseline")
            ].iloc[0]
            bicycle_metric = metrics.loc[
                metrics["city"].eq(city)
                & metrics["model_family"].eq(family)
                & metrics["information_set"].eq("bicycle_aware")
            ].iloc[0]
            records.append(
                {
                    "city": city,
                    "city_label": CITY_LABELS[city],
                    "model_family": family,
                    "model_label": MODEL_LABELS[family],
                    "observations": len(paired),
                    "test_days": int(paired["date"].nunique()),
                    "test_grids": int(paired["grid_id"].nunique()),
                    "mse_baseline_integer": float(
                        baseline_metric["mse_integer"]
                    ),
                    "mse_bicycle_integer": float(
                        bicycle_metric["mse_integer"]
                    ),
                    "delta_mse_integer": point,
                    "relative_mse_improvement": (
                        point / float(baseline_metric["mse_integer"])
                        if float(baseline_metric["mse_integer"]) > 0
                        else np.nan
                    ),
                    "bootstrap_standard_error": standard_error,
                    "normal_ci_low": normal_low,
                    "normal_ci_high": normal_high,
                    "percentile_ci_low": float(
                        np.quantile(values, (1 - confidence) / 2)
                    ),
                    "percentile_ci_high": float(
                        np.quantile(values, 1 - (1 - confidence) / 2)
                    ),
                    "one_sided_positive_p_value": one_sided_p,
                }
            )
            local = (
                paired.groupby(
                    [
                        "city",
                        "grid_id",
                        "x_index",
                        "y_index",
                        "centroid_longitude",
                        "centroid_latitude",
                    ],
                    observed=True,
                )
                .agg(
                    observations=("loss_difference", "size"),
                    mse_baseline_integer=(
                        "squared_error_integer_baseline",
                        "mean",
                    ),
                    mse_bicycle_integer=(
                        "squared_error_integer_bicycle",
                        "mean",
                    ),
                    delta_mse_integer=("loss_difference", "mean"),
                )
                .reset_index()
            )
            local["relative_mse_improvement"] = np.where(
                local["mse_baseline_integer"].gt(0),
                local["delta_mse_integer"] / local["mse_baseline_integer"],
                np.nan,
            )
            local["model_family"] = family
            local_records.append(local)
    contrasts = pd.DataFrame(records)
    contrasts["fdr_q_value"] = benjamini_hochberg(
        contrasts["one_sided_positive_p_value"].to_numpy(float)
    )
    contrasts["positive_improvement_screen"] = (
        contrasts["fdr_q_value"].le(0.05)
        & contrasts["normal_ci_low"].gt(0)
    )
    return (
        contrasts,
        pd.DataFrame(bootstrap_records),
        pd.concat(local_records, ignore_index=True),
    )


def _acceptance_checks(
    *,
    diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    bootstrap: pd.DataFrame,
    tuning: pd.DataFrame,
    metadata: list[dict[str, object]],
    thresholds: pd.DataFrame,
    config: dict[str, object],
    run_dir: Path,
) -> pd.DataFrame:
    expected = config["acceptance"]
    matched = predictions.loc[
        predictions["model_family"].isin(MODEL_ORDER)
    ]
    unique_test = predictions[
        ["city", "grid_id", "date", "observed_count"]
    ].drop_duplicates()
    expected_prediction_rows = int(
        expected["expected_test_observations"]
    ) * (len(MODEL_ORDER) * len(INFO_ORDER) + 1)
    recomputed = (
        predictions.groupby(
            ["city", "model_family", "information_set"], observed=True
        )["squared_error_integer"]
        .mean()
        .reset_index(name="recomputed_mse")
        .merge(
            metrics[
                ["city", "model_family", "information_set", "mse_integer"]
            ],
            on=["city", "model_family", "information_set"],
            how="left",
            validate="one_to_one",
        )
    )
    metric_error = float(
        (recomputed["recomputed_mse"] - recomputed["mse_integer"])
        .abs()
        .max()
    )
    model_files = list((run_dir / "models").glob("*.joblib"))
    selected = tuning.loc[tuning["selected"].astype(bool)]
    paired_counts = (
        matched.groupby(
            ["city", "model_family", "information_set"], observed=True
        )
        .size()
        .unstack("information_set")
    )
    aligned = bool(
        (paired_counts["baseline"] == paired_counts["bicycle_aware"]).all()
    )
    feature_match = all(
        item["features"]
        == [
            str(value)
            for value in config["information_sets"][item["information_set"]]
        ]
        for item in metadata
    )
    no_target_feature = all(
        str(config["target"]) not in item["features"] for item in metadata
    )
    checks = [
        (
            "Expected three cities",
            int(expected["expected_cities"]),
            diagnostics["city"].nunique(),
            diagnostics["city"].nunique() == int(expected["expected_cities"]),
        ),
        (
            "Expected unique test observations",
            int(expected["expected_test_observations"]),
            len(unique_test),
            len(unique_test) == int(expected["expected_test_observations"]),
        ),
        (
            "Expected prediction rows",
            expected_prediction_rows,
            len(predictions),
            len(predictions) == expected_prediction_rows,
        ),
        (
            "Expected metric rows",
            int(expected["expected_metric_rows"]),
            len(metrics),
            len(metrics) == int(expected["expected_metric_rows"]),
        ),
        (
            "Expected paired contrasts",
            int(expected["expected_pair_contrasts"]),
            len(contrasts),
            len(contrasts) == int(expected["expected_pair_contrasts"]),
        ),
        (
            "Expected bootstrap rows",
            int(expected["expected_bootstrap_rows"]),
            len(bootstrap),
            len(bootstrap) == int(expected["expected_bootstrap_rows"]),
        ),
        (
            "One selected hyperparameter per matched specification",
            len(MODEL_ORDER) * len(INFO_ORDER) * len(CITY_ORDER),
            len(selected),
            len(selected) == len(MODEL_ORDER) * len(INFO_ORDER) * len(CITY_ORDER),
        ),
        (
            "One serialized model per matched specification",
            len(MODEL_ORDER) * len(INFO_ORDER) * len(CITY_ORDER),
            len(model_files),
            len(model_files) == len(MODEL_ORDER) * len(INFO_ORDER) * len(CITY_ORDER),
        ),
        (
            "Frozen E04 training thresholds reused",
            3,
            int(thresholds["source_split"].eq("train").sum()),
            len(thresholds) == 3
            and thresholds["source_split"].eq("train").all(),
        ),
        (
            "No test data used in model selection",
            0,
            int(sum(bool(item["test_used_for_selection"]) for item in metadata)),
            not any(bool(item["test_used_for_selection"]) for item in metadata),
        ),
        (
            "Matched feature lists equal configured information sets",
            True,
            feature_match,
            feature_match,
        ),
        (
            "Prediction target excluded from every feature list",
            True,
            no_target_feature,
            no_target_feature,
        ),
        (
            "All predictions finite and nonnegative",
            True,
            bool(
                np.isfinite(predictions["prediction_continuous"]).all()
                and predictions["prediction_continuous"].ge(0).all()
            ),
            bool(
                np.isfinite(predictions["prediction_continuous"]).all()
                and predictions["prediction_continuous"].ge(0).all()
            ),
        ),
        (
            "All integer predictions lie on nonnegative integer lattice",
            True,
            bool(
                predictions["prediction_integer"].ge(0).all()
                and np.equal(
                    predictions["prediction_integer"],
                    np.floor(predictions["prediction_integer"]),
                ).all()
            ),
            bool(
                predictions["prediction_integer"].ge(0).all()
                and np.equal(
                    predictions["prediction_integer"],
                    np.floor(predictions["prediction_integer"]),
                ).all()
            ),
        ),
        (
            "Reported MSE reconciles with saved predictions",
            f"<= {expected['metric_recalculation_tolerance']}",
            metric_error,
            metric_error <= float(expected["metric_recalculation_tolerance"]),
        ),
        (
            "Baseline and bicycle-aware prediction keys aligned",
            True,
            aligned,
            aligned,
        ),
        (
            "All predictions are in held-out test period",
            "2022-07-01 to 2022-12-31",
            f"{predictions['date'].min().date()} to {predictions['date'].max().date()}",
            predictions["date"].min() == pd.Timestamp("2022-07-01")
            and predictions["date"].max() == pd.Timestamp("2022-12-31"),
        ),
        (
            "Each contrast has 1,000 bootstrap replicates",
            int(config["uncertainty"]["repetitions"]),
            int(
                bootstrap.groupby(["city", "model_family"], observed=True)
                .size()
                .min()
            ),
            bool(
                bootstrap.groupby(["city", "model_family"], observed=True)
                .size()
                .eq(int(config["uncertainty"]["repetitions"]))
                .all()
            ),
        ),
        (
            "External seasonal benchmark not labeled DELB-matched",
            False,
            bool(
                metrics.loc[
                    metrics["model_family"].eq("seasonal_naive_lag7"),
                    "matched_to_delb",
                ].any()
            ),
            not bool(
                metrics.loc[
                    metrics["model_family"].eq("seasonal_naive_lag7"),
                    "matched_to_delb",
                ].any()
            ),
        ),
        (
            "Every city retains 184 held-out days",
            184,
            int(diagnostics["test_days"].min()),
            diagnostics["test_days"].eq(184).all(),
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": index + 1,
                "criterion": criterion,
                "expected": target,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
            for index, (criterion, target, observed, passed) in enumerate(checks)
        ]
    )


def _plot_test_mse(
    metrics: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=False)
    width = 0.34
    for axis, city in zip(axes, CITY_ORDER):
        city_metrics = metrics.loc[
            metrics["city"].eq(city) & metrics["model_family"].isin(MODEL_ORDER)
        ]
        x = np.arange(len(MODEL_ORDER))
        baseline = [
            float(
                city_metrics.loc[
                    city_metrics["model_family"].eq(model)
                    & city_metrics["information_set"].eq("baseline"),
                    "mse_integer",
                ].iloc[0]
            )
            for model in MODEL_ORDER
        ]
        bicycle = [
            float(
                city_metrics.loc[
                    city_metrics["model_family"].eq(model)
                    & city_metrics["information_set"].eq("bicycle_aware"),
                    "mse_integer",
                ].iloc[0]
            )
            for model in MODEL_ORDER
        ]
        seasonal = float(
            metrics.loc[
                metrics["city"].eq(city)
                & metrics["model_family"].eq("seasonal_naive_lag7"),
                "mse_integer",
            ].iloc[0]
        )
        axis.bar(x - width / 2, baseline, width, color="#9EADBA", label="Baseline")
        axis.bar(
            x + width / 2,
            bicycle,
            width,
            color=CITY_COLORS[city],
            label="Bicycle-aware",
        )
        axis.axhline(
            seasonal,
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            label="Seasonal naive",
        )
        axis.set_title(CITY_LABELS[city])
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["State\nmean", "Poisson\nGLM", "NB\nGLM", "Gradient\nboosting"]
        )
        axis.set_ylabel("Held-out integer-prediction MSE")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Matched held-out prediction error under baseline and bicycle-aware information",
        y=0.985,
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    _save_figure(figure, path, dpi)


def _plot_improvement(
    contrasts: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    y = np.arange(len(MODEL_ORDER))
    for axis, city in zip(axes, CITY_ORDER):
        subset = (
            contrasts.loc[contrasts["city"].eq(city)]
            .set_index("model_family")
            .loc[MODEL_ORDER]
        )
        values = subset["delta_mse_integer"].to_numpy(float)
        lower = values - subset["normal_ci_low"].to_numpy(float)
        upper = subset["normal_ci_high"].to_numpy(float) - values
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=CITY_COLORS[city],
            ecolor="#555555",
            capsize=3,
        )
        axis.axvline(0, color="#333333", linewidth=0.9)
        axis.set_title(CITY_LABELS[city])
        axis.set_xlabel(r"$\Delta$MSE = baseline − bicycle-aware")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODEL_LABELS[value] for value in MODEL_ORDER])
    figure.suptitle(
        "Held-out MSE improvement from the matched bicycle information state",
        y=1.02,
        fontsize=12,
    )
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _plot_relative_improvement(
    contrasts: pd.DataFrame, path: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.23
    for city_index, city in enumerate(CITY_ORDER):
        subset = (
            contrasts.loc[contrasts["city"].eq(city)]
            .set_index("model_family")
            .loc[MODEL_ORDER]
        )
        axis.bar(
            x + (city_index - 1) * width,
            100 * subset["relative_mse_improvement"].to_numpy(float),
            width,
            color=CITY_COLORS[city],
            label=CITY_LABELS[city],
        )
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels(
        ["State mean", "Poisson GLM", "NB GLM", "Gradient boosting"]
    )
    axis.set_ylabel("Relative held-out MSE improvement (%)")
    axis.set_title("Empirical value of lagged bicycle information by model and city")
    axis.legend(frameon=False, ncol=3, loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _plot_monthly_improvement(
    predictions: pd.DataFrame,
    *,
    family: str,
    path: Path,
    dpi: int,
) -> None:
    subset = predictions.loc[predictions["model_family"].eq(family)].copy()
    paired = subset.loc[subset["information_set"].eq("baseline")][
        ["city", "grid_id", "date", "squared_error_integer"]
    ].merge(
        subset.loc[subset["information_set"].eq("bicycle_aware")][
            ["city", "grid_id", "date", "squared_error_integer"]
        ],
        on=["city", "grid_id", "date"],
        validate="one_to_one",
        suffixes=("_baseline", "_bicycle"),
    )
    paired["month"] = paired["date"].dt.to_period("M").astype(str)
    paired["delta"] = (
        paired["squared_error_integer_baseline"]
        - paired["squared_error_integer_bicycle"]
    )
    monthly = (
        paired.groupby(["city", "month"], observed=True)["delta"]
        .mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(8.8, 4.4))
    for city in CITY_ORDER:
        city_data = monthly.loc[monthly["city"].eq(city)]
        axis.plot(
            city_data["month"],
            city_data["delta"],
            marker="o",
            linewidth=1.8,
            color=CITY_COLORS[city],
            label=CITY_LABELS[city],
        )
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_ylabel("Monthly integer MSE improvement")
    axis.set_xlabel("Held-out month")
    axis.set_title(
        f"Temporal stability of bicycle-aware improvement: {MODEL_LABELS[family]}"
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _plot_daily_totals(
    predictions: pd.DataFrame,
    *,
    family: str,
    path: Path,
    dpi: int,
) -> None:
    subset = predictions.loc[predictions["model_family"].eq(family)].copy()
    daily = (
        subset.groupby(["city", "date", "information_set"], observed=True)
        .agg(
            observed=("observed_count", "sum"),
            predicted=("prediction_integer", "sum"),
        )
        .reset_index()
    )
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.0), sharex=True)
    for axis, city in zip(axes, CITY_ORDER):
        city_data = daily.loc[daily["city"].eq(city)]
        observed = city_data.drop_duplicates("date").sort_values("date")
        axis.plot(
            observed["date"],
            observed["observed"],
            color="#222222",
            linewidth=1.0,
            alpha=0.75,
            label="Observed",
        )
        for information_set, color, linestyle in [
            ("baseline", "#9EADBA", "--"),
            ("bicycle_aware", CITY_COLORS[city], "-"),
        ]:
            values = city_data.loc[
                city_data["information_set"].eq(information_set)
            ].sort_values("date")
            axis.plot(
                values["date"],
                values["predicted"],
                color=color,
                linewidth=1.2,
                linestyle=linestyle,
                label=INFO_LABELS[information_set],
            )
        axis.set_title(CITY_LABELS[city], loc="left")
        axis.set_ylabel("Daily city total")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Held-out date")
    figure.suptitle(
        f"Observed and predicted daily totals: {MODEL_LABELS[family]}",
        y=1.01,
        fontsize=12,
    )
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _plot_local_improvement(
    local: pd.DataFrame,
    *,
    family: str,
    path: Path,
    dpi: int,
) -> None:
    subset = local.loc[local["model_family"].eq(family)].copy()
    maximum = float(np.nanquantile(np.abs(subset["delta_mse_integer"]), 0.98))
    maximum = max(maximum, 1.0e-9)
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    image = None
    for axis, city in zip(axes, CITY_ORDER):
        city_data = subset.loc[subset["city"].eq(city)]
        image = axis.scatter(
            city_data["x_index"],
            city_data["y_index"],
            c=city_data["delta_mse_integer"],
            s=37,
            marker="s",
            cmap="RdBu_r",
            vmin=-maximum,
            vmax=maximum,
            linewidths=0,
        )
        axis.set_title(CITY_LABELS[city])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
    if image is not None:
        colorbar = figure.colorbar(
            image, ax=axes, orientation="horizontal", fraction=0.08, pad=0.12
        )
        colorbar.set_label("Local held-out integer MSE improvement")
    figure.suptitle(
        f"Spatial distribution of empirical bicycle-information value: {MODEL_LABELS[family]}",
        y=0.99,
        fontsize=12,
    )
    figure.subplots_adjust(bottom=0.19, top=0.84, wspace=0.10)
    _save_figure(figure, path, dpi)


def _plot_validation_selection(
    tuning: pd.DataFrame, path: Path, dpi: int
) -> None:
    selected = tuning.loc[tuning["selected"].astype(bool)].copy()
    selected["model_label"] = selected["model_family"].map(MODEL_LABELS)
    selected["information_label"] = selected["information_set"].map(INFO_LABELS)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), sharey=False)
    width = 0.34
    for axis, city in zip(axes, CITY_ORDER):
        city_data = selected.loc[selected["city"].eq(city)]
        x = np.arange(len(MODEL_ORDER))
        baseline = [
            float(
                city_data.loc[
                    city_data["model_family"].eq(model)
                    & city_data["information_set"].eq("baseline"),
                    "validation_mse_integer",
                ].iloc[0]
            )
            for model in MODEL_ORDER
        ]
        bicycle = [
            float(
                city_data.loc[
                    city_data["model_family"].eq(model)
                    & city_data["information_set"].eq("bicycle_aware"),
                    "validation_mse_integer",
                ].iloc[0]
            )
            for model in MODEL_ORDER
        ]
        axis.bar(x - width / 2, baseline, width, color="#9EADBA")
        axis.bar(x + width / 2, bicycle, width, color=CITY_COLORS[city])
        axis.set_title(CITY_LABELS[city])
        axis.set_xticks(x)
        axis.set_xticklabels(["State", "Poisson", "NB", "Boosting"])
        axis.set_ylabel("Selected validation integer MSE")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Validation-only selection of matched predictive specifications",
        y=1.02,
        fontsize=12,
    )
    figure.tight_layout()
    _save_figure(figure, path, dpi)


def _write_interpretation(
    run_dir: Path,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    lines = [
        "# E06 Matched Out-of-Sample Prediction Results",
        "",
        "## Design",
        "",
        "- Hyperparameters were selected using 2020--2021 training data and January--June 2022 validation data only.",
        "- Final models were refitted on training plus validation observations and evaluated once on July--December 2022.",
        "- Every matched pair used the E04 baseline state; the bicycle-aware member added only the frozen E04 lagged bicycle-flow state.",
        "- Predictions used for the primary DELB comparison were rounded half-up to nonnegative integers.",
        "- The lag-7 seasonal naive model is an external benchmark and is not described as DELB-matched.",
        "",
        "## Held-out matched comparisons",
        "",
    ]
    for city in CITY_ORDER:
        lines.extend([f"### {CITY_LABELS[city]}", ""])
        subset = contrasts.loc[contrasts["city"].eq(city)].set_index(
            "model_family"
        )
        for family in MODEL_ORDER:
            row = subset.loc[family]
            lines.append(
                f"- {MODEL_LABELS[family]}: baseline MSE "
                f"{row['mse_baseline_integer']:.5f}, bicycle-aware MSE "
                f"{row['mse_bicycle_integer']:.5f}, improvement "
                f"{row['delta_mse_integer']:.5f} "
                f"({100 * row['relative_mse_improvement']:.2f}%; "
                f"95% centered block interval "
                f"[{row['normal_ci_low']:.5f}, {row['normal_ci_high']:.5f}])."
            )
        seasonal = metrics.loc[
            metrics["city"].eq(city)
            & metrics["model_family"].eq("seasonal_naive_lag7")
        ].iloc[0]
        lines.extend(
            [
                f"- External lag-7 seasonal-naive MSE: {seasonal['mse_integer']:.5f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundaries",
            "",
            "- A positive empirical MSE improvement shows that the fitted model used the added bicycle state in the held-out period; it is not a causal bicycle effect.",
            "- The DELB reduction is an information-set property, whereas empirical improvement is model- and sample-dependent. Their relationship is evaluated explicitly in E07.",
            "- A lower theoretical floor does not guarantee that every fitted model will improve.",
            "- The block analysis preserves common city-day shocks but is based on one six-month test period; rolling-origin and alternative-lag robustness remain for E09.",
            "- E10 placebo experiments remain necessary before final claims about mobility-specific information.",
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
                "experiment": "E06",
                "status": "complete",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "run_directory": str(run_dir),
                "notes": (
                    "Matched baseline and bicycle-aware out-of-sample count "
                    "models with validation-only tuning and block uncertainty."
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
        run_id = started_at.strftime("%Y%m%d_%H%M%S_E06_predictive_models")
        run_dir = empirical_root / "runs" / run_id
    else:
        run_dir = resume_run.resolve()
        run_id = run_dir.name
    for directory in ["logs", "tables", "figures", "state", "models"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    command = (
        f"{sys.executable} empirical/run_e06_predictive_models.py "
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
    logger.info("Starting E06 matched predictive-model experiment")
    try:
        e02_pointer = empirical_root / str(config["source_e02_data_pointer"])
        data_dir = Path(
            e02_pointer.read_text(encoding="utf-8").strip()
        ).resolve()
        e04_run, thresholds = _load_e04_thresholds(empirical_root, config)
        all_predictions = []
        all_tuning = []
        all_metadata: list[dict[str, object]] = []
        all_metrics = []
        diagnostics = []
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
            diagnostics.append(diagnostic)
            logger.info(
                "%s prepared: %d train, %d validation, %d test rows across %d grids",
                city,
                diagnostic["training_rows"],
                diagnostic["validation_rows"],
                diagnostic["test_rows"],
                diagnostic["analysis_grids"],
            )
            predictions, tuning, metadata, metrics = _run_city(
                prepared,
                city=city,
                config=config,
                run_dir=run_dir,
                logger=logger,
            )
            all_predictions.append(predictions)
            all_tuning.append(tuning)
            all_metadata.extend(metadata)
            all_metrics.append(metrics)
            del raw, prepared
        predictions = pd.concat(all_predictions, ignore_index=True)
        tuning = pd.concat(all_tuning, ignore_index=True)
        metrics = pd.concat(all_metrics, ignore_index=True)
        diagnostics_frame = pd.DataFrame(diagnostics)
        predictions["date"] = pd.to_datetime(predictions["date"])
        contrasts, bootstrap, local = _paired_contrasts(
            predictions, metrics, config
        )
        acceptance = _acceptance_checks(
            diagnostics=diagnostics_frame,
            predictions=predictions,
            metrics=metrics,
            contrasts=contrasts,
            bootstrap=bootstrap,
            tuning=tuning,
            metadata=all_metadata,
            thresholds=thresholds,
            config=config,
            run_dir=run_dir,
        )
        table_dir = run_dir / "tables"
        predictions.sort_values(
            ["city", "model_family", "information_set", "date", "grid_id"],
            inplace=True,
        )
        predictions.to_parquet(
            table_dir / "test_predictions.parquet",
            index=False,
            compression="zstd",
        )
        metrics.to_csv(table_dir / "test_metrics.csv", index=False)
        contrasts.to_csv(
            table_dir / "paired_mse_contrasts.csv", index=False
        )
        bootstrap.to_parquet(
            table_dir / "paired_block_bootstrap.parquet",
            index=False,
            compression="zstd",
        )
        local.to_csv(table_dir / "local_test_metrics.csv", index=False)
        local.to_parquet(
            table_dir / "local_test_metrics.parquet",
            index=False,
            compression="zstd",
        )
        tuning.to_csv(table_dir / "validation_tuning_results.csv", index=False)
        pd.DataFrame(all_metadata).assign(
            selected_parameters_json=lambda frame: frame[
                "selected_parameters"
            ].map(lambda value: json.dumps(value, sort_keys=True)),
            features_json=lambda frame: frame["features"].map(json.dumps),
            refit_splits_json=lambda frame: frame["refit_splits"].map(json.dumps),
        ).drop(
            columns=["selected_parameters", "features", "refit_splits"]
        ).to_csv(
            table_dir / "selected_model_metadata.csv", index=False
        )
        diagnostics_frame.to_csv(
            table_dir / "city_input_diagnostics.csv", index=False
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
                "E06 acceptance gate failed: "
                + ", ".join(failed["criterion"].astype(str))
            )
        _publication_style()
        dpi = int(config["reporting"]["figure_dpi"])
        display_model = str(config["reporting"]["primary_display_model"])
        figure_dir = run_dir / "figures"
        _plot_test_mse(metrics, figure_dir / "e06_test_mse.png", dpi)
        _plot_improvement(
            contrasts, figure_dir / "e06_mse_improvement.png", dpi
        )
        _plot_relative_improvement(
            contrasts, figure_dir / "e06_relative_mse_improvement.png", dpi
        )
        _plot_monthly_improvement(
            predictions,
            family=display_model,
            path=figure_dir / "e06_monthly_improvement.png",
            dpi=dpi,
        )
        _plot_daily_totals(
            predictions,
            family=display_model,
            path=figure_dir / "e06_daily_city_totals.png",
            dpi=dpi,
        )
        _plot_local_improvement(
            local,
            family=display_model,
            path=figure_dir / "e06_local_mse_improvement.png",
            dpi=dpi,
        )
        _plot_validation_selection(
            tuning, figure_dir / "e06_validation_selection.png", dpi
        )
        _write_interpretation(run_dir, metrics, contrasts)
        elapsed = time.monotonic() - started_clock
        completed_at = datetime.now(timezone)
        status = {
            "run_id": run_id,
            "status": "complete",
            "acceptance_status": "PASS",
            "acceptance_checks": len(acceptance),
            "source_e02_data_directory": str(data_dir),
            "source_e04_run_directory": str(e04_run),
            "test_observations": int(
                predictions[
                    ["city", "grid_id", "date", "observed_count"]
                ].drop_duplicates().shape[0]
            ),
            "prediction_rows": len(predictions),
            "metric_rows": len(metrics),
            "paired_contrasts": len(contrasts),
            "bootstrap_replicates": len(bootstrap),
            "serialized_models": len(list((run_dir / "models").glob("*.joblib"))),
            "python_article_figure_groups": 7,
            "elapsed_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "article_figures_generated_by": "Python matplotlib",
        }
        (run_dir / "run_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        (empirical_root / "runs" / "latest_e06_run.txt").write_text(
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
            "E06 passed all %d checks with %d test predictions in %.1f seconds",
            len(acceptance),
            len(predictions),
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
        logger.exception("E06 failed; completed model checkpoints were retained.")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E06 matched out-of-sample predictive models"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("empirical/config/e06.yaml"),
    )
    parser.add_argument("--resume-run", type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.resume_run)


if __name__ == "__main__":
    main()
