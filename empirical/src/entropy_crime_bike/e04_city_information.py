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
from .figure_typography import normalize_chart_typography
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml

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
DOMAIN_LABELS = {
    "complete_crime_domain": "Complete crime domain",
    "bike_covered_training": "Training bicycle-covered domain",
}
PERIOD_LABELS = {
    "pooled_2020_2022": "2020–2022",
    "year_2020": "2020",
    "year_2021": "2021",
    "year_2022": "2022",
}
PERIOD_ORDER = [
    "pooled_2020_2022",
    "year_2020",
    "year_2021",
    "year_2022",
]
BOUND_METRICS = [
    "h0_miller_madow_bits",
    "hb_miller_madow_bits",
    "delta_h_miller_madow_raw_bits",
    "delta_h_projected_bits",
    "l0_exact_mse",
    "lb_exact_mse",
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
    logger = logging.getLogger("e04_city_information")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(
        run_dir / "logs" / "e04_city_information.log", encoding="utf-8"
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
    normalize_chart_typography(figure)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(
        path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def _period_mask(frame: pd.DataFrame, period: str) -> pd.Series:
    if period == "pooled_2020_2022":
        return pd.Series(True, index=frame.index)
    if period.startswith("year_"):
        return frame["year"].eq(int(period.split("_", 1)[1]))
    raise ValueError(f"Unknown E04 period: {period}")


def _domain_mask(frame: pd.DataFrame, domain: str) -> pd.Series:
    if domain == "complete_crime_domain":
        return pd.Series(True, index=frame.index)
    if domain == "bike_covered_training":
        return frame["bike_coverage_training"].astype(bool)
    raise ValueError(f"Unknown E04 domain: {domain}")


def _add_bound_metrics(
    frame: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    result = frame.copy()
    h0 = np.maximum(
        pd.to_numeric(
            result["h0_miller_madow_bits"], errors="raise"
        ).to_numpy(float),
        float(config["estimation"]["entropy_floor_bits"]),
    )
    hb_raw = np.maximum(
        pd.to_numeric(
            result["hb_miller_madow_bits"], errors="raise"
        ).to_numpy(float),
        float(config["estimation"]["entropy_floor_bits"]),
    )
    hb = np.minimum(hb_raw, h0)
    result["h0_for_bound_bits"] = h0
    result["hb_for_bound_bits"] = hb
    result["delta_h_projected_bits"] = h0 - hb
    tail_tolerance = float(
        config["numerics"]["lattice_tail_tolerance"]
    )
    root_tolerance = float(config["numerics"]["root_tolerance"])
    result["l0_exact_mse"] = [
        inverse_entropy_envelope(
            value,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        for value in h0
    ]
    result["lb_exact_mse"] = [
        inverse_entropy_envelope(
            value,
            tail_tolerance=tail_tolerance,
            root_tolerance=root_tolerance,
        )
        for value in hb
    ]
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


def _load_city_panel(
    data_dir: Path, city: str, target: str
) -> pd.DataFrame:
    columns = [
        "city",
        "grid_id",
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
        raise FileNotFoundError(f"No E02 panel files for {city}")
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files],
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _fit_city_bins(
    frame: pd.DataFrame,
    city: str,
    config: dict[str, object],
) -> tuple[tuple[float, float], dict[str, object]]:
    field = str(config["information_sets"]["bicycle_lag_field"])
    training = frame.loc[
        frame["split"].eq("train") & frame[field].notna()
    ]
    quantiles = config["information_sets"]["bicycle_positive_quantiles"]
    cut_points = fit_positive_bicycle_quantiles(
        training[field], quantiles
    )
    mapped = apply_bicycle_bins(training[field], cut_points)
    counts = pd.Series(mapped).value_counts(sort=False)
    record = {
        "city": city,
        "city_label": CITY_LABELS[city],
        "source_split": "train",
        "source_start": str(training["date"].min().date()),
        "source_end": str(training["date"].max().date()),
        "training_observations_with_lag": len(training),
        "training_positive_flow_observations": int(training[field].gt(0).sum()),
        "positive_tertile_1_upper": cut_points[0],
        "positive_tertile_2_upper": cut_points[1],
        "zero_bin_observations": int(counts.get("zero", 0)),
        "low_bin_observations": int(counts.get("low", 0)),
        "medium_bin_observations": int(counts.get("medium", 0)),
        "high_bin_observations": int(counts.get("high", 0)),
    }
    return (float(cut_points[0]), float(cut_points[1])), record


def _prepare_city(
    frame: pd.DataFrame,
    cut_points: tuple[float, float],
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    target = str(config["target"])
    crime_field = str(config["information_sets"]["crime_lag_field"])
    bicycle_field = str(
        config["information_sets"]["bicycle_lag_field"]
    )
    panel_rows = len(frame)
    missing_lag = frame[crime_field].isna() | frame[bicycle_field].isna()
    prepared = frame.loc[~missing_lag].copy()
    prepared["crime_lag_bin"] = crime_lag_bins(prepared[crime_field])
    prepared["bicycle_lag_bin"] = apply_bicycle_bins(
        prepared[bicycle_field], cut_points
    )
    if prepared[["crime_lag_bin", "bicycle_lag_bin"]].isna().any().any():
        raise AssertionError("E04 state mapping produced missing values.")
    target_values = pd.to_numeric(prepared[target], errors="raise")
    if (
        target_values.lt(0).any()
        or not np.equal(target_values, np.floor(target_values)).all()
    ):
        raise AssertionError("E04 target is not a nonnegative integer count.")
    diagnostic = {
        "city": str(frame["city"].iloc[0]),
        "panel_rows_e02": panel_rows,
        "rows_excluded_missing_strict_lag": int(missing_lag.sum()),
        "rows_eligible_e04": len(prepared),
        "minimum_date": str(prepared["date"].min().date()),
        "maximum_date": str(prepared["date"].max().date()),
        "target_minimum": int(target_values.min()),
        "target_maximum": int(target_values.max()),
        "training_covered_grids": int(
            prepared.loc[
                prepared["bike_coverage_training"], "grid_id"
            ].nunique()
        ),
        "complete_domain_grids": int(prepared["grid_id"].nunique()),
    }
    return prepared, diagnostic


def _spec_seed(base_seed: int, spec_id: str) -> int:
    return int((base_seed + zlib.crc32(spec_id.encode("utf-8"))) % 2**32)


def _run_specification(
    frame: pd.DataFrame,
    *,
    city: str,
    domain: str,
    period: str,
    config: dict[str, object],
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    spec_id = f"{city}__{domain}__{period}"
    point_path = run_dir / "state" / f"{spec_id}__point.parquet"
    bootstrap_path = run_dir / "state" / f"{spec_id}__bootstrap.parquet"
    diagnostics_path = run_dir / "state" / f"{spec_id}__diagnostics.json"
    if point_path.exists() and bootstrap_path.exists() and diagnostics_path.exists():
        logger.info("Loading completed E04 checkpoint: %s", spec_id)
        return (
            pd.read_parquet(point_path),
            pd.read_parquet(bootstrap_path),
            json.loads(diagnostics_path.read_text(encoding="utf-8")),
        )

    mask = _domain_mask(frame, domain) & _period_mask(frame, period)
    subset = frame.loc[mask].copy()
    if subset.empty:
        raise AssertionError(f"No observations for E04 specification {spec_id}")
    block_days = int(config["bootstrap"]["block_days"])
    subset["block_id"] = (
        (subset["date"] - subset["date"].min()).dt.days // block_days
    ).astype(np.int16)
    baseline_state = list(
        config["information_sets"]["baseline_state"]
    )
    codebook = ConditionalCodebook.from_frame(
        subset,
        target=str(config["target"]),
        baseline_state=baseline_state,
        bicycle_state_field=str(
            config["information_sets"]["bicycle_state_field"]
        ),
        block_field="block_id",
    )
    point = codebook.estimate(np.ones(codebook.block_count))
    point = _add_bound_metrics(point, config)
    point.insert(0, "spec_id", spec_id)
    point.insert(1, "city", city)
    point.insert(2, "city_label", CITY_LABELS[city])
    point.insert(3, "domain", domain)
    point.insert(4, "domain_label", DOMAIN_LABELS[domain])
    point.insert(5, "period", period)
    point.insert(6, "period_label", PERIOD_LABELS[period])
    point["period_start"] = str(subset["date"].min().date())
    point["period_end"] = str(subset["date"].max().date())
    point["grids"] = int(subset["grid_id"].nunique())
    point["days"] = int(subset["date"].nunique())
    point["blocks"] = codebook.block_count

    repetitions = int(config["bootstrap"]["repetitions"])
    chunk_size = int(
        config["bootstrap"]["matrix_chunk_repetitions"]
    )
    rng = np.random.default_rng(
        _spec_seed(int(config["random_seed"]), spec_id)
    )
    weights = rng.multinomial(
        codebook.block_count,
        np.full(codebook.block_count, 1.0 / codebook.block_count),
        size=repetitions,
    )
    bootstrap_parts = []
    for start in range(0, repetitions, chunk_size):
        stop = min(repetitions, start + chunk_size)
        part = codebook.estimate(weights[start:stop])
        part = _add_bound_metrics(part, config)
        part.insert(0, "replicate", np.arange(start + 1, stop + 1))
        bootstrap_parts.append(part)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    bootstrap.insert(0, "spec_id", spec_id)
    bootstrap.insert(1, "city", city)
    bootstrap.insert(2, "domain", domain)
    bootstrap.insert(3, "period", period)
    diagnostics = {
        "spec_id": spec_id,
        "city": city,
        "domain": domain,
        "period": period,
        "observations": len(subset),
        "grids": int(subset["grid_id"].nunique()),
        "days": int(subset["date"].nunique()),
        "blocks": codebook.block_count,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": _spec_seed(int(config["random_seed"]), spec_id),
        **codebook.sparsity_diagnostics(),
    }
    point.to_parquet(point_path, index=False, compression="zstd")
    bootstrap.to_parquet(
        bootstrap_path, index=False, compression="zstd"
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    logger.info(
        "Completed %s: n=%d, H0=%.4f, HB=%.4f, raw CMI=%.4f",
        spec_id,
        len(subset),
        point["h0_miller_madow_bits"].iloc[0],
        point["hb_miller_madow_bits"].iloc[0],
        point["delta_h_miller_madow_raw_bits"].iloc[0],
    )
    return point, bootstrap, diagnostics


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
            bootstrap["spec_id"].eq(point["spec_id"])
        ]
        record = point.to_dict()
        for metric in BOUND_METRICS:
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
        raw_values = subset["delta_h_miller_madow_raw_bits"]
        record["raw_cmi_bootstrap_nonpositive_share"] = float(
            raw_values.le(0).mean()
        )
        raw_lower = record[
            "delta_h_miller_madow_raw_bits_ci_low"
        ]
        if point["delta_h_miller_madow_raw_bits"] <= 0:
            record["information_evidence"] = "nonpositive_point_estimate"
        elif raw_lower > 0:
            record["information_evidence"] = "positive_with_centered_95pct_ci"
        else:
            record["information_evidence"] = "positive_but_ci_includes_zero"
        rows.append(record)
    return pd.DataFrame(rows)


def _acceptance_checks(
    estimates: pd.DataFrame,
    bootstrap: pd.DataFrame,
    thresholds: pd.DataFrame,
    city_diagnostics: pd.DataFrame,
    sparsity: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    expected_points = int(
        config["acceptance"]["expected_point_estimates"]
    )
    expected_bootstrap = int(
        config["acceptance"]["expected_bootstrap_replicates"]
    )
    identity_error = (
        estimates["h0_miller_madow_bits"]
        - estimates["hb_miller_madow_bits"]
        - estimates["delta_h_miller_madow_raw_bits"]
    ).abs()
    replicate_counts = bootstrap.groupby("spec_id")["replicate"].nunique()
    checks = [
        {
            "check": "Expected city-domain-period point estimates",
            "criterion": f"equals {expected_points}",
            "observed": len(estimates),
            "pass": len(estimates) == expected_points,
        },
        {
            "check": "Expected block-bootstrap replicate rows",
            "criterion": f"equals {expected_bootstrap}",
            "observed": len(bootstrap),
            "pass": len(bootstrap) == expected_bootstrap,
        },
        {
            "check": "Bootstrap repetitions per specification",
            "criterion": f"all equal {config['bootstrap']['repetitions']}",
            "observed": int(replicate_counts.min()),
            "pass": bool(
                replicate_counts.eq(
                    int(config["bootstrap"]["repetitions"])
                ).all()
            ),
        },
        {
            "check": "Conditional mutual-information identity",
            "criterion": (
                "maximum error <= "
                f"{config['acceptance']['cmi_identity_tolerance_bits']} bits"
            ),
            "observed": float(identity_error.max()),
            "pass": bool(
                identity_error.max()
                <= float(
                    config["acceptance"]["cmi_identity_tolerance_bits"]
                )
            ),
        },
        {
            "check": "Projected information gain is nonnegative",
            "criterion": "minimum >= 0",
            "observed": float(estimates["delta_h_projected_bits"].min()),
            "pass": bool(estimates["delta_h_projected_bits"].ge(0).all()),
        },
        {
            "check": "Bicycle-aware exact DELB ordering",
            "criterion": "LB <= L0 for every specification",
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
            "check": "Exact bound reductions are nonnegative",
            "criterion": "minimum >= 0",
            "observed": float(estimates["delta_l_exact_mse"].min()),
            "pass": bool(estimates["delta_l_exact_mse"].ge(-1e-12).all()),
        },
        {
            "check": "Training-only bicycle thresholds",
            "criterion": "all source_split values equal train",
            "observed": int(thresholds["source_split"].eq("train").sum()),
            "pass": bool(thresholds["source_split"].eq("train").all()),
        },
        {
            "check": "One threshold set per city",
            "criterion": "equals 3",
            "observed": thresholds["city"].nunique(),
            "pass": thresholds["city"].nunique() == 3,
        },
        {
            "check": "Strict-lag exclusions match one first day per grid",
            "criterion": "excluded rows equal complete-domain grids",
            "observed": int(
                (
                    city_diagnostics["rows_excluded_missing_strict_lag"]
                    == city_diagnostics["complete_domain_grids"]
                ).sum()
            ),
            "pass": bool(
                (
                    city_diagnostics["rows_excluded_missing_strict_lag"]
                    == city_diagnostics["complete_domain_grids"]
                ).all()
            ),
        },
        {
            "check": "All target counts nonnegative integers",
            "criterion": "minimum target >= 0",
            "observed": int(city_diagnostics["target_minimum"].min()),
            "pass": bool(city_diagnostics["target_minimum"].ge(0).all()),
        },
        {
            "check": "All required numerical estimates finite",
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
                        ]
                    ].to_numpy(float)
                ).all()
            ),
        },
        {
            "check": "Sparsity diagnostics cover every specification",
            "criterion": f"equals {expected_points}",
            "observed": len(sparsity),
            "pass": len(sparsity) == expected_points,
        },
        {
            "check": "Bootstrap normal-interval ordering",
            "criterion": "all lower endpoints <= upper endpoints",
            "observed": int(
                sum(
                    (
                        estimates[f"{metric}_ci_low"]
                        <= estimates[f"{metric}_ci_high"]
                    ).sum()
                    for metric in BOUND_METRICS
                )
            ),
            "pass": bool(
                all(
                    (
                        estimates[f"{metric}_ci_low"]
                        <= estimates[f"{metric}_ci_high"]
                    ).all()
                    for metric in BOUND_METRICS
                )
            ),
        },
    ]
    frame = pd.DataFrame(checks)
    frame["status"] = np.where(frame["pass"], "PASS", "FAIL")
    return frame.drop(columns="pass")


def _plot_primary_cmi(
    estimates: pd.DataFrame, path: Path, dpi: int
) -> None:
    subset = estimates.loc[
        estimates["domain"].eq("bike_covered_training")
        & estimates["period"].eq("pooled_2020_2022")
    ].set_index("city").loc[CITY_ORDER]
    values = subset["delta_h_miller_madow_raw_bits"].to_numpy()
    lower = (
        values
        - subset["delta_h_miller_madow_raw_bits_ci_low"].to_numpy()
    )
    upper = (
        subset["delta_h_miller_madow_raw_bits_ci_high"].to_numpy()
        - values
    )
    figure, axis = plt.subplots(figsize=(7.3, 4.5), constrained_layout=True)
    x = np.arange(len(CITY_ORDER))
    axis.bar(
        x,
        values,
        color=[CITY_COLORS[city] for city in CITY_ORDER],
        width=0.62,
    )
    axis.errorbar(
        x,
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#262626",
        capsize=4,
        linewidth=1.1,
    )
    axis.axhline(0, color="#7F7F7F", linewidth=0.9)
    axis.set_xticks(x, [CITY_LABELS[city] for city in CITY_ORDER])
    axis.set_ylabel("Conditional information gain (bits)")
    axis.set_title(
        "Lagged bicycle mobility and crime-count uncertainty",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
    axis.text(
        0,
        -0.20,
        "Miller–Madow point estimates; centered 95% block-bootstrap "
        "normal intervals; training bicycle-covered 1 km grids.",
        transform=axis.transAxes,
        fontsize=8.3,
        color="#595959",
    )
    _save_figure(figure, path, dpi)


def _plot_entropy_components(
    estimates: pd.DataFrame, path: Path, dpi: int
) -> None:
    subset = estimates.loc[
        estimates["domain"].eq("bike_covered_training")
        & estimates["period"].eq("pooled_2020_2022")
    ].set_index("city").loc[CITY_ORDER]
    x = np.arange(len(CITY_ORDER))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for offset, metric, label, color in [
        (
            -width / 2,
            "h0_miller_madow_bits",
            "Baseline information set",
            "#A5A5A5",
        ),
        (
            width / 2,
            "hb_miller_madow_bits",
            "Baseline + lagged bicycle flow",
            "#5B9BD5",
        ),
    ]:
        values = subset[metric].to_numpy()
        lower = values - subset[f"{metric}_ci_low"].to_numpy()
        upper = subset[f"{metric}_ci_high"].to_numpy() - values
        axis.bar(x + offset, values, width, label=label, color=color)
        axis.errorbar(
            x + offset,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#262626",
            capsize=3,
            linewidth=1,
        )
    axis.set_xticks(x, [CITY_LABELS[city] for city in CITY_ORDER])
    axis.set_ylabel("Conditional entropy (bits)")
    axis.set_title(
        "Crime-count uncertainty before and after bicycle information",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
    _save_figure(figure, path, dpi)


def _plot_exact_bounds(
    estimates: pd.DataFrame, path: Path, dpi: int
) -> None:
    subset = estimates.loc[
        estimates["domain"].eq("bike_covered_training")
        & estimates["period"].eq("pooled_2020_2022")
    ].set_index("city").loc[CITY_ORDER]
    x = np.arange(len(CITY_ORDER))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for offset, metric, label, color in [
        (-width / 2, "l0_exact_mse", "Baseline exact DELB", "#A5A5A5"),
        (width / 2, "lb_exact_mse", "Bicycle-aware exact DELB", "#1F4E78"),
    ]:
        values = subset[metric].to_numpy()
        lower = values - subset[f"{metric}_ci_low"].to_numpy()
        upper = subset[f"{metric}_ci_high"].to_numpy() - values
        axis.bar(x + offset, values, width, color=color, label=label)
        axis.errorbar(
            x + offset,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#262626",
            capsize=3,
            linewidth=1,
        )
    for index, city in enumerate(CITY_ORDER):
        reduction = subset.loc[city, "delta_l_exact_mse"]
        axis.text(
            index,
            max(
                subset.loc[city, "l0_exact_mse_ci_high"],
                subset.loc[city, "lb_exact_mse_ci_high"],
            )
            * 1.04,
            f"$\\Delta L={reduction:.4f}$",
            ha="center",
            fontsize=8.3,
        )
    axis.set_xticks(x, [CITY_LABELS[city] for city in CITY_ORDER])
    axis.set_ylabel("MSE lower bound")
    axis.set_title(
        "Exact discrete entropy lower bound with matched information sets",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
    _save_figure(figure, path, dpi)


def _plot_annual_cmi(
    estimates: pd.DataFrame, path: Path, dpi: int
) -> None:
    subset = estimates.loc[
        estimates["domain"].eq("bike_covered_training")
        & estimates["period"].isin(["year_2020", "year_2021", "year_2022"])
    ].copy()
    figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    years = [2020, 2021, 2022]
    for city in CITY_ORDER:
        city_subset = subset.loc[subset["city"].eq(city)].set_index(
            "period"
        )
        city_subset = city_subset.loc[
            [f"year_{year}" for year in years]
        ]
        values = city_subset["delta_h_miller_madow_raw_bits"].to_numpy()
        lower = (
            values
            - city_subset[
                "delta_h_miller_madow_raw_bits_ci_low"
            ].to_numpy()
        )
        upper = (
            city_subset[
                "delta_h_miller_madow_raw_bits_ci_high"
            ].to_numpy()
            - values
        )
        axis.errorbar(
            years,
            values,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=CITY_COLORS[city],
            label=CITY_LABELS[city],
        )
    axis.axhline(0, color="#7F7F7F", linewidth=0.9)
    axis.set_xticks(years)
    axis.set_ylabel("Conditional information gain (bits)")
    axis.set_title(
        "Annual bicycle information value",
        loc="left",
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
    _save_figure(figure, path, dpi)


def _plot_domain_sensitivity(
    estimates: pd.DataFrame, path: Path, dpi: int
) -> None:
    subset = estimates.loc[
        estimates["period"].eq("pooled_2020_2022")
    ].copy()
    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.2), constrained_layout=True
    )
    x = np.arange(len(CITY_ORDER))
    width = 0.34
    for offset, domain, color in [
        (-width / 2, "complete_crime_domain", "#A5A5A5"),
        (width / 2, "bike_covered_training", "#5B9BD5"),
    ]:
        ordered = subset.loc[subset["domain"].eq(domain)].set_index(
            "city"
        ).loc[CITY_ORDER]
        axes[0].bar(
            x + offset,
            ordered["delta_h_miller_madow_raw_bits"],
            width,
            color=color,
            label=DOMAIN_LABELS[domain],
        )
        axes[1].bar(
            x + offset,
            ordered["delta_l_exact_mse"],
            width,
            color=color,
            label=DOMAIN_LABELS[domain],
        )
    for axis in axes:
        axis.set_xticks(x, [CITY_LABELS[city] for city in CITY_ORDER])
        axis.grid(axis="y", color="#E7E6E6", linewidth=0.7)
        axis.axhline(0, color="#7F7F7F", linewidth=0.8)
    axes[0].set_ylabel("Conditional information gain (bits)")
    axes[0].set_title("(a) Entropy reduction", loc="left", fontweight="bold")
    axes[1].set_ylabel("Exact DELB reduction (MSE)")
    axes[1].set_title("(b) Bound reduction", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Sensitivity to the observed bicycle-system footprint",
        fontsize=13,
        fontweight="bold",
    )
    _save_figure(figure, path, dpi)


def _write_interpretation(
    run_dir: Path,
    estimates: pd.DataFrame,
    sparsity: pd.DataFrame,
) -> None:
    primary = estimates.loc[
        estimates["domain"].eq("bike_covered_training")
        & estimates["period"].eq("pooled_2020_2022")
    ].set_index("city").loc[CITY_ORDER]
    lines = [
        "# E04 City-Level Bicycle Information and DELB Results",
        "",
        "## Estimand",
        "",
        "The primary city-level estimand conditions on grid identity, lagged "
        "crime state, day of week, season, and holiday status. The augmented "
        "information set additionally contains lagged bicycle total-flow "
        "state. Positive-flow tertiles were fitted using 2020–2021 training "
        "observations only. The primary comparison uses grids with observed "
        "training-period bicycle activity; the complete crime domain is a "
        "prespecified sensitivity analysis.",
        "",
        "## Pooled 2020–2022 primary results",
        "",
    ]
    for city in CITY_ORDER:
        row = primary.loc[city]
        lines.extend(
            [
                f"### {CITY_LABELS[city]}",
                "",
                f"- Baseline conditional entropy: "
                f"{row['h0_miller_madow_bits']:.4f} bits.",
                f"- Bicycle-aware conditional entropy: "
                f"{row['hb_miller_madow_bits']:.4f} bits.",
                f"- Raw conditional information gain: "
                f"{row['delta_h_miller_madow_raw_bits']:.4f} bits "
                f"(centered 95% block-bootstrap normal interval "
                f"{row['delta_h_miller_madow_raw_bits_ci_low']:.4f} to "
                f"{row['delta_h_miller_madow_raw_bits_ci_high']:.4f}).",
                f"- Exact DELB reduction: "
                f"{row['delta_l_exact_mse']:.5f} MSE units "
                f"({100 * row['relative_l_reduction']:.2f}% of the "
                "matched baseline bound).",
                f"- Evidence classification: "
                f"`{row['information_evidence']}`.",
                "",
            ]
        )
    worst = sparsity.sort_values(
        "bicycle_observation_share_in_singleton_states",
        ascending=False,
    ).iloc[0]
    lines.extend(
        [
            "## Interpretation and limitations",
            "",
            "- A positive conditional information estimate means that lagged "
            "observed bicycle flow helps distinguish crime-count "
            "distributions after the prespecified baseline state is held "
            "fixed. It is not a causal effect.",
            "- The lower-bound reduction is a change in irreducible "
            "MSE-scale uncertainty. It is not the empirical improvement that "
            "a fitted forecasting model is guaranteed to realize.",
            "- Normal intervals are centered on the full-sample estimate and "
            "use the standard deviation of 7-day block-bootstrap replicates. "
            "Percentile endpoints and bootstrap bias are retained as "
            "diagnostics because discrete support loss can shift the raw "
            "bootstrap distribution.",
            f"- The largest observation share in singleton augmented states "
            f"was {100 * worst['bicycle_observation_share_in_singleton_states']:.2f}% "
            f"for `{worst['spec_id']}`. Sparse-state diagnostics must be "
            "considered when comparing annual and domain-specific estimates.",
            "- E04 establishes information value and bound changes only. "
            "Matched out-of-sample prediction and the predictability gap are "
            "evaluated in E06–E07.",
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
                "experiment": "E04",
                "status": "complete",
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "elapsed_seconds": round(elapsed, 3),
                "run_directory": str(run_dir),
                "notes": (
                    "City-level conditional entropy, bicycle CMI, exact "
                    "DELB reduction, dual-domain estimates, and block bootstrap."
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
            "E04_city_information"
        )
        run_dir = empirical_root / "runs" / run_id
    else:
        run_dir = resume_run.resolve()
        run_id = run_dir.name
    for directory in ["logs", "tables", "figures", "state"]:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    command = (
        f"{sys.executable} empirical/run_e04_city_information.py "
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
    logger.info("Starting E04 city-level information experiment")
    try:
        pointer = empirical_root / str(
            config["source_e02_data_pointer"]
        )
        data_dir = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
        all_points = []
        all_bootstrap = []
        threshold_records = []
        city_diagnostic_records = []
        sparsity_records = []
        for city in CITY_ORDER:
            city_frame = _load_city_panel(
                data_dir, city, str(config["target"])
            )
            cut_points, threshold_record = _fit_city_bins(
                city_frame, city, config
            )
            threshold_records.append(threshold_record)
            prepared, city_diagnostics = _prepare_city(
                city_frame, cut_points, config
            )
            city_diagnostic_records.append(city_diagnostics)
            for domain in config["domains"]:
                for period in config["periods"]:
                    point, bootstrap, diagnostics = _run_specification(
                        prepared,
                        city=city,
                        domain=str(domain),
                        period=str(period),
                        config=config,
                        run_dir=run_dir,
                        logger=logger,
                    )
                    all_points.append(point)
                    all_bootstrap.append(bootstrap)
                    sparsity_records.append(diagnostics)
            del city_frame, prepared

        points = pd.concat(all_points, ignore_index=True)
        bootstrap = pd.concat(all_bootstrap, ignore_index=True)
        estimates = _summarize_uncertainty(points, bootstrap, config)
        thresholds = pd.DataFrame(threshold_records)
        city_diagnostics = pd.DataFrame(city_diagnostic_records)
        sparsity = pd.DataFrame(sparsity_records)
        estimates["city_order"] = estimates["city"].map(
            {city: index for index, city in enumerate(CITY_ORDER)}
        )
        estimates["period_order"] = estimates["period"].map(
            {period: index for index, period in enumerate(PERIOD_ORDER)}
        )
        estimates.sort_values(
            ["city_order", "domain", "period_order"], inplace=True
        )
        estimates.drop(
            columns=["city_order", "period_order"], inplace=True
        )
        acceptance = _acceptance_checks(
            estimates,
            bootstrap,
            thresholds,
            city_diagnostics,
            sparsity,
            config,
        )
        table_dir = run_dir / "tables"
        estimates.to_csv(
            table_dir / "city_information_estimates.csv", index=False
        )
        bootstrap.to_parquet(
            table_dir / "bootstrap_replicates.parquet",
            index=False,
            compression="zstd",
        )
        bootstrap.to_csv(
            table_dir / "bootstrap_replicates.csv", index=False
        )
        thresholds.to_csv(
            table_dir / "bicycle_bin_thresholds.csv", index=False
        )
        city_diagnostics.to_csv(
            table_dir / "city_input_diagnostics.csv", index=False
        )
        sparsity.to_csv(
            table_dir / "state_sparsity_diagnostics.csv", index=False
        )
        acceptance.to_csv(
            table_dir / "acceptance_checklist.csv", index=False
        )
        primary = estimates.loc[
            estimates["domain"].eq("bike_covered_training")
            & estimates["period"].eq("pooled_2020_2022")
        ].copy()
        primary.to_csv(
            table_dir / "primary_city_results.csv", index=False
        )

        failed = acceptance.loc[acceptance["status"].ne("PASS")]
        if len(failed):
            raise AssertionError(
                "E04 acceptance gate failed: "
                + ", ".join(failed["check"].astype(str))
            )

        _publication_style()
        dpi = int(config["reporting"]["figure_dpi"])
        figure_dir = run_dir / "figures"
        _plot_primary_cmi(
            estimates, figure_dir / "e04_primary_bicycle_cmi.png", dpi
        )
        _plot_entropy_components(
            estimates, figure_dir / "e04_conditional_entropies.png", dpi
        )
        _plot_exact_bounds(
            estimates, figure_dir / "e04_exact_delb_reduction.png", dpi
        )
        _plot_annual_cmi(
            estimates, figure_dir / "e04_annual_bicycle_cmi.png", dpi
        )
        _plot_domain_sensitivity(
            estimates, figure_dir / "e04_domain_sensitivity.png", dpi
        )
        _write_interpretation(run_dir, estimates, sparsity)
        elapsed = time.monotonic() - started_clock
        completed_at = datetime.now(timezone)
        status = {
            "run_id": run_id,
            "status": "complete",
            "acceptance_status": "PASS",
            "acceptance_checks": len(acceptance),
            "source_e02_data_directory": str(data_dir),
            "point_estimates": len(estimates),
            "bootstrap_replicates": len(bootstrap),
            "elapsed_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "article_figures_generated_by": "Python matplotlib",
        }
        (run_dir / "run_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        (empirical_root / "runs" / "latest_e04_run.txt").write_text(
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
            "E04 passed all %d checks in %.1f seconds",
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
        logger.exception("E04 failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate city-level bicycle information and exact DELB"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-run", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, args.resume_run)


if __name__ == "__main__":
    main()
