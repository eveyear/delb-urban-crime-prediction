from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.stats import binomtest, nbinom
import yaml

from entropy_crime_bike.e06_predictive_models import CITY_LABELS, CITY_ORDER
from entropy_crime_bike.e09_robustness import _load_panel, _prepare_spec
from entropy_crime_bike.e10_placebo import (
    _state_codes,
    conditional_information,
    randomization_p_value,
    temporal_block_permutation,
)
from entropy_crime_bike.spatial_information import benjamini_hochberg


DESIGNS = ("temporal_block_permutation", "restricted_matched_blocks")


def _seed(base: int, label: str) -> int:
    return int((base + zlib.crc32(label.encode("utf-8"))) % 2**32)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def _read_pointer(root: Path, relative: str) -> Path:
    value = Path((root / relative).read_text(encoding="utf-8").strip())
    if not value.exists():
        raise FileNotFoundError(value)
    return value.resolve()


def _matrix_view(frame: pd.DataFrame, baseline_fields: list[str]) -> dict[str, object]:
    ordered = frame.sort_values(["grid_id", "date"]).reset_index(drop=True)
    grids = ordered["grid_id"].astype(str).drop_duplicates().tolist()
    counts = ordered.groupby("grid_id", observed=True).size()
    if counts.nunique() != 1:
        raise AssertionError("E18 requires equal-length complete grid panels.")
    days = int(counts.iloc[0])
    first = ordered["grid_id"].astype(str).eq(grids[0])
    return {
        "frame": ordered,
        "grids": grids,
        "days": days,
        "dates": pd.to_datetime(ordered.loc[first, "date"]).reset_index(drop=True),
        "target": ordered["crime_count_all"].to_numpy(np.int64),
        "baseline": _state_codes(ordered, baseline_fields),
        "bicycle": ordered["bicycle_lag_bin"].cat.codes.to_numpy(np.int16).reshape(
            len(grids), days
        ),
        "crime_lag": ordered["crime_lag_bin"].cat.codes.to_numpy(np.int16).reshape(
            len(grids), days
        ),
        "holiday": ordered["is_holiday"].astype(int).to_numpy(np.int8).reshape(
            len(grids), days
        ),
    }


def _block_groups(
    dates: pd.Series,
    crime_lag: np.ndarray,
    holiday: np.ndarray,
    block_days: int,
    minimum_blocks: int,
) -> tuple[list[list[int]], int]:
    complete = len(dates) - len(dates) % block_days
    starts = list(range(0, complete, block_days))
    metadata = []
    for block, start in enumerate(starts):
        date = dates.iloc[start]
        values = crime_lag[start : start + block_days]
        proportions = [float(np.mean(values == level)) for level in range(4)]
        metadata.append(
            {
                "block": block,
                "year": int(date.year),
                "season": int((int(date.month) % 12) // 3),
                "score": proportions[1] + 2 * proportions[2] + 3 * proportions[3]
                + 0.25 * float(holiday[start : start + block_days].sum()),
            }
        )
    meta = pd.DataFrame(metadata)
    groups: list[list[int]] = []
    used: set[int] = set()
    for _, stratum in meta.groupby(["year", "season"], sort=True):
        indices = stratum.sort_values("score")["block"].astype(int).tolist()
        if len(indices) >= minimum_blocks:
            groups.append(indices)
            used.update(indices)
    remaining = meta.loc[~meta["block"].isin(used)]
    for _, stratum in remaining.groupby("year", sort=True):
        indices = stratum.sort_values("score")["block"].astype(int).tolist()
        if len(indices) >= minimum_blocks:
            groups.append(indices)
            used.update(indices)
    return groups, len(starts)


def restricted_matched_blocks(
    bicycle_matrix: np.ndarray,
    dates: pd.Series,
    crime_lag_matrix: np.ndarray,
    holiday_matrix: np.ndarray,
    rng: np.random.Generator,
    *,
    block_days: int = 7,
    minimum_blocks: int = 5,
) -> tuple[np.ndarray, float]:
    values = np.asarray(bicycle_matrix)
    result = values.copy()
    movable = 0
    total_blocks = 0
    for grid in range(values.shape[0]):
        groups, blocks = _block_groups(
            dates,
            crime_lag_matrix[grid],
            holiday_matrix[grid],
            block_days,
            minimum_blocks,
        )
        total_blocks += blocks
        for group in groups:
            shift = int(rng.integers(1, len(group)))
            donors = np.roll(np.asarray(group), shift)
            for recipient, donor in zip(group, donors):
                r0, d0 = recipient * block_days, int(donor) * block_days
                result[grid, r0 : r0 + block_days] = values[
                    grid, d0 : d0 + block_days
                ]
            movable += len(group)
    return result, movable / max(1, total_blocks)


def _standardized_difference(a: np.ndarray, b: np.ndarray) -> float:
    denominator = math.sqrt((float(np.var(a)) + float(np.var(b))) / 2)
    if denominator == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / denominator)


def _run_empirical(
    views: dict[str, dict[str, object]], config: dict[str, object], development: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["restricted_randomization"]
    repetitions = 19 if development else int(cfg["repetitions"])
    rows, diagnostics = [], []
    for city in CITY_ORDER:
        view = views[city]
        y = np.asarray(view["target"])
        s = np.asarray(view["baseline"])
        b = np.asarray(view["bicycle"])
        _, _, observed = conditional_information(y, s, b.ravel())
        null_values, movable_values, balance_values = [], [], []
        for repetition in range(1, repetitions + 1):
            rng = np.random.default_rng(
                _seed(int(config["random_seed"]), f"E18A|{city}|{repetition}")
            )
            randomized, movable = restricted_matched_blocks(
                b,
                view["dates"],
                np.asarray(view["crime_lag"]),
                np.asarray(view["holiday"]),
                rng,
                block_days=int(cfg["block_days"]),
                minimum_blocks=int(cfg["minimum_blocks_per_stratum"]),
            )
            _, _, value = conditional_information(y, s, randomized.ravel())
            null_values.append(value)
            movable_values.append(movable)
            balance_values.append(
                abs(_standardized_difference(b.ravel(), randomized.ravel()))
            )
            rows.append(
                {
                    "city": city,
                    "replicate": repetition,
                    "null_cmi_bits": value,
                    "movable_block_share": movable,
                    "bicycle_state_smd": balance_values[-1],
                }
            )
        null = np.asarray(null_values)
        diagnostics.append(
            {
                "city": city,
                "observed_cmi_bits": observed,
                "null_mean_bits": float(null.mean()),
                "observed_minus_null_mean_bits": float(observed - null.mean()),
                "randomization_p_value": randomization_p_value(observed, null),
                "movable_block_share": float(np.mean(movable_values)),
                "maximum_bicycle_state_smd": float(max(balance_values)),
            }
        )
        print(f"E18-A complete: {city} ({repetitions} randomizations)", flush=True)
    summary = pd.DataFrame(diagnostics)
    summary["q_value_bh"] = benjamini_hochberg(summary["randomization_p_value"])
    summary["reject_fdr_005"] = summary["q_value_bh"].le(0.05)
    return pd.DataFrame(rows), summary


def _representative_view(
    view: dict[str, object], grid_count: int, days: int
) -> dict[str, object]:
    bicycle = np.asarray(view["bicycle"])
    score = bicycle.mean(axis=1)
    order = np.argsort(score)
    positions = np.linspace(0, len(order) - 1, min(grid_count, len(order))).round().astype(int)
    selected = np.sort(order[positions])
    panel_days = min(days, bicycle.shape[1])
    row_indices = np.concatenate(
        [np.arange(index * bicycle.shape[1], index * bicycle.shape[1] + panel_days) for index in selected]
    )
    frame = view["frame"].iloc[row_indices].copy().reset_index(drop=True)
    return _matrix_view(
        frame,
        ["grid_id", "crime_lag_bin", "day_of_week", "season", "is_holiday"],
    )


def _conditional_means(
    target: np.ndarray, baseline: np.ndarray, shrinkage: float
) -> np.ndarray:
    global_mean = float(np.mean(target))
    counts = np.bincount(baseline)
    sums = np.bincount(baseline, weights=target)
    means = (sums + shrinkage * global_mean) / (counts + shrinkage)
    return np.maximum(means[baseline], 0.02)


def _population_cmi(
    baseline: np.ndarray,
    bicycle: np.ndarray,
    base_mean: np.ndarray,
    dispersion: float,
    beta: float,
) -> float:
    centered = bicycle - np.mean(bicycle)
    means = np.clip(base_mean * np.exp(beta * centered), 0.01, 50.0)
    support_max = int(
        np.nanmax(nbinom.ppf(1 - 1e-10, dispersion, dispersion / (dispersion + means)))
    )
    support = np.arange(support_max + 1)
    probs = nbinom.pmf(
        support[None, :], dispersion, (dispersion / (dispersion + means))[:, None]
    )
    probs /= probs.sum(axis=1, keepdims=True)
    total = len(baseline)
    hb = 0.0
    h0 = 0.0
    pair = baseline * (int(bicycle.max()) + 1) + bicycle
    for code in np.unique(pair):
        mask = pair == code
        p = probs[mask].mean(axis=0)
        hb += mask.sum() / total * float(-np.dot(p[p > 0], np.log2(p[p > 0])))
    for code in np.unique(baseline):
        mask = baseline == code
        p = probs[mask].mean(axis=0)
        h0 += mask.sum() / total * float(-np.dot(p[p > 0], np.log2(p[p > 0])))
    return max(0.0, h0 - hb)


def _calibrate_beta(
    target_bits: float,
    baseline: np.ndarray,
    bicycle: np.ndarray,
    base_mean: np.ndarray,
    dispersion: float,
) -> tuple[float, float]:
    if target_bits == 0:
        return 0.0, 0.0
    fn = lambda beta: _population_cmi(
        baseline, bicycle, base_mean, dispersion, beta
    ) - target_bits
    upper = 0.25
    while fn(upper) < 0 and upper < 8:
        upper *= 2
    if fn(upper) < 0:
        candidates = np.linspace(0.0, upper, 65)
        achieved_values = np.asarray(
            [
                _population_cmi(
                    baseline, bicycle, base_mean, dispersion, candidate
                )
                for candidate in candidates
            ]
        )
        best = int(np.argmax(achieved_values))
        return float(candidates[best]), float(achieved_values[best])
    beta = float(brentq(fn, 0.0, upper))
    achieved = _population_cmi(
        baseline, bicycle, base_mean, dispersion, beta
    )
    return beta, achieved


def _randomized_matrix(
    design: str,
    view: dict[str, object],
    rng: np.random.Generator,
    config: dict[str, object],
) -> np.ndarray:
    matrix = np.asarray(view["bicycle"])
    if design == "temporal_block_permutation":
        return temporal_block_permutation(matrix, rng, 7)
    randomized, _ = restricted_matched_blocks(
        matrix,
        view["dates"],
        np.asarray(view["crime_lag"]),
        np.asarray(view["holiday"]),
        rng,
        block_days=int(config["restricted_randomization"]["block_days"]),
        minimum_blocks=int(
            config["restricted_randomization"]["minimum_blocks_per_stratum"]
        ),
    )
    return randomized


def _run_semi_synthetic(
    views: dict[str, dict[str, object]], config: dict[str, object], development: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config["semi_synthetic"]
    monte_carlo = 20 if development else int(cfg["monte_carlo_repetitions"])
    null_repetitions = 19 if development else int(cfg["randomization_repetitions"])
    targets = [0.0, 0.05] if development else [float(x) for x in cfg["target_cmi_bits"]]
    dispersion_multipliers = [1.0] if development else [
        float(x) for x in cfg["dispersion_multipliers"]
    ]
    rows = []
    for city in CITY_ORDER:
        view = _representative_view(
            views[city], int(cfg["representative_grids_per_city"]), int(cfg["days"])
        )
        empirical_y = np.asarray(view["target"])
        baseline = np.asarray(view["baseline"])
        bicycle_matrix = np.asarray(view["bicycle"])
        bicycle = bicycle_matrix.ravel().astype(int)
        base_mean = _conditional_means(
            empirical_y, baseline, float(cfg["mean_shrinkage"])
        )
        empirical_var = float(np.var(empirical_y))
        empirical_mean = float(np.mean(empirical_y))
        base_dispersion = (
            empirical_mean**2 / max(empirical_var - empirical_mean, 1e-3)
        )
        base_dispersion = float(np.clip(base_dispersion, 0.25, 20.0))
        for multiplier in dispersion_multipliers:
            dispersion = base_dispersion * multiplier
            for target_bits in targets:
                beta, achieved = _calibrate_beta(
                    target_bits, baseline, bicycle, base_mean, dispersion
                )
                means = np.clip(
                    base_mean * np.exp(beta * (bicycle - np.mean(bicycle))),
                    0.01,
                    50.0,
                )
                for repetition in range(1, monte_carlo + 1):
                    label = f"E18B|{city}|{multiplier}|{target_bits}|{repetition}"
                    rng_y = np.random.default_rng(
                        _seed(int(config["random_seed"]), label + "|Y")
                    )
                    probability = dispersion / (dispersion + means)
                    y = rng_y.negative_binomial(dispersion, probability).astype(int)
                    _, _, observed = conditional_information(
                        y, baseline, bicycle
                    )
                    for design in DESIGNS:
                        null = []
                        rng = np.random.default_rng(
                            _seed(int(config["random_seed"]), label + "|" + design)
                        )
                        for _ in range(null_repetitions):
                            randomized = _randomized_matrix(
                                design, view, rng, config
                            )
                            _, _, value = conditional_information(
                                y, baseline, randomized.ravel()
                            )
                            null.append(value)
                        rows.append(
                            {
                                "city": city,
                                "target_cmi_bits": target_bits,
                                "achieved_population_cmi_bits": achieved,
                                "target_cmi_reached": abs(achieved - target_bits)
                                <= 1e-6,
                                "dispersion_multiplier": multiplier,
                                "replicate": repetition,
                                "design": design,
                                "observed_cmi_bits": observed,
                                "null_mean_bits": float(np.mean(null)),
                                "randomization_p_value": randomization_p_value(
                                    observed, np.asarray(null)
                                ),
                                "reject_005": randomization_p_value(
                                    observed, np.asarray(null)
                                ) <= 0.05,
                            }
                        )
                print(
                    "E18-B complete: "
                    f"{city}, dispersion={multiplier}, target={target_bits:.2f}, "
                    f"achieved={achieved:.4f}",
                    flush=True,
                )
    datasets = pd.DataFrame(rows)
    summary = (
        datasets.groupby(
            [
                "city",
                "target_cmi_bits",
                "achieved_population_cmi_bits",
                "dispersion_multiplier",
                "design",
            ],
            as_index=False,
        )
        .agg(
            monte_carlo_repetitions=("replicate", "nunique"),
            rejection_rate=("reject_005", "mean"),
            mean_observed_cmi_bits=("observed_cmi_bits", "mean"),
            mean_null_cmi_bits=("null_mean_bits", "mean"),
        )
    )
    intervals = []
    for row in summary.itertuples(index=False):
        subset = datasets.loc[
            datasets["city"].eq(row.city)
            & datasets["target_cmi_bits"].eq(row.target_cmi_bits)
            & datasets["dispersion_multiplier"].eq(row.dispersion_multiplier)
            & datasets["design"].eq(row.design)
        ]
        result = binomtest(int(subset["reject_005"].sum()), len(subset))
        interval = result.proportion_ci(confidence_level=0.95, method="exact")
        intervals.append((interval.low, interval.high))
    summary["rejection_ci_low"] = [x[0] for x in intervals]
    summary["rejection_ci_high"] = [x[1] for x in intervals]
    return datasets, summary


def _figures(empirical: pd.DataFrame, simulation: pd.DataFrame, run_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(
        [CITY_LABELS[x] for x in empirical["city"]],
        empirical["observed_minus_null_mean_bits"],
        color=["#4472C4", "#ED7D31", "#70AD47"],
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Observed minus restricted-null mean (bits)")
    axes[0].set_title("(a) Empirical restricted randomization")
    for design, marker in zip(DESIGNS, ["o", "s"]):
        subset = simulation.loc[
            simulation["dispersion_multiplier"].eq(1.0)
            & simulation["design"].eq(design)
        ]
        grouped = subset.groupby("target_cmi_bits", as_index=False)["rejection_rate"].mean()
        axes[1].plot(
            grouped["target_cmi_bits"], grouped["rejection_rate"],
            marker=marker, label=design.replace("_", " ")
        )
    axes[1].axhline(0.05, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Target population CMI (bits)")
    axes[1].set_ylabel("Rejection rate")
    axes[1].set_title("(b) Semi-synthetic calibration and power")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "e18_restricted_calibration.pdf", bbox_inches="tight")
    fig.savefig(run_dir / "figures" / "e18_restricted_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(config_path: Path, development: bool = False) -> Path:
    started = time.monotonic()
    config = _load_yaml(config_path)
    project = config_path.resolve().parents[2]
    empirical_root = project / "empirical"
    mode = "development" if development else "formal"
    run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S") + f"_E18_{mode}"
    run_dir = empirical_root / "runs" / run_id
    for name in ["tables", "figures", "logs"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    e02 = _read_pointer(empirical_root, str(config["source_e02_data_pointer"]))
    views = {}
    for city in CITY_ORDER:
        panel = _load_panel(e02, city)
        prepared, _ = _prepare_spec(panel, "primary")
        views[city] = _matrix_view(
            prepared, list(config["information_sets"]["baseline_state"])
        )
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({"e18": config, "mode": mode, "source_e02": str(e02)}, sort_keys=False),
        encoding="utf-8",
    )
    empirical_replicates, empirical_summary = _run_empirical(
        views, config, development
    )
    minimum_share = float(config["restricted_randomization"]["minimum_movable_share"])
    balance_limit = float(config["restricted_randomization"]["balance_threshold"])
    if (empirical_summary["movable_block_share"] < minimum_share).any():
        raise AssertionError("E18 movable-block stopping rule triggered.")
    if (empirical_summary["maximum_bicycle_state_smd"] > balance_limit).any():
        raise AssertionError("E18 balance stopping rule triggered.")
    simulation_datasets, simulation_summary = _run_semi_synthetic(
        views, config, development
    )
    for name, frame in [
        ("empirical_replicates.parquet", empirical_replicates),
        ("semi_synthetic_datasets.parquet", simulation_datasets),
    ]:
        frame.to_parquet(run_dir / "tables" / name, index=False, compression="zstd")
    empirical_summary.to_csv(run_dir / "tables" / "empirical_summary.csv", index=False)
    simulation_summary.to_csv(run_dir / "tables" / "semi_synthetic_summary.csv", index=False)
    null_summary = simulation_summary.loc[simulation_summary["target_cmi_bits"].eq(0)].copy()
    null_summary["covers_nominal_005"] = (
        null_summary["rejection_ci_low"].le(0.05)
        & null_summary["rejection_ci_high"].ge(0.05)
    )
    checks = pd.DataFrame(
        [
            {"check": "movable_share", "passed": bool((empirical_summary["movable_block_share"] >= minimum_share).all())},
            {"check": "balance", "passed": bool((empirical_summary["maximum_bicycle_state_smd"] <= balance_limit).all())},
            {"check": "finite_empirical", "passed": bool(np.isfinite(empirical_summary.select_dtypes("number")).all().all())},
            {"check": "finite_simulation", "passed": bool(np.isfinite(simulation_summary.select_dtypes("number")).all().all())},
            {"check": "null_intervals_cover_005", "passed": bool(null_summary["covers_nominal_005"].all())},
        ]
    )
    checks.to_csv(run_dir / "tables" / "acceptance_checklist.csv", index=False)
    _figures(empirical_summary, simulation_summary, run_dir)
    elapsed = time.monotonic() - started
    status = {
        "experiment": "E18",
        "mode": mode,
        "status": "complete" if checks["passed"].all() else "complete_with_calibration_warning",
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not development:
        (empirical_root / "runs" / "latest_e18_run.txt").write_text(
            str(run_dir.resolve()) + "\n", encoding="utf-8"
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    print(run(args.config, development=args.development))


if __name__ == "__main__":
    main()
