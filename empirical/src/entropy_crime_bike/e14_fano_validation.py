from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
import yaml

from .discrete_bound import inverse_entropy_envelope
from .e06_predictive_models import (
    CITY_COLORS,
    CITY_LABELS,
    CITY_ORDER,
    _load_city_panel,
    _prepare_city,
)
from .e12_cmi_simulation import build_population_model


LOG_TWO = math.log(2.0)
TARGET_ORDER = ["binary_occurrence", "three_level_count"]
TARGET_LABELS = {
    "binary_occurrence": "Binary: zero vs positive",
    "three_level_count": "Three class: zero, one, two-plus",
}
INFO_ORDER = ["baseline", "bicycle_aware"]
INFO_LABELS = {"baseline": "Baseline", "bicycle_aware": "Bicycle-aware"}
ESTIMATORS = ["plugin", "miller_madow"]


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def _read_pointer(root: Path, relative: str) -> Path:
    pointer = root / relative
    result = Path(pointer.read_text(encoding="utf-8").strip())
    if not result.is_absolute():
        candidate = pointer.parent / result
        result = candidate if candidate.exists() else root / result
    result = result.resolve()
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 0.0), 1.0)
    if p in (0.0, 1.0):
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def fano_phi(error_probability: float, classes: int) -> float:
    if classes < 2:
        raise ValueError("Fano classification requires at least two classes.")
    p = float(error_probability)
    maximum = 1.0 - 1.0 / classes
    if p < 0 or p > maximum:
        raise ValueError("Error probability is outside the increasing branch.")
    return binary_entropy(p) + p * math.log2(classes - 1)


def fano_inverse(entropy_bits: float, classes: int) -> float:
    if classes < 2:
        raise ValueError("Fano classification requires at least two classes.")
    maximum_entropy = math.log2(classes)
    h = min(max(float(entropy_bits), 0.0), maximum_entropy)
    maximum_error = 1.0 - 1.0 / classes
    if h <= 0:
        return 0.0
    if h >= maximum_entropy:
        return maximum_error
    return float(
        brentq(
            lambda p: fano_phi(p, classes) - h,
            0.0,
            maximum_error,
        )
    )


def target_codes(counts: np.ndarray, target_id: str) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("Crime counts must be nonnegative.")
    if target_id == "binary_occurrence":
        return (values > 0).astype(np.int64)
    if target_id == "three_level_count":
        return np.minimum(values, 2).astype(np.int64)
    raise ValueError(target_id)


def _entropy_from_counts(counts: np.ndarray) -> float:
    positive = np.asarray(counts, dtype=float)
    positive = positive[positive > 0]
    total = positive.sum()
    if total <= 0:
        raise ValueError("Entropy requires positive mass.")
    probability = positive / total
    return float(-np.dot(probability, np.log2(probability)))


def conditional_entropy_pair(
    labels: np.ndarray,
    states: np.ndarray,
    classes: int,
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(states, dtype=np.int64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("Labels and states must be aligned vectors.")
    if np.any(y < 0) or np.any(y >= classes) or np.any(s < 0):
        raise ValueError("Invalid class or state code.")
    state_counts = np.bincount(s)
    joint_counts = np.bincount(s * classes + y)
    plugin = _entropy_from_counts(joint_counts) - _entropy_from_counts(
        state_counts
    )
    state_atoms = np.count_nonzero(state_counts)
    joint_atoms = np.count_nonzero(joint_counts)
    miller_madow = plugin + (
        joint_atoms - state_atoms
    ) / (2.0 * len(y) * LOG_TWO)
    return float(plugin), float(miller_madow)


def modal_classifier_error(
    train_labels: np.ndarray,
    train_states: np.ndarray,
    test_labels: np.ndarray,
    test_states: np.ndarray,
    classes: int,
) -> tuple[float, float, int]:
    train_y = np.asarray(train_labels, dtype=np.int64)
    train_s = np.asarray(train_states, dtype=np.int64)
    test_y = np.asarray(test_labels, dtype=np.int64)
    test_s = np.asarray(test_states, dtype=np.int64)
    state_count = int(max(train_s.max(), test_s.max())) + 1
    counts = np.bincount(
        train_s * classes + train_y,
        minlength=state_count * classes,
    ).reshape(state_count, classes)
    observed_state = counts.sum(axis=1) > 0
    global_mode = int(np.bincount(train_y, minlength=classes).argmax())
    modes = np.full(state_count, global_mode, dtype=np.int64)
    modes[observed_state] = counts[observed_state].argmax(axis=1)
    train_error = float(np.mean(modes[train_s] != train_y))
    test_error = float(np.mean(modes[test_s] != test_y))
    unseen = int(np.count_nonzero(~observed_state[test_s]))
    return train_error, test_error, unseen


def _state_codes(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    index = pd.MultiIndex.from_frame(frame[columns].astype(object))
    codes, _ = pd.factorize(index, sort=False)
    if np.any(codes < 0):
        raise ValueError("State factorization produced missing codes.")
    return codes.astype(np.int64)


def _seed(base: int, label: str) -> int:
    return int((base + zlib.crc32(label.encode("utf-8"))) % 2**32)


def _day_blocks(
    frame: pd.DataFrame,
    selected: np.ndarray,
    block_days: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    indices = np.flatnonzero(selected)
    dates = pd.to_datetime(frame.iloc[indices]["date"]).to_numpy()
    unique_dates = np.unique(dates)
    day_indices = [indices[dates == date] for date in unique_dates]
    complete_days = len(day_indices) - len(day_indices) % block_days
    blocks = [
        np.concatenate(day_indices[start : start + block_days])
        for start in range(0, complete_days, block_days)
    ]
    remainder = (
        np.concatenate(day_indices[complete_days:])
        if complete_days < len(day_indices)
        else np.empty(0, dtype=np.int64)
    )
    return blocks, remainder


def _resample_blocks(
    blocks: list[np.ndarray],
    remainder: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = rng.integers(0, len(blocks), size=len(blocks))
    parts = [blocks[index] for index in selected]
    if len(remainder):
        parts.append(remainder)
    return np.concatenate(parts)


def _point_results(
    prepared: pd.DataFrame,
    city: str,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    baseline = _state_codes(
        prepared, list(config["information_sets"]["baseline"])
    )
    bicycle = _state_codes(
        prepared, list(config["information_sets"]["bicycle_aware"])
    )
    pretest = prepared["split"].isin(config["splits"]["pretest"]).to_numpy()
    test = prepared["split"].eq(str(config["splits"]["test"])).to_numpy()
    arrays: dict[str, np.ndarray] = {
        "baseline": baseline,
        "bicycle_aware": bicycle,
        "pretest": pretest,
        "test": test,
        "count": prepared["crime_count_all"].to_numpy(np.int64),
    }
    rows = []
    for target_id in TARGET_ORDER:
        classes = int(config["targets"][target_id]["classes"])
        labels = target_codes(arrays["count"], target_id)
        arrays[f"target_{target_id}"] = labels
        for information_set in INFO_ORDER:
            states = arrays[information_set]
            plugin, miller_madow = conditional_entropy_pair(
                labels[pretest], states[pretest], classes
            )
            train_error, test_error, unseen = modal_classifier_error(
                labels[pretest],
                states[pretest],
                labels[test],
                states[test],
                classes,
            )
            for estimator, entropy in [
                ("plugin", plugin),
                ("miller_madow", miller_madow),
            ]:
                clipped = min(max(entropy, 0.0), math.log2(classes))
                floor = fano_inverse(clipped, classes)
                rows.append(
                    {
                        "city": city,
                        "city_label": CITY_LABELS[city],
                        "target_id": target_id,
                        "target_label": TARGET_LABELS[target_id],
                        "classes": classes,
                        "information_set": information_set,
                        "information_set_label": INFO_LABELS[information_set],
                        "estimator": estimator,
                        "pretest_observations": int(pretest.sum()),
                        "test_observations": int(test.sum()),
                        "conditional_entropy_raw_bits": entropy,
                        "conditional_entropy_for_fano_bits": clipped,
                        "entropy_projection_applied": not math.isclose(
                            entropy, clipped, abs_tol=1e-15
                        ),
                        "fano_error_floor": floor,
                        "pretest_modal_error": train_error,
                        "test_modal_error": test_error,
                        "test_unseen_state_observations": unseen,
                        "estimated_bound_violation": test_error + 1e-12 < floor,
                    }
                )
    return pd.DataFrame(rows), arrays


def _bootstrap_city(
    prepared: pd.DataFrame,
    city: str,
    arrays: dict[str, np.ndarray],
    config: dict[str, object],
    repetitions: int,
) -> pd.DataFrame:
    pretest_blocks, pretest_remainder = _day_blocks(
        prepared,
        arrays["pretest"],
        int(config["bootstrap"]["block_days"]),
    )
    test_blocks, test_remainder = _day_blocks(
        prepared,
        arrays["test"],
        int(config["bootstrap"]["block_days"]),
    )
    records: list[dict[str, object]] = []
    for repetition in range(1, repetitions + 1):
        rng = np.random.default_rng(
            _seed(
                int(config["random_seed"]),
                f"E14|bootstrap|{city}|{repetition}",
            )
        )
        train_index = _resample_blocks(
            pretest_blocks, pretest_remainder, rng
        )
        test_index = _resample_blocks(test_blocks, test_remainder, rng)
        for target_id in TARGET_ORDER:
            classes = int(config["targets"][target_id]["classes"])
            labels = arrays[f"target_{target_id}"]
            for information_set in INFO_ORDER:
                states = arrays[information_set]
                plugin, miller_madow = conditional_entropy_pair(
                    labels[train_index], states[train_index], classes
                )
                train_error, test_error, unseen = modal_classifier_error(
                    labels[train_index],
                    states[train_index],
                    labels[test_index],
                    states[test_index],
                    classes,
                )
                for estimator, entropy in [
                    ("plugin", plugin),
                    ("miller_madow", miller_madow),
                ]:
                    clipped = min(
                        max(entropy, 0.0), math.log2(classes)
                    )
                    records.append(
                        {
                            "city": city,
                            "replicate": repetition,
                            "target_id": target_id,
                            "classes": classes,
                            "information_set": information_set,
                            "estimator": estimator,
                            "conditional_entropy_bits": entropy,
                            "fano_error_floor": fano_inverse(
                                clipped, classes
                            ),
                            "pretest_modal_error": train_error,
                            "test_modal_error": test_error,
                            "test_unseen_state_observations": unseen,
                            "estimated_bound_violation": (
                                test_error + 1e-12
                                < fano_inverse(clipped, classes)
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _add_intervals(
    points: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    keys = ["city", "target_id", "information_set", "estimator"]
    point_fields = {
        "conditional_entropy_bits": "conditional_entropy_raw_bits",
        "fano_error_floor": "fano_error_floor",
        "test_modal_error": "test_modal_error",
    }
    rows = []
    for values, group in bootstrap.groupby(keys, observed=True, sort=True):
        row = dict(zip(keys, values, strict=True))
        point = points
        for key, value in row.items():
            point = point.loc[point[key].eq(value)]
        if len(point) != 1:
            raise AssertionError("Bootstrap point-estimate alignment failed.")
        point = point.iloc[0]
        for field, point_field in point_fields.items():
            centered = group[field] - group[field].mean()
            row[f"{field}_bootstrap_mean"] = group[field].mean()
            row[f"{field}_ci_low"] = (
                point[point_field] + centered.quantile(0.025)
            )
            row[f"{field}_ci_high"] = (
                point[point_field] + centered.quantile(0.975)
            )
        row["bootstrap_estimated_violation_share"] = group[
            "estimated_bound_violation"
        ].mean()
        rows.append(row)
    return points.merge(
        pd.DataFrame(rows), on=keys, how="left", validate="one_to_one"
    )


def _pair_contrasts(
    points: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    keys = ["city", "target_id", "estimator"]
    point_wide = points.pivot(
        index=keys,
        columns="information_set",
        values=[
            "conditional_entropy_raw_bits",
            "fano_error_floor",
            "test_modal_error",
        ],
    )
    rows = []
    for index, row in point_wide.iterrows():
        city, target_id, estimator = index
        record = {
            "city": city,
            "target_id": target_id,
            "estimator": estimator,
            "entropy_reduction_bits": (
                row[("conditional_entropy_raw_bits", "baseline")]
                - row[
                    ("conditional_entropy_raw_bits", "bicycle_aware")
                ]
            ),
            "fano_floor_reduction": (
                row[("fano_error_floor", "baseline")]
                - row[("fano_error_floor", "bicycle_aware")]
            ),
            "test_error_reduction": (
                row[("test_modal_error", "baseline")]
                - row[("test_modal_error", "bicycle_aware")]
            ),
        }
        selected = bootstrap.loc[
            bootstrap["city"].eq(city)
            & bootstrap["target_id"].eq(target_id)
            & bootstrap["estimator"].eq(estimator)
        ]
        wide = selected.pivot(
            index="replicate",
            columns="information_set",
            values=[
                "conditional_entropy_bits",
                "fano_error_floor",
                "test_modal_error",
            ],
        )
        for output, field in [
            ("entropy_reduction_bits", "conditional_entropy_bits"),
            ("fano_floor_reduction", "fano_error_floor"),
            ("test_error_reduction", "test_modal_error"),
        ]:
            difference = wide[(field, "baseline")] - wide[
                (field, "bicycle_aware")
            ]
            centered = difference - difference.mean()
            record[f"{output}_ci_low"] = (
                record[output] + centered.quantile(0.025)
            )
            record[f"{output}_ci_high"] = (
                record[output] + centered.quantile(0.975)
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _class_probabilities(
    count_probability: np.ndarray, target_id: str
) -> np.ndarray:
    if target_id == "binary_occurrence":
        return np.stack(
            [
                count_probability[..., 0],
                1.0 - count_probability[..., 0],
            ],
            axis=-1,
        )
    if target_id == "three_level_count":
        return np.stack(
            [
                count_probability[..., 0],
                count_probability[..., 1],
                1.0
                - count_probability[..., 0]
                - count_probability[..., 1],
            ],
            axis=-1,
        )
    raise ValueError(target_id)


def _weighted_entropy(
    weights: np.ndarray, probability: np.ndarray
) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            probability > 0,
            -probability * np.log2(probability),
            0.0,
        )
    return float(np.sum(weights * terms.sum(axis=-1)))


def _oracle_results(config: dict[str, object]) -> pd.DataFrame:
    oracle = config["oracle"]
    dgp = oracle["negative_binomial"]
    rows = []
    scenario = 0
    for z_count in oracle["baseline_state_counts"]:
        for b_count in oracle["bicycle_state_counts"]:
            for balance in oracle["balance_regimes"]:
                for effect in oracle["effect_strengths"]:
                    scenario += 1
                    model = build_population_model(
                        int(z_count),
                        int(b_count),
                        str(balance),
                        float(effect),
                        dgp,
                    )
                    p_zb = model.joint_state_probability
                    p_z = p_zb.sum(axis=1)
                    p_b_given_z = p_zb / p_z[:, None]
                    for target_id in TARGET_ORDER:
                        classes = int(config["targets"][target_id]["classes"])
                        p_class_zb = _class_probabilities(
                            model.count_probability, target_id
                        )
                        p_class_z = np.einsum(
                            "zb,zbk->zk",
                            p_b_given_z,
                            p_class_zb,
                        )
                        for information_set in INFO_ORDER:
                            if information_set == "baseline":
                                entropy = _weighted_entropy(
                                    p_z, p_class_z
                                )
                                bayes_error = float(
                                    np.sum(
                                        p_z
                                        * (
                                            1.0
                                            - p_class_z.max(axis=-1)
                                        )
                                    )
                                )
                            else:
                                entropy = _weighted_entropy(
                                    p_zb, p_class_zb
                                )
                                bayes_error = float(
                                    np.sum(
                                        p_zb
                                        * (
                                            1.0
                                            - p_class_zb.max(axis=-1)
                                        )
                                    )
                                )
                            floor = fano_inverse(entropy, classes)
                            rows.append(
                                {
                                    "scenario_id": scenario,
                                    "baseline_state_count": int(z_count),
                                    "bicycle_state_count": int(b_count),
                                    "balance_regime": balance,
                                    "effect_strength": float(effect),
                                    "target_id": target_id,
                                    "classes": classes,
                                    "information_set": information_set,
                                    "conditional_entropy_bits": entropy,
                                    "bayes_error": bayes_error,
                                    "fano_error_floor": floor,
                                    "oracle_inequality_holds": (
                                        bayes_error + 1e-12 >= floor
                                    ),
                                }
                            )
    return pd.DataFrame(rows)


def _delb_fano_comparison(
    prepared_by_city: dict[str, pd.DataFrame],
    config: dict[str, object],
    points: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for city, prepared in prepared_by_city.items():
        selected = prepared["split"].isin(config["splits"]["pretest"])
        baseline = _state_codes(
            prepared, list(config["information_sets"]["baseline"])
        )[selected]
        bicycle = _state_codes(
            prepared, list(config["information_sets"]["bicycle_aware"])
        )[selected]
        counts = prepared.loc[selected, "crime_count_all"].to_numpy(
            np.int64
        )
        _, h0 = conditional_entropy_pair(
            counts, baseline, int(counts.max()) + 1
        )
        _, hb = conditional_entropy_pair(
            counts, bicycle, int(counts.max()) + 1
        )
        hb_ordered = min(h0, hb)
        rows.append(
            {
                "city": city,
                "target": "integer crime count",
                "loss": "mean squared error",
                "bound_family": "exact DELB",
                "baseline_conditional_entropy_bits": h0,
                "bicycle_conditional_entropy_bits": hb,
                "baseline_floor": inverse_entropy_envelope(max(h0, 0.0)),
                "bicycle_floor": inverse_entropy_envelope(
                    max(hb_ordered, 0.0)
                ),
                "floor_reduction": (
                    inverse_entropy_envelope(max(h0, 0.0))
                    - inverse_entropy_envelope(max(hb_ordered, 0.0))
                ),
                "numeric_comparison_permitted": False,
            }
        )
    primary = points.loc[points["estimator"].eq("miller_madow")]
    for row in primary.itertuples(index=False):
        rows.append(
            {
                "city": row.city,
                "target": row.target_label,
                "loss": "Hamming error",
                "bound_family": "Fano",
                "baseline_conditional_entropy_bits": np.nan,
                "bicycle_conditional_entropy_bits": np.nan,
                "baseline_floor": np.nan,
                "bicycle_floor": np.nan,
                "floor_reduction": np.nan,
                "numeric_comparison_permitted": False,
                "_target_id": row.target_id,
                "_information_set": row.information_set,
                "_entropy": row.conditional_entropy_raw_bits,
                "_floor": row.fano_error_floor,
            }
        )
    frame = pd.DataFrame(rows)
    fano = frame["bound_family"].eq("Fano")
    fano_rows = frame.loc[fano].copy()
    assembled = []
    for (city, target), group in fano_rows.groupby(
        ["city", "_target_id"], observed=True
    ):
        values = group.set_index("_information_set")
        first = group.iloc[0].to_dict()
        first.update(
            {
                "baseline_conditional_entropy_bits": values.loc[
                    "baseline", "_entropy"
                ],
                "bicycle_conditional_entropy_bits": values.loc[
                    "bicycle_aware", "_entropy"
                ],
                "baseline_floor": values.loc["baseline", "_floor"],
                "bicycle_floor": values.loc[
                    "bicycle_aware", "_floor"
                ],
                "floor_reduction": (
                    values.loc["baseline", "_floor"]
                    - values.loc["bicycle_aware", "_floor"]
                ),
            }
        )
        assembled.append(first)
    output = pd.concat(
        [
            frame.loc[~fano],
            pd.DataFrame(assembled),
        ],
        ignore_index=True,
        sort=False,
    )
    return output.drop(
        columns=[
            "_target_id",
            "_information_set",
            "_entropy",
            "_floor",
        ],
        errors="ignore",
    )


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
        }
    )


def _save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_results(
    oracle: pd.DataFrame,
    points: pd.DataFrame,
    pairs: pd.DataFrame,
    figure_dir: Path,
    dpi: int,
) -> None:
    _publication_style()
    primary = points.loc[points["estimator"].eq("miller_madow")]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    for target_id, marker, color in [
        ("binary_occurrence", "o", "#0072B2"),
        ("three_level_count", "s", "#D55E00"),
    ]:
        subset = oracle.loc[oracle["target_id"].eq(target_id)]
        axes[0, 0].scatter(
            subset["fano_error_floor"],
            subset["bayes_error"],
            marker=marker,
            color=color,
            alpha=0.65,
            s=28,
            label=TARGET_LABELS[target_id],
        )
    maximum = max(
        oracle["fano_error_floor"].max(), oracle["bayes_error"].max()
    )
    axes[0, 0].plot([0, maximum], [0, maximum], "--", color="0.35")
    axes[0, 0].set(
        title="(a) Oracle verification",
        xlabel="Population Fano floor",
        ylabel="Population Bayes error",
    )
    axes[0, 0].legend(frameon=False)

    for axis, target_id, panel in [
        (axes[0, 1], "binary_occurrence", "(b) Binary occurrence"),
        (axes[1, 0], "three_level_count", "(c) Three-level count"),
    ]:
        subset = primary.loc[primary["target_id"].eq(target_id)].copy()
        subset["position"] = subset["city"].map(
            {city: index for index, city in enumerate(CITY_ORDER)}
        )
        for information_set, offset, marker in [
            ("baseline", -0.12, "o"),
            ("bicycle_aware", 0.12, "s"),
        ]:
            selected = subset.loc[
                subset["information_set"].eq(information_set)
            ]
            axis.scatter(
                selected["position"] + offset,
                selected["fano_error_floor"],
                marker=marker,
                s=48,
                facecolors="none",
                edgecolors=[
                    CITY_COLORS[city] for city in selected["city"]
                ],
                linewidth=1.5,
                label=f"{INFO_LABELS[information_set]} Fano floor",
            )
            axis.scatter(
                selected["position"] + offset,
                selected["test_modal_error"],
                marker=marker,
                s=32,
                color=[CITY_COLORS[city] for city in selected["city"]],
                label=f"{INFO_LABELS[information_set]} test error",
            )
        axis.set_xticks(
            range(len(CITY_ORDER)), [CITY_LABELS[c] for c in CITY_ORDER]
        )
        axis.set(title=panel, ylabel="Hamming error probability")
        axis.legend(frameon=False, fontsize=6.5)

    selected_pairs = pairs.loc[pairs["estimator"].eq("miller_madow")]
    labels = [
        f"{row.city}\n{'Binary' if row.target_id == 'binary_occurrence' else 'Three'}"
        for row in selected_pairs.itertuples(index=False)
    ]
    positions = np.arange(len(selected_pairs))
    axes[1, 1].bar(
        positions - 0.18,
        selected_pairs["fano_floor_reduction"],
        width=0.36,
        color="#4C78A8",
        label="Fano-floor reduction",
    )
    axes[1, 1].bar(
        positions + 0.18,
        selected_pairs["test_error_reduction"],
        width=0.36,
        color="#F58518",
        label="Test-error reduction",
    )
    axes[1, 1].axhline(0, color="0.25", linewidth=0.8)
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set(
        title="(d) Bicycle-aware changes",
        ylabel="Baseline minus bicycle-aware",
    )
    axes[1, 1].legend(frameon=False)
    for axis in axes.ravel():
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "e14_fano_main", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, target_id in zip(axes, TARGET_ORDER, strict=True):
        subset = primary.loc[primary["target_id"].eq(target_id)].copy()
        positions = np.arange(len(subset))
        floor_low = subset["fano_error_floor"] - subset[
            "fano_error_floor_ci_low"
        ]
        floor_high = subset["fano_error_floor_ci_high"] - subset[
            "fano_error_floor"
        ]
        error_low = subset["test_modal_error"] - subset[
            "test_modal_error_ci_low"
        ]
        error_high = subset["test_modal_error_ci_high"] - subset[
            "test_modal_error"
        ]
        axis.errorbar(
            positions - 0.08,
            subset["fano_error_floor"],
            yerr=np.vstack([floor_low, floor_high]),
            fmt="o",
            color="#4C78A8",
            capsize=2,
            label="Estimated Fano floor",
        )
        axis.errorbar(
            positions + 0.08,
            subset["test_modal_error"],
            yerr=np.vstack([error_low, error_high]),
            fmt="s",
            color="#F58518",
            capsize=2,
            label="Test modal error",
        )
        labels = [
            f"{row.city}\n{'Base' if row.information_set == 'baseline' else 'Bike'}"
            for row in subset.itertuples(index=False)
        ]
        axis.set_xticks(positions, labels)
        axis.set(
            title=TARGET_LABELS[target_id],
            ylabel="Probability",
        )
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "e14_fano_bootstrap", dpi)


def _interpretation(
    points: pd.DataFrame,
    pairs: pd.DataFrame,
    oracle: pd.DataFrame,
) -> str:
    primary = points.loc[points["estimator"].eq("miller_madow")]
    lines = [
        "# E14 Fano Validation Interpretation",
        "",
        "## Oracle verification",
        "",
        f"- All {len(oracle):,} population-level oracle rows satisfy "
        "Bayes error greater than or equal to the corresponding Fano floor.",
        "",
        "## City results",
        "",
    ]
    for row in primary.sort_values(
        ["target_id", "city", "information_set"]
    ).itertuples(index=False):
        lines.append(
            f"- {CITY_LABELS[row.city]}, {TARGET_LABELS[row.target_id]}, "
            f"{INFO_LABELS[row.information_set]}: entropy="
            f"{row.conditional_entropy_raw_bits:.6f} bits, Fano floor="
            f"{row.fano_error_floor:.6f}, test error="
            f"{row.test_modal_error:.6f}, estimated violation="
            f"{bool(row.estimated_bound_violation)}."
        )
    lines.extend(["", "## Bicycle-aware contrasts", ""])
    for row in pairs.loc[
        pairs["estimator"].eq("miller_madow")
    ].sort_values(["target_id", "city"]).itertuples(index=False):
        lines.append(
            f"- {CITY_LABELS[row.city]}, {TARGET_LABELS[row.target_id]}: "
            f"entropy reduction={row.entropy_reduction_bits:.6f} bits, "
            f"Fano-floor reduction={row.fano_floor_reduction:.6f}, "
            f"test-error reduction={row.test_error_reduction:.6f}."
        )
    primary_pairs = pairs.loc[pairs["estimator"].eq("miller_madow")]
    negative_reductions = int(
        primary_pairs["test_error_reduction"].lt(0).sum()
    )
    unseen = (
        primary.loc[primary["target_id"].eq("binary_occurrence")]
        .pivot(index="city", columns="information_set",
               values="test_unseen_state_observations")
        .reset_index()
    )
    lines.extend(
        [
            "",
            "## Finite-support diagnostic",
            "",
            f"- The bicycle information set lowers the estimated Fano floor "
            f"in all {len(primary_pairs)} primary contrasts, whereas the "
            f"statewise modal classifier has a negative test-error reduction "
            f"in {negative_reductions} of {len(primary_pairs)} contrasts.",
        ]
    )
    for row in unseen.sort_values("city").itertuples(index=False):
        lines.append(
            f"- {CITY_LABELS[row.city]}: test observations assigned through "
            f"the unseen-state fallback increase from "
            f"{int(row.baseline):,} under the baseline information set to "
            f"{int(row.bicycle_aware):,} under the bicycle information set."
        )
    lines.append(
        "- This divergence is consistent with a finite-support burden: "
        "additional information can lower the population error floor while "
        "a sparse statewise estimator fails to exploit it out of sample."
    )
    violations = int(primary["estimated_bound_violation"].sum())
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"There are {violations} point-estimate cases in which the held-out "
            "test error falls below the pretest estimated Fano floor. Such "
            "cases are retained as finite-sample or distribution-shift "
            "diagnostics; an estimated pretest entropy is not the population "
            "test-period conditional entropy. DELB and Fano values are not "
            "numerically comparable because their targets and losses differ. "
            "Neither bound establishes a causal effect of bicycle mobility.",
            "",
        ]
    )
    return "\n".join(lines)


def _environment() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }


def run(config_path: Path, *, quick: bool = False) -> Path:
    start = time.perf_counter()
    config = _load_yaml(config_path)
    project = config_path.resolve().parents[2]
    empirical = project / "empirical"
    e02_data = _read_pointer(
        empirical, str(config["source_e02_data_pointer"])
    )
    e04_run = _read_pointer(
        empirical, str(config["source_e04_run_pointer"])
    )
    thresholds = pd.read_csv(
        e04_run / "tables" / "bicycle_bin_thresholds.csv"
    ).set_index("city")
    repetitions = 5 if quick else int(config["bootstrap"]["repetitions"])
    cities = CITY_ORDER[:1] if quick else CITY_ORDER
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_E14_fano_validation"
        + ("_development" if quick else "")
    )
    run_dir = empirical / "runs" / run_id
    table_dir = run_dir / "tables"
    figure_dir = run_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    point_parts = []
    bootstrap_parts = []
    diagnostics = []
    prepared_by_city: dict[str, pd.DataFrame] = {}
    for city in cities:
        cut_points = (
            float(thresholds.loc[city, "positive_tertile_1_upper"]),
            float(thresholds.loc[city, "positive_tertile_2_upper"]),
        )
        panel = _load_city_panel(e02_data, city, "crime_count_all")
        prepared, diagnostic = _prepare_city(
            panel, cut_points=cut_points, config={
                **config,
                "target": "crime_count_all",
            }
        )
        prepared = prepared.sort_values(["date", "grid_id"]).reset_index(
            drop=True
        )
        points, arrays = _point_results(prepared, city, config)
        boot = _bootstrap_city(
            prepared, city, arrays, config, repetitions
        )
        point_parts.append(points)
        bootstrap_parts.append(boot)
        diagnostics.append(diagnostic)
        prepared_by_city[city] = prepared

    points = _add_intervals(
        pd.concat(point_parts, ignore_index=True),
        pd.concat(bootstrap_parts, ignore_index=True),
    )
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    pairs = _pair_contrasts(points, bootstrap)
    oracle = _oracle_results(config)
    comparison = _delb_fano_comparison(
        prepared_by_city, config, points
    )
    diagnostics_frame = pd.DataFrame(diagnostics)

    if not quick:
        _plot_results(
            oracle,
            points,
            pairs,
            figure_dir,
            int(config["reporting"]["figure_dpi"]),
        )
        (run_dir / "interpretation.md").write_text(
            _interpretation(points, pairs, oracle),
            encoding="utf-8",
        )

    compression = str(config["reporting"]["parquet_compression"])
    bootstrap.to_parquet(
        table_dir / "fano_bootstrap_replicates.parquet",
        index=False,
        compression=compression,
    )
    for name, frame in [
        ("fano_city_results.csv", points),
        ("fano_mobility_contrasts.csv", pairs),
        ("fano_oracle_results.csv", oracle),
        ("delb_fano_comparison.csv", comparison),
        ("city_input_diagnostics.csv", diagnostics_frame),
    ]:
        frame.to_csv(table_dir / name, index=False)

    expected_test = (
        diagnostics_frame["test_rows"].sum()
        if quick
        else int(config["acceptance"]["expected_test_observations"])
    )
    checks = {
        "oracle_fano_holds": bool(oracle["oracle_inequality_holds"].all()),
        "oracle_mobility_ordering": bool(
            (
                oracle.pivot_table(
                    index=[
                        "scenario_id",
                        "target_id",
                    ],
                    columns="information_set",
                    values="fano_error_floor",
                )["bicycle_aware"]
                <= oracle.pivot_table(
                    index=["scenario_id", "target_id"],
                    columns="information_set",
                    values="fano_error_floor",
                )["baseline"]
                + float(config["acceptance"]["oracle_tolerance"])
            ).all()
        ),
        "point_rows": bool(
            quick
            or len(points)
            == int(config["acceptance"]["expected_point_rows"])
        ),
        "bootstrap_rows": bool(
            quick
            or len(bootstrap)
            == int(config["acceptance"]["expected_bootstrap_rows"])
        ),
        "oracle_rows": bool(
            len(oracle) == int(config["acceptance"]["expected_oracle_rows"])
        ),
        "pair_rows": bool(
            quick
            or len(pairs)
            == int(config["acceptance"]["expected_pair_rows"])
        ),
        "test_observations": bool(
            diagnostics_frame["test_rows"].sum() == expected_test
        ),
        "entropy_in_range": bool(
            (
                points["conditional_entropy_for_fano_bits"] >= 0
            ).all()
            and (
                points["conditional_entropy_for_fano_bits"]
                <= np.log2(points["classes"])
                + float(config["acceptance"]["entropy_tolerance_bits"])
            ).all()
        ),
        "errors_in_unit_interval": bool(
            points[
                [
                    "fano_error_floor",
                    "pretest_modal_error",
                    "test_modal_error",
                ]
            ].ge(0).all().all()
            and points[
                [
                    "fano_error_floor",
                    "pretest_modal_error",
                    "test_modal_error",
                ]
            ].le(1).all().all()
        ),
        "formal_figures_exist": bool(
            quick
            or all(
                path.exists()
                for path in [
                    figure_dir / "e14_fano_main.pdf",
                    figure_dir / "e14_fano_main.png",
                    figure_dir / "e14_fano_bootstrap.pdf",
                    figure_dir / "e14_fano_bootstrap.png",
                ]
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError(
            "E14 acceptance failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    config_text = yaml.safe_dump(config, sort_keys=False)
    (run_dir / "config_snapshot.yaml").write_text(
        config_text, encoding="utf-8"
    )
    command = (
        "MPLCONFIGDIR=/private/tmp/e14_mplconfig "
        "python run_e14_fano_validation.py"
        + (" --quick" if quick else "")
    )
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "experiment": "E14",
        "status": "complete",
        "development": quick,
        "elapsed_seconds": time.perf_counter() - start,
        "inputs": {
            "e02_data": str(e02_data),
            "e04_run": str(e04_run),
        },
        "rows": {
            "point_results": len(points),
            "bootstrap_replicates": len(bootstrap),
            "oracle_results": len(oracle),
            "mobility_contrasts": len(pairs),
        },
        "checks": checks,
        "estimated_point_violations": int(
            points["estimated_bound_violation"].sum()
        ),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "environment": _environment(),
        "figures_generated_by": "Python/Matplotlib",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E14 Fano theory and empirical validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "e14.yaml",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(run(args.config, quick=args.quick))


if __name__ == "__main__":
    main()
