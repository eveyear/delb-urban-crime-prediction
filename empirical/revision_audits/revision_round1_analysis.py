from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from entropy_crime_bike.conditional_information import (
    apply_bicycle_bins,
    crime_lag_bins,
    fit_positive_bicycle_quantiles,
)
from entropy_crime_bike.discrete_bound import (
    LOG_TWO,
    discrete_gaussian_stats,
    inverse_entropy_envelope,
    lambda_for_second_moment,
)
from entropy_crime_bike.e10_placebo import _state_codes, conditional_information, randomization_p_value
from entropy_crime_bike.e18_restricted_randomization import _block_groups


ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "processed_data/e02/20260718_172231_E02_spatial_temporal_panel/grid_day_panel"
E06 = ROOT / "runs/20260718_193815_E06_predictive_models/tables/test_predictions.parquet"
E09 = ROOT / "runs/20260719_183554_E09_robustness/tables/theory_robustness_matrix.csv"
E12 = ROOT / "runs/20260720_115049_S7_synthesis/e12_summary.csv"
E14 = ROOT / "runs/20260720_111726_E14_fano_validation/tables/fano_city_results.csv"
E18 = ROOT / "runs/20260728_125347_E18_formal/tables/empirical_summary.csv"
OUT = ROOT / "runs/revision_round1_analysis"
TABLES = OUT / "tables"
CITY_ORDER = ["DC", "NY", "VAN"]
CITY_LABELS = {"DC": "Washington, DC", "NY": "New York City", "VAN": "Vancouver"}
BASE = ["grid_id", "crime_lag_bin", "day_of_week", "season", "is_holiday"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def conditional_entropy(frame: pd.DataFrame, target: str, states: list[str]) -> tuple[float, float, int, int]:
    joint = frame.groupby(states + [target], observed=True, sort=False).size().to_numpy()
    state = frame.groupby(states, observed=True, sort=False).size().to_numpy()
    n = len(frame)
    plugin = entropy(joint) - entropy(state)
    mm = plugin + (len(joint) - len(state)) / (2.0 * n * LOG_TWO)
    return plugin, mm, len(joint), len(state)


def support_metrics(frame: pd.DataFrame, target: str, baseline: list[str], bike: str) -> dict[str, float]:
    atom_counts = frame.groupby(baseline + [bike, target], observed=True, sort=False).size()
    k_obs = int(len(atom_counts))
    singleton_share = float(atom_counts[atom_counts.eq(1)].sum() / len(frame))
    by_z = frame.groupby(baseline, observed=True, sort=False)
    r = by_z[target].nunique()
    c = by_z[bike].nunique()
    nu = int(((r - 1) * (c - 1)).sum())
    return {
        "n": len(frame), "k_obs": k_obs, "k_obs_over_n": k_obs / len(frame),
        "singleton_observation_share": singleton_share, "nu": nu, "nu_over_n": nu / len(frame),
        "baseline_states": int(by_z.ngroups),
    }


def add_bins(frame: pd.DataFrame, quantiles=(1/3, 2/3), label="bicycle_lag_bin") -> tuple[pd.DataFrame, tuple[float, ...]]:
    training = frame.loc[frame["split"].eq("train") & frame["bike_total_flow_lag1"].notna()]
    cuts = tuple(float(x) for x in training.loc[training["bike_total_flow_lag1"].gt(0), "bike_total_flow_lag1"].quantile(quantiles))
    out = frame.loc[frame["crime_count_lag1"].notna() & frame["bike_total_flow_lag1"].notna()].copy()
    out["crime_lag_bin"] = crime_lag_bins(out["crime_count_lag1"])
    if len(cuts) == 2:
        out[label] = apply_bicycle_bins(out["bike_total_flow_lag1"], cuts)
    else:
        vals = pd.to_numeric(out["bike_total_flow_lag1"])
        names = ["zero"] + [f"q{i+1}" for i in range(len(cuts) + 1)]
        idx = np.where(vals.eq(0), 0, 1 + np.searchsorted(np.asarray(cuts), vals, side="left"))
        out[label] = pd.Categorical.from_codes(idx, categories=names, ordered=True)
    return out, cuts


def load_city(city: str) -> pd.DataFrame:
    files = sorted((PANEL / f"city={city}").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def estimate_spec(frame: pd.DataFrame, baseline: list[str], bike="bicycle_lag_bin") -> dict[str, float]:
    h0p, h0m, _, _ = conditional_entropy(frame, "crime_count_all", baseline)
    hbp, hbm, _, _ = conditional_entropy(frame, "crime_count_all", baseline + [bike])
    metrics = support_metrics(frame, "crime_count_all", baseline, bike)
    metrics.update({
        "h0_plugin_bits": h0p, "hb_plugin_bits": hbp, "cmi_plugin_bits": h0p - hbp,
        "h0_mm_bits": h0m, "hb_mm_bits": hbm, "cmi_mm_bits": h0m - hbm,
        "l0_mse": inverse_entropy_envelope(max(h0m, 1e-12)),
        "lb_mse": inverse_entropy_envelope(max(min(hbm, h0m), 1e-12)),
    })
    metrics["delta_l_mse"] = metrics["l0_mse"] - metrics["lb_mse"]
    return metrics


def matched_randomization(frame: pd.DataFrame, baseline: list[str], bike: str, reps: int, seed: int) -> dict[str, float]:
    ordered = frame.sort_values(["grid_id", "date"]).reset_index(drop=True)
    counts = ordered.groupby("grid_id", observed=True).size()
    if counts.nunique() != 1:
        raise AssertionError("Matched-block calibration requires equal-length grid panels.")
    grids, days = counts.size, int(counts.iloc[0])
    first_grid = ordered["grid_id"].astype(str).iloc[0]
    dates = pd.to_datetime(ordered.loc[ordered["grid_id"].astype(str).eq(first_grid), "date"]).reset_index(drop=True)
    bicycle_codes = pd.Categorical(ordered[bike]).codes.astype(np.int16).reshape(grids, days)
    crime_codes = pd.Categorical(ordered["crime_lag_bin"]).codes.astype(np.int16).reshape(grids, days)
    holidays = ordered["is_holiday"].astype(int).to_numpy(np.int8).reshape(grids, days)
    target = ordered["crime_count_all"].to_numpy(np.int64)
    state = _state_codes(ordered, baseline)
    _, _, observed = conditional_information(target, state, bicycle_codes.ravel())
    rng = np.random.default_rng(seed)
    null = np.empty(reps)
    movable = np.empty(reps)
    cached_groups = []
    total_blocks = 0
    movable_blocks = 0
    for grid in range(grids):
        groups_for_grid, blocks = _block_groups(dates, crime_codes[grid], holidays[grid], 7, 5)
        cached_groups.append(groups_for_grid)
        total_blocks += blocks
        movable_blocks += sum(len(group) for group in groups_for_grid)
    for r in range(reps):
        randomized = bicycle_codes.copy()
        for grid, groups_for_grid in enumerate(cached_groups):
            for group in groups_for_grid:
                shift = int(rng.integers(1, len(group)))
                donors = np.roll(np.asarray(group), shift)
                for recipient, donor in zip(group, donors):
                    r0, d0 = int(recipient) * 7, int(donor) * 7
                    randomized[grid, r0:r0 + 7] = bicycle_codes[grid, d0:d0 + 7]
        movable[r] = movable_blocks / max(1, total_blocks)
        _, _, null[r] = conditional_information(target, state, randomized.ravel())
    return {
        "observed_cmi_bits": observed,
        "null_mean_bits": float(null.mean()),
        "observed_minus_null_bits": float(observed - null.mean()),
        "p_value_plus_one": randomization_p_value(observed, null),
        "movable_block_share": float(movable.mean()),
        "repetitions": reps,
    }


def slack_decomposition(pred: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["city", "grid_id", "date", "day_of_week", "season", "is_holiday"]
    joined = pred.merge(panel[cols], on=["city", "grid_id", "date"], how="left", validate="many_to_one")
    rows = []
    for (city, model, info), g in joined.groupby(["city", "model_family", "information_set"], sort=False):
        g = g.copy()
        g["error"] = g["observed_count"].astype(int) - g["prediction_integer"].astype(int)
        states = ["grid_id", "crime_lag_bin", "day_of_week", "season", "is_holiday"]
        if info == "bicycle_aware": states.append("bicycle_lag_bin")
        he = entropy(g.groupby("error", observed=True).size().to_numpy())
        hf = entropy(g.groupby(states, observed=True).size().to_numpy())
        hef = entropy(g.groupby(states + ["error"], observed=True).size().to_numpy())
        mi = max(0.0, he + hf - hef)
        hcond = max(0.0, hef - hf)
        ep = g.groupby("error", observed=True).size().sort_index()
        errors = ep.index.to_numpy(float)
        p = ep.to_numpy(float) / len(g)
        d = float(np.dot(errors ** 2, p))
        lam = lambda_for_second_moment(d) if d > 1e-15 else math.inf
        if math.isinf(lam):
            env = 0.0; kl = 0.0
        else:
            st = discrete_gaussian_stats(lam)
            env = st.entropy_bits
            logq = (-lam * errors ** 2 - math.log(st.partition)) / LOG_TWO
            kl = max(0.0, float(np.dot(p, np.log2(p) - logq)))
        rows.append({
            "city": city, "model_family": model, "information_set": info,
            "n": len(g), "mse_second_moment": d, "conditional_entropy_bits": hcond,
            "error_entropy_bits": he, "envelope_entropy_bits": env,
            "history_dependence_bits": mi, "shape_mismatch_bits": kl,
            "universal_slack_bits": env - hcond,
            "identity_residual_bits": (env - hcond) - mi - kl,
        })
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest_paths = [E06, E09, E12, E14, E18] + sorted(PANEL.rglob("*.parquet"))
    pd.DataFrame([{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in manifest_paths]).to_csv(TABLES / "input_manifest.csv", index=False)

    primary_rows, lower_rows, month_rows, fine_rows, coverage_rows = [], [], [], [], []
    all_panels = []
    e18 = pd.read_csv(E18).set_index("city")
    e12 = pd.read_csv(E12)
    mm_null_bias = float(e12.loc[e12["estimator"].astype(str).str.contains("Miller", case=False), "mean_null_bias_bits"].iloc[0]) if "mean_null_bias_bits" in e12 else 0.1264
    for ci, city in enumerate(CITY_ORDER):
        raw = load_city(city)
        all_panels.append(raw)
        prepared, cuts = add_bins(raw)
        covered = prepared.loc[prepared["bike_coverage_training"]].copy().reset_index(drop=True)
        primary = estimate_spec(covered, BASE)
        primary.update({"city": city, "city_label": CITY_LABELS[city], "cut_points": json.dumps(cuts), "restricted_null_mean_bits": float(e18.loc[city, "null_mean_bits"]), "restricted_p_value": float(e18.loc[city, "randomization_p_value"]), "simulation_mm_null_bias_bits": mm_null_bias})
        primary_rows.append(primary)

        train_grid = covered.loc[covered["split"].eq("train")].groupby("grid_id")["crime_count_all"].mean()
        rank = train_grid.rank(method="first")
        strata = pd.qcut(rank, 5, labels=[f"crime_q{i}" for i in range(1, 6)])
        covered["spatial_stratum"] = covered["grid_id"].map(strata).astype(str)
        low_base = ["spatial_stratum", "crime_lag_bin", "day_of_week", "season", "is_holiday"]
        low = estimate_spec(covered, low_base)
        low.update(matched_randomization(covered, low_base, "bicycle_lag_bin", 199, 20260922 + ci))
        low.update({"city": city, "city_label": CITY_LABELS[city], "specification": "training-only crime-intensity quintile"})
        lower_rows.append(low)

        month_base = ["grid_id", "crime_lag_bin", "day_of_week", "month", "is_holiday"]
        month = estimate_spec(covered, month_base)
        month.update(matched_randomization(covered, month_base, "bicycle_lag_bin", 199, 20261022 + ci))
        month.update({"city": city, "city_label": CITY_LABELS[city], "specification": "calendar month replaces season"})
        month_rows.append(month)

        fine, fine_cuts = add_bins(raw, quantiles=(.2, .4, .6, .8), label="bicycle_fine_bin")
        fine = fine.loc[fine["bike_coverage_training"]].copy().reset_index(drop=True)
        fr = estimate_spec(fine, BASE, "bicycle_fine_bin")
        fr.update(matched_randomization(fine, BASE, "bicycle_fine_bin", 199, 20261122 + ci))
        fr.update({"city": city, "city_label": CITY_LABELS[city], "cut_points": json.dumps(fine_cuts), "specification": "zero plus positive-flow quintiles"})
        fine_rows.append(fr)

        complete = estimate_spec(prepared.reset_index(drop=True), BASE)
        share = len(covered) / len(prepared)
        implied = share * primary["cmi_mm_bits"]
        transition = prepared.groupby("grid_id").agg(training_covered=("bike_coverage_training", "first"), flow_2022=("bike_total_flow", lambda x: int(x[prepared.loc[x.index, "year"].eq(2022)].sum())))
        new_active = transition.loc[~transition.training_covered & transition.flow_2022.gt(0)]
        coverage_rows.append({"city": city, "covered_n": len(covered), "complete_n": len(prepared), "covered_share": share, "covered_cmi_bits": primary["cmi_mm_bits"], "implied_pooled_cmi_bits": implied, "reported_pooled_cmi_bits": complete["cmi_mm_bits"], "residual_bits": complete["cmi_mm_bits"] - implied, "outside_training_coverage_grids_active_in_2022": len(new_active), "their_2022_flow": int(new_active.flow_2022.sum())})

    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(TABLES / "primary_observed_support.csv", index=False)
    pd.DataFrame(lower_rows).to_csv(TABLES / "lower_cardinality_specification.csv", index=False)
    pd.DataFrame(month_rows).to_csv(TABLES / "month_control_specification.csv", index=False)
    pd.DataFrame(fine_rows).to_csv(TABLES / "finest_binning_matched_reference.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(TABLES / "coverage_dilution_decomposition.csv", index=False)

    panel_all = pd.concat(all_panels, ignore_index=True)
    pred = pd.read_parquet(E06)
    matched_families = {
        "state_mean", "poisson_glm", "negative_binomial_glm",
        "histogram_gradient_boosting_poisson",
    }
    pred = pred.loc[pred["model_family"].isin(matched_families)].copy()
    slack_decomposition(pred, panel_all).to_csv(TABLES / "empirical_two_slack_decomposition.csv", index=False)

    robust = pd.read_csv(E09)
    robust.loc[robust["spec_id"].isin(["primary", "bicycle_bins_median", "bicycle_bins_quintiles"]), ["city", "spec_id", "estimator", "observations", "delta_h_raw_bits", "delta_l_exact_mse", "bicycle_states", "bicycle_joint_atoms", "bicycle_observation_share_in_singleton_states"]].to_csv(TABLES / "bicycle_mapping_magnitudes.csv", index=False)

    fano = pd.read_csv(E14)
    audit = fano.loc[fano["estimator"].eq("miller_madow"), ["city", "target_id", "information_set", "test_observations", "test_unseen_state_observations"]].copy()
    audit["unseen_observation_share"] = audit["test_unseen_state_observations"] / audit["test_observations"]
    audit.to_csv(TABLES / "fano_unseen_observation_audit.csv", index=False)

    checks = [
        {"check": "primary_city_count", "value": len(primary_df), "expected": 3, "pass": len(primary_df) == 3},
        {"check": "primary_n_matches", "value": primary_df["n"].sum(), "expected": 452235, "pass": int(primary_df["n"].sum()) == 452235},
        {"check": "slack_rows", "value": len(pd.read_csv(TABLES / "empirical_two_slack_decomposition.csv")), "expected": 24, "pass": len(pd.read_csv(TABLES / "empirical_two_slack_decomposition.csv")) == 24},
        {"check": "slack_identity_max_abs", "value": float(pd.read_csv(TABLES / "empirical_two_slack_decomposition.csv")["identity_residual_bits"].abs().max()), "expected": 1e-8, "pass": float(pd.read_csv(TABLES / "empirical_two_slack_decomposition.csv")["identity_residual_bits"].abs().max()) < 1e-8},
        {"check": "all_support_ratios_valid", "value": True, "expected": True, "pass": bool(primary_df["k_obs_over_n"].between(0,1).all() and primary_df["singleton_observation_share"].between(0,1).all())},
    ]
    pd.DataFrame(checks).to_csv(TABLES / "acceptance_checks.csv", index=False)
    status = {"status": "complete" if all(x["pass"] for x in checks) else "failed", "checks": checks, "randomization_repetitions": 199, "note": "Routine revision sensitivities use 199 matched surrogates; existing primary restricted analysis retains 999."}
    encoder = lambda value: value.item() if isinstance(value, np.generic) else str(value)
    (OUT / "run_status.json").write_text(json.dumps(status, indent=2, default=encoder), encoding="utf-8")
    print(json.dumps(status, indent=2, default=encoder))


if __name__ == "__main__":
    main()
