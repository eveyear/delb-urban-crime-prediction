from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from revision_round1_analysis import BASE, CITY_LABELS, CITY_ORDER, add_bins, estimate_spec, load_city
from entropy_crime_bike.conditional_information import crime_lag_bins


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/revision_round1_analysis/tables"
E04 = ROOT / "runs/20260718_182751_E04_city_information/tables/city_information_estimates.csv"
E09_ROLLING = ROOT / "runs/20260719_183554_E09_robustness/tables/rolling_origin_results.csv"
E00_BIKE = ROOT / "runs/20260718_150723_E00_data_audit/tables/bicycle_file_audit.csv"
E07_GAPS = ROOT / "runs/20260718_204327_E07_theory_practice/tables/city_model_predictability_gaps.csv"


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def lower_cardinality_fdr() -> None:
    routine = pd.read_csv(OUT / "lower_cardinality_specification.csv")
    confirm = pd.read_csv(OUT / "lower_cardinality_ny_999_confirmation.csv")
    ny = confirm.iloc[0]
    routine.loc[routine["city"].eq("NY"), "p_value_plus_one"] = ny["p_value_plus_one"]
    routine.loc[routine["city"].eq("NY"), "repetitions"] = ny["repetitions"]
    routine.loc[routine["city"].eq("NY"), "null_mean_bits"] = ny["null_mean_bits"]
    routine.loc[routine["city"].eq("NY"), "observed_minus_null_bits"] = ny["observed_minus_null_bits"]
    routine["bh_q_value_three_city_family"] = bh_adjust(routine["p_value_plus_one"].to_numpy())
    routine["calibration_resolution_note"] = np.where(
        routine["city"].eq("NY"), "999-replicate confirmation", "199-replicate routine screen"
    )
    routine.to_csv(OUT / "lower_cardinality_fdr_reconciliation.csv", index=False)


def common_topcode() -> None:
    raw_by_city = {city: load_city(city) for city in CITY_ORDER}
    training_counts = pd.concat([
        frame.loc[frame["split"].eq("train") & frame["bike_coverage_training"], "crime_count_all"]
        for frame in raw_by_city.values()
    ], ignore_index=True)
    # A single, pre-test, pooled 99.5th-percentile cap limits cross-city support
    # differences while altering only the extreme upper tail.
    cap = int(math.ceil(float(training_counts.quantile(0.995))))
    rows = []
    for city in CITY_ORDER:
        original, _ = add_bins(raw_by_city[city])
        original = original.loc[original["bike_coverage_training"]].copy()
        topcoded = raw_by_city[city].copy()
        topcoded["crime_count_all"] = topcoded["crime_count_all"].clip(upper=cap)
        topcoded["crime_count_lag1"] = topcoded["crime_count_lag1"].clip(upper=cap)
        topcoded, _ = add_bins(topcoded)
        topcoded = topcoded.loc[topcoded["bike_coverage_training"]].copy()
        topcoded["crime_lag_bin"] = crime_lag_bins(topcoded["crime_count_lag1"])
        before = estimate_spec(original, BASE)
        after = estimate_spec(topcoded, BASE)
        rows.append({
            "city": city,
            "city_label": CITY_LABELS[city],
            "common_cap": cap,
            "cap_rule": "ceiling of pooled training-domain 99.5th percentile",
            "observations": len(topcoded),
            "target_observations_affected": int((original["crime_count_all"] > cap).sum()),
            "target_observation_share_affected": float((original["crime_count_all"] > cap).mean()),
            "original_cmi_mm_bits": before["cmi_mm_bits"],
            "topcoded_cmi_mm_bits": after["cmi_mm_bits"],
            "cmi_change_bits": after["cmi_mm_bits"] - before["cmi_mm_bits"],
            "original_delta_l_mse": before["delta_l_mse"],
            "topcoded_delta_l_mse": after["delta_l_mse"],
            "delta_l_change_mse": after["delta_l_mse"] - before["delta_l_mse"],
        })
    pd.DataFrame(rows).to_csv(OUT / "common_topcode_sensitivity.csv", index=False)


def temporal_audit() -> None:
    estimates = pd.read_csv(E04)
    annual = estimates.loc[
        estimates["domain"].eq("bike_covered_training")
        & estimates["period"].str.startswith("year_"),
        ["city", "city_label", "period", "sample_size", "delta_h_miller_madow_raw_bits",
         "delta_h_miller_madow_raw_bits_ci_low", "delta_h_miller_madow_raw_bits_ci_high",
         "delta_l_exact_mse", "relative_l_reduction"],
    ].copy()
    annual["year"] = annual["period"].str.replace("year_", "", regex=False).astype(int)
    annual["pandemic_period_role"] = annual["year"].map({2020: "pandemic onset", 2021: "pandemic transition", 2022: "later holdout year"})
    annual.to_csv(OUT / "pandemic_period_annual_sensitivity.csv", index=False)

    rolling = pd.read_csv(E09_ROLLING)
    rolling["origin"] = pd.to_datetime(rolling["origin"])
    quarterly = (
        rolling.assign(quarter=rolling["origin"].dt.to_period("Q").astype(str))
        .groupby(["city", "model_family", "quarter"], as_index=False)
        .agg(origins=("origin", "nunique"), mean_delta_mse=("delta_mse", "mean"),
             positive_origin_share=("delta_mse", lambda x: float((x > 0).mean())))
    )
    quarterly.to_csv(OUT / "rolling_2022_quarterly_sensitivity.csv", index=False)


def duplicate_audit() -> None:
    source = pd.read_csv(E00_BIKE)
    rows = []
    for city, frame in source.groupby("city", sort=False):
        total = int(frame["row_count"].sum())
        repeated = int(frame["repeated_full_row_signature_rows"].fillna(0).sum())
        rows.append({
            "city": city,
            "city_label": CITY_LABELS[city],
            "source_rows": total,
            "repeated_complete_signature_rows": repeated,
            "repeated_signature_share": repeated / total,
            "duplicate_official_ride_id_rows": int(frame["duplicate_source_ride_id_rows"].fillna(0).sum()),
            "maximum_endpoint_contribution_if_all_repeats_removed": 2 * repeated,
            "decision": (
                "retain: no official unique ride identifier; identical recorded attributes can represent distinct trips"
                if city == "VAN" else "no repeated complete signatures detected"
            ),
        })
    pd.DataFrame(rows).to_csv(OUT / "bicycle_duplicate_sensitivity_audit.csv", index=False)


def weather_feasibility() -> None:
    panels = {city: load_city(city) for city in CITY_ORDER}
    shared = set.intersection(*(set(frame.columns) for frame in panels.values()))
    weather_tokens = ("weather", "temperature", "temp_", "rain_", "rainfall", "snow", "precip", "wind", "humidity")
    weather_columns = sorted(c for c in shared if any(t in c.lower() for t in weather_tokens))
    pd.DataFrame([{
        "audit_scope": "accepted harmonized panel",
        "cities": ",".join(CITY_ORDER),
        "shared_columns": len(shared),
        "shared_weather_columns": json.dumps(weather_columns),
        "feasible_without_new_external_ingestion": bool(weather_columns),
        "resolution": (
            "run common weather-control sensitivity" if weather_columns
            else "state limitation; comparable forecast-time weather was not ingested in the frozen cross-city pipeline"
        ),
    }]).to_csv(OUT / "weather_control_feasibility.csv", index=False)


def evidence_reconciliations() -> None:
    pd.DataFrame([
        {
            "analysis": "Sparse-state known-truth estimator calibration",
            "estimand": "finite-sample CMI bias and rejection behavior under known truth",
            "reference": "simulation truth",
            "sample_scope": "prespecified synthetic sparse-state cells",
            "role": "estimator behavior; not empirical city significance",
            "source": "20260720_115049_S7_synthesis/e12_summary.csv",
        },
        {
            "analysis": "Primary empirical restricted randomization",
            "estimand": "observed city CMI relative to a baseline-state-matched reference",
            "reference": "999 matched 7-day surrogates",
            "sample_scope": "city-specific frozen bicycle-covered panels",
            "role": "primary empirical calibration",
            "source": "20260728_125347_E18_formal/tables/empirical_summary.csv",
        },
        {
            "analysis": "Truth-referenced scale-sample power",
            "estimand": "power and minimum detectable injected CMI",
            "reference": "known injected CMI with 199 surrogates; 999 at confirmation cells",
            "sample_scope": "500 m-day, 1 km-day, and 2 km-week designs; N=90,150,365,730",
            "role": "design resolution; not a re-test of the observed city estimates",
            "source": "20260728_182340_E19_MVP_stage5_mdcmi/minimum_detectable_cmi.csv",
        },
        {
            "analysis": "Lower-cardinality revision sensitivity",
            "estimand": "city CMI with training-only crime-intensity spatial strata",
            "reference": "199 matched surrogates; 999 for the key New York confirmation",
            "sample_scope": "same frozen city panels as the primary empirical analysis",
            "role": "state-complexity sensitivity",
            "source": "revision_round1_analysis/tables/lower_cardinality_fdr_reconciliation.csv",
        },
    ]).to_csv(OUT / "inference_design_reconciliation.csv", index=False)

    gaps = pd.read_csv(E07_GAPS)
    cols = ["city", "city_label", "model_family", "observations", "test_days", "test_grids",
            "l0_exact_mse", "lb_exact_mse", "delta_l_exact_mse", "mse_baseline_integer",
            "mse_bicycle_integer", "delta_mse_integer", "gap_baseline", "gap_bicycle", "gap_narrowing"]
    out = gaps[cols].copy()
    out["delta_l_scope"] = "held-out 2022 test window: 413 grids x 184 days = 75,992 observations across cities"
    out["interpretation"] = "bound shift and realized MSE shift are distinct quantities on the same matched test support"
    out.to_csv(OUT / "delta_l_scope_reconciliation.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lower_cardinality_fdr()
    common_topcode()
    temporal_audit()
    duplicate_audit()
    weather_feasibility()
    evidence_reconciliations()
    status = {
        "status": "complete",
        "outputs": [
            "lower_cardinality_fdr_reconciliation.csv",
            "common_topcode_sensitivity.csv",
            "pandemic_period_annual_sensitivity.csv",
            "rolling_2022_quarterly_sensitivity.csv",
            "bicycle_duplicate_sensitivity_audit.csv",
            "weather_control_feasibility.csv",
            "inference_design_reconciliation.csv",
            "delta_l_scope_reconciliation.csv",
        ],
    }
    (ROOT / "runs/revision_round1_analysis/secondary_audit_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
