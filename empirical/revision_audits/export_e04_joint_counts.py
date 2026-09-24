"""Export anonymous E04 joint-state frequencies and verify all 24 point estimates.

This exports no dates, grid identifiers, coordinates, or event-level records.
Baseline-state IDs are arbitrary and independently assigned within each
city/domain/period specification. The output reproduces E04 point estimates,
not bootstrap intervals or randomization tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from entropy_crime_bike.e04_city_information import (  # noqa: E402
    CITY_ORDER, _add_bound_metrics, _domain_mask, _fit_city_bins,
    _load_city_panel, _period_mask, _prepare_city,
)


def _conditional_entropy(joint: np.ndarray, state: np.ndarray) -> float:
    n = float(joint.sum())
    j = joint[joint > 0].astype(float)
    s = state[state > 0].astype(float)
    return float(-(j * np.log2(j / n)).sum() / n + (s * np.log2(s / n)).sum() / n)


def _verify_counts(table: pd.DataFrame, config: dict, expected: pd.Series) -> dict:
    base = table.groupby(["baseline_state_id", "target_count"], sort=False)["frequency"].sum()
    base_state = table.groupby("baseline_state_id", sort=False)["frequency"].sum()
    aug = table["frequency"].to_numpy(dtype=np.int64)
    aug_state = table.groupby(["baseline_state_id", "bicycle_bin"], sort=False)["frequency"].sum()
    n = int(aug.sum())
    h0_plugin = _conditional_entropy(base.to_numpy(), base_state.to_numpy())
    hb_plugin = _conditional_entropy(aug, aug_state.to_numpy())
    h0_mm = h0_plugin + (len(base) - len(base_state)) / (2 * n * np.log(2))
    hb_mm = hb_plugin + (len(aug) - len(aug_state)) / (2 * n * np.log(2))
    point = pd.DataFrame({
        "h0_miller_madow_bits": [h0_mm],
        "hb_miller_madow_bits": [hb_mm],
        "delta_h_miller_madow_raw_bits": [h0_mm - hb_mm],
    })
    bounds = _add_bound_metrics(point, config).iloc[0]
    checks = {
        "sample_size": float(n),
        "h0_plugin_bits": h0_plugin,
        "hb_plugin_bits": hb_plugin,
        "h0_miller_madow_bits": h0_mm,
        "hb_miller_madow_bits": hb_mm,
        "delta_h_miller_madow_raw_bits": h0_mm - hb_mm,
        "l0_exact_mse": float(bounds["l0_exact_mse"]),
        "lb_exact_mse": float(bounds["lb_exact_mse"]),
        "delta_l_exact_mse": float(bounds["delta_l_exact_mse"]),
    }
    for metric, value in checks.items():
        if not np.isclose(value, float(expected[metric]), atol=1e-9, rtol=0):
            raise AssertionError(f"{expected['spec_id']} {metric}: {value} != {expected[metric]}")
    return {
        "spec_id": expected["spec_id"],
        "observations": n,
        "joint_rows": len(table),
        "singleton_joint_rows": int(table["frequency"].eq(1).sum()),
        **checks,
        "max_absolute_difference": max(abs(value - float(expected[key])) for key, value in checks.items()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e02-dir", type=Path, default=REPO / "processed_data/e02/20260718_172231_E02_spatial_temporal_panel")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((REPO / "config/e04.yaml").read_text(encoding="utf-8"))
    estimates = pd.read_csv(REPO / "runs/20260718_182751_E04_city_information/tables/city_information_estimates.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = {}
    for city in CITY_ORDER:
        frame = _load_city_panel(args.e02_dir, city, str(config["target"]))
        cuts, _ = _fit_city_bins(frame, city, config)
        prepared, _ = _prepare_city(frame, cuts, config)
        for domain in config["domains"]:
            for period in config["periods"]:
                spec_id = f"{city}__{domain}__{period}"
                subset = prepared.loc[_domain_mask(prepared, domain) & _period_mask(prepared, period)]
                baseline = list(config["information_sets"]["baseline_state"])
                state_index = pd.MultiIndex.from_frame(subset[baseline].astype(object))
                codes, unique_states = pd.factorize(state_index, sort=False)
                # An independent, non-released random relabeling prevents the
                # arbitrary IDs from reflecting the panel's source-row order.
                labels = list(range(len(unique_states)))
                secrets.SystemRandom().shuffle(labels)
                anonymous_codes = np.asarray(labels, dtype=np.int32)[codes]
                count_frame = pd.DataFrame({
                    "baseline_state_id": anonymous_codes,
                    "bicycle_bin": subset["bicycle_lag_bin"].astype(str).to_numpy(),
                    "target_count": subset[str(config["target"])].astype(np.int32).to_numpy(),
                })
                table = (count_frame.value_counts(sort=False).rename("frequency")
                         .reset_index().sort_values(["baseline_state_id", "bicycle_bin", "target_count"]))
                expected = estimates.loc[estimates["spec_id"].eq(spec_id)]
                if len(expected) != 1:
                    raise AssertionError(f"Expected one E04 result for {spec_id}")
                rows.append(_verify_counts(table, config, expected.iloc[0]))
                path = args.output_dir / f"{spec_id}.csv.gz"
                table.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                print(f"{spec_id}: {len(table)} joint rows, n={len(subset)}", flush=True)
    if len(rows) != 24:
        raise AssertionError(f"Expected 24 specifications; got {len(rows)}")
    pd.DataFrame(rows).to_csv(args.output_dir / "validation_summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {len(rows)} specifications; max difference={max(x['max_absolute_difference'] for x in rows):.3g}")


if __name__ == "__main__":
    main()
