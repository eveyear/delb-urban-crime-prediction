"""Recompute the exact covered/outside-footprint CMI mixture (Reviewer 2 M3).

Uses the frozen 2020--2022 grid-day panel and the same Miller--Madow
estimator/lag bins as the revision analysis. No raw data are redistributed.
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "empirical/revision_audits"
sys.path.insert(0, str(ROOT / "empirical/src"))
sys.path.insert(0, str(SOURCE))
from revision_round1_analysis import BASE, CITY_ORDER, add_bins, estimate_spec, load_city  # noqa: E402


def main() -> None:
    rows = []
    for city in CITY_ORDER:
        prepared, _ = add_bins(load_city(city))
        covered = prepared.loc[prepared["bike_coverage_training"]]
        outside = prepared.loc[~prepared["bike_coverage_training"]]
        cmi_all = estimate_spec(prepared, BASE)["cmi_mm_bits"]
        cmi_covered = estimate_spec(covered, BASE)["cmi_mm_bits"]
        cmi_outside = estimate_spec(outside, BASE)["cmi_mm_bits"]
        covered_share = len(covered) / len(prepared)
        covered_component = covered_share * cmi_covered
        outside_component = (1 - covered_share) * cmi_outside
        identity_residual = cmi_all - covered_component - outside_component
        assert abs(identity_residual) < 1e-10, (city, identity_residual)
        rows.append({
            "city": city,
            "complete_n": len(prepared),
            "covered_n": len(covered),
            "outside_n": len(outside),
            "covered_share": covered_share,
            "covered_cmi_bits": cmi_covered,
            "outside_cmi_bits": cmi_outside,
            "covered_component_bits": covered_component,
            "outside_component_bits": outside_component,
            "complete_cmi_bits": cmi_all,
            "mixture_identity_residual_bits": identity_residual,
        })
    output = ROOT / "empirical/runs/reviewer2_m3_coverage_20260924/tables/coverage_components.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
