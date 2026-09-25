"""Rebuild revised charts from accepted, read-only experiment summaries.

Example::

    PYTHONPATH=empirical/src python empirical/rebuild_plot_typography.py \
      --runs-root /path/to/accepted/empirical/runs

This changes text presentation only; it does not rerun experiments or alter
statistical input tables. Figure 1--3 are handled by their data-free scripts.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from entropy_crime_bike.e19_figures import (  # noqa: E402
    _convergence_plot,
    _heatmap_plot,
    _power_mdcmi_plot,
    _support_plot,
)
from entropy_crime_bike.e12_cmi_simulation import _plot_main, _plot_supplement  # noqa: E402
from entropy_crime_bike.e04_city_information import _plot_domain_sensitivity  # noqa: E402
from entropy_crime_bike.e17_estimator_domain_sensitivity import _plot_estimator_sensitivity  # noqa: E402
from entropy_crime_bike.e10_placebo import _make_figures as plot_e10  # noqa: E402
from entropy_crime_bike.e09_robustness import _make_figures as plot_e09  # noqa: E402
from entropy_crime_bike.e08_crime_type_heterogeneity import _plot_theory_metric  # noqa: E402
from entropy_crime_bike.e07_theory_practice import _plot_gap_narrowing  # noqa: E402
from entropy_crime_bike.e13_support_null import _plot_results as plot_e13  # noqa: E402
from entropy_crime_bike.e14_fano_validation import _plot_results as plot_e14  # noqa: E402
from entropy_crime_bike.report_e02 import _plot_monthly  # noqa: E402
from entropy_crime_bike.e16_scale_normalization import (  # noqa: E402
    _plot_full_sample_heatmaps,
    _plot_stability,
)
from entropy_crime_bike.e16_randomization_calibration import _make_figures as plot_e16_randomization  # noqa: E402
from entropy_crime_bike.manuscript_six_item_figures import _combined_city_figure  # noqa: E402
from entropy_crime_bike.manuscript_figure_layout import (  # noqa: E402
    _plot_figure_3,
    _plot_figure_10,
    _publication_style,
)
from entropy_crime_bike.revision_figure6_readability import redraw as redraw_figure6  # noqa: E402
from entropy_crime_bike.revision_figure9_readability import redraw as redraw_figure9  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE.parent / "figures")
    args = parser.parse_args()
    runs = args.runs_root.resolve()
    figures = args.output_root.resolve()
    e11, e17, e19, s7, e16 = (figures / folder for folder in ("e11", "e17", "e19", "s7", "e16"))
    for folder in (e11, e17, e19, s7, e16):
        folder.mkdir(parents=True, exist_ok=True)

    e04 = runs / "20260718_182751_E04_city_information" / "tables"
    estimates = pd.read_csv(e04 / "city_information_estimates.csv")
    _combined_city_figure(estimates, e17, 300)
    _plot_domain_sensitivity(estimates, e17 / "e04_domain_sensitivity.pdf", 300)
    e17_results = runs / "20260727_144319_E17_estimator_domain_sensitivity" / "tables"
    _plot_estimator_sensitivity(
        pd.read_csv(e17_results / "estimator_null_summary.csv"),
        e17 / "e17_estimator_null_sensitivity.pdf", 300,
    )
    e12 = runs / "20260719_210746_E12_sparse_cmi" / "tables"
    estimator_summary = pd.read_csv(e12 / "estimator_summary.csv")
    randomization_datasets = pd.read_parquet(e12 / "randomization_dataset_summary.parquet")
    randomization_summary = pd.read_csv(e12 / "randomization_summary.csv")
    _plot_main(
        estimator_summary,
        randomization_datasets,
        randomization_summary,
        s7 / "e12_sparse_cmi_main",
        300,
    )
    _plot_supplement(estimator_summary, s7 / "e12_bias_rmse_supplement", 300)
    e02 = runs / "20260718_172231_E02_spatial_temporal_panel" / "tables"
    e05 = runs / "20260718_190356_E05_local_spatial_information" / "tables"
    spatial = pd.read_csv(e02 / "spatial_grid_totals.csv")
    eligibility = pd.read_csv(e05 / "grid_eligibility.csv")
    local_estimates = pd.read_csv(e05 / "local_information_estimates.csv")
    catalog = spatial[["city", "grid_id", "x_index", "y_index"]].drop_duplicates()
    _publication_style()
    _plot_figure_3(spatial, e11)
    _plot_figure_10(catalog, eligibility, local_estimates, e11)
    _plot_monthly(pd.read_csv(e02 / "monthly_patterns.csv"), e11 / "e02_monthly_patterns.png")
    e08 = runs / "20260718_220503_E08_crime_type_heterogeneity" / "tables"
    _plot_theory_metric(
        pd.read_csv(e08 / "crime_type_information_bounds.csv"),
        metric="delta_l_exact_mse", low="delta_l_normal_ci_low",
        high="delta_l_normal_ci_high", ylabel="Exact DELB reduction (MSE)",
        title="Crime-type heterogeneity in theoretical error-floor reduction",
        path=e11 / "e08_crime_type_delb_reduction.png", dpi=300,
    )
    e07 = runs / "20260718_204327_E07_theory_practice" / "tables"
    _plot_gap_narrowing(
        pd.read_csv(e07 / "city_model_predictability_gaps.csv"),
        e11 / "e07_gap_narrowing.png", 300,
    )
    e13 = runs / "20260720_102832_E13_support_null" / "tables"
    _plot_results_e13 = plot_e13
    _plot_results_e13(
        pd.read_csv(e13 / "local_support_summary.csv"),
        pd.read_csv(e13 / "city_support_summary.csv"),
        runs / "20260719_210746_E12_sparse_cmi", s7, 300,
    )
    e14 = runs / "20260720_111726_E14_fano_validation" / "tables"
    plot_e14(
        pd.read_csv(e14 / "fano_oracle_results.csv"),
        pd.read_csv(e14 / "fano_city_results.csv"),
        pd.read_csv(e14 / "fano_mobility_contrasts.csv"),
        s7, 300,
    )
    with tempfile.TemporaryDirectory(prefix="asb_legacy_figures_") as scratch:
        scratch_root = Path(scratch)
        (scratch_root / "figures").mkdir()
        e10 = runs / "20260719_190015_E10_placebo_null" / "tables"
        plot_e10(
            pd.read_parquet(e10 / "city_null_replicates.parquet"),
            pd.read_csv(e10 / "city_null_summary.csv"),
            pd.read_csv(e10 / "local_null_summary.csv"),
            pd.read_csv(e10 / "distant_lag_city.csv"),
            scratch_root, 300,
        )
        for source in (scratch_root / "figures").glob("e10_distant_lags.*"):
            shutil.copy2(source, e11 / source.name)
        e09 = runs / "20260719_183554_E09_robustness" / "tables"
        plot_e09(
            pd.read_csv(e09 / "theory_robustness_matrix.csv"),
            pd.read_csv(e09 / "block_length_sensitivity.csv"),
            pd.read_csv(e09 / "rolling_origin_results.csv"),
            scratch_root, 300,
        )
        for source in (scratch_root / "figures").glob("e09_rolling_origins.*"):
            shutil.copy2(source, e11 / source.name)
    e16_scale = runs / "20260723_102411_E16_S16_5_scale_normalization" / "tables"
    full_scale = pd.read_csv(e16_scale / "full_sample_scale_comparison.csv")
    stability = pd.read_csv(e16_scale / "sample_stability_summary.csv")
    _plot_full_sample_heatmaps(full_scale, e16, 300)
    _plot_stability(stability, e16, 300)
    e16_null = runs / "20260723_104215_E16_S16_6_randomization" / "tables"
    with tempfile.TemporaryDirectory(prefix="asb_e16_figures_") as scratch:
        scratch_root = Path(scratch)
        (scratch_root / "figures").mkdir()
        plot_e16_randomization(pd.read_csv(e16_null / "randomization_summary.csv"), scratch_root, 300)
        for source in (scratch_root / "figures").iterdir():
            shutil.copy2(source, e16 / source.name)
    redraw_figure6(
        runs / "20260719_190015_E10_placebo_null" / "tables",
        e11 / "e10_city_null_distributions.pdf",
    )
    redraw_figure9(
        runs / "20260718_193815_E06_predictive_models" / "tables" / "paired_mse_contrasts.csv",
        e11 / "e06_mse_improvement.pdf",
    )
    support = pd.read_csv(runs / "20260728_173641_E19_MVP_stage2_population" / "support_complexity.csv")
    convergence = pd.read_csv(runs / "20260728_174145_E19_MVP_stage3_monte_carlo" / "convergence_summary.csv")
    power = pd.read_csv(runs / "20260728_175658_E19_MVP_stage4_power" / "power_summary.csv")
    mdcmi = pd.read_csv(runs / "20260728_182340_E19_MVP_stage5_mdcmi" / "minimum_detectable_cmi.csv")
    _convergence_plot(convergence, e19, dpi=300)
    _power_mdcmi_plot(power, mdcmi, e19, dpi=300)
    _support_plot(support, power, e19, dpi=300)
    _heatmap_plot(convergence, e19, dpi=300)
    print("Rebuilt Figure 4--9 and Figure S1--S20 from accepted summaries.")


if __name__ == "__main__":
    main()
