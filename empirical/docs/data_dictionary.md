# Data Dictionary

## Standardized crime event

| Field | Type | Meaning |
|---|---|---|
| event_id | string | Deterministic event identifier, stable across reproducible reruns |
| source_event_id | string | Event identifier supplied by the source, when present |
| city | category | DC, NY, or VAN |
| source_year | integer | Year encoded in the source filename |
| event_datetime_local | datetime | Local event datetime |
| event_date | date | Local event calendar date |
| longitude | float | WGS84 longitude |
| latitude | float | WGS84 latitude |
| crime_type_raw | string | Original source category retained for provenance |
| crime_type_unified | category | Authoritative harmonized category |
| source_file | string | Absolute raw source path |
| source_file_sha256 | string | SHA-256 hash of the raw source file |
| source_row_number | integer | One-based physical source row, including the header offset |
| qc_flags | string | Pipe-delimited machine-readable quality-control flags |

## Standardized bicycle trip

| Field | Type | Meaning |
|---|---|---|
| trip_record_id | string | Deterministic record identifier, stable across reproducible reruns |
| source_ride_id | string | Formal ride identifier supplied by the source, when present |
| city | category | DC, NY, or VAN |
| source_month | string | YYYY-MM encoded in the source filename |
| started_at_local | datetime | Local trip start |
| ended_at_local | datetime | Local trip end |
| start_date | date | Local trip-start calendar date |
| end_date | date | Local trip-end calendar date, when available |
| start_station_id | string | Source station identifier when available |
| start_station_name | string | Source station name |
| start_longitude | float | Resolved WGS84 start longitude |
| start_latitude | float | Resolved WGS84 start latitude |
| start_coordinate_source | category | `source`, historical station lookup, `current_snapshot_retrofit`, `unresolved`, or `non_public_node` |
| start_flow_eligible | boolean | Whether the start endpoint may enter public spatial-flow aggregation |
| end_station_id | string | Source station identifier when available |
| end_station_name | string | Source station name |
| end_longitude | float | Resolved WGS84 end longitude |
| end_latitude | float | Resolved WGS84 end latitude |
| end_coordinate_source | category | Provenance class corresponding to the resolved end coordinates |
| end_flow_eligible | boolean | Whether the end endpoint may enter public spatial-flow aggregation |
| rideable_type | string | Bicycle type when available |
| member_type | string | Membership type when available |
| duration_seconds | float | Trip duration in seconds |
| source_file | string | Absolute raw source path |
| archive_member | string | CSV member name for an archived source, otherwise empty |
| source_file_sha256 | string | SHA-256 hash of the raw source file |
| source_row_number | integer | One-based physical row within the source CSV or archive member |
| qc_flags | string | Pipe-delimited flags, including imputation and duration diagnostics |

## Grid-day panel

The accepted E02 panel contains one row for every primary 1 km grid and local
calendar day from 2020-01-01 through 2022-12-31. The primary grid domain is
fixed using only 2020-2021 training-period crime locations.

| Field | Type | Meaning |
|---|---|---|
| city | category | DC, NY, or VAN |
| grid_id | string | Deterministic projected-grid identifier |
| x_index, y_index | integer | Grid indices anchored at the projected CRS origin |
| centroid_longitude, centroid_latitude | float | WGS84 grid-centroid coordinates |
| bike_coverage_training | boolean | At least one eligible bicycle endpoint in the grid during 2020-2021 |
| bike_coverage_ever | boolean | At least one eligible endpoint during 2020-2022 |
| date | date | Local calendar day |
| year, month, day_of_week, day_of_year, iso_week | integer | Calendar fields |
| is_weekend, is_holiday | boolean | Prespecified calendar controls |
| season | category | Winter, spring, summer, or autumn |
| split | category | Train, validation, or test |
| crime_count_property_theft | integer | Daily PROPERTY_THEFT count |
| crime_count_vehicle_theft | integer | Daily VEHICLE_THEFT count |
| crime_count_burglary | integer | Daily BURGLARY count |
| crime_count_all | integer | Sum of the three authoritative crime categories |
| bike_outflow | integer | Eligible trip starts in the grid on the day |
| bike_inflow | integer | Eligible trip ends in the grid on the day |
| bike_total_flow | integer | `bike_outflow + bike_inflow` |
| bike_net_flow | integer | `bike_inflow - bike_outflow` |
| bike_any_flow | boolean | Whether total observed endpoint flow is positive |
| crime_count_lag1 | nullable integer | Previous local-calendar day's crime count in the same grid |
| bike_total_flow_lag1 | nullable integer | Previous day's total bicycle flow in the same grid |

Zero-count grid-days are explicit rows, not missing observations. However, a
zero bicycle flow outside the observed system footprint does not imply zero
total human mobility; later experiments must use the coverage flags.

## E13 surrogate-support records

E13 contains one record for every E10 city or eligible city-grid surrogate.
All support fields are recomputed after randomization.

| Field | Type | Meaning |
|---|---|---|
| city | category | DC, NY, or VAN |
| grid_id | string | Grid identifier; `__CITY__` for pooled city records |
| null_design | category | Temporal block, spatial series, or circular-shift transformation |
| replicate | integer | Original E10 randomization replicate, 1--999 |
| observations | integer | Number of observations entering the statistic |
| support_atoms | integer | \(K_{\mathrm{obs}}=\#\{(C,Z,B)\}\) |
| support_atoms_per_observation | float | \(K_{\mathrm{obs}}/N\) |
| singleton_atom_count | integer | Number of observed atoms with frequency one |
| singleton_observation_share | float | Share of observations belonging to singleton atoms |
| effective_df | integer | \(\nu=\sum_z(r_z-1)(c_z-1)\) over observed baseline states |
| effective_df_per_observation | float | \(\nu/N\) |
| delta_h_raw_bits | float | Recomputed Miller--Madow CMI for the surrogate |
| e10_delta_h_raw_bits | float | Archived E10 value used for reconciliation |
| cmi_reconciliation_difference_bits | float | E13 minus E10 CMI |

## E14 Fano validation records

E14 uses the E02 integer count target without changing the DELB estimand. It
additionally constructs two prespecified finite-label targets for classification
loss: `binary_occurrence` is \(1\{C>0\}\), and `three_level_count` maps counts
to 0, 1, and 2 for \(C\geq2\).

| Field | Type | Meaning |
|---|---|---|
| city | category | DC, NY, or VAN |
| target_id | category | `binary_occurrence` or `three_level_count` |
| classes | integer | Fixed target cardinality, 2 or 3 |
| information_set | category | `baseline` or `bicycle_aware` |
| estimator | category | `plugin` or `miller_madow` |
| pretest_observations | integer | Train-plus-validation observations used to estimate states and entropy |
| test_observations | integer | Chronological July--December 2022 test observations |
| conditional_entropy_raw_bits | float | Estimated \(H(Y\mid\mathcal F)\), before inverse transformation |
| conditional_entropy_for_bound_bits | float | Entropy clipped only to the mathematical domain \([0,\log_2K]\) |
| fano_error_floor | float | \(\phi_K^{-1}(H)\) on \([0,1-1/K]\) |
| test_modal_error | float | Held-out Hamming error of the pretest statewise modal classifier |
| test_unseen_state_observations | integer | Test observations using the pretest global-mode fallback |
| estimated_bound_violation | boolean | Whether held-out error is below the estimated pretest floor; retained as a diagnostic |
| bootstrap_*_low, bootstrap_*_high | float | Centered seven-day block-bootstrap interval endpoints |
| bootstrap_*_raw_mean | float | Uncentered bootstrap mean retained to diagnose support loss |

The oracle table additionally reports exact probability-model entropy,
Bayes error, Fano floor, and `oracle_inequality_holds`. The contrast table
defines every reduction as baseline minus bicycle-aware. Hence a positive
Fano-floor reduction means that the richer information set lowers the
model-independent classification error floor; a positive test-error reduction
means that the fitted modal classifier improves.

## E15 concept-element records

E15 does not create empirical observations. Its `concept_elements.csv` is a
machine-readable record of the mathematical labels rendered into the formal
diagram.

| Field | Type | Meaning |
|---|---|---|
| key | category | `estimand`, `estimator`, or `null` |
| stage | integer | Display order from population quantity to randomization reference |
| title | string | Human-readable layer title |
| formula | string | Matplotlib mathtext expression rendered in the layer |
| line_1 | string | Primary interpretation or identity |
| line_2 | string | Layer-specific diagnostic meaning |
| line_3 | string | Boundary, sparsity, or recomputation statement |

These records are figure source metadata, not statistical results.

## S7 manuscript-synthesis records

S7 creates no new estimand or observation. It is a read-only publication
layer over accepted E12--E15 outputs.

| Artifact | Meaning |
|---|---|
| `e12_summary.csv` | Estimator-level null bias, randomization-null center, size, and power copied from accepted E12 summaries |
| `e13_summary.csv` | Support/null correlations and fixed-effect coefficient summaries copied from E13 |
| `e14_summary.csv` | Fano oracle counts, city-target contrasts, estimated violations, and test-error directions copied from E14 |
| `figure_manifest.csv` | Source path, destination path, and SHA-256 for each article PDF |
| `sections/generated/s7/*.tex` | Publication macros and tables generated from the three summaries |
| `manifest.json` | Accepted source runs, outputs, checks, environment, and elapsed time |

S7 files are presentation artifacts. Statistical changes must be made in the
upstream E12--E15 code and rerun; generated TeX or copied figures must not be
edited manually.

## E16 S16.2 multiscale panel records

S16.2 creates a geometrically nested common-core panel at 500 m, 1 km, and
2 km and at daily and complete ISO-week resolutions. All 1 km cells in a
retained 2 km parent must be present in the accepted E02 reference domain;
each retained 1 km cell is then divided into four explicit 500 m children.
Native-domain E02/E09 panels are referenced in
`native_domain_source_inventory.csv` and are not duplicated.

| Field | Type | Meaning |
|---|---|---|
| city | category | DC, NY, or VAN |
| grid_id | string | Resolution-specific deterministic projected-grid identifier |
| resolution_m | integer | 500, 1000, or 2000 m |
| nominal_area_km2 | float | 0.25, 1, or 4 km² for complete common-core cells |
| temporal_resolution | category | `day` or `week` |
| duration_days | integer | 1 or 7 |
| date | date | Local day or ISO-Monday period start |
| period_end | date | Sunday period end; present for weekly rows |
| days_in_period | integer | Seven for every retained weekly row |
| x_index, y_index | integer | Projected-grid indices at the stated resolution |
| split | category | Train, validation, or test; weekly assignment uses period end |
| is_holiday | boolean | Daily holiday indicator |
| contains_holiday | boolean | Whether a complete week contains at least one holiday |
| crime_count_* | integer | Counts summed inside the cell and period |
| bike_outflow, bike_inflow | integer | Eligible bicycle endpoints inside the cell and period |
| bike_total_flow | integer | Inflow plus outflow |
| bike_net_flow | integer | Inflow minus outflow; additive across cells and time |
| bike_coverage_training | boolean | Positive observed endpoint flow in the training period at this scale |
| crime_count_lag1 | nullable integer | Previous complete aggregation period's crime count |
| bike_total_flow_lag1 | nullable integer | Previous complete aggregation period's endpoint flow |

The hierarchy table contains `parent_2km_grid_id`, `parent_1km_grid_id`, and
`grid_500m_id`; every retained 2 km parent has exactly 4 one-kilometre and 16
five-hundred-metre descendants. Weekly panels exclude 1--5 January 2020 and
26--31 December 2022 because those dates do not form complete Monday--Sunday
periods within the study window. Consequently, daily and weekly totals are
not expected to be equal, but totals must be identical across all three
spatial resolutions within a temporal resolution.

S16.2 does not estimate entropy, CMI, DELB, randomization distributions, or
prediction models.

## E16 S16.3 sample-index records

S16.3 stores time-block membership rather than duplicating panel rows. A
downstream analysis joins `period_to_primary_block.parquet` to an S16.2 panel
by city, temporal resolution, and period start, then joins selected `block_id`
values from `primary_block_membership.parquet`.

| Field | Type | Meaning |
|---|---|---|
| city | category | DC, NY, or VAN |
| temporal_resolution | category | `day` or `week` |
| block_id | string | Seven-day (`D7`) or four-week (`W4`) block identifier |
| block_start, block_end | date | Inclusive block boundaries |
| periods_in_block | integer | 7 daily periods or 4 weekly periods |
| stratum_year | integer | Calendar year of the block midpoint |
| stratum_season | category | Season of the block midpoint |
| replicate | integer | 1--50 for thinning; 0 for the unique 100% universe |
| fraction | float | Target fraction: 0.2, 0.4, 0.6, 0.8, or 1.0 |
| selection_rank | integer | Position in the balanced randomized ordering |
| selected_blocks | integer | Number of complete blocks selected |
| realized_block_fraction | float | Selected divided by available blocks |
| selected_periods | integer | Selected blocks times periods per block |
| nominal_panel_rows | integer | Selected periods times common-core grids |

The daily primary universe contains 156 complete sequential seven-day blocks
and leaves four terminal dates outside the thinning universe. The weekly
universe contains 38 complete four-week blocks and leaves three terminal weeks
outside. Residual periods remain in S16.2 and the expanding-window design.

`expanding_window_index.csv` contains five nested chronological windows for
each city, spatial resolution, and temporal resolution. `sample_seeds.csv`
freezes all 300 city-by-temporal-by-replicate seeds. All three spatial
resolutions use identical selected periods within a city, temporal resolution,
replicate, and fraction. These indices contain no entropy, CMI, DELB,
randomization, or prediction results.

## E16 S16.4 entropy and DELB records

`primary_delb_estimates.parquet/csv` contains one row per estimator for every
fixed-coverage sample specification. `expanding_window_delb_estimates.csv`
uses the same result schema for chronological windows. Baseline and
bicycle-aware estimates always use the same observations.

| Field | Type | Meaning |
|---|---|---|
| design | category | Fixed block thinning, expanding window, or full-sample bootstrap |
| city | category | DC, NY, or VAN |
| resolution_m | integer | 500, 1000, or 2000 m |
| temporal_resolution | category | `day` or `week` |
| replicate | integer | Thinning or bootstrap replicate; 0 for deterministic full sample/window |
| fraction | float | Target sample fraction |
| estimator | category | `plugin` or `miller_madow` |
| observations | integer | Matched eligible grid-period observations |
| h0_raw_bits | float | Estimated baseline conditional entropy |
| hb_raw_bits | float | Estimated bicycle-aware conditional entropy |
| delta_h_raw_bits | float | Raw estimated CMI, `h0_raw_bits-hb_raw_bits` |
| h0_for_bound_bits, hb_for_bound_bits | float | Nonnegative ordered entropy inputs used only for bound inversion |
| projection_applied | boolean | Whether empirical entropy ordering required bound-only projection |
| l0_exact_mse, lb_exact_mse | float | Exact integer-lattice baseline and bicycle-aware DELBs |
| delta_l_exact_mse | float | Exact baseline-minus-bicycle DELB difference |
| l0_closed_mse, lb_closed_mse | float | Conservative closed-form lattice bounds |
| support_atoms | integer | Observed `(C,Z,B)` atoms |
| singleton_observation_share | float | Observations belonging to atoms appearing once |
| effective_df | integer | Sum of statewise conditional-table degrees of freedom |
| effective_df_per_observation | float | Effective degrees of freedom divided by observations |

The fixed-coverage design uses bicycle cut points fitted once from the full
2020--2021 training reference within each city and scale. Expanding windows
refit cut points using only observations available by the window endpoint.
`full_sample_block_bootstrap.parquet` contains 1,000 paired 28-day block
replicates for each city, spatial scale, time scale, and estimator. It does not
contain randomization-null surrogates.

## E16 S16.5 scale-normalized bounds and stability records

`primary_normalized_estimates.parquet/csv` preserves every S16.4 point estimate
and adds deterministic area--time normalization and sample-size stability
fields. Count-level DELBs remain the primary estimand. For cell area (A) in
km2 and period duration \(\tau\) in days, an MSE floor is divided by
\((A\tau)^2\), whereas an RMSE floor is divided by \(A\tau\). No bound is
divided by the number of observations.

| Field | Type | Meaning |
|---|---|---|
| area_km2 | float | Square grid-cell area implied by `resolution_m` |
| duration_days | float | One day for daily panels and seven days for weekly panels |
| area_time_km2_days | float | Product of area and duration used for deterministic target scaling |
| l0_*_intensity_mse, lb_*_intensity_mse | float | Baseline and bicycle-aware MSE floors for crimes per km2 per day |
| delta_l_*_intensity_mse | float | Difference between the corresponding intensity-MSE floors |
| l0_*_intensity_rmse, lb_*_intensity_rmse | float | Intensity-RMSE floors |
| delta_*_intensity_rmse | float | Difference between baseline and bicycle-aware intensity-RMSE floors |
| relative_reduction_* | float | Count-scale DELB reduction divided by the baseline DELB; invariant to deterministic scaling |
| discrete_entropy_power_ratio | float | \(2^{-2\Delta H}\), an entropy-power diagnostic |
| reference_* | float | Unique 100% estimate for the same city, space, time, and estimator cell |
| stability_* | float | Absolute relative deviation from that 100% estimate |

`sample_stability_summary.csv` reports the median and 2.5/97.5 percentiles over
the 50 repeated thinning designs at 20%--80%; its quantiles describe design
variability and are not population confidence intervals. The 100% specification
is deterministic and has zero reference deviation. `full_sample_bootstrap_intervals.csv`
contains 95% percentile intervals from 1,000 paired 28-day block-bootstrap
replicates for each of 36 scale--estimator cells. These intervals are sampling
diagnostics, not randomization-null intervals.

## E16 S16.6 multiscale randomization records

`randomization_replicates.parquet` contains one Miller--Madow CMI result per
city, spatial resolution, temporal resolution, sample fraction, null design,
and null replicate. Fractions below 100% use the prespecified lowest S16.3
thinning replicate (replicate 1); the full sample uses replicate 0. This choice
was fixed before S16.6 results were inspected. S16.5 separately characterizes
variation across all 50 thinning repetitions.

| Field | Type | Meaning |
|---|---|---|
| null_design | category | `temporal_block`, `spatial_series`, or `circular_shift` |
| null_replicate | integer | Randomization replicate within a scale and design |
| randomization_repetitions | integer | 199 for screening scales or 999 for prespecified key scales |
| canonical_sample_replicate | integer | S16.3 replicate 1 below full sample and 0 at full sample |
| observed_delta_h_bits | float | S16.4 Miller--Madow CMI for the matched canonical sample |
| null_delta_h_bits | float | Recomputed Miller--Madow CMI after randomization |
| null_delta_h_projected_bits | float | Nonnegative ordered entropy difference used only for DELB inversion |
| null_delta_l_exact_mse | float | Interpolated exact-lattice DELB reduction under the randomized state |
| support_atoms | integer | Observed `(C,Z,B)` atoms after randomization |
| singleton_observation_share | float | Share of observations in randomized singleton atoms |
| effective_df_per_observation | float | Randomized effective conditional-table degrees of freedom divided by N |

`randomization_summary.csv` has 270 rows: 90 frozen scale--sample cells times
three null mechanisms. It reports null means, standard deviations, percentile
limits, observed-minus-null differences, plus-one upper-tail p-values, global
BH q-values, and within-city-and-design BH q-values. These are
design-conditioned predictive-null diagnostics. The null mean need not be zero,
is not the population CMI, and is not an estimator-bias estimate. The three
transformations are not asserted to be exact conditional-randomization tests.

## E16 S16.7 manuscript-synthesis artifacts

S16.7 creates no new scientific observations. `e16_macros.tex` contains
single-source LaTeX commands computed from accepted S16.5/S16.6 tables.
`table_e16_sample_stability.tex` summarizes median relative deviation from the
matched 100% estimate; zero at 100% is definitional and not evidence of zero
population error. `table_e16_randomization.tex` summarizes the 270 registered
upper-tail comparisons and their global-FDR discovery counts.

`figure_copy_manifest.csv` records the source path, destination path, and
SHA-256 of every copied PDF, SVG, and PNG. A copied file is accepted only when
its hash equals the Python-generated source. The manuscript copies therefore
change neither plotted data nor rendering logic.
