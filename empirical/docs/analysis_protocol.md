# Prespecified Analysis Protocol

## Primary target

The primary target is the daily count of all observations whose authoritative
`crime_type_unified` value is one of `PROPERTY_THEFT`, `VEHICLE_THEFT`, or
`BURGLARY`. Category-specific analyses are prespecified secondary experiments.

## Primary information sets

The baseline information set contains lagged crime, day of week, season, and
holiday indicators. Lagged observed weather will be added when an authoritative
source has been archived and its forecast-availability interpretation is
documented. The mobility information set adds lagged bicycle total flow.

## Primary scales

The primary spatial unit is a deterministic 1 km projected grid and the primary
temporal unit is one local calendar day. Alternative 500 m and 2 km grids and
weekly aggregation are robustness specifications. A 250 m analysis is retained
as a diagnostic and is promoted only if its occupancy and effective sample size
meet prespecified reliability thresholds.

## Bicycle-system coverage

The complete crime-domain panel is retained for descriptive and sensitivity
analyses. The primary mobility-information comparison must also be reported on
the subset of grids with observed bicycle activity during the 2020-2021
training period (`bike_coverage_training = true`). This restriction is fixed
before validation and testing. A structural zero outside the bicycle-system
footprint must not be interpreted as observed zero total human mobility.

## Outcome and bound units

The theoretical lower bound is an MSE bound. Consequently,
`Predictability Gap = empirical MSE - DELB` and
`Information-use efficiency = DELB / empirical MSE`.

## Interpretation

Conditional mutual information and lower-bound reduction measure predictive
information value. They do not identify a causal effect of bicycle mobility on
crime.

## Numerical DELB implementation

The primary exact bound is computed with the centered discrete Gaussian.
For moderate and large \(\lambda\), the partition function is summed directly
over the integer lattice. For small \(\lambda\), the Poisson-dual theta series
is used to avoid an unnecessarily wide real-lattice truncation. The omitted
unnormalized sum is bounded by \(10^{-15}\), and one-dimensional inverse roots
are solved to a tolerance of \(10^{-12}\).

The dither-based value with the \(-1/12\) lattice correction is secondary and
must not be described as attainable. E03's unconditional E02 entropy values
are numerical range checks only; city-level conditional information estimands
are defined and estimated in E04.

## E04 city-level conditional information

The E04 baseline conditioning state is
`(grid_id, crime_lag_bin, day_of_week, season, is_holiday)`. Lagged crime
counts use the fixed states zero, one, two, and three-or-more. The augmented
state adds strict lag-1 bicycle total flow, represented by zero and three
positive-flow bins. Positive-flow tertiles are estimated separately by city
using only the 2020--2021 training sample and are frozen before evaluating any
2022 observation.

The primary point estimator is the Miller--Madow conditional entropy
correction. Plugin estimates and all support-atom counts are retained for
audit. Raw conditional mutual information is never truncated in descriptive
or inferential reporting. A nonnegative projection is permitted only when the
estimated entropy difference is inserted into the theoretical DELB ordering.

Uncertainty uses 1,000 nonoverlapping-calendar 7-day block-bootstrap
replicates for each city-domain-period specification. The reported 95% interval
is centered on the full-sample point estimate and uses the bootstrap standard
error. Raw percentile endpoints and bootstrap bias remain diagnostics because
resampling sparse discrete states can remove support atoms and shift entropy
estimates. Annual estimates with high singleton-state observation shares must
be described cautiously and subjected to E09 robustness and E10 placebo
analysis.

## E05 spatially localized information value

E05 is restricted to 1 km grids observed by the bicycle system during the
2020--2021 training period. Eligibility is fixed before calculating any local
information effect: 1,095 valid strict-lag days, at least 30 target crime
events, bicycle activity on at least 55 lagged days (5% of the sequence), no
continuous zero-bicycle interval longer than 365 days, and at least two
observed target and bicycle states. Every covered grid, including exclusions
and reason codes, remains in the eligibility table.

The resulting support-only gate contains 146 Washington, DC grids, 160 New
York City grids, and 33 Vancouver grids. Two Washington, DC grids that passed
the count, activity-day, and continuity thresholds were excluded because the
frozen bicycle discretization produced only one observed state.

Within each eligible grid, the baseline state contains lagged crime bin, day
of week, season, and holiday status; grid identity is omitted because it is
constant locally. E04's frozen city-specific bicycle cut points are reused
without refitting. Miller--Madow conditional entropy and raw CMI remain the
primary estimators. Exact point DELBs use the validated root solver. Bootstrap
DELB arrays use a separately validated monotone interpolation of
\(\log(1+L)\) solely to avoid hundreds of thousands of redundant root solves.

All grids in one city share the same 1,000 deterministic 7-day calendar-block
resamples, preserving common city-wide temporal shocks. One-sided positive-CMI
normal p-values are adjusted within city by Benjamini--Hochberg. A grid is
classified as a positive local bound reduction only when its adjusted p-value
is at most 0.05 and the centered two-sided 95% lower endpoints for both raw CMI
and exact DELB reduction exceed zero.

Spatial dependence is evaluated using row-standardized queen contiguity among
eligible grids. Global and local Moran statistics use 999 deterministic spatial
randomizations. Spatial clusters and all other E05 associations remain
descriptive rather than causal.

## E06 matched out-of-sample predictive models

E06 uses the complete training bicycle-covered 1 km domain rather than the
stricter local-entropy support gate. The three city samples contain 181
Washington, DC grids, 198 New York grids, and 34 Vancouver grids. Every
observation has a valid strict lag and belongs to the prespecified temporal
split: 2020--2021 training, January--June 2022 validation, or July--December
2022 testing.

Four model families are evaluated: a smoothed empirical conditional-state
mean, a Poisson fixed-effects model, a negative-binomial fixed-effects model,
and a Poisson histogram gradient-boosting model. Each family is fitted twice.
The baseline member uses
`(grid_id, crime_lag_bin, day_of_week, season, is_holiday)`, and the
bicycle-aware member adds only `bicycle_lag_bin`. The bicycle cut points are
read directly from the accepted E04 training-only table. A lag-7 seasonal
naive forecast is retained as an external benchmark but is not described as
matched to the DELB because it uses a different information set.

Hyperparameters minimize validation-period integer-prediction MSE within each
city, family, and information set. After selection, the model is refitted on
training plus validation observations and evaluated once on the untouched
test period. Continuous conditional means are retained for auxiliary metrics.
The primary theoretical comparison uses the prespecified nonnegative half-up
integer prediction rule and MSE. Auxiliary metrics are continuous MSE,
integer RMSE and MAE, mean Poisson deviance, exact-count accuracy, and the
predicted-to-observed total ratio.

For each city and model family, uncertainty for
\(\Delta\operatorname{MSE}=\operatorname{MSE}^{0}
-\operatorname{MSE}^{B}\) uses 1,000 paired resamples of nonoverlapping
7-day test-calendar blocks. Every sampled day retains all spatial grids,
thereby preserving common city-day shocks. One-sided positive-improvement
p-values are adjusted across the 12 city--model contrasts by
Benjamini--Hochberg, and the screening classification additionally requires
the centered 95% lower endpoint to exceed zero. This is a model-specific
out-of-sample performance test, not a causal bicycle effect. The relationship
between empirical improvement and the E04/E05 theoretical bound changes is
reserved for E07.

## E07 matched theory--practice comparison

The primary city-level predictability gap must be temporally and
informationally matched. E07 therefore re-estimates
\(H(C_t\mid\mathcal F^0)\), \(H(C_t\mid\mathcal F^B)\), \(L^0\), and \(L^B\)
on the exact E06 July--December 2022 test domain. Pooled 2020--2022 E04 bounds
must not be subtracted directly from held-out E06 MSE. The seasonal-naive
external benchmark remains excluded because its information set is not
matched to the DELB.

For each city and matched model family, define
\[
\operatorname{Gap}^j=\operatorname{MSE}^j-L^j,\qquad
\eta^j=\frac{L^j}{\operatorname{MSE}^j},\qquad j\in\{0,B\}.
\]
Here, \(\eta^j\) is termed information-use efficiency only in the restricted
sense of proximity to the theoretical floor. Define gap narrowing as
\[
\operatorname{Gap}^0-\operatorname{Gap}^B
=\Delta\operatorname{MSE}-\Delta L.
\]
The diagnostic ratio
\(\Delta\operatorname{MSE}/\Delta L\) is retained but is not bounded by one
and must not be described as a literal efficiency percentage.

City-level uncertainty uses 1,000 deterministic shared resamples of
nonoverlapping 7-day test-calendar blocks. For every resample, the conditional
entropy estimates, exact DELBs (through the validated monotone interpolation),
and both paired model MSEs are recomputed with identical block weights.
Percentile intervals describe the resulting joint bootstrap distribution.

Local theory--practice analysis prevents test-outcome leakage. E05's 339
prespecified eligible grids are retained, but local exact DELB reductions are
re-estimated using training plus validation observations ending 30 June 2022.
They are joined to E06 local MSE improvements calculated only from held-out
test predictions. For each city and model, association is summarized by
Spearman correlation with 2,000 paired-grid bootstrap resamples. The
asymptotic Spearman p-values are adjusted jointly across the 12 comparisons by
Benjamini--Hochberg; the bootstrap intervals remain the primary uncertainty
display.

For each model family, a pooled descriptive regression relates local
\(\Delta\operatorname{MSE}\) to pretest local \(\Delta L\), includes city
fixed effects, and uses HC3 covariance. The four theory--practice categories
use the city median \(\Delta L\) and city--model median
\(\Delta\operatorname{MSE}\); equality is assigned to the low group. These
correlations, regressions, and maps do not model cross-grid dependence and do
not identify a causal effect of bicycle activity on crime.

## E08 crime-type heterogeneity

E08 treats the existing `crime_type_unified` categories
`PROPERTY_THEFT`, `VEHICLE_THEFT`, and `BURGLARY` as authoritative secondary
outcomes. No category is reconstructed from raw labels. For each category,
the daily 1 km target is its original integer event count and the baseline
state contains the category's own strict lag-1 count bin, grid identity, day
of week, season, and holiday indicator. The bicycle-aware state adds only the
frozen E04 strict lag-1 total-flow bin. Thus the baseline history is not
silently borrowed from the combined-crime target.

The analysis uses the same prespecified training bicycle-covered domain as
E04 and E06: 181 Washington, DC grids, 198 New York City grids, and 34
Vancouver grids. Category counts must sum observation by observation to
`crime_count_all`. E08 reports a pooled 2020--2022 theoretical analysis and a
temporally matched July--December 2022 test analysis. The pooled analysis
describes general category heterogeneity; only the matched test DELBs enter
the predictability-gap comparison with held-out prediction MSE.

Conditional entropy uses the Miller--Madow estimator and the exact DELB uses
the E03-validated inverse discrete-Gaussian envelope. Each city and period has
1,000 deterministic resamples of shared nonoverlapping 7-day calendar
blocks. The same replicate indices are used for all three categories, so
pairwise differences in CMI and exact DELB reduction are paired rather than
formed from independent intervals. Two-sided category-heterogeneity
\(p\)-values and one-sided positive-information \(p\)-values are adjusted by
Benjamini--Hochberg.

The four matched E06 families are independently validation-tuned for every
city, category, and information set. Selected models are refitted on training
plus validation observations and evaluated once on the held-out test period.
The seasonal lag-7 benchmark is retained only as an explicitly nonmatched
reference. Primary predictions use nonnegative half-up integer rounding and
MSE. Joint 7-day test-block resamples yield paired category contrasts in
\(\Delta\operatorname{MSE}\) and gap narrowing. A positive model screen
requires a positive point improvement, a centered 95% lower endpoint above
zero, and a Benjamini--Hochberg-adjusted one-sided \(p\)-value no greater than
0.05.

Crime-type differences can reflect base rate, persistence, conditional-state
support, rounding, and model fit as well as the predictive content of bicycle
states. They are therefore heterogeneity results for prespecified outcomes,
not evidence that bicycle activity causally changes one crime type more than
another. Sparse categories can have positive theoretical information while
their fitted integer forecasts remain unchanged.

## E09 prespecified robustness analysis

E09 freezes its alternative-specification matrix before reading its outputs.
It evaluates 500 m and 2 km projected grids, weekly aggregation, an alternative
three-state crime-history mapping, median and quintile mobility states,
crime and bicycle lags of 2, 3, and 7 days, outbound/inbound/absolute-net
mobility measures, and 7/14/28-day bootstrap blocks. The 250 m grid remains
a diagnostic and is not promoted to an inferential specification.

The estimator comparison retains plug-in and Miller--Madow estimates and adds
a coherent Jeffreys--Dirichlet posterior calculation. One symmetric
Dirichlet posterior is defined over observed atoms of the full joint state;
all required marginals are obtained by aggregation from that same posterior.
Separate priors must not be fitted to baseline and augmented state spaces
when their entropy difference is interpreted as posterior conditional mutual
information.

Rolling-origin robustness uses twelve monthly 2022 forecast origins and an
expanding training window. Each model and information set reuses the
hyperparameters selected in accepted E06; no forecast month is used for
selection. E09 does not contain temporal or spatial permutations, surrogate
nulls, or distant-lag placebo claims. Those remain reserved for E10.

## E10 placebo and null calibration

E10 freezes three 999-replicate predictive null mechanisms. Seven-day
bicycle-state blocks are independently permuted within each grid, complete
grid time series are reassigned within city, and each grid receives an
independent circular shift of at least 31 days. The circular surrogate
preserves each grid's bicycle-state marginal distribution and circular
autocorrelation. Fourteen- and thirty-day bicycle lags are compared with
lag 1 on exactly matched samples.

Pooled tests use the primary E09 city state and local tests use the 339
prespecified E05-eligible grids. The upper-tail randomization p-value uses the
plus-one rule. City p-values are adjusted across all nine city-by-null tests;
local p-values are adjusted by Benjamini--Hochberg within city and null
mechanism. A stringent local joint screen requires rejection under all three
null mechanisms.

These placebo mechanisms are not exact conditional randomization tests
preserving the entire distribution of bicycle state given every baseline
conditioning state. A failure to exceed them limits claims of recent,
locally matched bicycle-specific information but does not prove conditional
independence. Conversely, rejection would remain predictive rather than
causal. Permutation-induced changes in sparse conditioning-state support must
be reported because they can reveal residual finite-sample entropy-estimator
bias.

## E11 manuscript synthesis

E11 is a read-only synthesis of accepted E02 and E04--E10 runs. It verifies
the acceptance gates and fixed source pointers before reading any result.
The manuscript uses E02 panel and descriptive summaries, E04 city
information estimates, E05 local estimates, E06 matched prediction outputs,
E07 predictability gaps and local theory--practice comparisons, E08
crime-type results, E09 robustness and rolling origins, and E10 null
calibration and distant lags. E03 is cited as the numerical validation stage
but does not contribute empirical point estimates.

Generated LaTeX tables and result macros are machine-written from CSV
outputs. Selected article figures are copied from the accepted experiment
runs after hash calculation. The only new article figure is the workflow
diagram, generated by the E11 Python/Matplotlib entry point. Manual redrawing
or editing of empirical figures is prohibited.

The accepted post-robustness E11 run is
`20260719_192814_E11_manuscript_synthesis`. It reports positive E09
specification point estimates but does not attribute them to recent local
bicycle-specific information because E10 produced 0/9 city-level and 0/339
local joint-null separations. The mathematical DELB and the empirical
mobility interpretation must remain explicitly separated.
