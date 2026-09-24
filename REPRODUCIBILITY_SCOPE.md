# Reproducibility scope

The published [Zenodo record](https://doi.org/10.5281/zenodo.22936346)
contains code, configurations, tests, and 24 anonymized primary E04
joint-state frequency tables. This GitHub repository additionally includes
the revised English manuscript, Supplementary Materials, final figures,
and selected revision scripts.

`derived_counts/` permits independent recalculation of primary
city-by-domain-by-period entropy, CMI, and DELB point estimates:

```sh
python empirical/revision_audits/verify_e04_joint_counts.py --counts-dir derived_counts
```

It does not provide the event-level inputs needed for bootstrap intervals,
randomization tests, multiscale or local analyses, held-out model fitting,
or the reviewer-requested Vancouver source-row sensitivity. Those analyses
require the cited external source data and frozen processed E01/E02 panels.
The 217-row sensitivity script is
`empirical/revision_audits/reviewer1_vancouver_source_row_dedup.py`; set
`DELB_VAN_RAW_DIR`, `DELB_E01_VAN_DIR`, and `DELB_E02_PANEL_DIR` before running.
The script checks the inputs and does not assert that identical source rows
represent duplicate journeys.

The revision's weather-field feasibility audit is
`empirical/revision_audits/audits/weather_control_feasibility.csv`.
It documents the lack of common forecast-time weather fields in the frozen
three-city panel, not the lack of public weather data.

Figures 1–3 can be rebuilt without city records:

```sh
python empirical/run_problem_chain_figure.py --output-dir /tmp/delb-figure1
python empirical/rebuild_submission_diagrams.py --output-root /tmp/delb-diagrams
```

The historical run identifiers in code/configurations are provenance labels;
they are not a promise that omitted raw or processed files are in this repo.
Do not describe either the Zenodo archive or this repository as a complete
independent reproduction package.
