# DELB Urban Crime Prediction

This repository accompanies the revised *Applied Sciences* manuscript
**“Calibrating Discrete Entropy Prediction Floors for Urban Crime Counts: A Multicity Evaluation”**
(manuscript applsci-4501207). The versioned code-and-derived-counts record is
[Zenodo 10.5281/zenodo.22936346](https://doi.org/10.5281/zenodo.22936346).

## Contents

- `applsci-4501207.tex` and `supplement.tex`: English main manuscript and independent
  Supplementary Materials. The matching PDFs are `applsci-4501207.pdf` and
  `supplement.pdf`. No Chinese manuscript or reviewer correspondence is included.
- `sections/`, `figures/`, and `Definitions/`: LaTeX inputs, generated tables,
  final figures, and journal-class dependencies.
- `empirical/src/`, `empirical/config/`, `empirical/run_*.py`, and
  `empirical/tests/`: analysis code, frozen configurations, entry points, and tests.
- `empirical/revision_audits/`: source-coordinate, coverage, Vancouver-row,
  and figure audit scripts. City-data-dependent scripts require separately
  obtained source data and frozen intermediate panels.
- `derived_counts/`: 24 anonymized city-by-domain-by-period joint-state
  frequency tables and a validation summary.
- `figure_code/figure_code_manifest.csv` and
  `empirical/docs/manuscript_table_provenance.csv`: figure/table provenance.

## Verify without city records

Use Python 3.12 and install the dependencies in `empirical/pyproject.toml`.
From the repository root:

```sh
python -m pip install -e ./empirical
python -m pytest empirical/tests
python examples/minimal_delb_cmi_simulation.py
python empirical/revision_audits/verify_e04_joint_counts.py --counts-dir derived_counts
python empirical/rebuild_submission_diagrams.py --output-root /tmp/delb-diagrams
python empirical/run_problem_chain_figure.py --output-dir /tmp/delb-figure1
```

The joint-state check independently recalculates the **primary** entropy,
conditional mutual information (CMI), and discrete entropy lower bound (DELB)
point estimates. It does not reproduce uncertainty intervals or tests.
The diagram commands use no city records. Run them to temporary output paths
to avoid overwriting the included final figures.

Compile the English article and supplement with:

```sh
latexmk -pdf applsci-4501207.tex
latexmk -pdf supplement.tex
```

## Scope and limits

The repository does **not** redistribute raw crime or bicycle trips,
processed event-level panels, fitted models, or complete randomization
replicates. The public sources are cited in the paper. Re-running bootstrap
intervals, randomization tests, multiscale/local analyses, held-out fitting,
or the 217-row Vancouver sensitivity requires separately obtained inputs.
See `REPRODUCIBILITY_SCOPE.md` and `empirical/README.md` for details.

The Zenodo release is a published code-and-counts archive, **not** a snapshot
of a final accepted manuscript commit. The GitHub revision is later than the
Zenodo release; archive and repository should be cited by their distinct
versions. No claim of complete independent reproduction is made.

## Rights

Original software code is MIT-licensed (`LICENSE-CODE.txt`). The anonymous
`derived_counts/` tables and validation summary are CC BY 4.0
(`LICENSE-DATA.txt`). Neither license automatically covers manuscript text,
figures, MDPI template files, or third-party source data.
