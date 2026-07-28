# DELB for Urban Crime Count Prediction

This repository accompanies the manuscript:

> *A Discrete Entropy Lower Bound for Urban Crime Count Prediction:
> Finite-Sample Calibration and Multicity Evaluation*

It contains the Applied Sciences manuscript source, Supplementary Materials,
accepted figures and generated tables, and the Python implementation of the
Discrete Entropy Lower Bound (DELB) empirical workflow.

## Repository layout

- `template.tex`: English Applied Sciences manuscript.
- `supplement.tex`: independent Supplementary Materials.
- `sections/`: manuscript sections and generated LaTeX tables.
- `figures/`: manuscript and supplementary figures.
- `Definitions/`: MDPI LaTeX class dependencies.
- `empirical/src/`: reusable analysis implementation.
- `empirical/config/`: experiment configurations.
- `empirical/run_*.py`: versioned experiment entry points.
- `empirical/tests/`: automated tests.
- `empirical/docs/analysis_protocol.md`: analysis protocol.
- `empirical/docs/data_dictionary.md`: data dictionary.

## Data

Raw crime and public-bike data are not redistributed in this repository.
They remain available from the District of Columbia, New York City, and
Vancouver police open-data portals and the Capital Bikeshare, Citi Bike, and
Mobi system-data portals cited in the manuscript. Place downloaded source
files under `data/raw/` and update `empirical/config/paths.yaml` if a different
local layout is used.

## Environment and tests

Python 3.12 is required.

```bash
python -m pip install -e ./empirical
python -m pytest empirical/tests
```

Individual experiments are run from the repository root. For example:

```bash
PYTHONPATH=empirical/src python empirical/run_e03_bound_validation.py \
  --config empirical/config/e03.yaml
```

The complete experiment sequence and commands are documented in
`empirical/README.md`.

## Manuscript compilation

Compile the English manuscript and supplement with pdfLaTeX:

```bash
latexmk -pdf template.tex
latexmk -pdf supplement.tex
```

## Reproducibility scope

Large raw data, processed panels, fitted objects, and full randomization
replicates are excluded from GitHub. The repository retains the code,
configuration, tests, manuscript tables, figures, and provenance
documentation needed to reconstruct them from the cited public sources.

## License

No reuse license has yet been assigned. The authors should add an appropriate
code and documentation license before archival release.
