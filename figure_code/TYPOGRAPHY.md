# Figure typography for the revised manuscript

The chart generator applies one semantic type scale at the manuscript's
approximately 6-inch source-to-page reference width: panel titles 9 pt, axis titles 8.5 pt,
tick labels 7.5 pt, legends 7.5 pt, and numeric annotations 7.5 pt. Source
font sizes are multiplied by the source-to-display width ratio, so wide
multi-panel figures retain approximately the same printed size. The same
DejaVu Sans family is used in the diagrams and statistical charts. Figure
captions remain LaTeX text and follow the journal template.

Figures 1--3 are schematic diagrams without axes or legends; their box/body
text is sized separately by role. Figure S14 is a raster chart because that
is the accepted asset format; it is regenerated at 300 dpi. Statistical
values, confidence intervals, limits, and the underlying accepted tables are
not recalculated by the typography rebuild.

From the repository root, with the accepted frozen run outputs available:

```bash
python empirical/run_problem_chain_figure.py --output-dir figures/problem_chain
python empirical/rebuild_submission_diagrams.py --output-root figures
PYTHONPATH=empirical/src python empirical/rebuild_plot_typography.py \
  --runs-root /path/to/accepted/empirical/runs --output-root figures
```

The last command rebuilds main Figures 4--9 and supplementary Figures S1--S20.
It reads frozen E02/E04/E05/E06/E07/E08/E09/E10/E12/E13/E14/E16/E17/E19
summary files. As explained in `REPRODUCIBILITY_SCOPE.md`, some underlying
city run inputs are not redistributed in the public repository.
