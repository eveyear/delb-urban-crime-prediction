# Compile the revised English manuscript

Upload `applsci-4501207.tex`, `supplement.tex`, `sections/`, `figures/`, and
`Definitions/` to Overleaf, preserving their relative paths. Select
`applsci-4501207.tex` as the main document and use pdfLaTeX. Compile it first
so that `applsci-4501207.aux` exists, then select and compile `supplement.tex`;
the supplement imports cross-references from the main article.

The local verification produced a 39-page main PDF and a 22-page independent
Supplementary Materials PDF. Source data and Python analyses are not needed
to compile these PDFs because the generated tables and figures are included.
The article and Supplement are separate publication files. See the main
`README.md` for reproducibility limits and licensing scopes.
