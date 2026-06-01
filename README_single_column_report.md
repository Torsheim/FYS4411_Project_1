# FYS4411 Project 1 single-column report update

This package updates the Project 1 report so that all result figures are one-column figures suitable for the two-column report format.

## Contents

- `scripts/plot_report_single_column.py`
  - Reads `data/results/*.csv`.
  - Writes vector PDF figures to `report/figures/`.
  - Writes LaTeX table snippets to `report/tables/`.
  - Uses a 10 pt base font and a figure size intended for `width=\columnwidth`.

- `report/project1_report.tex`
  - Full two-column report.
  - Every figure and table is referenced in the text.
  - Uses single-column figures rather than combined multi-panel figures.

- `report/project1_report.pdf`
  - Compiled report.

- `data/results/*.csv`
  - Result CSVs used for figures and tables.

## Usage

From the repository root:

```bash
python scripts/plot_report_single_column.py
cd report
pdflatex project1_report.tex
pdflatex project1_report.tex
```

The report uses inline `\captionof` figures to keep figures close to the relevant discussion.
