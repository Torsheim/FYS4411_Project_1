# FYS4411 Project 1: Variational Monte Carlo for trapped bosons

This repository contains the code, selected results, figures, tables, and report for Project 1 in FYS4411/FYS9411 Computational Physics II.

The project studies variational Monte Carlo calculations for trapped bosons. The implementation starts with the non-interacting harmonic oscillator, where exact results are available, and then adds importance sampling, optimization, blocking/bootstrap error analysis, a hard-core Jastrow factor, and one-body densities.

## Repository structure

```text
src/fys4411_project1/     Python package: wave functions, local energies, samplers, statistics, density
scripts/                  Reproduction scripts for project parts and report figures
tests/                    Unit tests
data/results/             Selected CSV result files used in the report
report/                   LaTeX report, figures, tables, and compiled PDF
docs/                     Usage notes and formula documentation
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Test the code

```bash
pytest -q
```

## Reproduce selected results

The selected result CSVs used in the report are committed in `data/results/`. To regenerate them from scratch, run the scripts below. The production runs can take a long time, especially the interacting `N=100` hard-core scans.

```bash
python scripts/run_project1_b.py --cycles 50000 --burn-in 5000
python scripts/compare_analytic_numerical_energy.py --cycles 3000 --burn-in 300 --n-particles 10 --dimensions 3 --alpha 0.45
python scripts/run_project1_c.py --n-particles 100 --cycles 50000 --burn-in 5000
python scripts/run_project1_c.py --n-particles 100 --alpha 0.45 --cycles 20000 --burn-in 2000 --output data/results/project1_c_importance_alpha045.csv
python scripts/run_project1_d.py --n-particles 100 --dimensions 3 --initial-alpha 0.7 --learning-rate 0.001 --iterations 20 --cycles 30000 --burn-in 3000 --output data/results/project1_d_optimization.csv
python scripts/run_project1_e.py --n-particles 100 --alpha 0.45 --cycles 50000 --burn-in 5000 --blocking-output data/results/project1_e_blocking_alpha045.csv --summary-output data/results/project1_e_summary_alpha045.csv
python scripts/run_project1_g.py --cycles 30000 --burn-in 3000
python scripts/run_project1_h.py --cycles 50000 --burn-in 5000 --alpha 0.5
python scripts/run_pair_distance.py --cycles 8000 --burn-in 800
```

## Regenerate report figures and tables

```bash
python scripts/make_all_report_outputs.py
```

This writes column-width PDF/PNG figures to `report/figures/` and LaTeX table snippets to `report/tables/`.

## Compile the report

```bash
cd report
pdflatex project1_report.tex
pdflatex project1_report.tex
cd ..
```

The compiled report is `report/project1_report.pdf`.

## Notes on feedback-driven fixes

The repository intentionally commits selected CSV files in `data/results/` so the figures and tables are reproducible. The report also includes an analytic-versus-finite-difference local-energy timing table and a pair-distance diagnostic to show the short-range role of the Jastrow factor more directly than the one-body density alone.
