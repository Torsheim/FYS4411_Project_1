# FYS4411 Project 1

Python implementation for Project 1 in FYS4411/FYS9411 Computational Physics II.

The code studies Variational Monte Carlo (VMC) methods for trapped bosons in harmonic oscillator traps. It starts with the non-interacting harmonic oscillator, then adds importance sampling, variational optimization, blocking/bootstrap error analysis, the hard-core Jastrow correlation, and one-body density calculations.

## Repository structure

```text
docs/                  Notes and usage guide
report/                LaTeX report skeleton, figures, tables, bibliography
src/fys4411_project1/  Reusable Python package
scripts/               Command-line scripts for project parts b-h
tests/                 Unit tests
data/raw/              Raw generated data
data/processed/        Processed data
data/results/          Selected final results
notebooks/             Optional exploratory notebooks
logs/                  Run logs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

## Quick sanity checks

One particle, 3D, non-interacting, exact optimum `alpha=0.5`:

```bash
python scripts/run_vmc.py --sampler brute --n-particles 1 --dimensions 3 --alpha 0.5 --cycles 5000 --burn-in 500 --step-size 1.0 --seed 1
```

Importance sampling version:

```bash
python scripts/run_vmc.py --sampler importance --n-particles 1 --dimensions 3 --alpha 0.5 --cycles 5000 --burn-in 500 --time-step 0.05 --seed 1
```

The exact non-interacting ground-state energy is `N*d/2` for a spherical `d`-dimensional trap.

## Main scripts

```bash
python scripts/run_project1_b.py --quick
python scripts/run_project1_c.py --quick
python scripts/run_project1_d.py --quick
python scripts/run_project1_e.py --quick
python scripts/run_project1_g.py --quick
python scripts/run_project1_h.py --quick
```

Remove `--quick` and increase cycles for production runs.
