# Code usage for Project 1

## 1. Install

```bash
cd ~/projects/FYS4411_Project_1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

## 2. Single VMC run

```bash
python scripts/run_vmc.py \
  --sampler brute \
  --n-particles 10 \
  --dimensions 3 \
  --alpha 0.5 \
  --cycles 10000 \
  --burn-in 1000 \
  --step-size 1.0 \
  --seed 1 \
  --output data/raw/example_energies.csv \
  --summary data/raw/example_summary.json
```

## 3. Project parts

### Project 1b: brute-force Metropolis

```bash
python scripts/run_project1_b.py --quick
python scripts/run_project1_b.py --cycles 50000 --burn-in 5000
```

### Analytic vs numerical local energy timing

```bash
python scripts/compare_analytic_numerical_energy.py \
  --n-particles 10 \
  --dimensions 3 \
  --alpha 0.5 \
  --cycles 3000
```

### Project 1c: importance sampling

```bash
python scripts/run_project1_c.py --quick
python scripts/run_project1_c.py --n-particles 100 --cycles 50000 --burn-in 5000
```

### Project 1d: optimize alpha

```bash
python scripts/run_project1_d.py --quick
python scripts/run_project1_d.py --n-particles 100 --iterations 20 --cycles 30000
```

### Project 1e: blocking/bootstrap

```bash
python scripts/run_project1_e.py --quick
python scripts/run_project1_e.py --cycles 100000 --burn-in 10000
```

### Project 1g: repulsive hard-core interaction

```bash
python scripts/run_project1_g.py --quick
python scripts/run_project1_g.py --cycles 50000 --burn-in 5000
```

### Project 1h: one-body densities

```bash
python scripts/run_project1_h.py --quick
python scripts/run_project1_h.py --cycles 50000 --burn-in 5000 --alpha 0.5
```

## 4. Notes

- Use `--quick` only to check that the code runs.
- For report-quality production data, increase `--cycles` and `--burn-in`.
- The exact ideal spherical result is `E = N*d/2` at `alpha = 0.5`.
- For the interacting elliptical case, the default values are `a/a_ho = 0.0043` and `beta = gamma = 2.82843`.
