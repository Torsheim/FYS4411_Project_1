# Terminal guide for adding the developed Project 1 code

Assume your local repository is here:

```bash
~/projects/FYS4411_Project_1
```

## 1. Copy/unzip the code overlay into your repository

Put `FYS4411_Project_1_developed_code.zip` inside your repository folder, then run:

```bash
cd ~/projects/FYS4411_Project_1
unzip -o FYS4411_Project_1_developed_code.zip
rm FYS4411_Project_1_developed_code.zip
```

This overwrites the placeholder files with the developed code.

## 2. Create a fresh virtual environment

```bash
cd ~/projects/FYS4411_Project_1
deactivate 2>/dev/null || true
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 3. Run the tests

```bash
pytest -q
```

Expected result:

```text
25 passed
```

## 4. Run a single sanity-check VMC calculation

```bash
python scripts/run_vmc.py \
  --sampler brute \
  --n-particles 1 \
  --dimensions 3 \
  --alpha 0.5 \
  --cycles 5000 \
  --burn-in 500 \
  --step-size 1.0 \
  --seed 1
```

Expected mean energy: approximately/exactly `1.5`.

## 5. Run quick versions of the project parts

```bash
python scripts/run_project1_b.py --quick
python scripts/run_project1_c.py --quick
python scripts/run_project1_d.py --quick
python scripts/run_project1_e.py --quick
python scripts/run_project1_g.py --quick
python scripts/run_project1_h.py --quick
```

## 6. Run production versions later

Increase Monte Carlo cycles and remove `--quick`, for example:

```bash
python scripts/run_project1_b.py --cycles 50000 --burn-in 5000
python scripts/run_project1_c.py --n-particles 100 --cycles 50000 --burn-in 5000
python scripts/run_project1_g.py --cycles 50000 --burn-in 5000
python scripts/run_project1_h.py --cycles 50000 --burn-in 5000 --alpha 0.5
```

## 7. Check files before committing later

You said you do not want to commit yet. To only inspect the changes:

```bash
git status
```

When you eventually want to commit:

```bash
git add .
git commit -m "Add Project 1 VMC implementation"
```
