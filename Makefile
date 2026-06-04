.RECIPEPREFIX := >
.PHONY: install test figures report clean

install:
> python -m pip install --upgrade pip
> python -m pip install -e ".[dev]"

test:
> pytest -q

figures:
> python scripts/make_all_report_outputs.py

report:
> cd report && pdflatex project1_report.tex && pdflatex project1_report.tex

clean:
> rm -rf .pytest_cache **/__pycache__ *.egg-info src/*.egg-info
> find report -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" \) -delete
