# Feedback fixes included in this update

This update addresses the main points from the 86/100 review:

1. **Selected result files committed**: the CSV files used by the report are included under `data/results/`, and `.gitignore` now permits selected CSVs there.
2. **Analytic-vs-numerical local-energy timing**: `analytic_vs_numerical_energy.csv` and `report/tables/table_1b_timing.tex` document agreement and CPU-time difference.
3. **Interacting benchmark context**: `report/tables/table_1g_benchmark_context.tex` gives quantitative small-`N` context and explains why a direct point-by-point DuBois--Glyde/Nilsen comparison is not available for this small grid.
4. **Root README improved**: setup, testing, regeneration, plotting, and compilation commands are now in the top-level README.
5. **Placeholder blocking script removed**: `scripts/analyze_blocking.py` is now a wrapper around the actual Project 1e analysis.
6. **Pair-distance diagnostic added**: `scripts/run_pair_distance.py`, `data/results/project1_h_pair_distance*.csv`, `report/figures/fig_1h_pair_distance.*`, and `report/tables/table_1h_pair_distance.tex` add a short-range diagnostic.
