#!/usr/bin/env python
"""Generate feedback-fix plots/tables for Project 1.

This script adds the figures/tables requested after feedback:
1. analytic-vs-numerical local-energy timing table,
2. pair-distance diagnostic with/without the Jastrow factor,
3. literature-benchmark context table for the interacting runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 10,
})

COLUMN_WIDTH_IN = 3.35
HEIGHT_IN = 2.45


def savefig(name: str) -> None:
    plt.tight_layout(pad=0.35)
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close()


def write_timing_table() -> None:
    path = RESULTS / "analytic_vs_numerical_energy.csv"
    df = pd.read_csv(path)
    analytic = df[df["mode"] == "analytic"].iloc[0]
    numerical = df[df["mode"] == "numerical"].iloc[0]
    speedup = numerical["elapsed_time"] / analytic["elapsed_time"]
    abs_diff = abs(numerical["mean_energy"] - analytic["mean_energy"])

    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Analytic and finite-difference local-energy comparison for the non-interacting harmonic oscillator. The run used $N={int(analytic['n_particles'])}$, $d={int(analytic['dimensions'])}$, $\alpha={analytic['alpha']:.2f}$, and {int(3000)} production cycles. The two estimators agree within numerical precision, while the analytic expression is substantially faster.}}
\label{{tab:1b-timing}}
\begin{{tabular}}{{lrrrr}}
\toprule
Method & $E$ & stderr & time [s] & rel. speed \\
\midrule
Analytic & {analytic['mean_energy']:.8f} & {analytic['standard_error_naive']:.5f} & {analytic['elapsed_time']:.3f} & $1.00\times$ \\
Numerical & {numerical['mean_energy']:.8f} & {numerical['standard_error_naive']:.5f} & {numerical['elapsed_time']:.3f} & ${speedup:.2f}\times$ slower \\
\bottomrule
\multicolumn{{5}}{{l}}{{\footnotesize Absolute energy difference: {abs_diff:.2e}.}}\\
\end{{tabular}}
\end{{table}}
"""
    (TABLES / "table_1b_timing.tex").write_text(tex)


def write_pair_distance_plot_and_table() -> None:
    hist_path = RESULTS / "project1_h_pair_distance.csv"
    summary_path = RESULTS / "project1_h_pair_distance_summary.csv"
    df = pd.read_csv(hist_path)
    summary = pd.read_csv(summary_path)

    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, HEIGHT_IN))
    ax = fig.add_subplot(111)
    labels = {"no_jastrow": "No Jastrow", "jastrow": "With Jastrow"}
    for key, sub in df.groupby("case"):
        sub = sub.sort_values("r")
        ax.plot(sub["r"], sub["probability_density"], label=labels.get(key, key))
    ax.set_xlabel(r"Pair distance $r_{ij}$")
    ax.set_ylabel("Probability density")
    ax.set_title("Pair-distance distribution")
    ax.legend(frameon=False)
    savefig("fig_1h_pair_distance")

    rows = []
    for _, row in summary.iterrows():
        case = "No Jastrow" if row["case"] == "no_jastrow" else "With Jastrow"
        rows.append(
            f"{case} & {row['mean_pair_distance']:.4f} & {row['median_pair_distance']:.4f} & {row['minimum_sampled_pair_distance']:.4f} & {row['acceptance_rate']:.3f} \\\\"
        )
    tex = """\\begin{table}[t]
\\centering
\\caption{Pair-distance diagnostics for the density run. The hard-core radius is small, $a/a_{\\mathrm{ho}}=0.0043$, so the one-body density changes weakly; the pair-distance distribution is a more direct check that the Jastrow factor affects short-range configurations.}
\\label{tab:1h-pair-distance}
\\begin{tabular}{lrrrr}
\\toprule
Case & mean $r_{ij}$ & median $r_{ij}$ & min. sampled & acc. \\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (TABLES / "table_1h_pair_distance.tex").write_text(tex)


def write_literature_context_table() -> None:
    df = pd.read_csv(RESULTS / "project1_g_interacting.csv")
    minima = df.loc[df.groupby("n_particles")["energy_per_particle"].idxmin()].sort_values("n_particles")
    rows = []
    for _, row in minima.iterrows():
        n = int(row["n_particles"])
        na = n * row["hard_core_radius"]
        excess = row["energy_per_particle"] - row["ideal_noninteracting_energy"] / n
        rows.append(
            f"{n} & {na:.3f} & {row['alpha']:.2f} & {row['energy_per_particle']:.6f} & {excess:.6f} \\\\"
        )
    tex = """\\begin{table}[t]
\\centering
\\caption{Interacting-run benchmark context. The published DuBois--Glyde benchmark for the same Rb hard-sphere diameter focuses on anisotropic-trap VMC values and GP comparisons over a much larger particle-number range, so the present $N=10,50,100$ data are used mainly as a small-$N$ code benchmark. The excess column is measured relative to the ideal anisotropic value $E_0/N=(2+\\gamma)/2=2.414215$.}
\\label{tab:1g-benchmark-context}
\\begin{tabular}{rrrrr}
\\toprule
$N$ & $Na/a_{\\mathrm{ho}}$ & $\\alpha_{\\min}$ & $E/N$ & excess \\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (TABLES / "table_1g_benchmark_context.tex").write_text(tex)


def main() -> None:
    write_timing_table()
    write_pair_distance_plot_and_table()
    write_literature_context_table()
    print(f"Wrote feedback figures to {FIGURES}")
    print(f"Wrote feedback tables to {TABLES}")


if __name__ == "__main__":
    main()
