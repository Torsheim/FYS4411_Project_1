r"""Create single-column report figures and LaTeX tables for FYS4411 Project 1.

Run from the repository root:
    python scripts/plot_report_single_column.py

The script reads data/results/*.csv and writes vector PDF figures to report/figures/
and table snippets to report/tables/. The figure dimensions and font sizes are
chosen for a 10 pt two-column LaTeX report where each figure is included with
width=\columnwidth.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

FIGSIZE = (3.35, 2.35)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.35,
    "lines.markersize": 4.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def savefig(name: str) -> None:
    plt.tight_layout(pad=0.25)
    plt.savefig(FIGURES / f"{name}.pdf")
    plt.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close()


def apply_axes_style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.tick_params(direction="in", top=False, right=False, length=3)


def exact_variational_energy_per_particle(alpha: float, dimensions: int) -> float:
    return dimensions * (alpha / 2.0 + 1.0 / (8.0 * alpha))


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_row(items: list[str]) -> str:
    return " & ".join(items) + r" \\" 


# -------------------- Project 1b --------------------
b = pd.read_csv(RESULTS / "project1_b_bruteforce.csv")
for d in sorted(b["dimensions"].unique()):
    subd = b[b["dimensions"] == d]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for n in sorted(subd["n_particles"].unique()):
        sub = subd[subd["n_particles"] == n].sort_values("alpha")
        ax.plot(sub["alpha"], sub["energy_per_particle"], marker="o", label=f"N={n}")
    ax.axhline(d / 2, linestyle="--", color="0.25", label=f"exact {d/2:g}")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E/N$")
    apply_axes_style(ax)
    ax.legend(loc="best", frameon=True, borderpad=0.25, handlelength=1.6)
    savefig(f"fig_1b_energy_{d}d")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for n in sorted(subd["n_particles"].unique()):
        sub = subd[subd["n_particles"] == n].sort_values("alpha")
        ax.plot(sub["alpha"], sub["acceptance_rate"], marker="o", label=f"N={n}")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("acceptance rate")
    apply_axes_style(ax)
    ax.legend(loc="best", frameon=True, borderpad=0.25, handlelength=1.6)
    savefig(f"fig_1b_acceptance_{d}d")

val = b[np.isclose(b["alpha"], 0.5)].copy().sort_values(["dimensions", "n_particles"])
rows = [latex_row([str(int(r.dimensions)), str(int(r.n_particles)), f"{r.energy_per_particle:.6f}", f"{r.dimensions/2:.1f}", f"{r.acceptance_rate:.3f}"]) for _, r in val.iterrows()]
write_text(TABLES / "table_1b_validation.tex", [
    r"\begin{center}",
    r"\captionof{table}{Non-interacting brute-force validation at $\alpha=0.5$. The exact energy per particle is $d/2$ in $d$ dimensions.}",
    r"\label{tab:1b-validation}",
    r"\setlength{\tabcolsep}{3.2pt}",
    r"\begin{tabular}{@{}ccccc@{}}",
    r"\toprule",
    r"$d$ & $N$ & $E/N$ & exact & acc. \\",
    r"\midrule",
    *rows,
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{center}",
])

# -------------------- Project 1c --------------------
for fname, tag, exact_kind in [
    ("project1_c_importance.csv", "alpha050", "ground"),
    ("project1_c_importance_alpha045.csv", "alpha045", "variational"),
]:
    c = pd.read_csv(RESULTS / fname)
    for d in sorted(c["dimensions"].unique()):
        sub = c[c["dimensions"] == d].sort_values("time_step")
        alpha = float(sub["alpha"].iloc[0])
        reference = d / 2 if exact_kind == "ground" else exact_variational_energy_per_particle(alpha, int(d))

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(sub["time_step"], sub["energy_per_particle"], marker="o", label="VMC")
        ax.axhline(reference, linestyle="--", color="0.25", label="analytic")
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(r"$E/N$")
        apply_axes_style(ax)
        ax.legend(loc="best", frameon=True, borderpad=0.25, handlelength=1.6)
        savefig(f"fig_1c_energy_{tag}_{d}d")

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(sub["time_step"], sub["acceptance_rate"], marker="o")
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel("acceptance rate")
        apply_axes_style(ax)
        savefig(f"fig_1c_acceptance_{tag}_{d}d")

c45 = pd.read_csv(RESULTS / "project1_c_importance_alpha045.csv")
sel = c45[(c45["dimensions"] == 3) & (c45["time_step"].isin([0.001, 0.01, 0.05, 0.1, 0.2]))].sort_values("time_step")
ref3 = exact_variational_energy_per_particle(0.45, 3)
rows = [latex_row([f"{r.time_step:.3f}", f"{r.energy_per_particle:.6f}", f"{r.acceptance_rate:.4f}", f"{r.energy_per_particle-ref3:+.2e}"]) for _, r in sel.iterrows()]
write_text(TABLES / "table_1c_alpha045_3d.tex", [
    r"\begin{center}",
    r"\captionof{table}{Importance-sampling time-step check for $N=100$, $d=3$, and $\alpha=0.45$. The last column is the deviation from the analytic variational value.}",
    r"\label{tab:1c-alpha045}",
    r"\setlength{\tabcolsep}{2.2pt}",
    r"\begin{tabular}{@{}cccc@{}}",
    r"\toprule",
    r"$\Delta t$ & $E/N$ & acc. & $\Delta(E/N)$ \\",
    r"\midrule",
    *rows,
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{center}",
])

# -------------------- Project 1d --------------------
dopt = pd.read_csv(RESULTS / "project1_d_optimization.csv").sort_values("iteration")
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(dopt["iteration"], dopt["alpha"], marker="o")
ax.axhline(0.5, linestyle="--", color="0.25")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$\alpha$")
apply_axes_style(ax)
savefig("fig_1d_alpha")

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(dopt["iteration"], dopt["energy"], marker="o")
ax.axhline(150.0, linestyle="--", color="0.25")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$E$")
apply_axes_style(ax)
savefig("fig_1d_energy")

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(dopt["iteration"], np.abs(dopt["gradient"]), marker="o")
ax.set_yscale("log")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$|\partial E/\partial\alpha|$")
apply_axes_style(ax)
savefig("fig_1d_gradient")

first = dopt.iloc[0]
last = dopt.iloc[-1]
write_text(TABLES / "table_1d_optimization.tex", [
    r"\begin{center}",
    r"\captionof{table}{Initial and final values of the $\alpha$ optimization for $N=100$ particles in three dimensions.}",
    r"\label{tab:1d-optimization}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}lccc@{}}",
    r"\toprule",
    r"step & $\alpha$ & $E$ & $|\partial E/\partial\alpha|$ \\",
    r"\midrule",
    latex_row(["initial", f"{first.alpha:.6f}", f"{first.energy:.6f}", f"{abs(first.gradient):.3g}"]),
    latex_row(["final", f"{last.alpha:.6f}", f"{last.energy:.6f}", f"{abs(last.gradient):.3g}"]),
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{center}",
])

# -------------------- Project 1e --------------------
eblock = pd.read_csv(RESULTS / "project1_e_blocking_alpha045.csv")
eblock = eblock[np.isfinite(eblock["standard_error"])]
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(eblock["block_size"], eblock["standard_error"], marker="o")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("block size")
ax.set_ylabel("standard error")
apply_axes_style(ax)
savefig("fig_1e_blocking_alpha045")

esum = pd.read_csv(RESULTS / "project1_e_summary_alpha045.csv").set_index("quantity")["value"].to_dict()
selected_blocks = eblock[eblock["block_size"].isin([1, 16, 128, 1024, 8192])]
rows = [latex_row([str(int(r.block_size)), str(int(r.n_blocks)), f"{r.standard_error:.5f}"]) for _, r in selected_blocks.iterrows()]
write_text(TABLES / "table_1e_statistics.tex", [
    r"\begin{center}",
    r"\captionof{table}{Selected blocking estimates for the $N=100$, $d=3$, $\alpha=0.45$ run, together with the bootstrap and final blocking errors from the summary file.}",
    r"\label{tab:1e-statistics}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}ccc@{}}",
    r"\toprule",
    r"block size & blocks & error \\",
    r"\midrule",
    *rows,
    r"\midrule",
    latex_row(["bootstrap", "--", f"{esum.get('bootstrap_error', float('nan')):.5f}"]),
    latex_row(["final block", "--", f"{esum.get('blocking_final_error', float('nan')):.5f}"]),
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{center}",
])

# -------------------- Project 1g --------------------
g = pd.read_csv(RESULTS / "project1_g_interacting.csv")
for n in sorted(g["n_particles"].unique()):
    sub = g[g["n_particles"] == n].sort_values("alpha")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sub["alpha"], sub["energy_per_particle"], marker="o", label="interacting")
    ideal_per_particle = (sub["ideal_noninteracting_energy"] / sub["n_particles"]).to_numpy()
    ax.plot(sub["alpha"], ideal_per_particle, linestyle="--", color="0.25", label="ideal")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E/N$")
    apply_axes_style(ax)
    ax.legend(loc="best", frameon=True, borderpad=0.25, handlelength=1.6)
    savefig(f"fig_1g_energy_N{int(n)}")

mins = g.loc[g.groupby("n_particles")["mean_energy"].idxmin()].sort_values("n_particles")
rows = []
for _, r in mins.iterrows():
    ideal_per = r.ideal_noninteracting_energy / r.n_particles
    rows.append(latex_row([str(int(r.n_particles)), f"{r.alpha:.2f}", f"{r.energy_per_particle:.6f}", f"{ideal_per:.6f}", f"{r.energy_per_particle-ideal_per:.6f}", f"{r.acceptance_rate:.3f}"]))
write_text(TABLES / "table_1g_minima.tex", [
    r"\begin{center}",
    r"\captionof{table}{Interacting hard-core minima from the scanned $\alpha$ grid. Energies are per particle and the ideal reference uses the anisotropic non-interacting ground state.}",
    r"\label{tab:1g-minima}",
    r"\setlength{\tabcolsep}{2.2pt}",
    r"\begin{tabular}{@{}cccccc@{}}",
    r"\toprule",
    r"$N$ & $\alpha$ & $E/N$ & ideal & excess & acc. \\",
    r"\midrule",
    *rows,
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{center}",
])

# -------------------- Project 1h --------------------
no = pd.read_csv(RESULTS / "project1_h_radial_density_no_jastrow.csv")
ja = pd.read_csv(RESULTS / "project1_h_radial_density_jastrow.csv")
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(no["bin_center"], no["density"], label="no Jastrow")
ax.plot(ja["bin_center"], ja["density"], label="with Jastrow")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\rho(r)$")
apply_axes_style(ax)
ax.legend(loc="best", frameon=True, borderpad=0.25, handlelength=1.6)
savefig("fig_1h_density")

fig, ax = plt.subplots(figsize=FIGSIZE)
r_no = no["bin_center"].to_numpy()
r_ja = ja["bin_center"].to_numpy()
prob_no = 4.0 * np.pi * r_no**2 * no["density"].to_numpy()
prob_ja = 4.0 * np.pi * r_ja**2 * ja["density"].to_numpy()
ax.plot(r_no, prob_no, label="no Jastrow")
ax.plot(r_ja, prob_ja, label="with Jastrow")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$4\pi r^2\rho(r)$")
apply_axes_style(ax)
ax.legend(loc="best", frameon=True, borderpad=0.25, handlelength=1.6)
savefig("fig_1h_radial_probability")

# Density table: peak locations for probability distribution.
idx_no = int(np.argmax(prob_no))
idx_ja = int(np.argmax(prob_ja))
write_text(TABLES / "table_1h_density.tex", [
    r"\begin{center}",
    r"\captionof{table}{Peak positions of the radial probability distributions with and without the Jastrow factor.}",
    r"\label{tab:1h-density}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}lcc@{}}",
    r"\toprule",
    r"case & $r_{\rm peak}$ & max $4\pi r^2\rho(r)$ \\",
    r"\midrule",
    latex_row(["no Jastrow", f"{r_no[idx_no]:.3f}", f"{prob_no[idx_no]:.3f}"]),
    latex_row(["with Jastrow", f"{r_ja[idx_ja]:.3f}", f"{prob_ja[idx_ja]:.3f}"]),
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{center}",
])

print(f"Wrote figures to {FIGURES}")
print(f"Wrote tables to {TABLES}")
