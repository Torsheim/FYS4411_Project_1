from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("data/results")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=200)
    plt.close()


def variational_energy_per_particle(alpha: float, dimensions: int) -> float:
    """Analytic non-interacting variational energy per particle for beta=1."""
    return dimensions * (alpha / 2.0 + 1.0 / (8.0 * alpha))


# ---------------------------------------------------------
# Fixed Project 1d: optimization should converge to alpha=0.5
# ---------------------------------------------------------
path = RESULTS / "project1_d_optimization.csv"
if path.exists():
    df = pd.read_csv(path).sort_values("iteration")

    plt.figure()
    plt.plot(df["iteration"], df["alpha"], marker="o")
    plt.axhline(0.5, linestyle="--", label="Expected optimum = 0.5")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\alpha$")
    plt.title("Project 1d: Fixed optimization path")
    plt.legend()
    savefig("project1_d_fixed_alpha_vs_iteration.png")

    plt.figure()
    plt.plot(df["iteration"], df["energy"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Project 1d: Fixed energy during optimization")
    savefig("project1_d_fixed_energy_vs_iteration.png")


# ---------------------------------------------------------
# Extra Project 1c: importance sampling with alpha=0.45
# ---------------------------------------------------------
path = RESULTS / "project1_c_importance_alpha045.csv"
if path.exists():
    df = pd.read_csv(path)

    for d in sorted(df["dimensions"].unique()):
        sub = df[df["dimensions"] == d].sort_values("time_step")
        alpha = float(sub["alpha"].iloc[0])
        exact_var = variational_energy_per_particle(alpha, int(d))

        plt.figure()
        plt.plot(sub["time_step"], sub["energy_per_particle"], marker="o", label="VMC")
        plt.axhline(exact_var, linestyle="--", label=f"Analytic variational E/N = {exact_var:.6f}")
        plt.xlabel("Time step")
        plt.ylabel("Energy per particle")
        plt.title(f"Project 1c: Importance sampling, alpha={alpha}, {d}D")
        plt.legend()
        savefig(f"project1_c_alpha045_energy_{d}D.png")

        plt.figure()
        plt.plot(sub["time_step"], sub["acceptance_rate"], marker="o")
        plt.xlabel("Time step")
        plt.ylabel("Acceptance rate")
        plt.title(f"Project 1c: Acceptance rate, alpha={alpha}, {d}D")
        savefig(f"project1_c_alpha045_acceptance_{d}D.png")


# ---------------------------------------------------------
# Project 1e: blocking plot for alpha=0.45
# ---------------------------------------------------------
path = RESULTS / "project1_e_blocking_alpha045.csv"
if path.exists():
    df = pd.read_csv(path)
    df = df[np.isfinite(df["standard_error"])]
    df = df[df["standard_error"] > 0]

    if len(df) > 0:
        plt.figure()
        plt.plot(df["block_size"], df["standard_error"], marker="o")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Block size")
        plt.ylabel("Standard error")
        plt.title("Project 1e: Blocking analysis, alpha=0.45")
        savefig("project1_e_blocking_alpha045.png")


# ---------------------------------------------------------
# Project 1h: one-body densities with and without Jastrow
# ---------------------------------------------------------
no_jastrow_path = RESULTS / "project1_h_radial_density_no_jastrow.csv"
jastrow_path = RESULTS / "project1_h_radial_density_jastrow.csv"

if no_jastrow_path.exists() and jastrow_path.exists():
    no_jastrow = pd.read_csv(no_jastrow_path)
    jastrow = pd.read_csv(jastrow_path)

    plt.figure()
    plt.plot(no_jastrow["bin_center"], no_jastrow["density"], label="No Jastrow")
    plt.plot(jastrow["bin_center"], jastrow["density"], label="With Jastrow")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\rho(r)$")
    plt.title("Project 1h: Radial one-body density")
    plt.legend()
    savefig("project1_h_radial_density.png")

    # Also plot the radial probability distribution 4*pi*r^2*rho(r),
    # which is often easier to visually interpret in 3D.
    r_no = no_jastrow["bin_center"].to_numpy()
    r_ja = jastrow["bin_center"].to_numpy()

    prob_no = 4.0 * np.pi * r_no**2 * no_jastrow["density"].to_numpy()
    prob_ja = 4.0 * np.pi * r_ja**2 * jastrow["density"].to_numpy()

    plt.figure()
    plt.plot(r_no, prob_no, label="No Jastrow")
    plt.plot(r_ja, prob_ja, label="With Jastrow")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$4\pi r^2 \rho(r)$")
    plt.title("Project 1h: Radial probability distribution")
    plt.legend()
    savefig("project1_h_radial_probability.png")


print(f"Saved fixed plots in {PLOTS.resolve()}")
