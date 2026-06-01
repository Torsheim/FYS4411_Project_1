from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("data/results")
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=200)
    plt.close()


# ----------------------------
# Project 1b: brute force VMC
# ----------------------------
path = RESULTS / "project1_b_bruteforce.csv"
if path.exists():
    df = pd.read_csv(path)

    for d in sorted(df["dimensions"].unique()):
        plt.figure()
        subd = df[df["dimensions"] == d]
        for n in sorted(subd["n_particles"].unique()):
            sub = subd[subd["n_particles"] == n].sort_values("alpha")
            plt.plot(sub["alpha"], sub["energy_per_particle"], marker="o", label=f"N={n}")
        plt.axhline(d / 2, linestyle="--", label=f"Exact E/N = {d/2:.1f}")
        plt.xlabel("alpha")
        plt.ylabel("Energy per particle")
        plt.title(f"Project 1b: Brute force VMC in {d}D")
        plt.legend()
        savefig(f"project1_b_energy_{d}D.png")

    for d in sorted(df["dimensions"].unique()):
        plt.figure()
        subd = df[df["dimensions"] == d]
        for n in sorted(subd["n_particles"].unique()):
            sub = subd[subd["n_particles"] == n].sort_values("alpha")
            plt.plot(sub["alpha"], sub["acceptance_rate"], marker="o", label=f"N={n}")
        plt.xlabel("alpha")
        plt.ylabel("Acceptance rate")
        plt.title(f"Project 1b: Acceptance rate in {d}D")
        plt.legend()
        savefig(f"project1_b_acceptance_{d}D.png")


# -----------------------------------
# Project 1c: importance sampling VMC
# -----------------------------------
path = RESULTS / "project1_c_importance.csv"
if path.exists():
    df = pd.read_csv(path)

    for d in sorted(df["dimensions"].unique()):
        plt.figure()
        sub = df[df["dimensions"] == d].sort_values("time_step")
        plt.plot(sub["time_step"], sub["energy_per_particle"], marker="o")
        plt.axhline(d / 2, linestyle="--", label=f"Exact E/N = {d/2:.1f}")
        plt.xlabel("Time step")
        plt.ylabel("Energy per particle")
        plt.title(f"Project 1c: Importance sampling energy in {d}D")
        plt.legend()
        savefig(f"project1_c_energy_{d}D.png")

        plt.figure()
        plt.plot(sub["time_step"], sub["acceptance_rate"], marker="o")
        plt.xlabel("Time step")
        plt.ylabel("Acceptance rate")
        plt.title(f"Project 1c: Acceptance rate in {d}D")
        savefig(f"project1_c_acceptance_{d}D.png")


# -------------------------
# Project 1d: optimization
# -------------------------
path = RESULTS / "project1_d_optimization.csv"
if path.exists():
    df = pd.read_csv(path).sort_values("iteration")

    plt.figure()
    plt.plot(df["iteration"], df["alpha"], marker="o")
    plt.axhline(0.5, linestyle="--", label="Expected optimum ~ 0.5")
    plt.xlabel("Iteration")
    plt.ylabel("alpha")
    plt.title("Project 1d: Optimization path")
    plt.legend()
    savefig("project1_d_alpha_vs_iteration.png")

    plt.figure()
    plt.plot(df["iteration"], df["energy"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Project 1d: Energy during optimization")
    savefig("project1_d_energy_vs_iteration.png")


# ----------------------------
# Project 1g: interacting case
# ----------------------------
path = RESULTS / "project1_g_interacting.csv"
if path.exists():
    df = pd.read_csv(path)

    for n in sorted(df["n_particles"].unique()):
        plt.figure()
        sub = df[df["n_particles"] == n].sort_values("alpha")
        plt.plot(sub["alpha"], sub["energy_per_particle"], marker="o", label="Interacting")
        if "ideal_noninteracting_energy" in sub.columns:
            ideal_per_particle = sub["ideal_noninteracting_energy"] / sub["n_particles"]
            plt.plot(sub["alpha"], ideal_per_particle, linestyle="--", marker="o", label="Ideal non-interacting")
        plt.xlabel("alpha")
        plt.ylabel("Energy per particle")
        plt.title(f"Project 1g: Interacting case, N={n}")
        plt.legend()
        savefig(f"project1_g_energy_N{n}.png")


# -------------------------
# Project 1h: one-body density
# -------------------------
path = RESULTS / "project1_h_density.csv"
if path.exists():
    df = pd.read_csv(path)

    possible_x = ["r", "radius", "bin_center", "x"]
    possible_y = ["density", "rho", "onebody_density"]
    possible_group = ["correlated", "with_jastrow", "jastrow", "case", "label"]

    xcol = next((c for c in possible_x if c in df.columns), None)
    ycol = next((c for c in possible_y if c in df.columns), None)
    gcol = next((c for c in possible_group if c in df.columns), None)

    if xcol and ycol:
        plt.figure()
        if gcol:
            for key, sub in df.groupby(gcol):
                sub = sub.sort_values(xcol)
                plt.plot(sub[xcol], sub[ycol], marker="o", label=str(key))
            plt.legend()
        else:
            df = df.sort_values(xcol)
            plt.plot(df[xcol], df[ycol], marker="o")
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title("Project 1h: One-body density")
        savefig("project1_h_density.png")


print(f"Plots saved in: {PLOTS.resolve()}")
