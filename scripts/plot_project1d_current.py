from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path("data/results")
plots = Path("plots")
plots.mkdir(exist_ok=True)

df = pd.read_csv(results / "project1_d_optimization.csv")
df = df.sort_values("iteration")

print(df[["iteration", "alpha", "energy", "gradient", "learning_rate"]].to_string(index=False))

plt.figure()
plt.plot(df["iteration"], df["alpha"], marker="o", label="Optimized alpha")
plt.axhline(0.5, linestyle="--", label="Exact optimum = 0.5")
plt.xlabel("Iteration")
plt.ylabel(r"$\alpha$")
plt.title("Project 1d: Optimized variational parameter")
plt.legend()
plt.tight_layout()
plt.savefig(plots / "project1_d_CURRENT_alpha_vs_iteration.png", dpi=200)
plt.close()

plt.figure()
plt.plot(df["iteration"], df["energy"], marker="o", label="VMC energy")
plt.axhline(150.0, linestyle="--", label="Exact energy = 150")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Project 1d: Energy during optimization")
plt.legend()
plt.tight_layout()
plt.savefig(plots / "project1_d_CURRENT_energy_vs_iteration.png", dpi=200)
plt.close()

plt.figure()
plt.plot(df["iteration"], abs(df["gradient"]), marker="o")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel(r"$|\partial E / \partial \alpha|$")
plt.title("Project 1d: Gradient magnitude")
plt.tight_layout()
plt.savefig(plots / "project1_d_CURRENT_gradient_vs_iteration.png", dpi=200)
plt.close()

print()
print("Saved:")
print(plots / "project1_d_CURRENT_alpha_vs_iteration.png")
print(plots / "project1_d_CURRENT_energy_vs_iteration.png")
print(plots / "project1_d_CURRENT_gradient_vs_iteration.png")
