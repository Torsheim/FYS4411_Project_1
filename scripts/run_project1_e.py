#!/usr/bin/env python
"""Project 1e: blocking and bootstrap analysis for VMC energies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fys4411_project1.io_utils import read_energy_csv
from fys4411_project1.samplers import VMCConfig, run_vmc
from fys4411_project1.statistics import blocking_analysis, bootstrap_mean, save_blocking_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--input", default=None, help="Optional energy CSV from scripts/run_vmc.py")
    parser.add_argument("--cycles", type=int, default=50_000)
    parser.add_argument("--burn-in", type=int, default=5_000)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--blocking-output", default="data/results/project1_e_blocking.csv")
    parser.add_argument("--summary-output", default="data/results/project1_e_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is not None:
        energies = read_energy_csv(args.input)
    else:
        cycles = 4_000 if args.quick else args.cycles
        burn_in = 400 if args.quick else args.burn_in
        config = VMCConfig(
            n_particles=args.n_particles,
            dimensions=3,
            alpha=args.alpha,
            beta=1.0,
            gamma=1.0,
            hard_core_radius=0.0,
            cycles=cycles,
            burn_in=burn_in,
            sampler="importance",
            time_step=0.05,
            local_energy_mode="analytic",
            seed=args.seed,
        )
        result = run_vmc(config)
        energies = result.energies
        result.save_energy_csv("data/raw/project1_e_energies.csv")

    blocking = blocking_analysis(energies)
    bootstrap_mean_value, bootstrap_error = bootstrap_mean(energies, n_bootstrap=1000, seed=args.seed)
    save_blocking_csv(blocking, args.blocking_output)

    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.summary_output).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["quantity", "value"])
        writer.writerow(["mean", bootstrap_mean_value])
        writer.writerow(["bootstrap_error", bootstrap_error])
        writer.writerow(["blocking_final_error", blocking.final_error])

    print(f"Mean energy: {bootstrap_mean_value}")
    print(f"Bootstrap error: {bootstrap_error}")
    print(f"Final blocking error: {blocking.final_error}")
    print(f"Saved blocking data to {args.blocking_output}")
    print(f"Saved summary to {args.summary_output}")


if __name__ == "__main__":
    main()
