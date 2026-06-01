#!/usr/bin/env python
"""Project 1g: hard-core repulsive interaction in an elliptical trap.

The default interacting parameters follow the project text:
    a/a_ho = 0.0043,
    beta = gamma = 2.82843.

The script scans alpha and writes data to data/results/project1_g_interacting.csv.
"""

from __future__ import annotations

import argparse

import numpy as np

from fys4411_project1.io_utils import write_dicts_csv
from fys4411_project1.local_energy import exact_energy_noninteracting
from fys4411_project1.samplers import VMCConfig, run_vmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cycles", type=int, default=30_000)
    parser.add_argument("--burn-in", type=int, default=3_000)
    parser.add_argument("--hard-core-radius", type=float, default=0.0043)
    parser.add_argument("--beta", type=float, default=2.82843)
    parser.add_argument("--gamma", type=float, default=2.82843)
    parser.add_argument("--time-step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=9988)
    parser.add_argument("--output", default="data/results/project1_g_interacting.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = [10] if args.quick else [10, 50, 100]
    alpha_values = [0.45, 0.5, 0.55] if args.quick else np.linspace(0.35, 0.65, 7)
    cycles = 2_000 if args.quick else args.cycles
    burn_in = 300 if args.quick else args.burn_in

    rows = []
    run_number = 0
    for n_particles in n_values:
        ideal_energy = exact_energy_noninteracting(n_particles, dimensions=3, gamma=args.gamma)
        for alpha in alpha_values:
            config = VMCConfig(
                n_particles=n_particles,
                dimensions=3,
                alpha=float(alpha),
                beta=args.beta,
                gamma=args.gamma,
                hard_core_radius=args.hard_core_radius,
                cycles=cycles,
                burn_in=burn_in,
                sampler="importance",
                time_step=args.time_step,
                local_energy_mode="analytic",
                seed=args.seed + run_number,
            )
            result = run_vmc(config)
            rows.append(
                {
                    "n_particles": n_particles,
                    "dimensions": 3,
                    "alpha": float(alpha),
                    "beta": args.beta,
                    "gamma": args.gamma,
                    "hard_core_radius": args.hard_core_radius,
                    "mean_energy": result.mean_energy,
                    "energy_per_particle": result.energy_per_particle,
                    "standard_error_naive": result.standard_error,
                    "variance": result.variance,
                    "acceptance_rate": result.acceptance_rate,
                    "elapsed_time": result.elapsed_time,
                    "ideal_noninteracting_energy": ideal_energy,
                    "difference_from_ideal": result.mean_energy - ideal_energy,
                }
            )
            print(rows[-1])
            run_number += 1

    write_dicts_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
