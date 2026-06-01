#!/usr/bin/env python
"""Project 1b: brute-force Metropolis calculations for the ideal trap.

The script scans alpha for selected particle numbers and dimensions and writes
results to data/results/project1_b_bruteforce.csv. Use --quick for a fast smoke
run; remove --quick and increase --cycles for production data.
"""

from __future__ import annotations

import argparse

import numpy as np

from fys4411_project1.io_utils import write_dicts_csv
from fys4411_project1.local_energy import (
    exact_energy_noninteracting,
    exact_variational_energy_noninteracting,
)
from fys4411_project1.samplers import VMCConfig, run_vmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a short smoke-test setup")
    parser.add_argument("--cycles", type=int, default=20_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/results/project1_b_bruteforce.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = [1, 10] if args.quick else [1, 10, 100, 500]
    dimensions_values = [1, 2, 3]
    alpha_values = [0.4, 0.5, 0.6] if args.quick else np.linspace(0.3, 0.7, 9)
    cycles = 2_000 if args.quick else args.cycles
    burn_in = 200 if args.quick else args.burn_in

    rows = []
    run_number = 0
    for dimensions in dimensions_values:
        for n_particles in n_values:
            for alpha in alpha_values:
                config = VMCConfig(
                    n_particles=n_particles,
                    dimensions=dimensions,
                    alpha=float(alpha),
                    beta=1.0,
                    gamma=1.0,
                    hard_core_radius=0.0,
                    cycles=cycles,
                    burn_in=burn_in,
                    step_size=args.step_size,
                    sampler="brute",
                    local_energy_mode="analytic",
                    seed=args.seed + run_number,
                )
                result = run_vmc(config)
                exact_ground = exact_energy_noninteracting(n_particles, dimensions)
                exact_variational = exact_variational_energy_noninteracting(
                    n_particles, dimensions, alpha=float(alpha)
                )
                rows.append(
                    {
                        "n_particles": n_particles,
                        "dimensions": dimensions,
                        "alpha": float(alpha),
                        "sampler": "brute",
                        "mean_energy": result.mean_energy,
                        "energy_per_particle": result.energy_per_particle,
                        "standard_error_naive": result.standard_error,
                        "variance": result.variance,
                        "acceptance_rate": result.acceptance_rate,
                        "elapsed_time": result.elapsed_time,
                        "exact_ground_energy": exact_ground,
                        "exact_variational_energy": exact_variational,
                        "energy_error_vs_variational": result.mean_energy - exact_variational,
                    }
                )
                print(rows[-1])
                run_number += 1

    write_dicts_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
