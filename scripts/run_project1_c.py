#!/usr/bin/env python
"""Project 1c: importance-sampling timestep study for the ideal trap."""

from __future__ import annotations

import argparse

from fys4411_project1.io_utils import write_dicts_csv
from fys4411_project1.local_energy import exact_energy_noninteracting
from fys4411_project1.samplers import VMCConfig, run_vmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cycles", type=int, default=20_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--output", default="data/results/project1_c_importance.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_steps = [0.01, 0.05] if args.quick else [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    dimensions_values = [1, 2, 3]
    cycles = 2_000 if args.quick else args.cycles
    burn_in = 200 if args.quick else args.burn_in

    rows = []
    run_number = 0
    for dimensions in dimensions_values:
        for time_step in time_steps:
            config = VMCConfig(
                n_particles=args.n_particles,
                dimensions=dimensions,
                alpha=args.alpha,
                beta=1.0,
                gamma=1.0,
                hard_core_radius=0.0,
                cycles=cycles,
                burn_in=burn_in,
                time_step=time_step,
                sampler="importance",
                local_energy_mode="analytic",
                seed=args.seed + run_number,
            )
            result = run_vmc(config)
            exact = exact_energy_noninteracting(args.n_particles, dimensions)
            rows.append(
                {
                    "n_particles": args.n_particles,
                    "dimensions": dimensions,
                    "alpha": args.alpha,
                    "time_step": time_step,
                    "sampler": "importance",
                    "mean_energy": result.mean_energy,
                    "energy_per_particle": result.energy_per_particle,
                    "standard_error_naive": result.standard_error,
                    "variance": result.variance,
                    "acceptance_rate": result.acceptance_rate,
                    "elapsed_time": result.elapsed_time,
                    "exact_ground_energy": exact,
                    "energy_error_vs_exact": result.mean_energy - exact,
                }
            )
            print(rows[-1])
            run_number += 1

    write_dicts_csv(rows, args.output)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
