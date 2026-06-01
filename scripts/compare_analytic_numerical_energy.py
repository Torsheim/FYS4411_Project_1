#!/usr/bin/env python
"""Compare analytic and finite-difference local energies for Project 1b."""

from __future__ import annotations

import argparse

from fys4411_project1.io_utils import write_dicts_csv
from fys4411_project1.samplers import VMCConfig, run_vmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3_000)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--dimensions", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--output", default="data/results/analytic_vs_numerical_energy.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for mode in ["analytic", "numerical"]:
        config = VMCConfig(
            n_particles=args.n_particles,
            dimensions=args.dimensions,
            alpha=args.alpha,
            beta=1.0,
            gamma=1.0,
            hard_core_radius=0.0,
            cycles=args.cycles,
            burn_in=args.burn_in,
            step_size=1.0,
            sampler="brute",
            local_energy_mode=mode,
            seed=args.seed,
        )
        result = run_vmc(config)
        row = {
            "mode": mode,
            "n_particles": args.n_particles,
            "dimensions": args.dimensions,
            "alpha": args.alpha,
            "mean_energy": result.mean_energy,
            "standard_error_naive": result.standard_error,
            "variance": result.variance,
            "acceptance_rate": result.acceptance_rate,
            "elapsed_time": result.elapsed_time,
        }
        rows.append(row)
        print(row)
    write_dicts_csv(rows, args.output)
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
