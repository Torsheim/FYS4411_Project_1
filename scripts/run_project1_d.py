#!/usr/bin/env python
"""Project 1d: optimize alpha with stochastic gradient descent."""

from __future__ import annotations

import argparse

from fys4411_project1.optimization import gradient_descent_alpha, save_optimization_csv
from fys4411_project1.samplers import VMCConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--initial-alpha", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=20_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--dimensions", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--sampler", choices=["brute", "importance"], default="importance")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--output", default="data/results/project1_d_optimization.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycles = 2_000 if args.quick else args.cycles
    burn_in = 200 if args.quick else args.burn_in
    iterations = 4 if args.quick else args.iterations

    config = VMCConfig(
        n_particles=args.n_particles,
        dimensions=args.dimensions,
        alpha=args.initial_alpha,
        beta=1.0,
        gamma=1.0,
        hard_core_radius=0.0,
        cycles=cycles,
        burn_in=burn_in,
        step_size=1.0,
        time_step=0.05,
        sampler=args.sampler,
        local_energy_mode="analytic",
        seed=args.seed,
    )
    steps = gradient_descent_alpha(
        initial_alpha=args.initial_alpha,
        base_config=config,
        learning_rate=args.learning_rate,
        iterations=iterations,
        min_alpha=0.1,
        max_alpha=1.5,
    )
    for step in steps:
        print(step)
    save_optimization_csv(steps, args.output)
    print(f"Saved {len(steps)} optimization steps to {args.output}")


if __name__ == "__main__":
    main()
