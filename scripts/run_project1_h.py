#!/usr/bin/env python
"""Project 1h: compute one-body densities with and without the Jastrow factor."""

from __future__ import annotations

import argparse

from fys4411_project1.density import radial_density, save_density_csv
from fys4411_project1.samplers import VMCConfig, run_vmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=2.82843)
    parser.add_argument("--gamma", type=float, default=2.82843)
    parser.add_argument("--hard-core-radius", type=float, default=0.0043)
    parser.add_argument("--cycles", type=int, default=20_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def run_density_case(args: argparse.Namespace, hard_core_radius: float, label: str) -> None:
    cycles = 2_000 if args.quick else args.cycles
    burn_in = 300 if args.quick else args.burn_in
    config = VMCConfig(
        n_particles=args.n_particles,
        dimensions=3,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        hard_core_radius=hard_core_radius,
        cycles=cycles,
        burn_in=burn_in,
        sampler="importance",
        time_step=0.02,
        local_energy_mode="analytic",
        seed=args.seed + (1 if hard_core_radius > 0.0 else 0),
        store_positions=True,
    )
    result = run_vmc(config)
    assert result.positions is not None
    density = radial_density(result.positions, bins=args.bins, dimensions=3)
    output = f"data/results/project1_h_radial_density_{label}.csv"
    save_density_csv(density, output)
    print(f"{label}: E = {result.mean_energy}, acceptance = {result.acceptance_rate}")
    print(f"Saved radial density to {output}")


def main() -> None:
    args = parse_args()
    run_density_case(args, hard_core_radius=0.0, label="no_jastrow")
    run_density_case(args, hard_core_radius=args.hard_core_radius, label="jastrow")


if __name__ == "__main__":
    main()
