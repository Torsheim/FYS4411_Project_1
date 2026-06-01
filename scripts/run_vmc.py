#!/usr/bin/env python
"""Run one VMC calculation from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from fys4411_project1.samplers import VMCConfig, run_vmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", choices=["brute", "importance"], default="brute")
    parser.add_argument("--n-particles", type=int, default=1)
    parser.add_argument("--dimensions", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--hard-core-radius", type=float, default=0.0)
    parser.add_argument("--cycles", type=int, default=10_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument("--local-energy-mode", choices=["analytic", "numerical"], default="analytic")
    parser.add_argument("--numerical-step", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for sampled energies")
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON path for summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VMCConfig(
        n_particles=args.n_particles,
        dimensions=args.dimensions,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        hard_core_radius=args.hard_core_radius,
        cycles=args.cycles,
        burn_in=args.burn_in,
        step_size=args.step_size,
        time_step=args.time_step,
        sampler=args.sampler,
        local_energy_mode=args.local_energy_mode,
        numerical_step=args.numerical_step,
        seed=args.seed,
    )
    result = run_vmc(config)
    result.print_summary()
    if args.output is not None:
        result.save_energy_csv(args.output)
        print(f"Saved energies to {args.output}")
    if args.summary is not None:
        result.save_summary_json(args.summary)
        print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
