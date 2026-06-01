#!/usr/bin/env python
"""Generate pair-distance histograms with and without the hard-core Jastrow factor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fys4411_project1.samplers import VMCConfig, run_vmc


def pair_distances(samples: np.ndarray) -> np.ndarray:
    """Return all pair distances from stored samples.

    samples has shape (n_samples, n_particles, dimensions).
    """
    distances: list[np.ndarray] = []
    for positions in samples:
        n = positions.shape[0]
        for i in range(n - 1):
            diff = positions[i + 1 :] - positions[i]
            distances.append(np.linalg.norm(diff, axis=1))
    return np.concatenate(distances)


def histogram(distances: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    counts, edges = np.histogram(distances, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame({"r": centers, "probability_density": counts})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=12000)
    parser.add_argument("--burn-in", type=int, default=1200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=2.82843)
    parser.add_argument("--gamma", type=float, default=2.82843)
    parser.add_argument("--hard-core-radius", type=float, default=0.0043)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--r-max", type=float, default=4.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--output", default="data/results/project1_h_pair_distance.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bins = np.linspace(0.0, args.r_max, args.bins + 1)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []

    cases = [
        ("no_jastrow", 0.0, args.seed),
        ("jastrow", args.hard_core_radius, args.seed + 1),
    ]

    for label, hard_core, seed in cases:
        config = VMCConfig(
            n_particles=args.n_particles,
            dimensions=3,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            hard_core_radius=hard_core,
            cycles=args.cycles,
            burn_in=args.burn_in,
            step_size=args.step_size,
            sampler="brute",
            seed=seed,
            store_positions=True,
        )
        result = run_vmc(config)
        distances = pair_distances(result.positions)
        df = histogram(distances, bins=bins)
        df.insert(0, "case", label)
        frames.append(df)
        summaries.append(
            {
                "case": label,
                "n_particles": args.n_particles,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "hard_core_radius": hard_core,
                "cycles": args.cycles,
                "burn_in": args.burn_in,
                "mean_pair_distance": float(np.mean(distances)),
                "median_pair_distance": float(np.median(distances)),
                "minimum_sampled_pair_distance": float(np.min(distances)),
                "acceptance_rate": result.acceptance_rate,
                "elapsed_time": result.elapsed_time,
            }
        )
        print(summaries[-1])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    summary_output = output.with_name(output.stem + "_summary.csv")
    pd.DataFrame(summaries).to_csv(summary_output, index=False)
    print(f"Saved {output}")
    print(f"Saved {summary_output}")


if __name__ == "__main__":
    main()
