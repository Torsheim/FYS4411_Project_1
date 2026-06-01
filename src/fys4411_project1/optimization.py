"""Optimization routines for variational parameters."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from fys4411_project1.samplers import VMCConfig, VMCRunResult, run_vmc, with_updates


@dataclass(frozen=True)
class ScanPoint:
    alpha: float
    mean_energy: float
    standard_error: float
    variance: float
    acceptance_rate: float
    elapsed_time: float


@dataclass(frozen=True)
class OptimizationStep:
    iteration: int
    alpha: float
    energy: float
    gradient: float
    learning_rate: float
    acceptance_rate: float


def energy_gradient_alpha(result: VMCRunResult) -> float:
    """Estimate d<E>/d alpha from sampled energies and O_alpha.

    The standard VMC identity is

        d<E>/d alpha = 2 ( <E_L O_alpha> - <E_L><O_alpha> ),

    where O_alpha = d log(Psi_T)/d alpha.
    """

    if result.alpha_derivatives is None:
        raise ValueError("result does not contain alpha derivatives")
    energies = result.energies
    operators = result.alpha_derivatives
    return float(2.0 * (np.mean(energies * operators) - np.mean(energies) * np.mean(operators)))


def scan_alpha(alpha_values: ArrayLike, base_config: VMCConfig) -> list[ScanPoint]:
    """Run independent VMC calculations for many alpha values."""

    points: list[ScanPoint] = []
    for alpha in np.asarray(alpha_values, dtype=float):
        config = with_updates(base_config, alpha=float(alpha), store_alpha_derivative=False)
        result = run_vmc(config)
        points.append(
            ScanPoint(
                alpha=float(alpha),
                mean_energy=result.mean_energy,
                standard_error=result.standard_error,
                variance=result.variance,
                acceptance_rate=result.acceptance_rate,
                elapsed_time=result.elapsed_time,
            )
        )
    return points


def best_scan_point(points: list[ScanPoint]) -> ScanPoint:
    """Return the scan point with the lowest mean energy."""

    if not points:
        raise ValueError("points cannot be empty")
    return min(points, key=lambda point: point.mean_energy)


def gradient_descent_alpha(
    initial_alpha: float,
    base_config: VMCConfig,
    learning_rate: float = 0.05,
    iterations: int = 10,
    min_alpha: float = 0.05,
    max_alpha: float = 2.0,
) -> list[OptimizationStep]:
    """Optimize alpha using stochastic gradient descent from VMC samples."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    alpha = float(initial_alpha)
    steps: list[OptimizationStep] = []

    for iteration in range(iterations):
        config = with_updates(base_config, alpha=alpha, store_alpha_derivative=True)
        result = run_vmc(config)
        gradient = energy_gradient_alpha(result)
        steps.append(
            OptimizationStep(
                iteration=iteration,
                alpha=alpha,
                energy=result.mean_energy,
                gradient=gradient,
                learning_rate=learning_rate,
                acceptance_rate=result.acceptance_rate,
            )
        )
        alpha = float(np.clip(alpha - learning_rate * gradient, min_alpha, max_alpha))

    return steps


def save_scan_csv(points: list[ScanPoint], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "alpha",
                "mean_energy",
                "standard_error",
                "variance",
                "acceptance_rate",
                "elapsed_time",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.alpha,
                    point.mean_energy,
                    point.standard_error,
                    point.variance,
                    point.acceptance_rate,
                    point.elapsed_time,
                ]
            )


def save_optimization_csv(steps: list[OptimizationStep], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "alpha", "energy", "gradient", "learning_rate", "acceptance_rate"])
        for step in steps:
            writer.writerow(
                [
                    step.iteration,
                    step.alpha,
                    step.energy,
                    step.gradient,
                    step.learning_rate,
                    step.acceptance_rate,
                ]
            )
