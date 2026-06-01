"""Variational Monte Carlo samplers.

Two samplers are implemented:

1. Brute-force Metropolis sampling.
2. Importance sampling using a Langevin proposal and the Fokker-Planck
   Green's-function correction.

Both samplers move one particle at a time. Energies are recorded once per
Monte Carlo cycle after all particles have been offered a move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fys4411_project1.local_energy import LocalEnergyMode, local_energy
from fys4411_project1.wavefunction import (
    drift_force_particle,
    has_hard_core_overlap,
    log_derivative_alpha,
    log_wavefunction,
)

SamplerName = Literal["brute", "importance"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VMCConfig:
    """Configuration for a VMC run."""

    n_particles: int = 1
    dimensions: int = 3
    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 1.0
    hard_core_radius: float = 0.0
    cycles: int = 10_000
    burn_in: int = 1_000
    step_size: float = 1.0
    time_step: float = 0.05
    diffusion_constant: float = 0.5
    sampler: SamplerName = "brute"
    local_energy_mode: LocalEnergyMode = "analytic"
    numerical_step: float = 1.0e-4
    seed: int | None = None
    store_positions: bool = False
    store_alpha_derivative: bool = False

    def validated(self) -> "VMCConfig":
        """Return a validated copy of the config."""

        if self.n_particles < 1:
            raise ValueError("n_particles must be positive")
        if self.dimensions not in (1, 2, 3):
            raise ValueError("dimensions must be 1, 2, or 3")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.hard_core_radius < 0.0:
            raise ValueError("hard_core_radius cannot be negative")
        if self.cycles < 1:
            raise ValueError("cycles must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in cannot be negative")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if self.diffusion_constant <= 0.0:
            raise ValueError("diffusion_constant must be positive")
        if self.sampler not in ("brute", "importance"):
            raise ValueError("sampler must be 'brute' or 'importance'")
        if self.local_energy_mode not in ("analytic", "numerical"):
            raise ValueError("local_energy_mode must be 'analytic' or 'numerical'")
        return self


@dataclass
class VMCRunResult:
    """Container for VMC results."""

    energies: FloatArray
    acceptance_rate: float
    elapsed_time: float
    config: VMCConfig
    alpha_derivatives: FloatArray | None = None
    positions: FloatArray | None = None

    @property
    def mean_energy(self) -> float:
        return float(np.mean(self.energies))

    @property
    def variance(self) -> float:
        if self.energies.size < 2:
            return float("nan")
        return float(np.var(self.energies, ddof=1))

    @property
    def standard_error(self) -> float:
        if self.energies.size < 2:
            return float("nan")
        return float(np.std(self.energies, ddof=1) / np.sqrt(self.energies.size))

    @property
    def energy_per_particle(self) -> float:
        return self.mean_energy / self.config.n_particles

    def summary(self) -> dict[str, float | int | str | None]:
        """Return a flat summary dictionary."""

        return {
            "sampler": self.config.sampler,
            "n_particles": self.config.n_particles,
            "dimensions": self.config.dimensions,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "gamma": self.config.gamma,
            "hard_core_radius": self.config.hard_core_radius,
            "cycles": self.config.cycles,
            "burn_in": self.config.burn_in,
            "mean_energy": self.mean_energy,
            "energy_per_particle": self.energy_per_particle,
            "standard_error_naive": self.standard_error,
            "variance": self.variance,
            "acceptance_rate": self.acceptance_rate,
            "elapsed_time": self.elapsed_time,
            "local_energy_mode": self.config.local_energy_mode,
            "seed": self.config.seed,
        }

    def print_summary(self) -> None:
        """Pretty-print a summary to stdout."""

        print(json.dumps(self.summary(), indent=2, sort_keys=True))

    def save_energy_csv(self, path: str | Path) -> None:
        """Save sampled energies to CSV."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["cycle", "energy"])
            for cycle, energy in enumerate(self.energies):
                writer.writerow([cycle, f"{energy:.16e}"])

    def save_summary_json(self, path: str | Path) -> None:
        """Save run summary and full config to JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "config": asdict(self.config),
        }
        output.write_text(json.dumps(data, indent=2, sort_keys=True))


def initial_positions(
    config: VMCConfig,
    rng: np.random.Generator,
    max_attempts: int = 20_000,
) -> FloatArray:
    """Generate an initial configuration with no hard-core overlap."""

    # Approximate width of |exp(-alpha x^2)|^2 = exp(-2 alpha x^2).
    sigma = 1.0 / math.sqrt(4.0 * config.alpha)

    for _ in range(max_attempts):
        positions = rng.normal(0.0, sigma, size=(config.n_particles, config.dimensions))
        if not has_hard_core_overlap(positions, config.hard_core_radius):
            return positions.astype(float)

    raise RuntimeError(
        "could not generate non-overlapping initial positions; try a smaller "
        "hard_core_radius or a more dilute initial distribution"
    )


def run_vmc(config: VMCConfig) -> VMCRunResult:
    """Run a VMC simulation according to config."""

    config = config.validated()
    rng = np.random.default_rng(config.seed)

    if config.sampler == "brute":
        return _run_brute_force(config, rng)
    return _run_importance_sampling(config, rng)


def _record_sample(
    positions: FloatArray,
    config: VMCConfig,
    energies: FloatArray,
    sample_index: int,
    stored_positions: list[FloatArray] | None,
    alpha_derivatives: FloatArray | None,
) -> None:
    energies[sample_index] = local_energy(
        positions,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        hard_core_radius=config.hard_core_radius,
        mode=config.local_energy_mode,
        numerical_step=config.numerical_step,
    )
    if stored_positions is not None:
        stored_positions.append(positions.copy())
    if alpha_derivatives is not None:
        alpha_derivatives[sample_index] = log_derivative_alpha(positions, beta=config.beta)


def _run_brute_force(config: VMCConfig, rng: np.random.Generator) -> VMCRunResult:
    positions = initial_positions(config, rng)
    log_psi = log_wavefunction(
        positions,
        alpha=config.alpha,
        beta=config.beta,
        hard_core_radius=config.hard_core_radius,
    )
    if log_psi == -math.inf:
        raise RuntimeError("initial wave function was zero")

    energies = np.empty(config.cycles, dtype=float)
    alpha_derivatives = np.empty(config.cycles, dtype=float) if config.store_alpha_derivative else None
    stored_positions: list[FloatArray] | None = [] if config.store_positions else None

    accepted = 0
    attempted = 0
    sample_index = 0
    start = time.perf_counter()

    total_cycles = config.burn_in + config.cycles
    for cycle in range(total_cycles):
        for particle in range(config.n_particles):
            old_particle_position = positions[particle].copy()
            proposal = old_particle_position + rng.uniform(
                -config.step_size, config.step_size, size=config.dimensions
            )
            positions[particle] = proposal
            new_log_psi = log_wavefunction(
                positions,
                alpha=config.alpha,
                beta=config.beta,
                hard_core_radius=config.hard_core_radius,
            )

            attempted += 1
            log_acceptance = 2.0 * (new_log_psi - log_psi)
            if new_log_psi != -math.inf and math.log(rng.random()) < min(0.0, log_acceptance):
                log_psi = new_log_psi
                accepted += 1
            else:
                positions[particle] = old_particle_position

        if cycle >= config.burn_in:
            _record_sample(
                positions,
                config,
                energies,
                sample_index,
                stored_positions,
                alpha_derivatives,
            )
            sample_index += 1

    elapsed = time.perf_counter() - start
    return VMCRunResult(
        energies=energies,
        acceptance_rate=accepted / attempted,
        elapsed_time=elapsed,
        config=config,
        alpha_derivatives=alpha_derivatives,
        positions=np.asarray(stored_positions) if stored_positions is not None else None,
    )


def _run_importance_sampling(config: VMCConfig, rng: np.random.Generator) -> VMCRunResult:
    positions = initial_positions(config, rng)
    log_psi = log_wavefunction(
        positions,
        alpha=config.alpha,
        beta=config.beta,
        hard_core_radius=config.hard_core_radius,
    )
    if log_psi == -math.inf:
        raise RuntimeError("initial wave function was zero")

    energies = np.empty(config.cycles, dtype=float)
    alpha_derivatives = np.empty(config.cycles, dtype=float) if config.store_alpha_derivative else None
    stored_positions: list[FloatArray] | None = [] if config.store_positions else None

    accepted = 0
    attempted = 0
    sample_index = 0
    start = time.perf_counter()

    dt = config.time_step
    sqrt_dt = math.sqrt(dt)
    diffusion = config.diffusion_constant
    greens_denominator = 4.0 * diffusion * dt

    total_cycles = config.burn_in + config.cycles
    for cycle in range(total_cycles):
        for particle in range(config.n_particles):
            old_particle_position = positions[particle].copy()
            old_force = drift_force_particle(
                positions,
                particle=particle,
                alpha=config.alpha,
                beta=config.beta,
                hard_core_radius=config.hard_core_radius,
            )

            proposal = (
                old_particle_position
                + diffusion * old_force * dt
                + rng.normal(0.0, sqrt_dt, size=config.dimensions)
            )
            positions[particle] = proposal
            new_log_psi = log_wavefunction(
                positions,
                alpha=config.alpha,
                beta=config.beta,
                hard_core_radius=config.hard_core_radius,
            )

            attempted += 1
            if new_log_psi == -math.inf:
                positions[particle] = old_particle_position
                continue

            new_force = drift_force_particle(
                positions,
                particle=particle,
                alpha=config.alpha,
                beta=config.beta,
                hard_core_radius=config.hard_core_radius,
            )

            if not np.all(np.isfinite(new_force)):
                positions[particle] = old_particle_position
                continue

            forward_difference = proposal - old_particle_position - diffusion * old_force * dt
            backward_difference = old_particle_position - proposal - diffusion * new_force * dt
            log_greens_ratio = (
                float(np.dot(forward_difference, forward_difference))
                - float(np.dot(backward_difference, backward_difference))
            ) / greens_denominator
            log_acceptance = 2.0 * (new_log_psi - log_psi) + log_greens_ratio

            if math.log(rng.random()) < min(0.0, log_acceptance):
                log_psi = new_log_psi
                accepted += 1
            else:
                positions[particle] = old_particle_position

        if cycle >= config.burn_in:
            _record_sample(
                positions,
                config,
                energies,
                sample_index,
                stored_positions,
                alpha_derivatives,
            )
            sample_index += 1

    elapsed = time.perf_counter() - start
    return VMCRunResult(
        energies=energies,
        acceptance_rate=accepted / attempted,
        elapsed_time=elapsed,
        config=config,
        alpha_derivatives=alpha_derivatives,
        positions=np.asarray(stored_positions) if stored_positions is not None else None,
    )


def with_updates(config: VMCConfig, **kwargs: object) -> VMCConfig:
    """Return a copy of config with selected fields replaced."""

    return replace(config, **kwargs)
