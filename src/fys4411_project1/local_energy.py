"""Analytic and numerical local-energy expressions.

Natural units are used throughout: hbar = m = omega_perp = 1.

For a spherical non-interacting harmonic oscillator in d dimensions,

    H = 1/2 sum_i (-nabla_i^2 + r_i^2),

and the Gaussian trial function exp(-alpha r^2) gives

    E_L = alpha N d + (1/2 - 2 alpha^2) sum_i r_i^2.

For the interacting three-dimensional problem with an elliptical trap,

    H = sum_i 1/2 (-nabla_i^2 + x_i^2 + y_i^2 + gamma^2 z_i^2)
        + sum_{i<j} V_int(r_ij),

where V_int is represented by the hard-core boundary condition in the
Jastrow factor.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from fys4411_project1.wavefunction import (
    as_positions,
    gradient_log_gaussian,
    has_hard_core_overlap,
    log_wavefunction,
)

LocalEnergyMode = Literal["analytic", "numerical"]


def external_potential(positions: ArrayLike, gamma: float = 1.0) -> float:
    """Return the harmonic oscillator external potential.

    In 1D and 2D this is 1/2 sum r_i^2. In 3D this is
    1/2 sum (x_i^2 + y_i^2 + gamma^2 z_i^2).
    """

    r = as_positions(positions)
    if r.shape[1] == 1:
        return float(0.5 * np.sum(r[:, 0] ** 2))
    if r.shape[1] == 2:
        return float(0.5 * np.sum(r[:, 0] ** 2 + r[:, 1] ** 2))
    return float(0.5 * np.sum(r[:, 0] ** 2 + r[:, 1] ** 2 + gamma**2 * r[:, 2] ** 2))


def laplacian_gaussian_over_gaussian(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
) -> np.ndarray:
    """Return (nabla_i^2 phi_i) / phi_i for each particle."""

    r = as_positions(positions)
    alpha = float(alpha)
    n_particles, dimensions = r.shape
    values = np.empty(n_particles, dtype=float)

    if dimensions == 1:
        values[:] = 4.0 * alpha**2 * r[:, 0] ** 2 - 2.0 * alpha
    elif dimensions == 2:
        values[:] = 4.0 * alpha**2 * (r[:, 0] ** 2 + r[:, 1] ** 2) - 4.0 * alpha
    else:
        beta = float(beta)
        values[:] = (
            4.0 * alpha**2 * (r[:, 0] ** 2 + r[:, 1] ** 2 + beta**2 * r[:, 2] ** 2)
            - 2.0 * alpha * (2.0 + beta)
        )
    return values


def local_energy_noninteracting(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Analytic local energy without the Jastrow/hard-core interaction."""

    r = as_positions(positions)
    alpha = float(alpha)
    n_particles, dimensions = r.shape

    if dimensions < 3:
        r2 = float(np.sum(r**2))
        return float(alpha * dimensions * n_particles + (0.5 - 2.0 * alpha**2) * r2)

    beta = float(beta)
    gamma = float(gamma)
    x2y2 = float(np.sum(r[:, 0] ** 2 + r[:, 1] ** 2))
    z2 = float(np.sum(r[:, 2] ** 2))
    kinetic_constant = alpha * n_particles * (2.0 + beta)
    xy_term = (0.5 - 2.0 * alpha**2) * x2y2
    z_term = (0.5 * gamma**2 - 2.0 * alpha**2 * beta**2) * z2
    return float(kinetic_constant + xy_term + z_term)


def local_energy_interacting(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    gamma: float = 1.0,
    hard_core_radius: float = 0.0043,
) -> float:
    """Analytic local energy for the full Jastrow trial wave function.

    The hard-sphere interaction is handled through the boundary condition in
    the Jastrow factor. If a configuration violates r_ij > a, the local energy
    is returned as infinity.
    """

    r = as_positions(positions)
    a = float(hard_core_radius)
    if a <= 0.0:
        return local_energy_noninteracting(r, alpha=alpha, beta=beta, gamma=gamma)
    if has_hard_core_overlap(r, a):
        return math.inf

    n_particles, dimensions = r.shape
    grad_phi = gradient_log_gaussian(r, alpha=alpha, beta=beta)
    lap_phi = laplacian_gaussian_over_gaussian(r, alpha=alpha, beta=beta)

    # Vectorized over all partners j for each particle k. This keeps the
    # expression close to the analytic formula while avoiding a slow inner
    # Python loop.
    laplacian_over_psi_sum = 0.0
    for k in range(n_particles):
        diff = r[k] - r
        distances = np.linalg.norm(diff, axis=1)
        mask = np.ones(n_particles, dtype=bool)
        mask[k] = False
        diff = diff[mask]
        distances = distances[mask]
        if np.any(distances <= a):
            return math.inf

        u1 = a / (distances * (distances - a))
        u2 = -a * (2.0 * distances - a) / (distances**2 * (distances - a) ** 2)
        unit_vectors = diff / distances[:, None]
        corr_gradient = np.sum(unit_vectors * u1[:, None], axis=0)
        corr_laplacian = float(np.sum(u2 + (dimensions - 1.0) * u1 / distances))

        laplacian_k = (
            lap_phi[k]
            + 2.0 * float(np.dot(grad_phi[k], corr_gradient))
            + float(np.dot(corr_gradient, corr_gradient))
            + corr_laplacian
        )
        laplacian_over_psi_sum += laplacian_k

    kinetic_energy = -0.5 * laplacian_over_psi_sum
    return float(kinetic_energy + external_potential(r, gamma=gamma))


def local_energy(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    gamma: float = 1.0,
    hard_core_radius: float = 0.0,
    mode: LocalEnergyMode = "analytic",
    numerical_step: float = 1.0e-4,
) -> float:
    """Return local energy using either analytic or finite-difference kinetic energy."""

    if mode == "analytic":
        if hard_core_radius > 0.0:
            return local_energy_interacting(positions, alpha, beta, gamma, hard_core_radius)
        return local_energy_noninteracting(positions, alpha, beta, gamma)
    if mode == "numerical":
        return local_energy_numerical(
            positions,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            hard_core_radius=hard_core_radius,
            step=numerical_step,
        )
    raise ValueError("mode must be 'analytic' or 'numerical'")


def local_energy_numerical(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    gamma: float = 1.0,
    hard_core_radius: float = 0.0,
    step: float = 1.0e-4,
) -> float:
    """Finite-difference local energy.

    This is mainly intended for Project 1b comparisons against the analytic
    local energy. It computes

        (1/Psi) d^2 Psi/dx^2 = [Psi(x+h)/Psi(x) + Psi(x-h)/Psi(x) - 2]/h^2

    using log-wavefunction ratios.
    """

    r = as_positions(positions).copy()
    h = float(step)
    if h <= 0.0:
        raise ValueError("finite-difference step must be positive")

    log_psi0 = log_wavefunction(r, alpha, beta, hard_core_radius)
    if log_psi0 == -math.inf:
        return math.inf

    laplacian_over_psi = 0.0
    n_particles, dimensions = r.shape
    for i in range(n_particles):
        for d in range(dimensions):
            r[i, d] += h
            log_plus = log_wavefunction(r, alpha, beta, hard_core_radius)
            r[i, d] -= 2.0 * h
            log_minus = log_wavefunction(r, alpha, beta, hard_core_radius)
            r[i, d] += h

            ratio_plus = 0.0 if log_plus == -math.inf else math.exp(log_plus - log_psi0)
            ratio_minus = 0.0 if log_minus == -math.inf else math.exp(log_minus - log_psi0)
            laplacian_over_psi += (ratio_plus + ratio_minus - 2.0) / h**2

    kinetic_energy = -0.5 * laplacian_over_psi
    return float(kinetic_energy + external_potential(r, gamma=gamma))


def exact_energy_noninteracting(
    n_particles: int,
    dimensions: int,
    gamma: float = 1.0,
) -> float:
    """Return exact ground-state energy for the non-interacting trap."""

    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2, or 3")
    if dimensions < 3:
        return 0.5 * n_particles * dimensions
    return 0.5 * n_particles * (2.0 + float(gamma))


def exact_variational_energy_noninteracting(
    n_particles: int,
    dimensions: int,
    alpha: float,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Analytic expectation value for the Gaussian trial function.

    This is useful for validating Monte Carlo scans. For spherical traps and
    beta = gamma = 1 it reduces to

        E(alpha) = N d [alpha/2 + 1/(8 alpha)].
    """

    alpha = float(alpha)
    if alpha <= 0.0:
        return math.inf
    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2, or 3")

    if dimensions < 3:
        return float(n_particles * dimensions * (0.5 * alpha + 1.0 / (8.0 * alpha)))

    beta = float(beta)
    gamma = float(gamma)
    if beta <= 0.0:
        return math.inf
    xy_mean = n_particles * 2.0 / (4.0 * alpha)
    z_mean = n_particles / (4.0 * alpha * beta)
    return float(
        alpha * n_particles * (2.0 + beta)
        + (0.5 - 2.0 * alpha**2) * xy_mean
        + (0.5 * gamma**2 - 2.0 * alpha**2 * beta**2) * z_mean
    )
