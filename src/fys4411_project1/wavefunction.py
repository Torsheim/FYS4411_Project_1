"""Trial wave functions, log derivatives, and drift forces.

The trial state is

    Psi_T(R) = prod_i exp[-alpha (x_i^2 + y_i^2 + beta z_i^2)]
               prod_{i<j} f(r_ij),

where f(r) = 1 - a/r for r > a and f(r) = 0 for r <= a.

The implementation uses log wave functions where possible. This is more stable
for large particle numbers than multiplying many small factors directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrapParameters:
    """Parameters defining the trial wave function and trap.

    alpha:
        Variational Gaussian width.
    beta:
        Anisotropy parameter in the trial wave function. For the spherical trap
        beta = 1. For the interacting elliptical Project 1 case, beta is often
        initialized to gamma = 2.82843.
    hard_core_radius:
        Hard-core diameter in oscillator units. Set to zero for no Jastrow
        factor.
    """

    alpha: float
    beta: float = 1.0
    hard_core_radius: float = 0.0


def as_positions(positions: ArrayLike) -> FloatArray:
    """Return positions as a validated float array of shape (N, dim)."""

    array = np.asarray(positions, dtype=float)
    if array.ndim != 2:
        raise ValueError("positions must have shape (n_particles, dimensions)")
    if array.shape[0] < 1:
        raise ValueError("at least one particle is required")
    if array.shape[1] not in (1, 2, 3):
        raise ValueError("only one, two, and three dimensions are supported")
    return array


def gaussian_exponent_argument(positions: ArrayLike, beta: float = 1.0) -> float:
    """Return sum_i (x_i^2 + y_i^2 + beta z_i^2).

    In one and two dimensions this reduces to the ordinary squared radius.
    """

    r = as_positions(positions)
    if r.shape[1] == 1:
        return float(np.sum(r[:, 0] ** 2))
    if r.shape[1] == 2:
        return float(np.sum(r[:, 0] ** 2 + r[:, 1] ** 2))
    return float(np.sum(r[:, 0] ** 2 + r[:, 1] ** 2 + beta * r[:, 2] ** 2))


def log_gaussian_wavefunction(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
) -> float:
    """Return log of the Gaussian part of the trial wave function."""

    return -float(alpha) * gaussian_exponent_argument(positions, beta)


def gaussian_wavefunction(positions: ArrayLike, alpha: float, beta: float = 1.0) -> float:
    """Return the Gaussian part of the trial wave function."""

    return float(math.exp(log_gaussian_wavefunction(positions, alpha, beta)))


def pair_distances(positions: ArrayLike) -> FloatArray:
    """Return the upper-triangular pair distances r_ij for i < j."""

    r = as_positions(positions)
    n_particles = r.shape[0]
    if n_particles < 2:
        return np.empty(0, dtype=float)
    i, j = np.triu_indices(n_particles, k=1)
    return np.linalg.norm(r[i] - r[j], axis=1)


def has_hard_core_overlap(positions: ArrayLike, hard_core_radius: float) -> bool:
    """Return True if at least one pair has r_ij <= hard_core_radius."""

    if hard_core_radius <= 0.0:
        return False
    distances = pair_distances(positions)
    return bool(np.any(distances <= hard_core_radius))


def log_jastrow_factor(positions: ArrayLike, hard_core_radius: float) -> float:
    """Return log of the hard-sphere Jastrow factor.

    The value is -inf if any pair violates the hard-core constraint.
    """

    a = float(hard_core_radius)
    if a <= 0.0:
        return 0.0
    distances = pair_distances(positions)
    if distances.size == 0:
        return 0.0
    if np.any(distances <= a):
        return -math.inf
    return float(np.sum(np.log1p(-a / distances)))


def log_wavefunction(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    hard_core_radius: float = 0.0,
) -> float:
    """Return log Psi_T for the full trial wave function."""

    log_jastrow = log_jastrow_factor(positions, hard_core_radius)
    if not math.isfinite(log_jastrow):
        return -math.inf
    return log_gaussian_wavefunction(positions, alpha, beta) + log_jastrow


def wavefunction(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    hard_core_radius: float = 0.0,
) -> float:
    """Return Psi_T for the full trial wave function."""

    log_psi = log_wavefunction(positions, alpha, beta, hard_core_radius)
    if log_psi == -math.inf:
        return 0.0
    return float(math.exp(log_psi))


def gradient_log_gaussian(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
) -> FloatArray:
    """Return grad_i log phi_i for every particle.

    Shape is the same as positions.
    """

    r = as_positions(positions)
    grad = -2.0 * float(alpha) * r.copy()
    if r.shape[1] == 3:
        grad[:, 2] *= float(beta)
    return grad


def jastrow_u_prime(distance: float, hard_core_radius: float) -> float:
    """Return u'(r) for u(r) = log(1 - a/r)."""

    a = float(hard_core_radius)
    r = float(distance)
    if a <= 0.0:
        return 0.0
    if r <= a:
        return math.inf
    return a / (r * (r - a))


def jastrow_u_double_prime(distance: float, hard_core_radius: float) -> float:
    """Return u''(r) for u(r) = log(1 - a/r)."""

    a = float(hard_core_radius)
    r = float(distance)
    if a <= 0.0:
        return 0.0
    if r <= a:
        return math.inf
    return -a * (2.0 * r - a) / (r**2 * (r - a) ** 2)


def gradient_log_jastrow(positions: ArrayLike, hard_core_radius: float) -> FloatArray:
    """Return grad_i log J for every particle."""

    r = as_positions(positions)
    grad = np.zeros_like(r)
    a = float(hard_core_radius)
    if a <= 0.0 or r.shape[0] < 2:
        return grad

    n_particles = r.shape[0]
    for i in range(n_particles - 1):
        for j in range(i + 1, n_particles):
            diff = r[i] - r[j]
            distance = float(np.linalg.norm(diff))
            if distance <= a:
                # The wave function is zero. The formal gradient is singular,
                # but returning inf makes mistakes visible during debugging.
                grad[i, :] = math.inf
                grad[j, :] = -math.inf
                continue
            contribution = diff / distance * jastrow_u_prime(distance, a)
            grad[i] += contribution
            grad[j] -= contribution
    return grad


def gradient_log_wavefunction(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    hard_core_radius: float = 0.0,
) -> FloatArray:
    """Return grad_i log Psi_T for every particle."""

    return gradient_log_gaussian(positions, alpha, beta) + gradient_log_jastrow(
        positions, hard_core_radius
    )


def drift_force(
    positions: ArrayLike,
    alpha: float,
    beta: float = 1.0,
    hard_core_radius: float = 0.0,
) -> FloatArray:
    """Return the quantum force F = 2 grad Psi_T / Psi_T = 2 grad log Psi_T."""

    return 2.0 * gradient_log_wavefunction(positions, alpha, beta, hard_core_radius)


def gradient_log_jastrow_particle(
    positions: ArrayLike,
    particle: int,
    hard_core_radius: float,
) -> FloatArray:
    """Return grad_k log J for one particle k.

    This is O(N) and is used by the importance sampler.
    """

    r = as_positions(positions)
    if particle < 0 or particle >= r.shape[0]:
        raise IndexError("particle index out of range")
    grad = np.zeros(r.shape[1], dtype=float)
    a = float(hard_core_radius)
    if a <= 0.0 or r.shape[0] < 2:
        return grad

    diff = r[particle] - r
    distances = np.linalg.norm(diff, axis=1)
    mask = np.ones(r.shape[0], dtype=bool)
    mask[particle] = False
    distances = distances[mask]
    diff = diff[mask]
    if np.any(distances <= a):
        return np.full(r.shape[1], math.inf, dtype=float)
    u1 = a / (distances * (distances - a))
    grad += np.sum(diff / distances[:, None] * u1[:, None], axis=0)
    return grad


def gradient_log_wavefunction_particle(
    positions: ArrayLike,
    particle: int,
    alpha: float,
    beta: float = 1.0,
    hard_core_radius: float = 0.0,
) -> FloatArray:
    """Return grad_k log Psi_T for one particle k."""

    r = as_positions(positions)
    gaussian = gradient_log_gaussian(r, alpha, beta)[particle]
    return gaussian + gradient_log_jastrow_particle(r, particle, hard_core_radius)


def drift_force_particle(
    positions: ArrayLike,
    particle: int,
    alpha: float,
    beta: float = 1.0,
    hard_core_radius: float = 0.0,
) -> FloatArray:
    """Return the quantum force for one particle."""

    return 2.0 * gradient_log_wavefunction_particle(
        positions, particle, alpha, beta, hard_core_radius
    )


def log_derivative_alpha(
    positions: ArrayLike,
    beta: float = 1.0,
) -> float:
    """Return O_alpha = d log(Psi_T) / d alpha.

    For this trial function the Jastrow factor is independent of alpha.
    """

    return -gaussian_exponent_argument(positions, beta)
