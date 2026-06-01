import math

import numpy as np

from fys4411_project1.local_energy import (
    exact_energy_noninteracting,
    exact_variational_energy_noninteracting,
    local_energy_interacting,
    local_energy_noninteracting,
    local_energy_numerical,
)


def test_exact_energy_at_optimal_alpha_1d():
    positions = np.array([[0.1], [0.2], [0.3]])
    energy = local_energy_noninteracting(positions, alpha=0.5)
    assert energy == 1.5


def test_exact_energy_at_optimal_alpha_2d():
    positions = np.array([[0.1, 0.2], [0.4, 0.5]])
    energy = local_energy_noninteracting(positions, alpha=0.5)
    assert energy == 2.0


def test_exact_energy_at_optimal_alpha_3d():
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    energy = local_energy_noninteracting(positions, alpha=0.5)
    assert energy == 3.0


def test_anisotropic_exact_constant_when_beta_gamma_and_alpha_half():
    gamma = 2.82843
    positions = np.array([[0.1, 0.2, 0.3], [0.4, -0.5, 0.6]])
    energy = local_energy_noninteracting(positions, alpha=0.5, beta=gamma, gamma=gamma)
    assert math.isclose(energy, 0.5 * 2 * (2.0 + gamma), rel_tol=1e-14)


def test_numerical_local_energy_matches_analytic():
    positions = np.array([[0.1, -0.4, 0.2], [0.3, 0.2, -0.1]])
    analytic = local_energy_noninteracting(positions, alpha=0.4)
    numerical = local_energy_numerical(positions, alpha=0.4, step=1e-5)
    assert math.isclose(numerical, analytic, rel_tol=1e-5, abs_tol=1e-5)


def test_interacting_reduces_to_noninteracting_when_a_zero():
    positions = np.array([[0.1, -0.4, 0.2], [0.3, 0.2, -0.1]])
    assert local_energy_interacting(positions, alpha=0.5, hard_core_radius=0.0) == local_energy_noninteracting(
        positions, alpha=0.5
    )


def test_interacting_energy_is_finite_for_valid_configuration():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, -0.2], [-0.3, 0.9, 0.4]])
    energy = local_energy_interacting(
        positions,
        alpha=0.5,
        beta=2.82843,
        gamma=2.82843,
        hard_core_radius=0.0043,
    )
    assert math.isfinite(energy)


def test_exact_energy_helpers():
    assert exact_energy_noninteracting(10, 3) == 15.0
    assert exact_energy_noninteracting(10, 2) == 10.0
    assert math.isclose(exact_variational_energy_noninteracting(10, 3, alpha=0.5), 15.0)
