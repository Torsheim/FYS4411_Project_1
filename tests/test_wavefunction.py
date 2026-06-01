import math

import numpy as np

from fys4411_project1.wavefunction import (
    drift_force,
    gaussian_wavefunction,
    gradient_log_wavefunction,
    has_hard_core_overlap,
    jastrow_u_double_prime,
    jastrow_u_prime,
    log_gaussian_wavefunction,
    log_jastrow_factor,
    log_wavefunction,
)


def test_log_gaussian_wavefunction_origin():
    positions = np.zeros((2, 3))
    assert log_gaussian_wavefunction(positions, alpha=0.5) == 0.0
    assert gaussian_wavefunction(positions, alpha=0.5) == 1.0


def test_log_gaussian_wavefunction_3d_beta():
    positions = np.array([[1.0, 2.0, 3.0]])
    assert log_gaussian_wavefunction(positions, alpha=0.5, beta=2.0) == -0.5 * (
        1.0 + 4.0 + 2.0 * 9.0
    )


def test_hard_core_overlap_detection():
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert has_hard_core_overlap(positions, 0.2)
    assert not has_hard_core_overlap(positions, 0.05)


def test_log_jastrow_factor():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert math.isclose(log_jastrow_factor(positions, 0.5), math.log(1.0 - 0.5 / 2.0))
    assert log_jastrow_factor(positions, 2.0) == -math.inf


def test_drift_force_gaussian_only():
    positions = np.array([[1.0, -2.0, 3.0]])
    force = drift_force(positions, alpha=0.5, beta=2.0, hard_core_radius=0.0)
    expected = np.array([[-2.0, 4.0, -12.0]])
    assert np.allclose(force, expected)


def test_gradient_log_wavefunction_with_jastrow_is_finite():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    grad = gradient_log_wavefunction(positions, alpha=0.5, hard_core_radius=0.1)
    assert np.all(np.isfinite(grad))


def test_jastrow_derivatives_against_finite_difference():
    a = 0.2
    r = 1.3
    h = 1.0e-6
    u = lambda x: math.log(1.0 - a / x)
    numerical_first = (u(r + h) - u(r - h)) / (2.0 * h)
    numerical_second = (u(r + h) - 2.0 * u(r) + u(r - h)) / h**2
    assert math.isclose(jastrow_u_prime(r, a), numerical_first, rel_tol=1e-7)
    assert math.isclose(jastrow_u_double_prime(r, a), numerical_second, rel_tol=1e-3)


def test_log_wavefunction_zero_when_overlap():
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert log_wavefunction(positions, alpha=0.5, hard_core_radius=0.2) == -math.inf
