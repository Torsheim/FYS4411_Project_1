import math

from fys4411_project1.samplers import VMCConfig, run_vmc


def test_brute_force_sampler_exact_energy_at_alpha_half():
    config = VMCConfig(
        n_particles=2,
        dimensions=3,
        alpha=0.5,
        cycles=50,
        burn_in=10,
        step_size=1.0,
        sampler="brute",
        seed=123,
    )
    result = run_vmc(config)
    assert math.isclose(result.mean_energy, 3.0, rel_tol=0.0, abs_tol=1e-12)
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_importance_sampler_exact_energy_at_alpha_half():
    config = VMCConfig(
        n_particles=2,
        dimensions=2,
        alpha=0.5,
        cycles=50,
        burn_in=10,
        time_step=0.05,
        sampler="importance",
        seed=456,
    )
    result = run_vmc(config)
    assert math.isclose(result.mean_energy, 2.0, rel_tol=0.0, abs_tol=1e-12)
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_sampler_stores_alpha_derivatives_and_positions():
    config = VMCConfig(
        n_particles=2,
        dimensions=1,
        alpha=0.5,
        cycles=5,
        burn_in=1,
        sampler="brute",
        seed=1,
        store_positions=True,
        store_alpha_derivative=True,
    )
    result = run_vmc(config)
    assert result.positions is not None
    assert result.positions.shape == (5, 2, 1)
    assert result.alpha_derivatives is not None
    assert result.alpha_derivatives.shape == (5,)
