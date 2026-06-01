import numpy as np

from fys4411_project1.density import axial_density, radial_density


def test_radial_density_normalization_shape():
    samples = np.zeros((4, 3, 3))
    samples[:, :, 0] = 0.5
    density = radial_density(samples, bins=4, r_max=2.0, dimensions=3)
    assert density.bin_centers.shape == (4,)
    assert density.density.shape == (4,)


def test_axial_density_shape():
    samples = np.random.default_rng(1).normal(size=(5, 4, 3))
    density = axial_density(samples, axis=2, bins=5)
    assert density.bin_centers.shape == (5,)
