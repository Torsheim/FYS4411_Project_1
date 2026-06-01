"""One-body density estimators."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HistogramDensity:
    bin_centers: FloatArray
    density: FloatArray
    counts: FloatArray
    bin_edges: FloatArray


def radial_distances(samples: ArrayLike) -> FloatArray:
    """Return all single-particle radial distances from stored samples.

    samples must have shape (n_samples, n_particles, dimensions).
    """

    array = np.asarray(samples, dtype=float)
    if array.ndim != 3:
        raise ValueError("samples must have shape (n_samples, n_particles, dimensions)")
    return np.linalg.norm(array, axis=2).ravel()


def radial_density(
    samples: ArrayLike,
    bins: int = 80,
    r_max: float | None = None,
    dimensions: int | None = None,
) -> HistogramDensity:
    """Estimate the spherically averaged one-body density rho(r).

    The returned density is normalized so that integral rho(r) dV = 1 for one
    randomly selected particle. In 3D, dV = 4 pi r^2 dr; in 2D, dV = 2 pi r dr;
    in 1D, the radial histogram is normalized over r >= 0.
    """

    array = np.asarray(samples, dtype=float)
    if array.ndim != 3:
        raise ValueError("samples must have shape (n_samples, n_particles, dimensions)")
    if dimensions is None:
        dimensions = array.shape[2]
    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2, or 3")

    radii = radial_distances(array)
    if r_max is None:
        r_max = float(np.max(radii))
    counts, edges = np.histogram(radii, bins=bins, range=(0.0, r_max))
    shell_volumes: FloatArray
    if dimensions == 1:
        # Histogram of |x|, normalized on r >= 0.
        shell_volumes = edges[1:] - edges[:-1]
    elif dimensions == 2:
        shell_volumes = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    else:
        shell_volumes = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)

    probability = counts / np.sum(counts)
    density = probability / shell_volumes
    centers = 0.5 * (edges[1:] + edges[:-1])
    return HistogramDensity(
        bin_centers=centers,
        density=density,
        counts=counts.astype(float),
        bin_edges=edges,
    )


def axial_density(
    samples: ArrayLike,
    axis: int = 2,
    bins: int = 80,
    value_range: tuple[float, float] | None = None,
) -> HistogramDensity:
    """Estimate a one-dimensional marginal density along one coordinate axis."""

    array = np.asarray(samples, dtype=float)
    if array.ndim != 3:
        raise ValueError("samples must have shape (n_samples, n_particles, dimensions)")
    if axis < 0 or axis >= array.shape[2]:
        raise ValueError("axis is outside the position dimensionality")
    values = array[:, :, axis].ravel()
    counts, edges = np.histogram(values, bins=bins, range=value_range, density=False)
    widths = edges[1:] - edges[:-1]
    probability = counts / np.sum(counts)
    density_values = probability / widths
    centers = 0.5 * (edges[1:] + edges[:-1])
    return HistogramDensity(
        bin_centers=centers,
        density=density_values,
        counts=counts.astype(float),
        bin_edges=edges,
    )


def save_density_csv(density: HistogramDensity, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bin_center", "density", "counts", "bin_left", "bin_right"])
        for center, value, count, left, right in zip(
            density.bin_centers,
            density.density,
            density.counts,
            density.bin_edges[:-1],
            density.bin_edges[1:],
            strict=True,
        ):
            writer.writerow([center, value, count, left, right])
