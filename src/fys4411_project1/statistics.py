"""Statistical analysis helpers for Monte Carlo data."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BlockingResult:
    """Result from a blocking analysis."""

    block_sizes: FloatArray
    means: FloatArray
    variances: FloatArray
    standard_errors: FloatArray
    n_blocks: NDArray[np.int64]

    @property
    def final_error(self) -> float:
        """Return the last finite blocking error estimate."""

        finite = np.isfinite(self.standard_errors)
        if not np.any(finite):
            return float("nan")
        return float(self.standard_errors[finite][-1])


def as_1d_float_array(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size == 0:
        raise ValueError("values cannot be empty")
    return array


def mean_and_error(values: ArrayLike) -> tuple[float, float]:
    """Return sample mean and naive standard error of the mean."""

    x = as_1d_float_array(values)
    if x.size < 2:
        return float(np.mean(x)), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(x.size))


def autocovariance(values: ArrayLike, max_lag: int | None = None) -> FloatArray:
    """Return the normalized autocovariance C(t)/C(0) up to max_lag."""

    x = as_1d_float_array(values)
    n = x.size
    if max_lag is None:
        max_lag = min(n - 1, 1_000)
    max_lag = min(max_lag, n - 1)
    centered = x - np.mean(x)
    variance = float(np.dot(centered, centered) / n)
    if variance == 0.0:
        return np.ones(max_lag + 1)

    result = np.empty(max_lag + 1, dtype=float)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag) / variance)
    return result


def blocking_analysis(values: ArrayLike, max_block_size: int | None = None) -> BlockingResult:
    """Perform a simple blocking analysis.

    For block size B, the input series is split into floor(N/B) blocks. The
    standard error is estimated from the variance of the block means.
    """

    x = as_1d_float_array(values)
    n = x.size
    if max_block_size is None:
        max_block_size = max(1, n // 4)
    max_block_size = max(1, min(max_block_size, n))

    # Use powers of two plus the full max size for compact, readable output.
    block_sizes: list[int] = []
    b = 1
    while b <= max_block_size:
        block_sizes.append(b)
        b *= 2
    if block_sizes[-1] != max_block_size:
        block_sizes.append(max_block_size)

    means: list[float] = []
    variances: list[float] = []
    errors: list[float] = []
    n_blocks_list: list[int] = []

    for block_size in block_sizes:
        n_blocks = n // block_size
        n_blocks_list.append(n_blocks)
        if n_blocks < 2:
            means.append(float(np.mean(x)))
            variances.append(float("nan"))
            errors.append(float("nan"))
            continue
        trimmed = x[: n_blocks * block_size]
        block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
        means.append(float(np.mean(block_means)))
        variances.append(float(np.var(block_means, ddof=1)))
        errors.append(float(np.std(block_means, ddof=1) / math.sqrt(n_blocks)))

    return BlockingResult(
        block_sizes=np.asarray(block_sizes, dtype=float),
        means=np.asarray(means, dtype=float),
        variances=np.asarray(variances, dtype=float),
        standard_errors=np.asarray(errors, dtype=float),
        n_blocks=np.asarray(n_blocks_list, dtype=np.int64),
    )


def bootstrap_mean(
    values: ArrayLike,
    n_bootstrap: int = 2_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap estimate of mean and standard error."""

    x = as_1d_float_array(values)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    bootstrap_means = x[indices].mean(axis=1)
    return float(np.mean(x)), float(np.std(bootstrap_means, ddof=1))


def save_blocking_csv(result: BlockingResult, path: str) -> None:
    """Save blocking result to CSV."""

    import csv
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["block_size", "n_blocks", "mean", "variance", "standard_error"])
        for row in zip(
            result.block_sizes,
            result.n_blocks,
            result.means,
            result.variances,
            result.standard_errors,
            strict=True,
        ):
            writer.writerow(row)
