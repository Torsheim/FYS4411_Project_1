import math

import numpy as np

from fys4411_project1.statistics import autocovariance, blocking_analysis, bootstrap_mean, mean_and_error


def test_mean_and_error():
    mean, error = mean_and_error(np.array([1.0, 2.0, 3.0]))
    assert mean == 2.0
    assert error > 0.0


def test_blocking_analysis():
    result = blocking_analysis(np.arange(16.0), max_block_size=8)
    assert result.block_sizes[0] == 1
    assert result.n_blocks[0] == 16
    assert math.isfinite(result.final_error)


def test_bootstrap_mean_constant_series():
    mean, error = bootstrap_mean(np.ones(20), n_bootstrap=100, seed=1)
    assert mean == 1.0
    assert error == 0.0


def test_autocovariance_first_entry():
    corr = autocovariance(np.arange(10.0), max_lag=3)
    assert corr[0] == 1.0
    assert corr.shape == (4,)
