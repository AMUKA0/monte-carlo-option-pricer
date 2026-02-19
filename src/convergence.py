"""Convergence diagnostics for Monte Carlo discounted payoffs.

For discounted payoff samples X_i, this module computes the expanding
running mean and 95% confidence interval bands:
    mean_n +/- 1.96 * s_n / sqrt(n)
where s_n is the sample standard deviation from the first n samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayF64 = np.ndarray


@dataclass(frozen=True)
class ConvergenceSeries:
    """Running Monte Carlo estimate and 95% confidence bands."""

    n: ArrayF64
    mean: ArrayF64
    ci_low: ArrayF64
    ci_high: ArrayF64


def running_mean(discounted_payoffs: ArrayF64) -> ArrayF64:
    """Return expanding running means of discounted payoffs."""
    x = np.asarray(discounted_payoffs, dtype=np.float64)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("discounted_payoffs must be a non-empty 1D array")

    counts = np.arange(1, x.size + 1, dtype=np.float64)
    cumsum = np.cumsum(x)
    return cumsum / counts


def running_ci_95(discounted_payoffs: ArrayF64) -> ConvergenceSeries:
    """Return expanding 95% CI bands using CLT on discounted payoffs."""
    x = np.asarray(discounted_payoffs, dtype=np.float64)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("discounted_payoffs must be a non-empty 1D array")

    counts = np.arange(1, x.size + 1, dtype=np.float64)
    mean = running_mean(x)

    cumsum_sq = np.cumsum(x * x)
    centered_ss = cumsum_sq - counts * mean * mean

    denom = np.maximum(counts - 1.0, 1.0)
    sample_var = centered_ss / denom
    sample_var[0] = 0.0
    sample_var = np.maximum(sample_var, 0.0)

    stderr = np.sqrt(sample_var / counts)
    half_width = 1.96 * stderr

    return ConvergenceSeries(
        n=counts,
        mean=mean,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
    )
