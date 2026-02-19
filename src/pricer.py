"""Monte Carlo pricing for European options under risk-neutral GBM.

Price estimator:
    V0 = exp(-rT) * E[payoff(S_T)]
with a 95% confidence interval from the CLT on discounted payoffs:
    mean +/- 1.96 * standard_error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.gbm import simulate_terminal_prices


ArrayF64 = np.ndarray


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with standard error and 95% confidence interval."""

    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int


def call_payoff(ST: ArrayF64, K: float) -> ArrayF64:
    """Vectorized European call payoff max(ST - K, 0)."""
    if K <= 0.0:
        raise ValueError("K must be > 0")
    return np.maximum(ST - K, 0.0)


def put_payoff(ST: ArrayF64, K: float) -> ArrayF64:
    """Vectorized European put payoff max(K - ST, 0)."""
    if K <= 0.0:
        raise ValueError("K must be > 0")
    return np.maximum(K - ST, 0.0)


def _mc_result_from_discounted_payoffs(discounted_payoffs: ArrayF64) -> MCResult:
    """Compute mean, standard error, and 95% CI from discounted payoffs."""
    n_paths = int(discounted_payoffs.shape[0])
    if n_paths < 1:
        raise ValueError("discounted_payoffs must contain at least one value")

    mean = float(np.mean(discounted_payoffs))
    if n_paths == 1:
        stderr = 0.0
    else:
        sample_std = float(np.std(discounted_payoffs, ddof=1))
        stderr = sample_std / np.sqrt(n_paths)

    half_width = 1.96 * stderr
    return MCResult(
        price=mean,
        stderr=stderr,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
        n_paths=n_paths,
    )


def price_call_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Price a European call via risk-neutral Monte Carlo under GBM."""
    ST = simulate_terminal_prices(S0=S0, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed)
    discount = np.exp(-r * T)
    discounted_payoffs = discount * call_payoff(ST, K)
    return _mc_result_from_discounted_payoffs(discounted_payoffs)


def price_put_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Price a European put via risk-neutral Monte Carlo under GBM."""
    ST = simulate_terminal_prices(S0=S0, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed)
    discount = np.exp(-r * T)
    discounted_payoffs = discount * put_payoff(ST, K)
    return _mc_result_from_discounted_payoffs(discounted_payoffs)
