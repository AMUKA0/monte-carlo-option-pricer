"""Geometric Brownian Motion simulators under the risk-neutral measure.

Risk-neutral GBM terminal model:
    S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z),
    where Z ~ N(0, 1).
"""

from __future__ import annotations

import numpy as np


ArrayF64 = np.ndarray


def _validate_common_inputs(S0: float, sigma: float, T: float, n_paths: int) -> None:
    """Validate shared model inputs."""
    if S0 <= 0.0:
        raise ValueError("S0 must be > 0")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if T <= 0.0:
        raise ValueError("T must be > 0")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")


def simulate_terminal_prices(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> ArrayF64:
    """Simulate terminal prices under risk-neutral GBM.

    Uses the closed-form terminal distribution of GBM.
    """
    _validate_common_inputs(S0=S0, sigma=sigma, T=T, n_paths=n_paths)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T) * z
    terminal = S0 * np.exp(drift + diffusion)

    return terminal.astype(np.float64, copy=False)


def simulate_price_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> ArrayF64:
    """Simulate GBM paths under the risk-neutral measure.

    Returns an array with shape (n_steps + 1, n_paths), including S0 at t=0.
    """
    _validate_common_inputs(S0=S0, sigma=sigma, T=T, n_paths=n_paths)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = T / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_steps, n_paths))

    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion

    paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    paths[0, :] = S0
    paths[1:, :] = S0 * np.exp(np.cumsum(log_returns, axis=0))

    return paths
