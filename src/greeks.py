"""Finite-difference Greeks for European options under GBM using CRN.

Central finite differences are used for sensitivities:
    Delta ~= (V(S0 + dS) - V(S0 - dS)) / (2 dS)
    Vega  ~= (V(sigma + dsigma) - V(sigma - dsigma)) / (2 dsigma)
where each bumped valuation reuses the same normal draws (common random
numbers) to reduce Monte Carlo noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.pricer import call_payoff, put_payoff


OptionType = Literal["call", "put"]
ArrayF64 = np.ndarray


@dataclass(frozen=True)
class Greeks:
    """Monte Carlo finite-difference Greek estimates."""

    delta: float
    vega: float


def _terminal_from_z(S0: float, r: float, sigma: float, T: float, z: ArrayF64) -> ArrayF64:
    """Construct terminal GBM prices from pre-sampled standard normals."""
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def _discounted_price_from_z(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    z: ArrayF64,
    option_type: OptionType,
) -> float:
    """Estimate discounted option value using fixed normal samples."""
    ST = _terminal_from_z(S0=S0, r=r, sigma=sigma, T=T, z=z)
    if option_type == "call":
        payoff = call_payoff(ST, K)
    elif option_type == "put":
        payoff = put_payoff(ST, K)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(np.exp(-r * T) * np.mean(payoff))


def estimate_delta_vega_fd(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    dS: float = 0.5,
    dsigma: float = 0.01,
    option_type: OptionType = "call",
    seed: int | None = None,
) -> Greeks:
    """Estimate Delta and Vega using central differences with CRN."""
    if S0 <= 0.0:
        raise ValueError("S0 must be > 0")
    if K <= 0.0:
        raise ValueError("K must be > 0")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if T <= 0.0:
        raise ValueError("T must be > 0")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if dS <= 0.0:
        raise ValueError("dS must be > 0")
    if dsigma <= 0.0:
        raise ValueError("dsigma must be > 0")
    if S0 - dS <= 0.0:
        raise ValueError("Central difference requires S0 - dS > 0")
    if sigma - dsigma < 0.0:
        raise ValueError("Central difference requires sigma - dsigma >= 0")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    value_s_up = _discounted_price_from_z(
        S0=S0 + dS,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        z=z,
        option_type=option_type,
    )
    value_s_dn = _discounted_price_from_z(
        S0=S0 - dS,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        z=z,
        option_type=option_type,
    )
    delta = (value_s_up - value_s_dn) / (2.0 * dS)

    value_v_up = _discounted_price_from_z(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma + dsigma,
        T=T,
        z=z,
        option_type=option_type,
    )
    value_v_dn = _discounted_price_from_z(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma - dsigma,
        T=T,
        z=z,
        option_type=option_type,
    )
    vega = (value_v_up - value_v_dn) / (2.0 * dsigma)

    return Greeks(delta=float(delta), vega=float(vega))
