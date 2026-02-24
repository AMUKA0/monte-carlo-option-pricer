import math

import numpy as np

from src.gbm import simulate_price_paths, simulate_terminal_prices
from src.pricer import call_payoff, price_call_mc, price_put_mc, put_payoff


def test_payoff_correctness() -> None:
    st = np.array([80.0, 100.0, 120.0], dtype=np.float64)
    k = 100.0

    call = call_payoff(st, k)
    put = put_payoff(st, k)

    np.testing.assert_allclose(call, np.array([0.0, 0.0, 20.0], dtype=np.float64))
    np.testing.assert_allclose(put, np.array([20.0, 0.0, 0.0], dtype=np.float64))


def test_sigma_zero_matches_deterministic_pricing() -> None:
    s0 = 100.0
    k = 110.0
    r = 0.03
    t = 1.0

    call = price_call_mc(S0=s0, K=k, r=r, sigma=0.0, T=t, n_paths=1000, seed=7)
    put = price_put_mc(S0=s0, K=k, r=r, sigma=0.0, T=t, n_paths=1000, seed=7)

    st_det = s0 * math.exp(r * t)
    call_det = math.exp(-r * t) * max(st_det - k, 0.0)
    put_det = math.exp(-r * t) * max(k - st_det, 0.0)

    assert abs(call.price - call_det) < 1e-12
    assert abs(put.price - put_det) < 1e-12


def test_gbm_shape_and_positivity_smoke() -> None:
    terminal = simulate_terminal_prices(S0=100.0, r=0.03, sigma=0.25, T=1.0, n_paths=5000, seed=1)
    paths = simulate_price_paths(
        S0=100.0,
        r=0.03,
        sigma=0.25,
        T=1.0,
        n_steps=64,
        n_paths=300,
        seed=1,
    )

    assert terminal.shape == (5000,)
    assert paths.shape == (65, 300)
    assert np.all(terminal > 0.0)
    assert np.all(paths > 0.0)
    np.testing.assert_allclose(paths[0], 100.0)


def test_ci_sanity() -> None:
    result = price_call_mc(S0=100.0, K=110.0, r=0.03, sigma=0.25, T=1.0, n_paths=20000, seed=123)

    assert result.price >= 0.0
    assert result.stderr >= 0.0
    assert result.ci_low <= result.price <= result.ci_high

    half_width = 0.5 * (result.ci_high - result.ci_low)
    assert abs(half_width - 1.96 * result.stderr) < 1e-12
