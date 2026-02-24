"""Entrypoint for generating a Monte Carlo option pricing report.

This script runs risk-neutral GBM Monte Carlo pricing for European call/put,
computes finite-difference Greeks with CRN, and saves evidence plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.convergence import running_ci_95
from src.gbm import simulate_price_paths, simulate_terminal_prices
from src.greeks import estimate_delta_vega_fd
from src.market_data import fetch_live_market_snapshot
from src.plotting import (
    plot_convergence_with_ci,
    plot_delta_vs_s0,
    plot_discounted_payoff_hist,
    plot_gbm_paths_fan,
    plot_terminal_price_hist,
    plot_vega_vs_sigma,
)
from src.pricer import call_payoff, price_call_mc, price_put_mc


def run_report(
    S0: float = 100.0,
    K: float = 110.0,
    r: float = 0.03,
    sigma: float = 0.25,
    T: float = 1.0,
    n_paths: int = 200_000,
    seed: int = 42,
    use_live: bool = False,
    ticker: str = "AAPL",
    lookback_days: int = 365,
) -> None:
    """Run pricing, Greeks, convergence diagnostics, and figure generation."""
    live_snapshot = None
    if use_live:
        live_snapshot = fetch_live_market_snapshot(
            ticker=ticker,
            lookback_days=lookback_days,
            fallback_rate=r,
        )
        S0 = live_snapshot.spot
        sigma = live_snapshot.sigma_annualized
        r = live_snapshot.risk_free_rate

    # 1) Price call/put with confidence intervals
    call_result = price_call_mc(S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed)
    put_result = price_put_mc(S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed)

    # 2) Build discounted payoff sample for convergence/histogram diagnostics
    terminal_prices = simulate_terminal_prices(
        S0=S0, r=r, sigma=sigma, T=T, n_paths=n_paths, seed=seed
    )
    discounted_call_payoffs = np.exp(-r * T) * call_payoff(terminal_prices, K)
    convergence = running_ci_95(discounted_call_payoffs)

    # 3) Greeks at base point + sweeps
    base_greeks = estimate_delta_vega_fd(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths=120_000,
        dS=0.5,
        dsigma=0.01,
        option_type="call",
        seed=seed,
    )

    s0_values = np.linspace(60.0, 160.0, 13)
    delta_values = np.array(
        [
            estimate_delta_vega_fd(
                S0=float(s),
                K=K,
                r=r,
                sigma=sigma,
                T=T,
                n_paths=40_000,
                dS=0.5,
                dsigma=0.01,
                option_type="call",
                seed=seed,
            ).delta
            for s in s0_values
        ],
        dtype=np.float64,
    )

    sigma_values = np.linspace(0.05, 0.60, 12)
    vega_values = np.array(
        [
            estimate_delta_vega_fd(
                S0=S0,
                K=K,
                r=r,
                sigma=float(vol),
                T=T,
                n_paths=40_000,
                dS=0.5,
                dsigma=0.01,
                option_type="call",
                seed=seed,
            ).vega
            for vol in sigma_values
        ],
        dtype=np.float64,
    )

    # 4) Sample paths for fan chart
    paths = simulate_price_paths(
        S0=S0,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=252,
        n_paths=400,
        seed=seed,
    )

    # 5) Save figures
    fig_paths = [
        plot_gbm_paths_fan(paths),
        plot_terminal_price_hist(terminal_prices),
        plot_discounted_payoff_hist(discounted_call_payoffs),
        plot_convergence_with_ci(
            convergence.n,
            convergence.mean,
            convergence.ci_low,
            convergence.ci_high,
        ),
        plot_delta_vs_s0(s0_values, delta_values),
        plot_vega_vs_sigma(sigma_values, vega_values),
    ]

    # 6) Console summary
    print("Monte Carlo Option Pricer Report")
    print("=" * 33)
    if live_snapshot is not None:
        print(
            "Live inputs: "
            f"ticker={live_snapshot.ticker}, "
            f"spot={live_snapshot.spot:.4f}, "
            f"hist_sigma={live_snapshot.sigma_annualized:.4f}, "
            f"rf={live_snapshot.risk_free_rate:.4f}, "
            f"obs={live_snapshot.n_obs}"
        )
    print(f"Params: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}, n_paths={n_paths}")
    print(
        "Call price: "
        f"{call_result.price:.6f} "
        f"(95% CI: [{call_result.ci_low:.6f}, {call_result.ci_high:.6f}], "
        f"stderr={call_result.stderr:.6f})"
    )
    print(
        "Put price:  "
        f"{put_result.price:.6f} "
        f"(95% CI: [{put_result.ci_low:.6f}, {put_result.ci_high:.6f}], "
        f"stderr={put_result.stderr:.6f})"
    )
    print(f"Call Delta (FD, CRN): {base_greeks.delta:.6f}")
    print(f"Call Vega  (FD, CRN): {base_greeks.vega:.6f}")
    print("Saved figures:")
    for path in fig_paths:
        print(f"- {Path(path)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for report execution."""
    parser = argparse.ArgumentParser(description="Monte Carlo option pricer report")
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=110.0)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n-paths", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-live", action="store_true")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--lookback-days", type=int, default=365)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_report(
        S0=args.S0,
        K=args.K,
        r=args.r,
        sigma=args.sigma,
        T=args.T,
        n_paths=args.n_paths,
        seed=args.seed,
        use_live=args.use_live,
        ticker=args.ticker,
        lookback_days=args.lookback_days,
    )
