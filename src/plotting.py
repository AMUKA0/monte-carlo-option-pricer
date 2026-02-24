"""Matplotlib plotting utilities for Monte Carlo evidence outputs.

Each function creates a single figure and saves it to results/figures by default.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ArrayF64 = np.ndarray
DEFAULT_FIG_DIR = Path("results") / "figures"


def _prepare_output_path(filename: str, output_dir: str | Path = DEFAULT_FIG_DIR) -> Path:
    """Create output directory and return full PNG path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def plot_gbm_paths_fan(
    paths: ArrayF64,
    filename: str = "gbm_paths_fan.png",
    output_dir: str | Path = DEFAULT_FIG_DIR,
) -> Path:
    """Save a fan chart of simulated GBM paths."""
    if paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 1:
        raise ValueError("paths must be a 2D array with shape (n_steps+1, n_paths)")

    n_steps, n_paths = paths.shape[0] - 1, paths.shape[1]
    t = np.linspace(0.0, 1.0, n_steps + 1)
    max_lines = min(100, n_paths)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, paths[:, :max_lines], linewidth=0.8, alpha=0.25, color="tab:blue")

    q05 = np.percentile(paths, 5, axis=1)
    q50 = np.percentile(paths, 50, axis=1)
    q95 = np.percentile(paths, 95, axis=1)
    ax.fill_between(t, q05, q95, color="tab:orange", alpha=0.2, label="5%-95% band")
    ax.plot(t, q50, color="tab:red", linewidth=1.8, label="Median")

    ax.set_title("GBM Path Fan Chart")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Price")
    ax.legend()

    out = _prepare_output_path(filename, output_dir)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_terminal_price_hist(
    terminal_prices: ArrayF64,
    filename: str = "terminal_price_hist.png",
    output_dir: str | Path = DEFAULT_FIG_DIR,
    bins: int = 60,
) -> Path:
    """Save histogram of terminal prices S_T."""
    if terminal_prices.ndim != 1 or terminal_prices.size < 1:
        raise ValueError("terminal_prices must be a non-empty 1D array")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(terminal_prices, bins=bins, density=False, color="tab:blue", alpha=0.8)
    ax.set_title("Terminal Price Distribution")
    ax.set_xlabel("Terminal Price $S_T$")
    ax.set_ylabel("Frequency")

    out = _prepare_output_path(filename, output_dir)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_discounted_payoff_hist(
    discounted_payoffs: ArrayF64,
    filename: str = "discounted_payoff_hist.png",
    output_dir: str | Path = DEFAULT_FIG_DIR,
    bins: int = 60,
) -> Path:
    """Save histogram of discounted option payoffs."""
    if discounted_payoffs.ndim != 1 or discounted_payoffs.size < 1:
        raise ValueError("discounted_payoffs must be a non-empty 1D array")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(discounted_payoffs, bins=bins, density=False, color="tab:green", alpha=0.8)
    ax.set_title("Discounted Payoff Distribution")
    ax.set_xlabel("Discounted Payoff")
    ax.set_ylabel("Frequency")

    out = _prepare_output_path(filename, output_dir)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_convergence_with_ci(
    n: ArrayF64,
    running_mean: ArrayF64,
    ci_low: ArrayF64,
    ci_high: ArrayF64,
    filename: str = "convergence_ci.png",
    output_dir: str | Path = DEFAULT_FIG_DIR,
) -> Path:
    """Save convergence plot with running estimate and 95% CI band."""
    if n.ndim != 1 or running_mean.ndim != 1 or ci_low.ndim != 1 or ci_high.ndim != 1:
        raise ValueError("All inputs must be 1D arrays")
    if not (n.size == running_mean.size == ci_low.size == ci_high.size and n.size > 0):
        raise ValueError("Input arrays must have matching positive lengths")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n, running_mean, color="tab:blue", linewidth=1.4, label="Running mean")
    ax.fill_between(n, ci_low, ci_high, color="tab:blue", alpha=0.2, label="95% CI")
    ax.set_title("Monte Carlo Convergence")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Price Estimate")
    ax.legend()

    out = _prepare_output_path(filename, output_dir)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_delta_vs_s0(
    s0_values: ArrayF64,
    delta_values: ArrayF64,
    filename: str = "delta_vs_s0.png",
    output_dir: str | Path = DEFAULT_FIG_DIR,
) -> Path:
    """Save Delta sensitivity curve versus spot S0."""
    if s0_values.ndim != 1 or delta_values.ndim != 1:
        raise ValueError("s0_values and delta_values must be 1D arrays")
    if s0_values.size != delta_values.size or s0_values.size < 1:
        raise ValueError("s0_values and delta_values must have same non-zero length")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s0_values, delta_values, marker="o", markersize=3, linewidth=1.4, color="tab:purple")
    ax.set_title("Delta vs Spot Price")
    ax.set_xlabel("Spot Price $S_0$")
    ax.set_ylabel("Delta")

    out = _prepare_output_path(filename, output_dir)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_vega_vs_sigma(
    sigma_values: ArrayF64,
    vega_values: ArrayF64,
    filename: str = "vega_vs_sigma.png",
    output_dir: str | Path = DEFAULT_FIG_DIR,
) -> Path:
    """Save Vega sensitivity curve versus volatility sigma."""
    if sigma_values.ndim != 1 or vega_values.ndim != 1:
        raise ValueError("sigma_values and vega_values must be 1D arrays")
    if sigma_values.size != vega_values.size or sigma_values.size < 1:
        raise ValueError("sigma_values and vega_values must have same non-zero length")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigma_values, vega_values, marker="o", markersize=3, linewidth=1.4, color="tab:orange")
    ax.set_title("Vega vs Volatility")
    ax.set_xlabel("Volatility $sigma$")
    ax.set_ylabel("Vega")

    out = _prepare_output_path(filename, output_dir)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
