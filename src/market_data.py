"""Live market data helpers for optional report parameterization.

Data source: Yahoo Finance via yfinance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiveMarketSnapshot:
    """Live inputs inferred from market data."""

    ticker: str
    spot: float
    sigma_annualized: float
    risk_free_rate: float
    n_obs: int


def _close_series(ticker: str, period: str) -> np.ndarray:
    """Download adjusted close prices as a 1D float array."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for --use-live mode") from exc

    history = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if history.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'")

    close = history["Close"]
    if hasattr(close, "to_numpy"):
        close_values = close.to_numpy(dtype=float).reshape(-1)
    else:
        close_values = np.asarray(close, dtype=float).reshape(-1)

    close_values = close_values[np.isfinite(close_values)]
    if close_values.size < 2:
        raise ValueError(f"Not enough close prices for ticker '{ticker}'")
    return close_values


def fetch_live_market_snapshot(
    ticker: str,
    lookback_days: int = 365,
    trading_days: int = 252,
    rate_ticker: str = "^IRX",
    fallback_rate: float = 0.03,
) -> LiveMarketSnapshot:
    """Fetch spot and estimate annualized volatility/risk-free rate.

    Volatility is historical realized vol from daily log returns over the
    specified lookback period.
    """
    if lookback_days < 30:
        raise ValueError("lookback_days must be >= 30")
    if trading_days < 200:
        raise ValueError("trading_days must be >= 200")

    period = f"{lookback_days}d"
    closes = _close_series(ticker=ticker, period=period)
    spot = float(closes[-1])

    log_returns = np.diff(np.log(closes))
    if log_returns.size < 2:
        raise ValueError("Not enough observations to estimate volatility")
    sigma_annualized = float(np.std(log_returns, ddof=1) * np.sqrt(trading_days))
    if sigma_annualized < 0.0:
        raise ValueError("Estimated volatility was negative")

    risk_free_rate = fallback_rate
    try:
        rate_closes = _close_series(ticker=rate_ticker, period="5d")
        # ^IRX is quoted in percentage points (e.g. 5.10 means 5.10%).
        risk_free_rate = float(rate_closes[-1] / 100.0)
    except Exception:
        risk_free_rate = fallback_rate

    return LiveMarketSnapshot(
        ticker=ticker,
        spot=spot,
        sigma_annualized=sigma_annualized,
        risk_free_rate=risk_free_rate,
        n_obs=int(log_returns.size),
    )
