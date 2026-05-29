"""
LOCKED signal library — the universe of weak signals strategy.py can compose.

Each signal:
  - Pure function of PriceData (+ a small set of typed kwargs).
  - Returns a wide DataFrame indexed by prices.dates, columns prices.tickers.
  - Higher values = stronger BUY signal (sign convention is uniform across the library).
  - Cross-sectional z-scoring is the responsibility of the caller (strategy.py),
    not of the signal itself — so a strategy can compose raw signals or transformed ones.

Adding a signal to the library means widening the proposer's typed move surface.
The agent can then propose 'add signal X to enabled_signals' as a structural move
rather than reinventing the implementation from scratch.

DO NOT MODIFY — this defines the comparability surface across all strategy runs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from harness.data import PriceData


# ----------------------------- individual signals -----------------------------

def momentum_12_1(prices: PriceData, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Classic 12-month-minus-1-month momentum. Skip avoids short-term reversal contamination."""
    p = prices.close
    return np.log(p.shift(skip) / p.shift(skip + lookback))


def momentum_3m(prices: PriceData, lookback: int = 63) -> pd.DataFrame:
    """3-month price return. No skip — closer to current trend."""
    return np.log(prices.close / prices.close.shift(lookback))


def reversion_5d(prices: PriceData, lookback: int = 5) -> pd.DataFrame:
    """Short-term reversal: oversold (negative recent return) gets a positive signal."""
    return -np.log(prices.close / prices.close.shift(lookback))


def reversion_21d(prices: PriceData, lookback: int = 21) -> pd.DataFrame:
    """1-month reversal. Slower than 5d; captures different overreaction horizons."""
    return -np.log(prices.close / prices.close.shift(lookback))


def lowvol_20d(prices: PriceData, lookback: int = 20) -> pd.DataFrame:
    """Negative realized vol — high signal = LOW vol = preferred (low-vol anomaly)."""
    return -prices.returns.rolling(lookback).std()


def lowvol_60d(prices: PriceData, lookback: int = 60) -> pd.DataFrame:
    """Longer-window low-vol bias. Less noisy than 20d."""
    return -prices.returns.rolling(lookback).std()


def trend_consistency(prices: PriceData, lookback: int = 60) -> pd.DataFrame:
    """Fraction of positive return days in window minus 0.5. Captures 'smooth' trends."""
    pos = (prices.returns > 0).astype(float)
    return pos.rolling(lookback).mean() - 0.5


def acceleration(prices: PriceData, short_lb: int = 63, long_lb: int = 252) -> pd.DataFrame:
    """3-month return minus 12-month return: positive if recent trend stronger than long trend."""
    short_ret = np.log(prices.close / prices.close.shift(short_lb))
    long_ret = np.log(prices.close / prices.close.shift(long_lb))
    return short_ret - long_ret


def vol_adjusted_momentum(prices: PriceData, mom_lb: int = 252, vol_lb: int = 60) -> pd.DataFrame:
    """12-month momentum scaled by realized vol. Sharpe-ratio-shaped per-stock signal."""
    mom = np.log(prices.close / prices.close.shift(mom_lb))
    vol = prices.returns.rolling(vol_lb).std().replace(0.0, np.nan)
    return mom / vol


# ----------------------------- registry -----------------------------

SignalFn = Callable[..., pd.DataFrame]

SIGNAL_LIBRARY: dict[str, SignalFn] = {
    "momentum_12_1": momentum_12_1,
    "momentum_3m": momentum_3m,
    "reversion_5d": reversion_5d,
    "reversion_21d": reversion_21d,
    "lowvol_20d": lowvol_20d,
    "lowvol_60d": lowvol_60d,
    "trend_consistency": trend_consistency,
    "acceleration": acceleration,
    "vol_adjusted_momentum": vol_adjusted_momentum,
}


def get_signal(name: str) -> SignalFn:
    if name not in SIGNAL_LIBRARY:
        raise KeyError(f"Unknown signal {name!r}. Available: {sorted(SIGNAL_LIBRARY)}")
    return SIGNAL_LIBRARY[name]


def zscore_cs(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per row. Robust to NaN, returns NaN-filled-with-zero standardized values."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1).replace(0.0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)
