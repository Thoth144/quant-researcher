"""
Performance metrics. Pure functions over daily-return series / equity curves.

LOCKED — do not modify. Metric definitions are the comparability contract.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio. Returns 0.0 if no variance (degenerate strategy)."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / TRADING_DAYS_PER_YEAR
    sigma = excess.std()
    if sigma == 0 or not np.isfinite(sigma):
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / sigma)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown as a positive fraction (e.g. 0.32 = -32%)."""
    r = returns.fillna(0.0)
    if len(r) == 0:
        return 0.0
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(-dd.min())


def calmar(returns: pd.Series) -> float:
    """Annualized return / max drawdown. 0 if no drawdown (uninformative)."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    ann_return = float((1 + r).prod() ** (TRADING_DAYS_PER_YEAR / len(r)) - 1)
    dd = max_drawdown(r)
    if dd == 0:
        return 0.0
    return ann_return / dd


def turnover(weights: pd.DataFrame) -> float:
    """Annualized one-way turnover. weights: index=dates, columns=tickers."""
    if len(weights) < 2:
        return 0.0
    daily = weights.diff().abs().sum(axis=1) / 2.0  # one-way
    return float(daily.mean() * TRADING_DAYS_PER_YEAR)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of strictly-positive return days. Excludes NaN."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((1 + r).prod() ** (TRADING_DAYS_PER_YEAR / len(r)) - 1)


def annualized_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
