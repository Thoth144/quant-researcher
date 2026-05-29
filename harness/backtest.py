"""
Vectorized walk-forward backtester.

LOCKED — do not modify. This file defines the question every strategy is judged
against. Changing it invalidates all prior run comparisons.

Contract:
    run_backtest(strategy_fn, params, seed=None) -> BacktestResult

    strategy_fn must have signature:
        (prices: PriceData, params: Any) -> pd.DataFrame
    returning target weights with:
        - index: a subset of prices.dates
        - columns: a subset of prices.tickers
        - values: target portfolio weight at close of that date
                  (long positive, short negative, sum-abs should be <=~1 for sane sizing)

Walk-forward split:
    - In-sample:  2010-01-01 .. 2021-12-31  (agent may inspect freely)
    - Out-of-sample: 2022-01-01 .. 2024-12-31  (the gating window)

Stochastic replication:
    seed=None         -> deterministic, full universe, single result
    seed=int          -> random 80% sub-universe (reproducible per seed),
                         enabling paired-CI decisions across N replicates
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from harness.data import PriceData, load_panel
from harness import metrics as M

IN_SAMPLE_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2024-12-31")

DEFAULT_COST_BPS = 5.0           # round-trip cost approx 10 bps
DEFAULT_SUBUNIVERSE_PCT = 0.80   # for seeded replicates


@dataclass(frozen=True)
class WindowMetrics:
    sharpe: float
    calmar: float
    max_drawdown: float
    annualized_return: float
    annualized_vol: float
    hit_rate: float
    turnover: float
    n_days: int


@dataclass(frozen=True)
class BacktestResult:
    in_sample: WindowMetrics
    oos: WindowMetrics
    seed: int | None
    n_tickers_used: int
    cost_bps: float

    def to_dict(self) -> dict:
        return {
            "in_sample": asdict(self.in_sample),
            "oos": asdict(self.oos),
            "seed": self.seed,
            "n_tickers_used": self.n_tickers_used,
            "cost_bps": self.cost_bps,
        }

    @property
    def primary_metric(self) -> float:
        """The single number the decision gate compares across runs."""
        return self.oos.sharpe


def _sample_universe(tickers: list[str], pct: float, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    n = max(1, int(len(tickers) * pct))
    idx = rng.choice(len(tickers), size=n, replace=False)
    return [tickers[i] for i in sorted(idx)]


def _filter_prices(prices: PriceData, tickers: list[str]) -> PriceData:
    """Subset PriceData to a given ticker list. Returns same shape, fewer columns."""
    return PriceData(
        close=prices.close[tickers],
        volume=prices.volume[tickers],
        returns=prices.returns[tickers],
        membership=prices.membership[tickers],
    )


def _compute_window_metrics(
    portfolio_returns: pd.Series, weights: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> WindowMetrics:
    mask = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
    r = portfolio_returns[mask]
    w = weights.loc[weights.index.isin(r.index)]
    return WindowMetrics(
        sharpe=M.sharpe(r),
        calmar=M.calmar(r),
        max_drawdown=M.max_drawdown(r),
        annualized_return=M.annualized_return(r),
        annualized_vol=M.annualized_vol(r),
        hit_rate=M.hit_rate(r),
        turnover=M.turnover(w),
        n_days=int(len(r)),
    )


def run_backtest(
    strategy_fn: Callable[[PriceData, Any], pd.DataFrame],
    params: Any,
    seed: int | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    subuniverse_pct: float = DEFAULT_SUBUNIVERSE_PCT,
    prices: PriceData | None = None,
) -> BacktestResult:
    """
    Run a vectorized walk-forward backtest.

    Mechanics:
      1. Call strategy_fn(prices, params) -> target weights at date t.
      2. Lag weights by one day so signals at t become positions at t+1
         (no look-ahead in the returns calculation).
      3. Portfolio return at t = sum(weight_t * asset_return_t).
      4. Subtract transaction cost = cost_bps/10000 * |weight delta|/2 per day.
      5. Slice into in-sample / OOS windows, compute metrics for each.
    """
    if prices is None:
        prices = load_panel(min_tickers=1)

    if seed is not None:
        chosen = _sample_universe(prices.tickers, subuniverse_pct, seed)
        prices = _filter_prices(prices, chosen)

    raw_weights = strategy_fn(prices, params)

    # Defensive: align to prices, fill missing positions with 0, clip wild values.
    weights = raw_weights.reindex(index=prices.dates, columns=prices.tickers).fillna(0.0)
    weights = weights.replace([np.inf, -np.inf], 0.0)

    # Lag one day to avoid look-ahead.
    positions = weights.shift(1).fillna(0.0)

    # Portfolio gross return per day.
    gross = (positions * prices.returns.fillna(0.0)).sum(axis=1)

    # Transaction cost on weight changes (one-way turnover * cost rate).
    weight_delta = positions.diff().abs().sum(axis=1).fillna(0.0)
    cost = (cost_bps / 10000.0) * weight_delta / 2.0
    portfolio_returns = gross - cost

    in_sample = _compute_window_metrics(
        portfolio_returns, positions, prices.dates.min(), IN_SAMPLE_END
    )
    oos = _compute_window_metrics(portfolio_returns, positions, OOS_START, OOS_END)

    return BacktestResult(
        in_sample=in_sample,
        oos=oos,
        seed=seed,
        n_tickers_used=len(prices.tickers),
        cost_bps=cost_bps,
    )
