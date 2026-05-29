"""
The editable surface. The agent mutates THIS file.

Strategy: compose a subset of signals from harness/signals.py (LOCKED), z-score them
cross-sectionally, blend, build a long-short portfolio with periodic rebalance.

Typed mutation surface (improvement #7):
  - enabled_signals: which of harness/signals.py's library is in this strategy
  - weights:         per-signal blend coefficient
  - signal_kwargs:   per-signal hyperparameter overrides (lookbacks etc.) when overriding defaults
  - combine_mode:    'linear' (default), 'sign_vote', 'sharpe_weighted'
  - portfolio:       long_pct, short_pct, rebalance_days, gross_leverage

This means a structural mutation looks like:
    "add 'acceleration' to enabled_signals with weight 0.4"
not just "tweak w_momentum from 1.0 to 1.3".

Contract that must NOT break (the harness depends on it):
  - PARAMS is a module-level StrategyParams instance
  - generate_signals(prices, params) returns a wide DataFrame of target weights
"""

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np
import pandas as pd

from harness.data import PriceData
from harness.signals import SIGNAL_LIBRARY, get_signal, zscore_cs


@dataclass(frozen=True)
class StrategyParams:
    # Which signals from harness/signals.py to enable.
    enabled_signals: tuple[str, ...] = (
        "momentum_12_1", "reversion_5d", "lowvol_20d",
    )

    # Blend weights per enabled signal. Missing entries default to 0.0 (signal silenced).
    weights: dict[str, float] = field(default_factory=lambda: {
        "momentum_12_1": 1.0,
        "reversion_5d": 0.5,
        "lowvol_20d": 0.3,
    })

    # Per-signal hyperparameter override. Each value is a dict of kwargs for that signal.
    # If a key is absent, the signal uses its library default.
    signal_kwargs: dict[str, dict] = field(default_factory=dict)

    # How to combine z-scored signals into a single composite per ticker per date.
    combine_mode: Literal["linear", "sign_vote", "sharpe_weighted"] = "linear"

    # Portfolio construction
    long_pct: float = 0.20
    short_pct: float = 0.20
    rebalance_days: int = 21
    gross_leverage: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


PARAMS = StrategyParams()


# --------------------------- signal composition ---------------------------

_VALID_COMBINE_MODES = {"linear", "sign_vote", "sharpe_weighted"}


def _build_composite(prices: PriceData, params: StrategyParams) -> pd.DataFrame:
    """Compute z-scored per-signal panels, blend per combine_mode."""
    if not params.enabled_signals:
        raise ValueError("enabled_signals is empty — must contain at least one signal name from SIGNAL_LIBRARY")
    if params.combine_mode not in _VALID_COMBINE_MODES:
        raise ValueError(
            f"Unknown combine_mode {params.combine_mode!r}; "
            f"expected one of {sorted(_VALID_COMBINE_MODES)}"
        )

    panels: dict[str, pd.DataFrame] = {}
    for name in params.enabled_signals:
        fn = get_signal(name)
        kwargs = params.signal_kwargs.get(name, {})
        raw = fn(prices, **kwargs)
        panels[name] = zscore_cs(raw).fillna(0.0)

    if params.combine_mode == "sign_vote":
        votes = sum(np.sign(panels[n]) * params.weights.get(n, 0.0) for n in panels)
        return votes  # type: ignore[return-value]

    if params.combine_mode == "sharpe_weighted":
        composite = pd.DataFrame(0.0, index=prices.dates, columns=prices.tickers)
        for n, panel in panels.items():
            w = params.weights.get(n, 0.0)
            sd = panel.stack().std() or 1.0
            composite = composite + w * (panel / sd)
        return composite

    # default: linear
    composite = pd.DataFrame(0.0, index=prices.dates, columns=prices.tickers)
    for n, panel in panels.items():
        composite = composite + params.weights.get(n, 0.0) * panel
    return composite


# --------------------------- portfolio construction ---------------------------

def _signal_to_weights(signal: pd.DataFrame, long_pct: float, short_pct: float, gross: float) -> pd.DataFrame:
    n_cols = signal.shape[1]
    if n_cols == 0:
        return signal * 0.0
    n_long = max(1, int(n_cols * long_pct))
    n_short = max(1, int(n_cols * short_pct))

    ranks = signal.rank(axis=1, method="first", ascending=False)
    long_mask = ranks <= n_long
    short_mask = ranks > (n_cols - n_short)

    longs = long_mask.astype(float).div(long_mask.sum(axis=1).replace(0, np.nan), axis=0)
    shorts = -short_mask.astype(float).div(short_mask.sum(axis=1).replace(0, np.nan), axis=0)

    weights = (longs + shorts).fillna(0.0) * (gross / 2.0)
    return weights


def _apply_rebalance(weights: pd.DataFrame, rebalance_days: int) -> pd.DataFrame:
    if rebalance_days <= 1:
        return weights
    keep_rows = np.zeros(len(weights), dtype=bool)
    keep_rows[::rebalance_days] = True
    mask = pd.DataFrame(
        np.broadcast_to(keep_rows[:, None], weights.shape),
        index=weights.index, columns=weights.columns,
    )
    return weights.where(mask).ffill().fillna(0.0)


# --------------------------- main entrypoint ---------------------------

def generate_signals(prices: PriceData, params: StrategyParams) -> pd.DataFrame:
    composite = _build_composite(prices, params)
    weights = _signal_to_weights(composite, params.long_pct, params.short_pct, params.gross_leverage)
    return _apply_rebalance(weights, params.rebalance_days)
