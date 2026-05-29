"""Signal library + strategy composition tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from harness.data import PriceData
from harness.signals import SIGNAL_LIBRARY, get_signal, zscore_cs
from strategy import PARAMS, StrategyParams, generate_signals


@pytest.fixture
def synthetic_prices():
    """Tiny deterministic price panel: 5 tickers, 400 days, geometric random walk."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    tickers = ["A", "B", "C", "D", "E"]
    log_rets = rng.normal(0.0005, 0.015, size=(400, 5))
    close = pd.DataFrame(100 * np.exp(np.cumsum(log_rets, axis=0)), index=dates, columns=tickers)
    volume = pd.DataFrame(1e6, index=dates, columns=tickers)
    returns = close.pct_change()
    membership = pd.DataFrame(True, index=dates, columns=tickers)
    return PriceData(close=close, volume=volume, returns=returns, membership=membership)


def test_all_signals_return_correct_shape(synthetic_prices):
    for name, fn in SIGNAL_LIBRARY.items():
        out = fn(synthetic_prices)
        assert isinstance(out, pd.DataFrame), f"{name} returned {type(out)}, expected DataFrame"
        assert out.shape == synthetic_prices.close.shape, \
            f"{name} shape {out.shape} != prices.close shape {synthetic_prices.close.shape}"


def test_all_signals_have_finite_values_after_warmup(synthetic_prices):
    """Past the largest lookback window, signals should produce mostly finite values."""
    for name, fn in SIGNAL_LIBRARY.items():
        out = fn(synthetic_prices).iloc[300:]  # past any reasonable warmup
        finite_pct = np.isfinite(out.values).mean()
        assert finite_pct > 0.9, f"{name} only {finite_pct:.1%} finite past warmup"


def test_get_signal_known_and_unknown():
    assert get_signal("momentum_12_1") is SIGNAL_LIBRARY["momentum_12_1"]
    with pytest.raises(KeyError):
        get_signal("nonexistent_signal_xyz")


def test_zscore_cs_mean_zero_std_one(synthetic_prices):
    raw = SIGNAL_LIBRARY["momentum_3m"](synthetic_prices).iloc[300:]
    z = zscore_cs(raw)
    # Per-row mean ~0, std ~1 (after warmup, ignoring rows where everything is NaN)
    finite_rows = z.dropna(how="all")
    assert (finite_rows.mean(axis=1).abs() < 1e-6).all()
    assert ((finite_rows.std(axis=1) - 1.0).abs() < 1e-6).all()


def test_strategy_composes_default_signals(synthetic_prices):
    weights = generate_signals(synthetic_prices, PARAMS)
    assert weights.shape == synthetic_prices.close.shape
    # Positions should sum to approximately 0 per row (market-neutral long-short)
    # past the warmup window
    row_sums = weights.iloc[300:].sum(axis=1)
    # Allow small tolerance for numerical noise and integer-bucket sizes
    assert (row_sums.abs() < 0.1).all()


def test_strategy_supports_adding_signals(synthetic_prices):
    """Add a 4th signal to the typed mutation surface and verify it composes."""
    p = replace(
        PARAMS,
        enabled_signals=PARAMS.enabled_signals + ("acceleration",),
        weights={**PARAMS.weights, "acceleration": 0.3},
    )
    out = generate_signals(synthetic_prices, p)
    assert out.shape == synthetic_prices.close.shape


def test_strategy_rejects_empty_signal_set(synthetic_prices):
    p = replace(PARAMS, enabled_signals=())
    with pytest.raises(ValueError, match="enabled_signals is empty"):
        generate_signals(synthetic_prices, p)


def test_strategy_rejects_unknown_combine_mode(synthetic_prices):
    p = replace(PARAMS, combine_mode="not_a_real_mode")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown combine_mode"):
        generate_signals(synthetic_prices, p)


def test_sign_vote_combine_mode_works(synthetic_prices):
    p = replace(PARAMS, combine_mode="sign_vote")
    out = generate_signals(synthetic_prices, p)
    assert out.shape == synthetic_prices.close.shape
