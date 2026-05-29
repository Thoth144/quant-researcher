"""
Panel data loader. Reads per-ticker parquets from data/cache/ into wide DataFrames
indexed by trading date, columns by ticker.

LOCKED — do not modify. This is part of the harness defining run comparability.

Point-in-time S&P 500 membership (improvement #6):
  - load_membership() reconstructs per-date membership from the current pinned
    universe + data/sp500_changes.parquet (Wikipedia historical additions/removals).
  - load_panel(apply_membership=True) masks non-member ticker-dates to NaN so
    signals and ranking naturally exclude them.
  - If sp500_changes.parquet is absent, the universe is treated as permanent
    (survivorship-biased) and a warning is printed once.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_ROOT / "cache"
UNIVERSE_FILE = DATA_ROOT / "sp500_universe.txt"
CHANGES_FILE = DATA_ROOT / "sp500_changes.parquet"


@dataclass(frozen=True)
class PriceData:
    """Wide-format daily panel. Index = trading dates, columns = tickers."""
    close: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame      # simple close-to-close
    membership: pd.DataFrame   # bool DataFrame, True where the ticker was an S&P 500 member on that date

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index


@lru_cache(maxsize=1)
def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(
            f"Universe file missing at {UNIVERSE_FILE}. Run: uv run python -m data.prepare"
        )
    return [line.strip() for line in UNIVERSE_FILE.read_text().splitlines() if line.strip()]


def _build_membership_panel(trading_dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """
    For each (date, ticker), True iff the ticker was an S&P 500 member on that date.

    Reconstruction:
      - For tickers with no recorded events: True iff in current pinned universe
        (assumption: their membership is permanent within the window).
      - For tickers with events: chronologically apply the events. Initial state
        (before the first event) = True if first event is 'remove', False if 'add'.
    """
    current = set(load_universe())
    if not CHANGES_FILE.exists():
        # No historical changes — fall back to permanent-membership (survivorship-biased).
        # Log once via stderr to keep the harness's stdout quiet.
        import sys as _sys
        print(
            f"warning: {CHANGES_FILE.name} missing; falling back to survivorship-biased "
            "permanent-membership. Run `uv run python -m data.prepare_membership`.",
            file=_sys.stderr,
        )
        return pd.DataFrame(
            np.broadcast_to(np.array([t in current for t in tickers], dtype=bool), (len(trading_dates), len(tickers))),
            index=trading_dates, columns=tickers,
        )

    changes = pd.read_parquet(CHANGES_FILE)
    changes["date"] = pd.to_datetime(changes["date"])

    panel = pd.DataFrame(
        np.zeros((len(trading_dates), len(tickers)), dtype=bool),
        index=trading_dates, columns=tickers,
    )

    events_by_ticker = changes.groupby("ticker")

    for ticker in tickers:
        if ticker in events_by_ticker.groups:
            events = events_by_ticker.get_group(ticker).sort_values("date")
            first_action = events.iloc[0]["action"]
            state = (first_action == "remove")  # was a member before its first 'remove'
            series = pd.Series(state, index=trading_dates)
            for _, ev in events.iterrows():
                if ev["action"] == "add":
                    series.loc[series.index >= ev["date"]] = True
                elif ev["action"] == "remove":
                    series.loc[series.index >= ev["date"]] = False
            panel[ticker] = series
        else:
            # No events recorded — permanent member iff currently in universe
            panel[ticker] = (ticker in current)

    return panel


@lru_cache(maxsize=4)
def load_panel(min_tickers: int = 0, apply_membership: bool = True) -> PriceData:
    """
    Load all cached tickers into a wide panel.

    apply_membership=True (default): point-in-time S&P 500 membership masks
        close/volume/returns to NaN for ticker-dates outside the membership window.
        This removes survivorship bias. Set False only for diagnostics.

    min_tickers: fail loudly if fewer than this many tickers have data.
    """
    universe = load_universe()
    closes, volumes = {}, {}
    for t in universe:
        p = CACHE_DIR / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty or "Close" not in df.columns:
            continue
        closes[t] = df["Close"]
        volumes[t] = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)

    if len(closes) < min_tickers:
        raise RuntimeError(
            f"Only {len(closes)} tickers loaded (need >={min_tickers}). "
            f"Run: uv run python -m data.prepare"
        )

    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex_like(close)
    returns = close.pct_change()

    membership = _build_membership_panel(close.index, list(close.columns))

    if apply_membership:
        close = close.where(membership)
        volume = volume.where(membership)
        returns = returns.where(membership)

    return PriceData(close=close, volume=volume, returns=returns, membership=membership)
