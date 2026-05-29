"""
One-time data preparation for quant-researcher.

Pins the S&P 500 universe (fetched once from Wikipedia, then frozen to
data/sp500_universe.txt) and downloads adjusted daily bars 2010-01-01 to
2024-12-31 via yfinance. Each ticker cached to its own parquet so partial
downloads are resumable.

Usage:
    uv run python -m data.prepare                  # full universe + full range
    uv run python -m data.prepare --num-tickers 5  # tiny smoke subset
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_ROOT = Path(__file__).parent
UNIVERSE_FILE = DATA_ROOT / "sp500_universe.txt"
CACHE_DIR = DATA_ROOT / "cache"

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (quant-researcher pinned-universe fetch; +https://github.com/colbymchenry)"

# Fallback if Wikipedia fetch fails. 100 large-cap S&P 500 names as of late 2024.
# Not the full 500 — but enough for a real backtest. Replace by deleting
# sp500_universe.txt and re-running with network access.
FALLBACK_UNIVERSE: list[str] = sorted([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "UNH", "XOM", "WMT", "PG", "MA", "HD", "CVX", "ABBV", "AVGO",
    "LLY", "PEP", "KO", "MRK", "COST", "ADBE", "BAC", "PFE", "TMO", "CSCO",
    "ACN", "MCD", "ABT", "CRM", "DHR", "WFC", "DIS", "VZ", "TXN", "NEE",
    "NKE", "BMY", "PM", "AMD", "ORCL", "QCOM", "RTX", "T", "UPS", "HON",
    "LIN", "INTC", "UNP", "SBUX", "LOW", "IBM", "MS", "GS", "INTU", "AMGN",
    "CAT", "BA", "AXP", "DE", "GE", "BLK", "ELV", "ISRG", "GILD", "BKNG",
    "ADP", "PLD", "TJX", "MDLZ", "SYK", "SCHW", "C", "REGN", "VRTX", "PYPL",
    "LMT", "MMC", "ZTS", "CB", "CI", "MO", "SO", "DUK", "BSX", "EOG",
    "EQIX", "AON", "PGR", "ETN", "ITW", "ICE", "APD", "FCX", "NOC", "EMR",
])


def _fetch_wikipedia() -> list[str]:
    r = requests.get(WIKIPEDIA_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    # Newer pandas requires file-like wrapper around raw HTML string
    tables = pd.read_html(io.StringIO(r.text))
    return sorted(tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())


def ensure_universe() -> list[str]:
    """Return pinned tickers. Cached to sp500_universe.txt after first call."""
    if UNIVERSE_FILE.exists():
        tickers = [line.strip() for line in UNIVERSE_FILE.read_text().splitlines() if line.strip()]
        print(f"Universe: {len(tickers)} tickers loaded from {UNIVERSE_FILE.name}")
        return tickers

    print("Universe: fetching S&P 500 list from Wikipedia (one-time)...")
    try:
        tickers = _fetch_wikipedia()
        source = "Wikipedia"
    except Exception as e:
        print(f"  Wikipedia fetch failed ({type(e).__name__}: {e}); using FALLBACK_UNIVERSE")
        tickers = list(FALLBACK_UNIVERSE)
        source = "fallback (100 large-caps)"

    UNIVERSE_FILE.write_text("\n".join(tickers) + "\n")
    print(f"Universe: pinned {len(tickers)} tickers from {source} to {UNIVERSE_FILE.name} (commit this file)")
    return tickers


def download_one(ticker: str) -> bool:
    """Download a single ticker's adjusted daily bars. Idempotent."""
    out = CACHE_DIR / f"{ticker}.parquet"
    if out.exists():
        return True
    try:
        df = yf.download(
            ticker, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False, threads=False,
        )
        if df.empty:
            print(f"  {ticker}: empty (delisted or no data)")
            return False
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_parquet(out)
        return True
    except Exception as e:
        print(f"  {ticker}: failed ({e})")
        return False


def download_universe(tickers: list[str], pause: float = 0.0) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for t in tickers if (CACHE_DIR / f"{t}.parquet").exists())
    needed = len(tickers) - existing
    print(f"Bars: {existing}/{len(tickers)} already cached, downloading {needed}")
    if needed == 0:
        return

    ok = existing
    for i, ticker in enumerate(tickers):
        if (CACHE_DIR / f"{ticker}.parquet").exists():
            continue
        if download_one(ticker):
            ok += 1
        if (i + 1) % 25 == 0:
            print(f"  progress: {ok}/{len(tickers)} done")
        if pause > 0:
            time.sleep(pause)

    print(f"Bars: {ok}/{len(tickers)} cached at {CACHE_DIR}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-tickers", type=int, default=-1, help="Limit to first N tickers (smoke test)")
    p.add_argument("--pause", type=float, default=0.0, help="Seconds between requests (avoid rate limits)")
    args = p.parse_args()

    tickers = ensure_universe()
    if args.num_tickers > 0:
        tickers = tickers[: args.num_tickers]
        print(f"Smoke subset: {len(tickers)} tickers")

    download_universe(tickers, pause=args.pause)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
