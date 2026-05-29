"""
Scrape Wikipedia's S&P 500 'Selected changes' table and persist as a
historical-changes parquet. Combined with the current pinned universe, this
lets the harness reconstruct approximate point-in-time membership for any date
in [2010-01-01, 2024-12-31].

Removes the largest survivorship bias from backtests: a ticker that ENTERED the
S&P 500 in 2018 (e.g., post-IPO) is excluded from 2010-2017 portfolios; a ticker
that EXITED in 2015 is included only through 2015 even though it's not in
today's pinned universe.

Limitations:
  - Wikipedia's table only goes back so far; gaps before ~2000 are likely.
  - Some changes are misclassified or undated. We fail open: when a ticker's
    historical status is unknown, treat it as a member if it has price data.

Usage:
    uv run python -m data.prepare_membership
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

from data.prepare import WIKIPEDIA_URL, USER_AGENT

DATA_ROOT = Path(__file__).parent
CHANGES_FILE = DATA_ROOT / "sp500_changes.parquet"


def _fetch_changes_table() -> pd.DataFrame:
    """Wikipedia page has two tables: [0] current constituents, [1] selected changes."""
    r = requests.get(WIKIPEDIA_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    if len(tables) < 2:
        raise RuntimeError(f"Wikipedia returned {len(tables)} tables; expected at least 2")
    changes = tables[1]
    return changes


def _normalize_changes(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape Wikipedia's two-row-header changes table into long form:
        date | ticker | action ('add' | 'remove')

    The raw table has a MultiIndex header like:
        ('Date',)               | ('Added', 'Ticker') | ('Added', 'Security') | ('Removed', 'Ticker') | ('Removed', 'Security') | ('Reason',)
    We extract just the date + ticker + add/remove rows.
    """
    df = raw.copy()
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(c).strip() for c in col if str(c) != "nan"]).strip()
            for col in df.columns.values
        ]

    # Find date / added-ticker / removed-ticker columns by fuzzy match
    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    add_col = next(
        (c for c in df.columns if "added" in str(c).lower() and "ticker" in str(c).lower()),
        None,
    )
    rem_col = next(
        (c for c in df.columns if "removed" in str(c).lower() and "ticker" in str(c).lower()),
        None,
    )
    if not all([date_col, add_col, rem_col]):
        raise RuntimeError(
            f"Could not identify columns in changes table. Got: {list(df.columns)}"
        )

    rows: list[dict] = []
    for _, r in df.iterrows():
        date = pd.to_datetime(r[date_col], errors="coerce")
        if pd.isna(date):
            continue
        add_t = str(r[add_col]).strip()
        rem_t = str(r[rem_col]).strip()
        if add_t and add_t.lower() not in ("nan", "none", ""):
            rows.append({"date": date, "ticker": add_t.replace(".", "-"), "action": "add"})
        if rem_t and rem_t.lower() not in ("nan", "none", ""):
            rows.append({"date": date, "ticker": rem_t.replace(".", "-"), "action": "remove"})

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["date", "ticker"]).sort_values("date").reset_index(drop=True)
    return out


def main() -> int:
    print(f"Fetching S&P 500 changes from Wikipedia...")
    raw = _fetch_changes_table()
    print(f"  raw changes table: {len(raw)} rows")
    changes = _normalize_changes(raw)
    n_add = int((changes["action"] == "add").sum())
    n_rem = int((changes["action"] == "remove").sum())
    print(f"  normalized: {len(changes)} events ({n_add} adds, {n_rem} removes)")
    print(f"  date range: {changes['date'].min().date()} → {changes['date'].max().date()}")

    changes.to_parquet(CHANGES_FILE, index=False)
    print(f"  saved to {CHANGES_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
