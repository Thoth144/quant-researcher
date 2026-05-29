"""
Tiny Shakespeare data: downloads karpathy/char-rnn's input.txt, caches locally,
splits into train/val.

LOCKED — do not modify. Definitions are the comparability contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

DATA_ROOT = Path(__file__).parent.parent / "data"
SHAKESPEARE_CACHE = DATA_ROOT / "shakespeare_input.txt"

SOURCE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TRAIN_FRAC = 0.9


@dataclass(frozen=True)
class ShakespeareData:
    train_text: str
    val_text: str
    full_text: str

    @property
    def n_train_bytes(self) -> int:
        return len(self.train_text.encode("utf-8"))

    @property
    def n_val_bytes(self) -> int:
        return len(self.val_text.encode("utf-8"))


def _ensure_downloaded() -> Path:
    if SHAKESPEARE_CACHE.exists() and SHAKESPEARE_CACHE.stat().st_size > 100_000:
        return SHAKESPEARE_CACHE
    SHAKESPEARE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=30) as resp:
        body = resp.read()
    SHAKESPEARE_CACHE.write_bytes(body)
    return SHAKESPEARE_CACHE


@lru_cache(maxsize=1)
def load() -> ShakespeareData:
    """Download (once) and split tiny shakespeare into train/val."""
    path = _ensure_downloaded()
    full = path.read_text(encoding="utf-8")
    n = len(full)
    split_at = int(n * TRAIN_FRAC)
    return ShakespeareData(
        train_text=full[:split_at],
        val_text=full[split_at:],
        full_text=full,
    )
