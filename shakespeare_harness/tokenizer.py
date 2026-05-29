"""
Character-level tokenizer over the tiny-shakespeare corpus.

LOCKED — do not modify. The vocab definition is the comparability contract:
two runs are comparable only if they tokenize the same way.

Vocab = the unique ASCII chars present in the full corpus (~65 chars for tiny
shakespeare). Deterministic via sorted() so successive loads give the same
char→id mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from shakespeare_harness.data import load


@dataclass(frozen=True)
class CharTokenizer:
    chars: tuple[str, ...]          # sorted unique chars from full corpus
    stoi: dict[str, int]
    itos: tuple[str, ...]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


@lru_cache(maxsize=1)
def get_tokenizer() -> CharTokenizer:
    full = load().full_text
    chars = tuple(sorted(set(full)))
    stoi = {c: i for i, c in enumerate(chars)}
    return CharTokenizer(chars=chars, stoi=stoi, itos=chars)
