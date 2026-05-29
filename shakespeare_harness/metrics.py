"""
val_bpb (validation bits per byte) — the LOWER-better primary metric.

LOCKED — do not modify. Definition is the comparability contract.

bpb is vocab-size-independent so architectural changes (different tokenizers,
different vocab) remain fairly comparable. For char-level tokenizers on ASCII
text, byte_count == char_count, so bpb == loss / ln(2) per character.
"""

from __future__ import annotations

import math


def loss_to_bpb(cross_entropy_nats_per_token: float, bytes_per_token: float = 1.0) -> float:
    """
    Convert nat-per-token cross-entropy loss to bits-per-byte.
      bpb = (nats / token) * (tokens / byte) / ln(2)
          = (nats / token) / (bytes / token) / ln(2)
    """
    if bytes_per_token <= 0:
        return float("inf")
    return cross_entropy_nats_per_token / (math.log(2) * bytes_per_token)
