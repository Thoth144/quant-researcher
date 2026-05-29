"""
Cross-session memory (#3).

Surfaces relevant prior attempts from ALL sessions to the proposer, beyond the
current session's recent_decided list. Stops the agent from rediscovering moves
that were tried (and worked OR failed) in earlier runs.

Strategy:
  1. Extract feature tokens from current strategy.py — signal names from
     enabled_signals, the combine_mode value.
  2. Query candidates_fts (FTS5 index over diff + rationale) for any candidate
     mentioning those tokens, from sessions OTHER than the current.
  3. Rank: kept-with-larger-|Δ| first, then discarded-with-larger-|Δ|.
  4. Format for inclusion in the proposer's user message.

Read-only. Safe to run alongside an active research session (sqlite WAL).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from researcher.runs import DB_PATH, connect


@dataclass(frozen=True)
class CrossSessionExample:
    session_id: int
    candidate_id: int
    decision: str
    accepted: bool
    primary_delta: float | None
    rationale: str | None


def _extract_features(strategy_src: str) -> list[str]:
    """
    Pull signal names from enabled_signals tuple + the combine_mode value.
    Returns lowercase, dedup'd, FTS5-safe tokens (no quotes/parens).
    """
    features: set[str] = set()

    # enabled_signals = (...) — grab quoted strings inside
    m = re.search(r"enabled_signals[^=]*=\s*\(([^)]+)\)", strategy_src, re.DOTALL)
    if m:
        for sig in re.findall(r'["\']([A-Za-z_][\w]*)["\']', m.group(1)):
            features.add(sig.lower())

    # combine_mode = "..."  (single or double quoted)
    m = re.search(r'combine_mode[^=]*=\s*["\']([A-Za-z_][\w]*)["\']', strategy_src)
    if m:
        features.add(m.group(1).lower())

    # FTS5: drop any with non-alphanumeric-underscore
    return [f for f in sorted(features) if re.fullmatch(r"[A-Za-z_][\w]*", f)]


def find_similar_attempts(
    strategy_src: str,
    exclude_session_id: int | None = None,
    limit: int = 8,
    db_path: Path = DB_PATH,
) -> list[CrossSessionExample]:
    """
    Search candidates_fts for prior attempts mentioning features of the current strategy.
    Returns up to `limit` examples, ranked by informativeness (kept > discard, larger |Δ| first).
    """
    features = _extract_features(strategy_src)
    if not features:
        return []

    # FTS5 OR query — matches any candidate whose diff or rationale mentions any feature
    fts_query = " OR ".join(features)

    base_query = (
        "SELECT c.session_id, c.id AS cid, c.decision, c.accepted, "
        "       c.decision_payload_json, c.rationale "
        "FROM candidates c "
        "JOIN candidates_fts fts ON fts.rowid = c.id "
        "WHERE candidates_fts MATCH ? AND c.decision IS NOT NULL "
    )
    params: list = [fts_query]
    if exclude_session_id is not None:
        base_query += "AND c.session_id != ? "
        params.append(exclude_session_id)
    base_query += "ORDER BY c.accepted DESC, c.id DESC LIMIT ?"
    params.append(limit * 4)  # over-fetch; we'll re-rank below

    with connect(db_path) as conn:
        try:
            rows = conn.execute(base_query, params).fetchall()
        except Exception:
            # FTS5 syntax errors on weird tokens — fail open
            return []

    out: list[CrossSessionExample] = []
    seen: set[tuple[int, int]] = set()
    for r in rows:
        key = (r["session_id"], r["cid"])
        if key in seen:
            continue
        seen.add(key)
        payload = json.loads(r["decision_payload_json"]) if r["decision_payload_json"] else {}
        out.append(CrossSessionExample(
            session_id=r["session_id"], candidate_id=r["cid"],
            decision=r["decision"], accepted=bool(r["accepted"]),
            primary_delta=payload.get("primary_delta"),
            rationale=r["rationale"],
        ))

    # Final ranking: kept first (largest |Δ|), then discarded (largest |Δ|)
    out.sort(key=lambda x: (
        not x.accepted,                       # False (= kept) sorts first
        -abs(x.primary_delta or 0.0),         # larger magnitude first
    ))
    return out[:limit]


def format_cross_session(examples: list[CrossSessionExample]) -> str:
    if not examples:
        return "(no relevant prior attempts found across other sessions)"
    lines = []
    for e in examples:
        delta = f"{e.primary_delta:+.4f}" if e.primary_delta is not None else "n/a"
        rat = (e.rationale or "(no rationale)").replace("\n", " ").strip()[:160]
        marker = "KEEP" if e.accepted else e.decision.upper()[:8]
        lines.append(f"  [{marker} Δ={delta}] s{e.session_id} #{e.candidate_id}: {rat}")
    return "\n".join(lines)
