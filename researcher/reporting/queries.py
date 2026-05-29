"""
Shared read-only DB queries used by the TUI dashboard and HTML report generator.

All queries safe to run concurrently with an active research session (sqlite WAL).
No writes from this module — observability only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from researcher import runs
from researcher.runs import connect


@dataclass(frozen=True)
class SessionSummary:
    id: int
    started_at: str
    ended_at: str | None
    proposer: str
    n_iterations_requested: int | None
    replicate_seeds: int | None
    n_candidates: int
    n_decided: int
    n_kept: int
    n_inconclusive: int
    n_discard: int
    hit_rate: float
    notes: str | None
    alive: bool   # heuristic: ended_at is null AND has recent activity


@dataclass(frozen=True)
class CandidateView:
    id: int
    parent_id: int | None
    decision: str | None
    accepted: bool
    primary_delta: float | None
    rationale: str | None
    diff: str
    created_at: str
    mean_metric: float | None
    n_trials: int


@dataclass(frozen=True)
class TrialView:
    seed: int | None
    status: str
    primary_metric: float | None
    metrics: dict | None


def session_metric_label(session_id: int, db_path: Path | None = None) -> tuple[str, str]:
    """
    Return (metric_name, metric_format) for a session's domain.
    Reads sessions.proposer_config_json -> looks up domain in DOMAINS.
    Falls back to ('OOS Sharpe', '+.4f') if domain unknown (pre-Domain sessions).
    """
    db_path = db_path or runs.DB_PATH
    if not Path(db_path).exists():
        return ("OOS Sharpe", "+.4f")
    with connect(db_path) as conn:
        r = conn.execute(
            "SELECT proposer_config_json FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if not r or not r["proposer_config_json"]:
        return ("OOS Sharpe", "+.4f")
    try:
        cfg = json.loads(r["proposer_config_json"])
        domain_name = cfg.get("domain", "finance")
    except (json.JSONDecodeError, TypeError):
        return ("OOS Sharpe", "+.4f")
    # Lookup without import-time circular risk
    from researcher.domain import DOMAINS
    d = DOMAINS.get(domain_name)
    if d is None:
        return ("OOS Sharpe", "+.4f")
    return (d.primary_metric_name, d.primary_metric_format)


def list_sessions(db_path: Path | None = None) -> list[SessionSummary]:
    db_path = db_path or runs.DB_PATH
    if not Path(db_path).exists():
        return []
    out: list[SessionSummary] = []
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, started_at, ended_at, proposer, n_iterations_requested, "
            "       replicate_seeds, notes FROM sessions ORDER BY id"
        ).fetchall()
        for r in rows:
            stats = _session_stats(conn, r["id"])
            out.append(SessionSummary(
                id=r["id"], started_at=r["started_at"], ended_at=r["ended_at"],
                proposer=r["proposer"],
                n_iterations_requested=r["n_iterations_requested"],
                replicate_seeds=r["replicate_seeds"], notes=r["notes"],
                alive=(r["ended_at"] is None),
                **stats,
            ))
    return out


def _session_stats(conn, session_id: int) -> dict:
    r = conn.execute(
        "SELECT "
        "  COUNT(*) AS n_candidates, "
        "  SUM(CASE WHEN decision IS NOT NULL THEN 1 ELSE 0 END) AS n_decided, "
        "  SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) AS n_kept, "
        "  SUM(CASE WHEN decision = 'inconclusive' THEN 1 ELSE 0 END) AS n_inconclusive, "
        "  SUM(CASE WHEN decision = 'discard' THEN 1 ELSE 0 END) AS n_discard "
        "FROM candidates WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    n_decided = r["n_decided"] or 0
    n_kept = r["n_kept"] or 0
    return {
        "n_candidates": r["n_candidates"] or 0,
        "n_decided": n_decided,
        "n_kept": n_kept,
        "n_inconclusive": r["n_inconclusive"] or 0,
        "n_discard": r["n_discard"] or 0,
        "hit_rate": (n_kept / n_decided) if n_decided else 0.0,
    }


def get_session(session_id: int, db_path: Path | None = None) -> SessionSummary | None:
    for s in list_sessions(db_path):
        if s.id == session_id:
            return s
    return None


def candidates_for_session(
    session_id: int, limit: int | None = None, db_path: Path | None = None,
) -> list[CandidateView]:
    db_path = db_path or runs.DB_PATH
    if not Path(db_path).exists():
        return []
    with connect(db_path) as conn:
        q = (
            "SELECT c.id, c.parent_id, c.decision, c.accepted, c.decision_payload_json, "
            "       c.rationale, c.diff, c.created_at, "
            "       AVG(t.primary_metric) AS mean_metric, "
            "       COUNT(t.id) AS n_trials "
            "FROM candidates c LEFT JOIN trials t ON t.candidate_id = c.id AND t.status = 'ok' "
            "WHERE c.session_id = ? "
            "GROUP BY c.id "
            "ORDER BY c.id"
        )
        rows = conn.execute(q, (session_id,)).fetchall()

    out: list[CandidateView] = []
    for r in rows:
        payload = json.loads(r["decision_payload_json"]) if r["decision_payload_json"] else {}
        out.append(CandidateView(
            id=r["id"], parent_id=r["parent_id"],
            decision=r["decision"], accepted=bool(r["accepted"]),
            primary_delta=payload.get("primary_delta"),
            rationale=r["rationale"], diff=r["diff"] or "",
            created_at=r["created_at"],
            mean_metric=r["mean_metric"], n_trials=r["n_trials"] or 0,
        ))
    if limit:
        out = out[-limit:]
    return out


def trials_for_candidate(candidate_id: int, db_path: Path | None = None) -> list[TrialView]:
    db_path = db_path or runs.DB_PATH
    if not Path(db_path).exists():
        return []
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT seed, status, primary_metric, metrics_json "
            "FROM trials WHERE candidate_id = ? ORDER BY seed",
            (candidate_id,),
        ).fetchall()
    return [
        TrialView(
            seed=r["seed"], status=r["status"], primary_metric=r["primary_metric"],
            metrics=json.loads(r["metrics_json"]) if r["metrics_json"] else None,
        )
        for r in rows
    ]


def leaderboard(session_id: int, limit: int = 10, db_path: Path | None = None) -> list[CandidateView]:
    """Accepted candidates only, sorted by mean OOS metric descending."""
    cands = [c for c in candidates_for_session(session_id, db_path=db_path) if c.accepted]
    cands.sort(key=lambda c: (c.mean_metric is None, -(c.mean_metric or 0.0)))
    return cands[:limit]


def lineage_tree(session_id: int, db_path: Path | None = None) -> dict[int | None, list[int]]:
    """parent_id -> list of child_ids, walking forward from the baseline."""
    cands = candidates_for_session(session_id, db_path=db_path)
    tree: dict[int | None, list[int]] = {}
    for c in cands:
        tree.setdefault(c.parent_id, []).append(c.id)
    return tree
