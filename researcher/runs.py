"""
SQLite provenance store for the research loop.

Two grain levels:
  - `candidates`: one row per strategy proposal (one diff + rationale + decision).
  - `trials`:     one row per backtest execution. N trials per candidate (one per seed).

FTS5 over candidate diff+rationale supports retrieval-augmented proposer prompts
("show me prior attempts similar to what we're about to try").

Hit rate = accepted candidates / total candidates with a decision. Direct proxy
for proposer reasoning quality across sessions.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(__file__).parent.parent / "runs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT,
    proposer TEXT NOT NULL,
    proposer_config_json TEXT,
    n_iterations_requested INTEGER,
    replicate_seeds INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    parent_id INTEGER REFERENCES candidates(id),
    diff TEXT NOT NULL,
    rationale TEXT,
    decision TEXT,
    decision_payload_json TEXT,
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_candidates_session ON candidates(session_id);
CREATE INDEX IF NOT EXISTS idx_candidates_decision ON candidates(decision);

CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id),
    seed INTEGER,
    status TEXT NOT NULL,
    primary_metric REAL,
    metrics_json TEXT,
    params_json TEXT,
    log TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_trials_candidate ON trials(candidate_id);

CREATE TABLE IF NOT EXISTS signal_attribution (
    id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id),
    signal_name TEXT NOT NULL,
    standalone_oos_sharpe REAL,
    weight_in_candidate REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_attribution_candidate ON signal_attribution(candidate_id);

CREATE VIRTUAL TABLE IF NOT EXISTS candidates_fts USING fts5(
    diff, rationale, content='candidates', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS candidates_ai AFTER INSERT ON candidates BEGIN
    INSERT INTO candidates_fts(rowid, diff, rationale)
    VALUES (new.id, new.diff, coalesce(new.rationale, ''));
END;
CREATE TRIGGER IF NOT EXISTS candidates_au AFTER UPDATE OF diff, rationale ON candidates BEGIN
    INSERT INTO candidates_fts(candidates_fts, rowid, diff, rationale)
    VALUES ('delete', old.id, old.diff, coalesce(old.rationale, ''));
    INSERT INTO candidates_fts(rowid, diff, rationale)
    VALUES (new.id, new.diff, coalesce(new.rationale, ''));
END;
"""


@dataclass(frozen=True)
class CandidateRecord:
    id: int
    parent_id: int | None
    diff: str
    rationale: str | None
    decision: str | None
    primary_delta: float | None
    accepted: bool


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    # Sentinel default + call-time resolution so monkeypatching `runs.DB_PATH`
    # actually takes effect (Python default-arg values are bound at def time).
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.executescript(SCHEMA)
        yield conn
    finally:
        conn.close()


def start_session(
    proposer: str, proposer_config: dict, n_iterations: int, replicate_seeds: int,
    notes: str | None = None, db_path: Path = DB_PATH,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO sessions(proposer, proposer_config_json, n_iterations_requested, replicate_seeds, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (proposer, json.dumps(proposer_config), n_iterations, replicate_seeds, notes),
        )
        return cur.lastrowid


def end_session(session_id: int, db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))


def insert_candidate(
    session_id: int, parent_id: int | None, diff: str, rationale: str | None,
    db_path: Path = DB_PATH,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO candidates(session_id, parent_id, diff, rationale) VALUES (?, ?, ?, ?)",
            (session_id, parent_id, diff, rationale),
        )
        return cur.lastrowid


def insert_trial(
    candidate_id: int, seed: int | None, status: str, primary_metric: float | None,
    metrics: dict | None, params: dict | None, log: str | None,
    db_path: Path = DB_PATH,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO trials(candidate_id, seed, status, primary_metric, metrics_json, params_json, log) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                candidate_id, seed, status, primary_metric,
                json.dumps(metrics) if metrics is not None else None,
                json.dumps(params) if params is not None else None,
                log,
            ),
        )
        return cur.lastrowid


def record_decision(
    candidate_id: int, decision: str, payload: dict, accepted: bool, db_path: Path = DB_PATH,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE candidates SET decision = ?, decision_payload_json = ?, accepted = ? WHERE id = ?",
            (decision, json.dumps(payload), 1 if accepted else 0, candidate_id),
        )


def trials_for_candidate(candidate_id: int, db_path: Path = DB_PATH) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(conn.execute(
            "SELECT * FROM trials WHERE candidate_id = ? ORDER BY seed",
            (candidate_id,),
        ))


def recent_candidates(
    session_id: int, limit: int = 20, only_decided: bool = True, db_path: Path = DB_PATH,
) -> list[CandidateRecord]:
    where = "session_id = ?" + (" AND decision IS NOT NULL" if only_decided else "")
    with connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT id, parent_id, diff, rationale, decision, decision_payload_json, accepted "
            f"FROM candidates WHERE {where} ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        payload = json.loads(r["decision_payload_json"]) if r["decision_payload_json"] else {}
        out.append(CandidateRecord(
            id=r["id"], parent_id=r["parent_id"], diff=r["diff"], rationale=r["rationale"],
            decision=r["decision"], primary_delta=payload.get("primary_delta"),
            accepted=bool(r["accepted"]),
        ))
    return out


def current_best(session_id: int, db_path: Path = DB_PATH) -> CandidateRecord | None:
    """Most recent accepted candidate in this session (the active parent for next proposal)."""
    with connect(db_path) as conn:
        r = conn.execute(
            "SELECT id, parent_id, diff, rationale, decision, decision_payload_json, accepted "
            "FROM candidates WHERE session_id = ? AND accepted = 1 ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if r is None:
        return None
    payload = json.loads(r["decision_payload_json"]) if r["decision_payload_json"] else {}
    return CandidateRecord(
        id=r["id"], parent_id=r["parent_id"], diff=r["diff"], rationale=r["rationale"],
        decision=r["decision"], primary_delta=payload.get("primary_delta"),
        accepted=True,
    )


def hit_rate(session_id: int, db_path: Path = DB_PATH) -> tuple[int, int, float]:
    """Returns (n_accepted, n_decided, ratio). Ignores inconclusive and crashes-only."""
    with connect(db_path) as conn:
        r = conn.execute(
            "SELECT "
            "  SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) AS hits, "
            "  SUM(CASE WHEN decision IS NOT NULL THEN 1 ELSE 0 END) AS decided "
            "FROM candidates WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    hits = r["hits"] or 0
    decided = r["decided"] or 0
    ratio = (hits / decided) if decided else 0.0
    return hits, decided, ratio


def leaderboard(session_id: int, limit: int = 10, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Top accepted candidates by mean primary_metric across their trials."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT c.id, c.rationale, AVG(t.primary_metric) AS mean_metric, COUNT(t.id) AS n_trials "
            "FROM candidates c JOIN trials t ON t.candidate_id = c.id "
            "WHERE c.session_id = ? AND c.accepted = 1 AND t.status = 'ok' "
            "GROUP BY c.id ORDER BY mean_metric DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
