"""
Prompt artifact + PromptRegistry.

A Prompt is the unit GEPA evolves. Each one has provenance (parent_id, source),
a content hash (so we dedupe identical text across mutations), and an optional
last-measured FitnessScore JSON blob.

Storage extends the existing runs.db with two new tables. We do NOT mutate
the existing `sessions` table; instead a join table `session_prompts` links
sessions to the prompt that drove them. This keeps the original schema stable
and the gepa extension purely additive.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from researcher.runs import DB_PATH

GEPA_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES prompts(id),
    source TEXT NOT NULL,                    -- 'seed' | 'mutation:<operator>' | 'gepa:reflective'
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,       -- dedupes identical text across mutations
    fitness_json TEXT,                       -- last-measured FitnessScore, NULL until evaluated
    n_evaluations INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_prompts_parent ON prompts(parent_id);

CREATE TABLE IF NOT EXISTS session_prompts (
    session_id INTEGER PRIMARY KEY REFERENCES sessions(id),
    prompt_id INTEGER NOT NULL REFERENCES prompts(id)
);

CREATE TABLE IF NOT EXISTS gepa_generations (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,                 -- groups generations of one GEPA invocation
    generation INTEGER NOT NULL,
    prompt_id INTEGER NOT NULL REFERENCES prompts(id),
    fitness_json TEXT,
    is_parent INTEGER NOT NULL DEFAULT 0,    -- selected as parent for next generation?
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_generations_run ON gepa_generations(run_id, generation);
"""


def content_hash(text: str) -> str:
    """Stable hash of prompt content. Used for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Prompt:
    """Evolvable artifact. content_hash is derived; never set manually."""
    id: int | None
    parent_id: int | None
    source: str
    content: str
    content_hash: str = field(init=False)
    fitness_json: str | None = None
    n_evaluations: int = 0

    def __post_init__(self):
        # frozen dataclass: bypass __setattr__ via object's
        object.__setattr__(self, "content_hash", content_hash(self.content))


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Connect to runs.db and ensure the gepa schema exists alongside the base schema."""
    # Ensure base runs.py schema first (so FK targets like sessions(id) exist)
    from researcher import runs as _runs
    with _runs.connect(db_path) as base:
        pass  # creates sessions/candidates/trials if missing

    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.executescript(GEPA_SCHEMA)
        yield conn
    finally:
        conn.close()


class PromptRegistry:
    """All Prompt persistence funnels through this class. Dedupe on content_hash."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def register(self, content: str, source: str, parent_id: int | None = None) -> Prompt:
        """Insert a new prompt or return the existing one with the same content hash."""
        h = content_hash(content)
        with connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id, parent_id, source, content, fitness_json, n_evaluations "
                "FROM prompts WHERE content_hash = ?", (h,),
            ).fetchone()
            if existing:
                return self._row_to_prompt(existing)

            cur = conn.execute(
                "INSERT INTO prompts(parent_id, source, content, content_hash) "
                "VALUES (?, ?, ?, ?)",
                (parent_id, source, content, h),
            )
            new_id = cur.lastrowid

        return Prompt(
            id=new_id, parent_id=parent_id, source=source, content=content,
            fitness_json=None, n_evaluations=0,
        )

    def get(self, prompt_id: int) -> Prompt | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, parent_id, source, content, fitness_json, n_evaluations "
                "FROM prompts WHERE id = ?", (prompt_id,),
            ).fetchone()
        return self._row_to_prompt(row) if row else None

    def get_by_hash(self, h: str) -> Prompt | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, parent_id, source, content, fitness_json, n_evaluations "
                "FROM prompts WHERE content_hash = ?", (h,),
            ).fetchone()
        return self._row_to_prompt(row) if row else None

    def update_fitness(self, prompt_id: int, fitness: dict) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE prompts SET fitness_json = ?, n_evaluations = n_evaluations + 1 "
                "WHERE id = ?",
                (json.dumps(fitness), prompt_id),
            )

    def attach_to_session(self, session_id: int, prompt_id: int) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO session_prompts(session_id, prompt_id) VALUES (?, ?)",
                (session_id, prompt_id),
            )

    def lineage_of(self, prompt_id: int) -> list[Prompt]:
        """Walk up parent_id chain. Oldest ancestor first."""
        chain: list[Prompt] = []
        current_id: int | None = prompt_id
        while current_id is not None:
            p = self.get(current_id)
            if p is None:
                break
            chain.append(p)
            current_id = p.parent_id
        return list(reversed(chain))

    def leaderboard(self, limit: int = 10, primary_metric: str = "hit_rate") -> list[dict]:
        """Top prompts by primary_metric in their fitness_json."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, source, fitness_json, n_evaluations "
                "FROM prompts WHERE fitness_json IS NOT NULL"
            ).fetchall()
        out = []
        for r in rows:
            f = json.loads(r["fitness_json"]) if r["fitness_json"] else {}
            out.append({
                "id": r["id"], "source": r["source"], "metric": f.get(primary_metric),
                "fitness": f, "n_evaluations": r["n_evaluations"],
            })
        out = [x for x in out if x["metric"] is not None]
        out.sort(key=lambda x: x["metric"], reverse=True)
        return out[:limit]

    def log_generation(
        self, run_id: int, generation: int, prompt_id: int,
        fitness: dict | None, is_parent: bool,
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO gepa_generations(run_id, generation, prompt_id, fitness_json, is_parent) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, generation, prompt_id, json.dumps(fitness) if fitness else None, 1 if is_parent else 0),
            )

    def generations(self, run_id: int) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT generation, prompt_id, fitness_json, is_parent "
                "FROM gepa_generations WHERE run_id = ? ORDER BY generation, id",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_prompt(row: sqlite3.Row) -> Prompt:
        return Prompt(
            id=row["id"], parent_id=row["parent_id"], source=row["source"],
            content=row["content"], fitness_json=row["fitness_json"],
            n_evaluations=row["n_evaluations"],
        )
