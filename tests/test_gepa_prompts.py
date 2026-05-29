"""Registry CRUD + dedupe + lineage tests."""

import tempfile
from pathlib import Path

import pytest

from researcher.gepa.prompts import PromptRegistry, content_hash


@pytest.fixture
def reg(tmp_path):
    return PromptRegistry(db_path=tmp_path / "runs.db")


def test_register_returns_prompt_with_id(reg):
    p = reg.register("hello prompt", source="seed")
    assert p.id is not None
    assert p.content == "hello prompt"
    assert p.content_hash == content_hash("hello prompt")
    assert p.source == "seed"
    assert p.parent_id is None


def test_register_dedupes_identical_content(reg):
    a = reg.register("same content", source="seed")
    b = reg.register("same content", source="mutation:noop")
    assert a.id == b.id  # dedupe wins
    # Source on the dedupe path returns the ORIGINAL row's source
    assert b.source == "seed"


def test_get_by_id_and_hash(reg):
    p = reg.register("findable", source="seed")
    assert reg.get(p.id).id == p.id
    assert reg.get_by_hash(p.content_hash).id == p.id
    assert reg.get(99999) is None


def test_update_fitness(reg):
    p = reg.register("scored", source="seed")
    reg.update_fitness(p.id, {"hit_rate": 0.42, "n_decided": 10})
    refetched = reg.get(p.id)
    assert refetched.n_evaluations == 1
    assert refetched.fitness_json is not None
    assert '"hit_rate": 0.42' in refetched.fitness_json


def test_lineage(reg):
    a = reg.register("a", source="seed")
    b = reg.register("b", source="mutation", parent_id=a.id)
    c = reg.register("c", source="mutation", parent_id=b.id)
    lineage = reg.lineage_of(c.id)
    assert [p.id for p in lineage] == [a.id, b.id, c.id]


def test_leaderboard_orders_by_primary_metric(reg):
    a = reg.register("a", source="seed")
    b = reg.register("b", source="seed")
    c = reg.register("c", source="seed")
    reg.update_fitness(a.id, {"hit_rate": 0.1})
    reg.update_fitness(b.id, {"hit_rate": 0.5})
    reg.update_fitness(c.id, {"hit_rate": 0.3})
    lb = reg.leaderboard(limit=10, primary_metric="hit_rate")
    assert [r["id"] for r in lb] == [b.id, c.id, a.id]


def test_leaderboard_skips_unscored(reg):
    a = reg.register("a", source="seed")
    b = reg.register("b", source="seed")
    reg.update_fitness(a.id, {"hit_rate": 0.2})
    # b has no fitness set
    lb = reg.leaderboard(limit=10)
    assert [r["id"] for r in lb] == [a.id]


def test_generations_logging(reg):
    a = reg.register("a", source="seed")
    reg.log_generation(run_id=1, generation=0, prompt_id=a.id, fitness={"hit_rate": 0.3}, is_parent=True)
    rows = reg.generations(run_id=1)
    assert len(rows) == 1
    assert rows[0]["is_parent"] == 1
