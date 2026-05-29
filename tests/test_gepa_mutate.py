"""Mutator determinism + valid-output tests."""

import pytest

from researcher.gepa.mutate import DeterministicMockMutator
from researcher.gepa.prompts import PromptRegistry


@pytest.fixture
def reg(tmp_path):
    return PromptRegistry(db_path=tmp_path / "runs.db")


def test_mutate_returns_n_children(reg):
    parent = reg.register("## Output format\nuse XML\n", source="seed")
    mut = DeterministicMockMutator(registry=reg, seed=0)
    kids = mut.mutate(parent, n=3)
    assert len(kids) == 3
    for k in kids:
        assert k.id is not None
        assert k.parent_id == parent.id
        assert k.source.startswith("mutation:")


def test_mutator_is_deterministic_for_same_seed(reg):
    parent = reg.register("hello\n\n## Output format\nx", source="seed")
    mut_a = DeterministicMockMutator(registry=reg, seed=42)
    mut_b = DeterministicMockMutator(registry=reg, seed=42)
    kids_a = mut_a.mutate(parent, n=4)
    kids_b = mut_b.mutate(parent, n=4)
    # Same seed → same content (registry dedupes so same id)
    assert [k.content for k in kids_a] == [k.content for k in kids_b]


def test_different_seeds_diverge(reg):
    parent = reg.register(
        "Hello there.\n\nThis is paragraph two.\n\nThis is paragraph three.\n\n## Output format\nXML.\n",
        source="seed",
    )
    mut_a = DeterministicMockMutator(registry=reg, seed=1)
    mut_b = DeterministicMockMutator(registry=reg, seed=999)
    contents_a = [k.content for k in mut_a.mutate(parent, n=5)]
    contents_b = [k.content for k in mut_b.mutate(parent, n=5)]
    # Highly likely to diverge on at least one child
    assert contents_a != contents_b


def test_child_provenance_tracks_parent(reg):
    parent = reg.register("seed text", source="seed")
    mut = DeterministicMockMutator(registry=reg, seed=0)
    kids = mut.mutate(parent, n=2)
    for k in kids:
        assert k.parent_id == parent.id
        lineage = reg.lineage_of(k.id)
        assert lineage[0].id == parent.id
        assert lineage[-1].id == k.id
