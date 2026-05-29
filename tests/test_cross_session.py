"""Cross-session memory: feature extraction + FTS5 retrieval."""

import pytest

from researcher.cross_session import (
    CrossSessionExample,
    _extract_features,
    find_similar_attempts,
    format_cross_session,
)
from researcher.runs import insert_candidate, record_decision, start_session


# ---------------------------- feature extraction ----------------------------

def test_extract_signals_from_enabled_signals_tuple():
    src = '''
    @dataclass(frozen=True)
    class StrategyParams:
        enabled_signals: tuple[str, ...] = (
            "momentum_12_1", "reversion_5d", "lowvol_20d",
        )
    '''
    feats = _extract_features(src)
    assert "momentum_12_1" in feats
    assert "reversion_5d" in feats
    assert "lowvol_20d" in feats


def test_extract_combine_mode():
    src = '''
    combine_mode: str = "sign_vote"
    '''
    feats = _extract_features(src)
    assert "sign_vote" in feats


def test_extract_handles_single_quotes():
    src = "enabled_signals = ('a_signal', 'b_signal')"
    feats = _extract_features(src)
    assert "a_signal" in feats
    assert "b_signal" in feats


def test_extract_returns_empty_for_unparseable_src():
    assert _extract_features("totally unrelated source code") == []


# ---------------------------- FTS5 retrieval ----------------------------

@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    """Stand up a runs.db with two sessions of distinct candidates."""
    db = tmp_path / "runs.db"
    # Monkeypatch the module-level DB_PATH everywhere
    import researcher.runs
    import researcher.cross_session
    monkeypatch.setattr(researcher.runs, "DB_PATH", db)
    monkeypatch.setattr(researcher.cross_session, "DB_PATH", db)

    # Session 1: tried various momentum variants
    s1 = start_session("stub", {}, n_iterations=10, replicate_seeds=3, db_path=db)
    cid_a = insert_candidate(s1, None, "diff A", "Added momentum_12_1 with weight 1.2", db_path=db)
    record_decision(cid_a, "keep", {"primary_delta": 0.15}, accepted=True, db_path=db)
    cid_b = insert_candidate(s1, cid_a, "diff B", "Tried reversion_5d at higher weight", db_path=db)
    record_decision(cid_b, "discard", {"primary_delta": -0.4}, accepted=False, db_path=db)

    # Session 2: current session, should be excluded
    s2 = start_session("stub", {}, n_iterations=10, replicate_seeds=3, db_path=db)
    cid_c = insert_candidate(s2, None, "diff C", "current session momentum_12_1 attempt", db_path=db)
    record_decision(cid_c, "keep", {"primary_delta": 0.05}, accepted=True, db_path=db)

    return {"db": db, "s1": s1, "s2": s2, "cid_a": cid_a, "cid_b": cid_b, "cid_c": cid_c}


def test_find_similar_returns_prior_session_only(populated_db):
    src = 'enabled_signals = ("momentum_12_1",)'
    out = find_similar_attempts(
        src, exclude_session_id=populated_db["s2"], limit=10, db_path=populated_db["db"],
    )
    ids = [e.candidate_id for e in out]
    assert populated_db["cid_a"] in ids  # from session 1
    assert populated_db["cid_c"] not in ids  # excluded — current session


def test_find_similar_ranks_keeps_first(populated_db):
    src = 'enabled_signals = ("momentum_12_1", "reversion_5d")'
    out = find_similar_attempts(
        src, exclude_session_id=populated_db["s2"], limit=10, db_path=populated_db["db"],
    )
    # cid_a was kept (accepted=True), cid_b was discarded — kept should come first
    accepted_first = [e for e in out if e.accepted] + [e for e in out if not e.accepted]
    assert out == accepted_first


def test_find_similar_returns_empty_when_no_features(populated_db):
    out = find_similar_attempts(
        "no signals here", exclude_session_id=None,
        limit=10, db_path=populated_db["db"],
    )
    assert out == []


def test_format_cross_session_empty():
    assert "no relevant prior attempts" in format_cross_session([])


def test_format_cross_session_renders():
    examples = [
        CrossSessionExample(
            session_id=3, candidate_id=42, decision="keep", accepted=True,
            primary_delta=0.12, rationale="added vol_adjusted_momentum",
        ),
        CrossSessionExample(
            session_id=2, candidate_id=17, decision="discard", accepted=False,
            primary_delta=-0.31, rationale="too many signals",
        ),
    ]
    text = format_cross_session(examples)
    assert "KEEP" in text
    assert "+0.1200" in text
    assert "vol_adjusted_momentum" in text
    assert "s3" in text
    assert "s2" in text
