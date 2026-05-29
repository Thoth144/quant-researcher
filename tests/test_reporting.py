"""Reporting queries + HTML rendering tests."""

import lxml.etree as ET
import pytest

from researcher.reporting.queries import (
    candidates_for_session, leaderboard, lineage_tree, list_sessions,
    trials_for_candidate,
)
from researcher.runs import (
    insert_candidate, insert_trial, record_decision, start_session,
)


@pytest.fixture
def populated(tmp_path, monkeypatch):
    db = tmp_path / "runs.db"
    import researcher.runs
    monkeypatch.setattr(researcher.runs, "DB_PATH", db)

    # Session 1: a baseline + 1 kept mutation + 1 discarded
    s1 = start_session("stub", {}, n_iterations=5, replicate_seeds=3, db_path=db)
    base = insert_candidate(s1, None, "(baseline)", "baseline", db_path=db)
    record_decision(base, "baseline", {"primary_delta": None}, accepted=True, db_path=db)
    for seed in (1, 2, 3):
        insert_trial(base, seed, "ok", 0.20, {"oos": {"sharpe": 0.20}}, {}, None, db_path=db)

    kept = insert_candidate(s1, base, "diff A", "added vol signal", db_path=db)
    record_decision(kept, "keep", {"primary_delta": 0.15}, accepted=True, db_path=db)
    for seed in (1, 2, 3):
        insert_trial(kept, seed, "ok", 0.35, {"oos": {"sharpe": 0.35}}, {}, None, db_path=db)

    disc = insert_candidate(s1, kept, "diff B", "too many signals", db_path=db)
    record_decision(disc, "discard", {"primary_delta": -0.25}, accepted=False, db_path=db)
    for seed in (1, 2, 3):
        insert_trial(disc, seed, "ok", 0.10, {"oos": {"sharpe": 0.10}}, {}, None, db_path=db)

    return {"db": db, "s1": s1, "base": base, "kept": kept, "disc": disc}


def test_list_sessions(populated):
    sessions = list_sessions(db_path=populated["db"])
    assert len(sessions) == 1
    s = sessions[0]
    assert s.n_candidates == 3
    assert s.n_decided == 3
    assert s.n_kept == 2  # baseline + kept mutation
    assert s.n_discard == 1
    assert s.hit_rate == pytest.approx(2 / 3)


def test_candidates_for_session_includes_metrics(populated):
    cands = candidates_for_session(populated["s1"], db_path=populated["db"])
    assert len(cands) == 3
    by_id = {c.id: c for c in cands}
    assert by_id[populated["base"]].mean_metric == pytest.approx(0.20)
    assert by_id[populated["kept"]].mean_metric == pytest.approx(0.35)
    assert by_id[populated["kept"]].accepted is True
    assert by_id[populated["disc"]].accepted is False
    assert by_id[populated["disc"]].primary_delta == pytest.approx(-0.25)


def test_leaderboard_sorts_descending(populated):
    lb = leaderboard(populated["s1"], db_path=populated["db"])
    assert [c.id for c in lb] == [populated["kept"], populated["base"]]


def test_lineage_tree_structure(populated):
    tree = lineage_tree(populated["s1"], db_path=populated["db"])
    assert tree.get(None) == [populated["base"]]
    assert tree.get(populated["base"]) == [populated["kept"]]
    assert tree.get(populated["kept"]) == [populated["disc"]]


def test_trials_for_candidate(populated):
    trials = trials_for_candidate(populated["kept"], db_path=populated["db"])
    assert len(trials) == 3
    assert all(t.status == "ok" for t in trials)
    assert all(t.primary_metric == pytest.approx(0.35) for t in trials)


def test_html_report_parses_as_valid_html(populated, tmp_path):
    from scripts.report import render_html
    html_text = render_html(populated["s1"])
    # Parse with lxml HTML parser (forgiving)
    parser = ET.HTMLParser()
    doc = ET.fromstring(html_text.encode(), parser)
    assert doc is not None
    # Smoke-check key sections rendered
    assert "Research session" in html_text
    assert "added vol signal" in html_text       # rationale rendered
    assert "diff A" in html_text                  # diff rendered
    assert "Δ=+0.1500" in html_text or "+0.1500" in html_text  # kept Δ
    assert "BASELINE" in html_text
    assert "KEEP" in html_text
    assert "DISCARD" in html_text
