"""Adaptive replication (curriculum) tests."""

from unittest.mock import patch

import pytest

from researcher.decide import TrialSummary
from researcher.runner import CandidateRun, _run_one_seed, run_candidate


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    db = tmp_path / "runs.db"
    import researcher.runs
    monkeypatch.setattr(researcher.runs, "DB_PATH", db)
    return db


def _patch_run_seed(metrics_by_seed: dict[int, float | None]):
    """Patch _run_one_seed to return synthetic metrics per seed (no real backtests)."""
    def fake(seed, timeout, domain=None):
        m = metrics_by_seed.get(seed)
        if m is None:
            return "crashed", None, None, None, "synthetic crash"
        return "ok", m, {"oos": {"sharpe": m}}, {"depth": 4}, "log"
    return patch("researcher.runner._run_one_seed", side_effect=fake)


def _make_test_domain(tmp_path):
    """Construct a Domain that writes to tmp_path so tests don't touch real strategy files."""
    import dataclasses
    from researcher.domain import FINANCE
    fake_strategy = tmp_path / "strategy.py"
    fake_backup = tmp_path / "strategy.py.parent_backup"
    fake_strategy.write_text("# stub")
    return dataclasses.replace(
        FINANCE, strategy_file=fake_strategy, parent_backup_file=fake_backup,
    )


def test_early_screen_triggers_on_clearly_worse_seed1(populated_db, monkeypatch, tmp_path):
    test_domain = _make_test_domain(tmp_path)
    from researcher.runs import start_session
    sid = start_session("test", {}, n_iterations=1, replicate_seeds=3, db_path=populated_db)

    parent_trials = [
        TrialSummary(seed=1, status="ok", primary_metric=0.50),
        TrialSummary(seed=2, status="ok", primary_metric=0.45),
        TrialSummary(seed=3, status="ok", primary_metric=0.55),
    ]
    # Candidate seed-1 = 0.10 (gap = -0.40 < -0.20 threshold) → early-screen triggers
    with _patch_run_seed({1: 0.10, 2: 0.40, 3: 0.50}):
        result = run_candidate(
            session_id=sid, parent_id=None, parent_src="parent", candidate_src="child",
            rationale="test", seeds=[1, 2, 3],
            parent_trials=parent_trials, screen_gap=-0.20, domain=test_domain,
        )

    assert result.early_decision is not None
    assert result.early_decision.action == "discard"
    assert "early-screen" in result.early_decision.reason
    assert len(result.trials) == 3
    # Seed 1 ran, seeds 2 & 3 were skipped
    assert result.trials[0].status == "ok"
    assert result.trials[1].status == "skipped"
    assert result.trials[2].status == "skipped"


def test_early_screen_does_not_trigger_when_close(populated_db, monkeypatch, tmp_path):
    test_domain = _make_test_domain(tmp_path)
    from researcher.runs import start_session
    sid = start_session("test", {}, n_iterations=1, replicate_seeds=3, db_path=populated_db)

    parent_trials = [
        TrialSummary(seed=1, status="ok", primary_metric=0.50),
        TrialSummary(seed=2, status="ok", primary_metric=0.45),
        TrialSummary(seed=3, status="ok", primary_metric=0.55),
    ]
    # Candidate seed-1 = 0.40 (gap = -0.10, NOT below -0.20 threshold) → all seeds run
    with _patch_run_seed({1: 0.40, 2: 0.50, 3: 0.45}):
        result = run_candidate(
            session_id=sid, parent_id=None, parent_src="parent", candidate_src="child",
            rationale="test", seeds=[1, 2, 3],
            parent_trials=parent_trials, screen_gap=-0.20, domain=test_domain,
        )

    assert result.early_decision is None
    assert all(t.status == "ok" for t in result.trials)
    assert len(result.trials) == 3


def test_early_screen_skipped_without_parent_trials(populated_db, monkeypatch, tmp_path):
    test_domain = _make_test_domain(tmp_path)
    from researcher.runs import start_session
    sid = start_session("test", {}, n_iterations=1, replicate_seeds=3, db_path=populated_db)

    # No parent_trials provided → curriculum disabled regardless of seed-1 metric
    with _patch_run_seed({1: -2.0, 2: -2.0, 3: -2.0}):
        result = run_candidate(
            session_id=sid, parent_id=None, parent_src="parent", candidate_src="child",
            rationale="test", seeds=[1, 2, 3],
            parent_trials=None, domain=test_domain,
        )

    assert result.early_decision is None
    assert len(result.trials) == 3
    assert all(t.status == "ok" for t in result.trials)


def test_early_screen_handles_crashed_seed1(populated_db, monkeypatch, tmp_path):
    test_domain = _make_test_domain(tmp_path)
    from researcher.runs import start_session
    sid = start_session("test", {}, n_iterations=1, replicate_seeds=3, db_path=populated_db)

    parent_trials = [
        TrialSummary(seed=1, status="ok", primary_metric=0.50),
        TrialSummary(seed=2, status="ok", primary_metric=0.45),
        TrialSummary(seed=3, status="ok", primary_metric=0.55),
    ]
    # Seed-1 crashes → curriculum can't screen, runs remaining seeds (decide.py handles crash)
    with _patch_run_seed({1: None, 2: 0.40, 3: 0.50}):
        result = run_candidate(
            session_id=sid, parent_id=None, parent_src="parent", candidate_src="child",
            rationale="test", seeds=[1, 2, 3],
            parent_trials=parent_trials, screen_gap=-0.20, domain=test_domain,
        )

    assert result.early_decision is None
    assert result.trials[0].status == "crashed"
    assert result.trials[1].status == "ok"
    assert result.trials[2].status == "ok"
