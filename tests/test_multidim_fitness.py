"""FitnessScore multi-dim extension + extraction tests."""

import json

import pytest

from researcher.gepa.fitness import FitnessScore, _score_from_session
from researcher.runs import (
    insert_candidate, insert_trial, record_decision, start_session,
)


def test_fitness_score_has_new_dims():
    f = FitnessScore(
        hit_rate=0.3, mean_kept_delta=0.1,
        n_kept=3, n_decided=10, n_inconclusive=2, n_crashes=0,
        terminal_metric=0.5, time_to_first_keep=2, wall_seconds=60.0,
        oos_is_sharpe_gap=-0.1, max_drawdown=0.25,
        turnover_annualized=3.5, stability_score=0.8,
    )
    d = f.to_dict()
    for key in ("oos_is_sharpe_gap", "max_drawdown", "turnover_annualized", "stability_score"):
        assert key in d


def test_pareto_components_marks_direction_correctly():
    f = FitnessScore(
        hit_rate=0.5, mean_kept_delta=0.1,
        n_kept=2, n_decided=10, n_inconclusive=0, n_crashes=0,
        terminal_metric=0.5, time_to_first_keep=1, wall_seconds=1.0,
        oos_is_sharpe_gap=0.0, max_drawdown=0.2,
        turnover_annualized=2.0, stability_score=0.5,
    )
    components = f.pareto_components()
    # higher-is-better
    assert components["hit_rate"][1] is True
    assert components["mean_kept_delta"][1] is True
    assert components["terminal_metric"][1] is True
    assert components["stability_score"][1] is True
    # lower-is-better
    assert components["max_drawdown"][1] is False
    assert components["turnover_annualized"][1] is False


def test_dataclass_defaults_keep_back_compat():
    """Old code constructing FitnessScore without new fields must still work."""
    f = FitnessScore(
        hit_rate=0.5, mean_kept_delta=0.1,
        n_kept=2, n_decided=10, n_inconclusive=0, n_crashes=0,
        terminal_metric=0.5, time_to_first_keep=1, wall_seconds=1.0,
    )
    assert f.oos_is_sharpe_gap == 0.0
    assert f.max_drawdown == 0.0
    assert f.turnover_annualized == 0.0
    assert f.stability_score == 0.0


def test_score_from_session_extracts_multidim(tmp_path, monkeypatch):
    """End-to-end: stand up a synthetic session in runs.db, verify dims are pulled correctly."""
    db = tmp_path / "runs.db"
    import researcher.runs
    import researcher.gepa.fitness
    monkeypatch.setattr(researcher.runs, "DB_PATH", db)
    monkeypatch.setattr(researcher.gepa.fitness, "runs", researcher.runs)

    sid = start_session("stub", {}, n_iterations=2, replicate_seeds=3, db_path=db)

    # Baseline (accepted, no delta)
    base_id = insert_candidate(sid, None, "(baseline)", "baseline", db_path=db)
    record_decision(base_id, "baseline", {"primary_delta": None}, accepted=True, db_path=db)
    for s in [1, 2, 3]:
        insert_trial(
            base_id, seed=s, status="ok", primary_metric=0.40,
            metrics={
                "in_sample": {"sharpe": 0.55, "max_drawdown": 0.15, "turnover": 1.5},
                "oos": {"sharpe": 0.40, "max_drawdown": 0.22, "turnover": 2.1},
            },
            params={}, log=None, db_path=db,
        )

    # An accepted candidate with very low stability (high stddev across seeds)
    cid = insert_candidate(sid, base_id, "diff", "tweaked weights", db_path=db)
    record_decision(cid, "keep", {"primary_delta": 0.20}, accepted=True, db_path=db)
    # Per-seed: 0.5, 0.6, 0.4 — mean 0.5, stddev ~0.1 -> stability ~ 1 - 0.1/0.6 ~ 0.83
    insert_trial(cid, 1, "ok", 0.5, {"in_sample": {"sharpe": 0.7, "max_drawdown": 0.10, "turnover": 1.0},
                                      "oos": {"sharpe": 0.5, "max_drawdown": 0.18, "turnover": 1.8}}, {}, None, db_path=db)
    insert_trial(cid, 2, "ok", 0.6, {"in_sample": {"sharpe": 0.7, "max_drawdown": 0.10, "turnover": 1.0},
                                      "oos": {"sharpe": 0.6, "max_drawdown": 0.15, "turnover": 1.7}}, {}, None, db_path=db)
    insert_trial(cid, 3, "ok", 0.4, {"in_sample": {"sharpe": 0.7, "max_drawdown": 0.10, "turnover": 1.0},
                                      "oos": {"sharpe": 0.4, "max_drawdown": 0.20, "turnover": 1.9}}, {}, None, db_path=db)

    f = _score_from_session(sid, wall_seconds=120.0)
    assert f.hit_rate == 1.0
    assert f.terminal_metric == pytest.approx(0.5)
    # OOS - IS gap: 0.5 - 0.7 = -0.2 (overfitting)
    assert f.oos_is_sharpe_gap == pytest.approx(-0.2, abs=0.001)
    # Max drawdown: mean of [0.18, 0.15, 0.20] = ~0.177
    assert f.max_drawdown == pytest.approx(0.177, abs=0.001)
    # Turnover: mean of [1.8, 1.7, 1.9] = ~1.8
    assert f.turnover_annualized == pytest.approx(1.8, abs=0.001)
    # Stability: 1 - stdev(0.5,0.6,0.4)/(0.5+0.1) — stdev=0.1, so 1 - 0.1/0.6 = 0.833
    assert 0.7 < f.stability_score < 0.9
