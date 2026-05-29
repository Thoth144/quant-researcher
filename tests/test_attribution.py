"""Per-signal attribution tests."""

from unittest.mock import patch

import pytest

from researcher.attribution import (
    SignalAttribution, attribute_candidate, format_attribution, get_attribution,
)


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    db = tmp_path / "runs.db"
    import researcher.runs
    monkeypatch.setattr(researcher.runs, "DB_PATH", db)
    # Seed a candidate so foreign-key references work
    from researcher.runs import insert_candidate, start_session
    sid = start_session("test", {}, n_iterations=1, replicate_seeds=1, db_path=db)
    cid = insert_candidate(sid, None, "diff", "test candidate", db_path=db)
    return {"db": db, "session_id": sid, "candidate_id": cid}


def test_attribute_candidate_writes_one_row_per_signal(populated_db):
    """Mocked subprocess; verify SignalAttribution objects + DB rows match."""
    fake_results = iter([
        ("ok", 0.45),   # momentum_12_1 standalone
        ("ok", -0.10),  # reversion_5d standalone (a drag)
        ("ok", 0.20),   # lowvol_20d standalone
    ])

    def fake_run(*args, **kwargs):
        return next(fake_results)

    with patch("researcher.attribution._run_attribution_seed", side_effect=fake_run):
        attrs = attribute_candidate(
            candidate_id=populated_db["candidate_id"],
            enabled_signals=["momentum_12_1", "reversion_5d", "lowvol_20d"],
            weights={"momentum_12_1": 1.0, "reversion_5d": 0.5, "lowvol_20d": 0.3},
        )

    assert len(attrs) == 3
    by_name = {a.signal_name: a for a in attrs}
    assert by_name["momentum_12_1"].standalone_oos_sharpe == pytest.approx(0.45)
    assert by_name["reversion_5d"].standalone_oos_sharpe == pytest.approx(-0.10)
    assert by_name["reversion_5d"].weight_in_candidate == pytest.approx(0.5)

    # Verify DB persistence
    retrieved = get_attribution(populated_db["candidate_id"])
    assert len(retrieved) == 3


def test_get_attribution_orders_by_standalone_descending(populated_db):
    fake_results = iter([("ok", 0.1), ("ok", 0.7), ("ok", 0.3)])

    def fake_run(*args, **kwargs):
        return next(fake_results)

    with patch("researcher.attribution._run_attribution_seed", side_effect=fake_run):
        attribute_candidate(
            candidate_id=populated_db["candidate_id"],
            enabled_signals=["a", "b", "c"],
            weights={"a": 0.3, "b": 0.5, "c": 0.2},
        )

    retrieved = get_attribution(populated_db["candidate_id"])
    assert [r.signal_name for r in retrieved] == ["b", "c", "a"]


def test_attribute_handles_failed_signal(populated_db):
    """One subprocess failure shouldn't break the whole attribution."""
    fake_results = iter([("ok", 0.3), ("crashed", None), ("ok", 0.5)])

    def fake_run(*args, **kwargs):
        return next(fake_results)

    with patch("researcher.attribution._run_attribution_seed", side_effect=fake_run):
        attrs = attribute_candidate(
            candidate_id=populated_db["candidate_id"],
            enabled_signals=["a", "b", "c"],
            weights={"a": 0.5, "b": 0.5, "c": 0.5},
        )

    by_name = {a.signal_name: a for a in attrs}
    assert by_name["b"].standalone_oos_sharpe is None


def test_format_attribution_flags_drag_signals():
    attrs = [
        SignalAttribution("momentum_12_1", 0.45, 1.0),
        SignalAttribution("reversion_5d", -0.10, 0.5),     # weighted but negative → drag
        SignalAttribution("lowvol_20d", 0.20, 0.0),        # not enabled but evaluated
    ]
    text = format_attribution(attrs)
    assert "momentum_12_1" in text
    assert "+0.4500" in text
    assert "drag" in text                                   # flagged
    assert "not enabled" in text                            # flagged


def test_format_attribution_empty():
    assert "no per-signal" in format_attribution([])
