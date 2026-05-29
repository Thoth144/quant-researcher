"""Correctness tests for the paired-CI decision gate."""

import pytest

from researcher.decide import TrialSummary, decide


def T(seed: int, metric: float, status: str = "ok") -> TrialSummary:
    return TrialSummary(seed=seed, status=status, primary_metric=metric)


def test_clear_win_all_seeds_better():
    parent = [T(1, 0.8), T(2, 0.9), T(3, 1.0)]
    cand = [T(1, 1.0), T(2, 1.1), T(3, 1.2)]
    d = decide(parent, cand)
    assert d.action == "keep"
    assert d.primary_delta > 0
    assert d.ci_low > 0


def test_clear_loss_all_seeds_worse():
    parent = [T(1, 1.0), T(2, 1.1), T(3, 1.2)]
    cand = [T(1, 0.5), T(2, 0.4), T(3, 0.6)]
    d = decide(parent, cand)
    assert d.action == "discard"
    assert d.primary_delta < 0
    assert d.ci_high < 0


def test_noise_straddles_zero():
    # Mixed signs, small magnitudes — should be inconclusive
    parent = [T(1, 1.00), T(2, 1.00), T(3, 1.00)]
    cand = [T(1, 1.02), T(2, 0.98), T(3, 1.01)]
    d = decide(parent, cand)
    assert d.action == "inconclusive"
    assert d.ci_low < 0 < d.ci_high


def test_candidate_crash_is_immediate_discard():
    parent = [T(1, 1.0), T(2, 1.1), T(3, 1.2)]
    cand = [T(1, 1.5), T(2, None, status="crashed"), T(3, 1.6)]
    d = decide(parent, cand)
    assert d.action == "discard"
    assert "crashed" in d.reason


def test_single_seed_is_inconclusive():
    # Not enough data to compute a CI
    parent = [T(1, 0.9)]
    cand = [T(1, 1.5)]
    d = decide(parent, cand)
    assert d.action == "inconclusive"
    assert d.n_pairs == 1


def test_zero_variance_strict_win():
    # All deltas identical and positive — should be 'keep' via sign branch
    parent = [T(1, 1.0), T(2, 1.0), T(3, 1.0)]
    cand = [T(1, 1.1), T(2, 1.1), T(3, 1.1)]
    d = decide(parent, cand)
    assert d.action == "keep"
    assert d.primary_delta == pytest.approx(0.1)


def test_drops_parent_crash_seeds():
    # Seed 2 parent crashed; should still decide on seeds 1 and 3
    parent = [T(1, 1.0), T(2, None, status="crashed"), T(3, 1.0)]
    cand = [T(1, 1.5), T(2, 1.6), T(3, 1.5)]
    d = decide(parent, cand)
    assert d.n_pairs == 2  # only seeds 1 and 3 usable
    assert d.action in ("keep", "inconclusive")  # depends on n=2 CI width
