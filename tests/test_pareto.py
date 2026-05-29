"""Pareto dominance + frontier + selector tests."""

import pytest

from researcher.gepa.fitness import FitnessScore
from researcher.gepa.pareto import dominates, frontier, select_from_frontier


def F(hit_rate=0.5, mean_kept_delta=0.1, terminal_metric=0.5,
      stability_score=0.5, max_drawdown=0.2, turnover_annualized=2.0,
      **kwargs):
    """Convenience constructor with required-but-uninteresting fields defaulted."""
    return FitnessScore(
        hit_rate=hit_rate, mean_kept_delta=mean_kept_delta,
        n_kept=5, n_decided=10, n_inconclusive=0, n_crashes=0,
        terminal_metric=terminal_metric, time_to_first_keep=3, wall_seconds=60.0,
        stability_score=stability_score, max_drawdown=max_drawdown,
        turnover_annualized=turnover_annualized, **kwargs,
    )


def test_dominates_strictly_better_in_all_dims():
    a = F(hit_rate=0.6, terminal_metric=0.8, stability_score=0.9,
          max_drawdown=0.1, turnover_annualized=1.0, mean_kept_delta=0.2)
    b = F(hit_rate=0.3, terminal_metric=0.2, stability_score=0.5,
          max_drawdown=0.3, turnover_annualized=3.0, mean_kept_delta=0.05)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_no_dominance_when_one_dim_worse():
    # a is better on hit_rate, b is better on max_drawdown — neither dominates
    a = F(hit_rate=0.7, max_drawdown=0.3)
    b = F(hit_rate=0.5, max_drawdown=0.1)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_no_self_dominance():
    # Equal scores: not strictly better in any dim → no dominance
    a = F(hit_rate=0.5)
    b = F(hit_rate=0.5)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_frontier_single_dominant_winner():
    scores = {
        1: F(hit_rate=0.9, terminal_metric=0.9, stability_score=0.9,
             max_drawdown=0.05, turnover_annualized=0.5, mean_kept_delta=0.3),
        2: F(hit_rate=0.5, terminal_metric=0.5, stability_score=0.5,
             max_drawdown=0.2, turnover_annualized=2.0, mean_kept_delta=0.1),
        3: F(hit_rate=0.3, terminal_metric=0.3, stability_score=0.3,
             max_drawdown=0.3, turnover_annualized=3.0, mean_kept_delta=0.05),
    }
    assert frontier(scores) == [1]


def test_frontier_multiple_non_dominated():
    # Each is best at one dimension → all on the frontier
    scores = {
        1: F(hit_rate=0.9, max_drawdown=0.3, turnover_annualized=3.0),  # best hit_rate
        2: F(hit_rate=0.5, max_drawdown=0.05, turnover_annualized=3.0), # best max_dd
        3: F(hit_rate=0.5, max_drawdown=0.3, turnover_annualized=0.5),  # best turnover
    }
    assert frontier(scores) == [1, 2, 3]


def test_select_first_is_deterministic():
    scores = {
        7: F(hit_rate=0.5, max_drawdown=0.05),
        3: F(hit_rate=0.9, max_drawdown=0.3),
    }
    # Both on frontier (each best at one dim); select_first returns lowest id
    assert select_from_frontier(scores, mode="first") == 3


def test_select_scalarized_uses_weights():
    scores = {
        1: F(hit_rate=0.9, max_drawdown=0.3),  # high hit_rate, high DD
        2: F(hit_rate=0.5, max_drawdown=0.05), # lower hit_rate, much better DD
    }
    # Both on frontier. With default weights, hit_rate weight (3.0) >> max_drawdown weight (-1)
    # so candidate 1 wins.
    assert select_from_frontier(scores, mode="scalarized") == 1
    # With heavy max_drawdown penalty, candidate 2 wins
    heavy_dd_penalty = {"hit_rate": 1.0, "max_drawdown": -10.0}
    assert select_from_frontier(scores, mode="scalarized", weights=heavy_dd_penalty) == 2


def test_select_random_returns_frontier_member():
    import random
    scores = {
        1: F(hit_rate=0.9, max_drawdown=0.3),
        2: F(hit_rate=0.5, max_drawdown=0.05),
        3: F(hit_rate=0.3, max_drawdown=0.4),  # dominated by 1
    }
    rng = random.Random(0)
    picked = select_from_frontier(scores, mode="random", rng=rng)
    assert picked in {1, 2}  # 3 is dominated, never picked


def test_select_unknown_mode_raises():
    scores = {1: F()}
    with pytest.raises(ValueError, match="Unknown selector mode"):
        select_from_frontier(scores, mode="moonshot")


def test_empty_scores_raises():
    with pytest.raises(ValueError, match="empty"):
        select_from_frontier({}, mode="first")
