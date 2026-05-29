"""
Pareto dominance + frontier maintenance over multi-dim FitnessScore.

A Pareto-optimal selector is the right move whenever the metric you care about
isn't a single scalar. For GEPA over the prompt: a prompt with slightly lower
hit_rate but much better stability_score and lower turnover may be preferable
to one that scores higher on the single scalar.

Dominance rule:
  A dominates B iff for every Pareto dimension d:
    - if d is higher-is-better: A.d >= B.d
    - if d is lower-is-better:  A.d <= B.d
  AND there exists at least one d where A is strictly better than B.

Selection from the frontier:
  - 'first'      — first frontier entry (stable, useful for tests)
  - 'random'     — uniform random pick (exploration)
  - 'crowding'   — pick the entry with the most-spread neighbors (NSGA-II flavor)
  - 'scalarized' — weighted-sum collapse to single value, pick max (back-compat)

The Pareto layer is OPTIONAL — gepa/loop.py still defaults to single-best by
primary_metric. selector_mode='pareto' opts in.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

from researcher.gepa.fitness import FitnessScore


def dominates(a: FitnessScore, b: FitnessScore) -> bool:
    """True iff `a` Pareto-dominates `b` across all dimensions in pareto_components()."""
    a_comp = a.pareto_components()
    b_comp = b.pareto_components()
    strictly_better_in_one = False
    for dim, (a_val, higher_better) in a_comp.items():
        b_val, _ = b_comp[dim]
        if higher_better:
            if a_val < b_val:
                return False
            if a_val > b_val:
                strictly_better_in_one = True
        else:
            if a_val > b_val:
                return False
            if a_val < b_val:
                strictly_better_in_one = True
    return strictly_better_in_one


def frontier(scores: dict[int, FitnessScore]) -> list[int]:
    """
    Return ids of non-dominated entries (the Pareto frontier).
    Input: {id: FitnessScore}. Output: list of ids, sorted ascending.
    """
    ids = list(scores.keys())
    front: list[int] = []
    for i in ids:
        dominated = False
        for j in ids:
            if i == j:
                continue
            if dominates(scores[j], scores[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front)


def _scalarize(score: FitnessScore, weights: dict[str, float] | None = None) -> float:
    """Weighted-sum collapse for the 'scalarized' selector mode and tie-breaking."""
    w = weights or {
        "hit_rate": 3.0,
        "mean_kept_delta": 2.0,
        "terminal_metric": 1.0,
        "stability_score": 1.0,
        "max_drawdown": -1.0,         # negative weight = penalty
        "turnover_annualized": -0.1,  # mild penalty
    }
    return sum(w[k] * v for k, (v, _) in score.pareto_components().items() if k in w)


def _crowding_distances(front_ids: list[int], scores: dict[int, FitnessScore]) -> dict[int, float]:
    """
    NSGA-II crowding distance: for each frontier entry, sum normalized inter-neighbor
    distances across dimensions. Larger distance = more isolated = preserve (diversity).
    """
    if len(front_ids) <= 2:
        return {i: math.inf for i in front_ids}

    dist = {i: 0.0 for i in front_ids}
    dim_names = list(scores[front_ids[0]].pareto_components().keys())

    for dim in dim_names:
        vals = {i: scores[i].pareto_components()[dim][0] for i in front_ids}
        sorted_ids = sorted(front_ids, key=lambda i: vals[i])
        vmin, vmax = vals[sorted_ids[0]], vals[sorted_ids[-1]]
        if vmax - vmin == 0:
            continue
        dist[sorted_ids[0]] = math.inf
        dist[sorted_ids[-1]] = math.inf
        for k in range(1, len(sorted_ids) - 1):
            prev_v = vals[sorted_ids[k - 1]]
            next_v = vals[sorted_ids[k + 1]]
            dist[sorted_ids[k]] += (next_v - prev_v) / (vmax - vmin)

    return dist


def select_from_frontier(
    scores: dict[int, FitnessScore],
    mode: str = "scalarized",
    rng: random.Random | None = None,
    weights: dict[str, float] | None = None,
) -> int:
    """
    Pick one id from the Pareto frontier of `scores`. Returns the chosen id.

    Modes:
      'first'      — lowest id on the frontier (deterministic; for tests)
      'random'     — uniform pick from frontier
      'crowding'   — pick the entry with the largest crowding distance (most isolated)
      'scalarized' — pick the entry with the largest weighted-sum score (default)
    """
    if not scores:
        raise ValueError("Cannot select from empty scores")
    front = frontier(scores)
    if not front:
        # Shouldn't happen if scores is non-empty, but be defensive
        front = list(scores.keys())

    if mode == "first":
        return front[0]
    if mode == "random":
        return (rng or random).choice(front)
    if mode == "crowding":
        d = _crowding_distances(front, scores)
        # Tie-break by scalarized score for reproducibility
        return max(front, key=lambda i: (d[i], _scalarize(scores[i], weights)))
    if mode == "scalarized":
        return max(front, key=lambda i: _scalarize(scores[i], weights))
    raise ValueError(f"Unknown selector mode: {mode!r}")
