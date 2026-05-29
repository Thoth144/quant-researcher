"""
Paired-CI prompt-vs-prompt comparison.

Decision grain: prompts (not strategy candidates as in researcher/decide.py).
Pairing axis: evaluator_seed. For each seed in `seeds`, the same RNG draws
apply to both prompts via MockProposer(prompt_X, evaluator_seed=s) — so
seed-paired fitness deltas isolate the prompt's effect from run noise.

Output mirrors researcher/decide.py shape so a future Pareto layer can reason
across both grains uniformly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

from scipy import stats

from researcher.gepa.fitness import Evaluator, FitnessScore
from researcher.gepa.prompts import Prompt


@dataclass(frozen=True)
class PromptComparison:
    action: str               # 'a_better' | 'b_better' | 'inconclusive'
    reason: str
    primary_delta: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    n_pairs: int
    a_scores: list[float]     # per-seed primary metric for A
    b_scores: list[float]     # per-seed primary metric for B

    def to_dict(self) -> dict:
        return asdict(self)


def compare_prompts(
    a: Prompt, b: Prompt, evaluator: Evaluator,
    seeds: list[int], alpha: float = 0.05,
    primary_metric: str = "hit_rate",
) -> PromptComparison:
    """
    Evaluate both prompts across the same `seeds`, then run a paired t-test on
    the primary-metric delta (b - a).

    'a_better' if CI for (b - a) is strictly < 0.
    'b_better' if CI for (b - a) is strictly > 0.
    'inconclusive' otherwise.
    """
    a_scores: list[float] = []
    b_scores: list[float] = []
    for s in seeds:
        fa = evaluator.evaluate(a, evaluator_seed=s)
        fb = evaluator.evaluate(b, evaluator_seed=s)
        ma = getattr(fa, primary_metric)
        mb = getattr(fb, primary_metric)
        if not (math.isfinite(ma) and math.isfinite(mb)):
            continue
        a_scores.append(ma)
        b_scores.append(mb)

    return _decide_from_paired(a_scores, b_scores, alpha)


def compare_prompts_from_cached_scores(
    a_scores: Iterable[float], b_scores: Iterable[float], alpha: float = 0.05,
) -> PromptComparison:
    """Same decision logic, but skipping evaluation (use cached per-seed metrics)."""
    a_scores = list(a_scores)
    b_scores = list(b_scores)
    if len(a_scores) != len(b_scores):
        raise ValueError(f"a/b score lists must be equal length, got {len(a_scores)} vs {len(b_scores)}")
    return _decide_from_paired(a_scores, b_scores, alpha)


def _decide_from_paired(
    a_scores: list[float], b_scores: list[float], alpha: float,
) -> PromptComparison:
    n = len(a_scores)
    if n < 2:
        return PromptComparison(
            action="inconclusive", reason=f"only {n} usable seed pairs; need >=2",
            primary_delta=None, ci_low=None, ci_high=None, p_value=None,
            n_pairs=n, a_scores=a_scores, b_scores=b_scores,
        )

    deltas = [bs - as_ for as_, bs in zip(a_scores, b_scores)]
    mean_delta = sum(deltas) / n

    var = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        if mean_delta > 0:
            action, reason = "b_better", "all replicates strictly favor B (zero variance)"
        elif mean_delta < 0:
            action, reason = "a_better", "all replicates strictly favor A (zero variance)"
        else:
            action, reason = "inconclusive", "zero delta across all replicates"
        return PromptComparison(
            action=action, reason=reason, primary_delta=mean_delta,
            ci_low=mean_delta, ci_high=mean_delta, p_value=None,
            n_pairs=n, a_scores=a_scores, b_scores=b_scores,
        )

    se = sd / math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    half = t_crit * se
    ci_low, ci_high = mean_delta - half, mean_delta + half
    t_stat = mean_delta / se
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))

    if ci_low > 0:
        action, reason = "b_better", f"Δ(B-A) CI=[{ci_low:.4f}, {ci_high:.4f}] strictly positive"
    elif ci_high < 0:
        action, reason = "a_better", f"Δ(B-A) CI=[{ci_low:.4f}, {ci_high:.4f}] strictly negative"
    else:
        action, reason = "inconclusive", f"Δ(B-A) CI=[{ci_low:.4f}, {ci_high:.4f}] straddles 0"

    return PromptComparison(
        action=action, reason=reason, primary_delta=mean_delta,
        ci_low=ci_low, ci_high=ci_high, p_value=p_value,
        n_pairs=n, a_scores=a_scores, b_scores=b_scores,
    )
