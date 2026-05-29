"""
Paired-CI decision gate.

Compares a candidate strategy against its parent across N replicate seeds, where
each seed corresponds to a fixed sub-universe of the backtest universe. Same seed
on parent and candidate = same backtest scenario = directly comparable.

Decision is on the domain's primary metric (higher-better; e.g. OOS Sharpe for finance,
−val_bpb for shakespeare). Three outcomes:
  - 'keep'         : paired-mean-delta CI strictly above 0  (candidate confidently better)
  - 'discard'      : either any candidate trial crashed, or CI strictly below 0
  - 'inconclusive' : CI straddles 0 (can't tell within noise budget)

Only 'keep' sets accepted=True. 'inconclusive' is logged as a distinct outcome so
we can measure how often the proposer wastes replicates on indecisive mutations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

from scipy import stats


@dataclass(frozen=True)
class TrialSummary:
    seed: int | None
    status: str           # 'ok' | 'crashed' | 'timeout'
    primary_metric: float | None


@dataclass(frozen=True)
class Decision:
    action: str           # 'keep' | 'discard' | 'inconclusive'
    reason: str           # human-readable
    primary_delta: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    n_pairs: int

    def to_dict(self) -> dict:
        return asdict(self)


def _by_seed(trials: Iterable[TrialSummary]) -> dict[int | None, TrialSummary]:
    return {t.seed: t for t in trials}


def decide(
    parent_trials: list[TrialSummary],
    candidate_trials: list[TrialSummary],
    alpha: float = 0.05,
) -> Decision:
    """
    Gate the candidate against the parent. Assumes seeds align across the two lists
    (each seed must appear in both for that pair to be usable).
    """
    # Any candidate crash is an immediate discard. Buggy strategies are not kept.
    crashed = [t for t in candidate_trials if t.status != "ok"]
    if crashed:
        seeds = [t.seed for t in crashed]
        return Decision(
            action="discard", reason=f"candidate crashed on seed(s) {seeds}",
            primary_delta=None, ci_low=None, ci_high=None, p_value=None, n_pairs=0,
        )

    parent_by_seed = _by_seed(parent_trials)
    cand_by_seed = _by_seed(candidate_trials)

    pairs: list[tuple[float, float]] = []
    for seed, cand in cand_by_seed.items():
        par = parent_by_seed.get(seed)
        if par is None or par.status != "ok":
            continue
        if (
            cand.primary_metric is None or par.primary_metric is None
            or not math.isfinite(cand.primary_metric) or not math.isfinite(par.primary_metric)
        ):
            continue
        pairs.append((par.primary_metric, cand.primary_metric))

    n = len(pairs)
    if n < 2:
        return Decision(
            action="inconclusive", reason=f"only {n} usable seed pair(s); need >=2 for CI",
            primary_delta=None, ci_low=None, ci_high=None, p_value=None, n_pairs=n,
        )

    deltas = [c - p for p, c in pairs]
    mean_delta = sum(deltas) / n

    var = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        # All deltas identical. Use sign directly; CI is degenerate.
        if mean_delta > 0:
            action, reason = "keep", "all replicates strictly better (zero variance)"
        elif mean_delta < 0:
            action, reason = "discard", "all replicates strictly worse (zero variance)"
        else:
            action, reason = "inconclusive", "zero delta across all replicates"
        return Decision(
            action=action, reason=reason, primary_delta=mean_delta,
            ci_low=mean_delta, ci_high=mean_delta, p_value=None, n_pairs=n,
        )

    se = sd / math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    half = t_crit * se
    ci_low, ci_high = mean_delta - half, mean_delta + half

    # Two-sided paired t-test p-value for completeness
    t_stat = mean_delta / se
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))

    if ci_low > 0:
        action, reason = "keep", f"primary-metric delta CI=[{ci_low:.4f}, {ci_high:.4f}] strictly positive"
    elif ci_high < 0:
        action, reason = "discard", f"primary-metric delta CI=[{ci_low:.4f}, {ci_high:.4f}] strictly negative"
    else:
        action, reason = "inconclusive", f"primary-metric delta CI=[{ci_low:.4f}, {ci_high:.4f}] straddles 0"

    return Decision(
        action=action, reason=reason, primary_delta=mean_delta,
        ci_low=ci_low, ci_high=ci_high, p_value=p_value, n_pairs=n,
    )
