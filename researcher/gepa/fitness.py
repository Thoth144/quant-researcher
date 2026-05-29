"""
FitnessEvaluator — measures how well a prompt performs as a research-loop proposer.

Two evaluators:
  - FitnessEvaluator (mode='real'): runs the actual research loop using MockProposer
    keyed by the prompt. Real backtests, real runs.db rows, real wall time.
  - SyntheticFitnessEvaluator: computes deterministic fitness from prompt features
    only. Skips backtests. For unit tests and fast GEPA-loop smoke runs.

Both return the same FitnessScore shape so downstream comparison code is unified.

Real-mode discipline:
  - Every evaluation snapshots strategy.py before running, restores after.
    This guarantees each prompt is evaluated from the same parent baseline,
    independent of what prior evaluations did to strategy.py.
  - Sequential only. Concurrent evaluations would race on strategy.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol

from researcher import runs
from researcher.gepa._mock_proposer import MockProposer
from researcher.gepa.prompts import Prompt, PromptRegistry
from researcher.loop import run_loop

REPO_ROOT = Path(__file__).parent.parent.parent
STRATEGY_FILE = REPO_ROOT / "strategy.py"


@dataclass(frozen=True)
class FitnessScore:
    """
    Multi-dimensional fitness — Pareto-aware. Primary metric is hit_rate.
    All other fields enable Pareto selection and stability gating.
    """
    hit_rate: float                   # n_kept / n_decided                       PRIMARY
    mean_kept_delta: float            # mean OOS-Sharpe Δ across kept candidates
    n_kept: int
    n_decided: int
    n_inconclusive: int
    n_crashes: int
    terminal_metric: float            # parent's mean OOS Sharpe at session end
    time_to_first_keep: int | None    # iteration index of first 'keep', None if never
    wall_seconds: float

    # Stability / overfitting dimensions (improvement #8)
    oos_is_sharpe_gap: float = 0.0    # terminal_oos_sharpe - terminal_in_sample_sharpe
                                      #   negative = OOS worse than IS (overfitting risk)
                                      #   ~0 or positive = strategy generalizes
    max_drawdown: float = 0.0         # of terminal accepted strategy's OOS equity curve
                                      #   reported as positive fraction (e.g. 0.32 = -32%)
    turnover_annualized: float = 0.0  # of terminal accepted strategy's OOS positions
    stability_score: float = 0.0      # 1 - stddev(per-seed OOS Sharpe)/(|mean| + eps)
                                      #   1.0 = perfectly consistent across seeds, 0 = noisy

    def to_dict(self) -> dict:
        return asdict(self)

    def pareto_components(self) -> dict[str, tuple[float, bool]]:
        """
        Return the (value, higher_is_better) for each Pareto dimension.
        Used by gepa.pareto for dominance computation.
        """
        return {
            "hit_rate":          (self.hit_rate, True),
            "mean_kept_delta":   (self.mean_kept_delta, True),
            "terminal_metric":   (self.terminal_metric, True),
            "stability_score":   (self.stability_score, True),
            "max_drawdown":      (self.max_drawdown, False),   # lower DD better
            "turnover_annualized": (self.turnover_annualized, False),  # lower turnover better
        }


class Evaluator(Protocol):
    def evaluate(self, prompt: Prompt, evaluator_seed: int) -> FitnessScore: ...


# --------------------------- real evaluator ---------------------------

class FitnessEvaluator:
    """Runs the actual research loop with a MockProposer derived from the prompt."""

    def __init__(
        self,
        seed_strategy_src: str,
        registry: PromptRegistry,
        n_iterations: int = 10,
        backtest_seeds: list[int | None] | None = None,
        timeout_sec: int = 120,
    ):
        self.seed_strategy_src = seed_strategy_src
        self.registry = registry
        self.n_iterations = n_iterations
        self.backtest_seeds = backtest_seeds if backtest_seeds is not None else [1, 2, 3]
        self.timeout_sec = timeout_sec

    def evaluate(self, prompt: Prompt, evaluator_seed: int) -> FitnessScore:
        snapshot = STRATEGY_FILE.read_text()
        STRATEGY_FILE.write_text(self.seed_strategy_src)
        t0 = time.time()
        try:
            proposer = MockProposer(prompt=prompt, evaluator_seed=evaluator_seed)
            session_id = run_loop(
                proposer=proposer,
                n_iterations=self.n_iterations,
                seeds=self.backtest_seeds,
                session_notes=f"gepa:fitness prompt={prompt.content_hash[:8]} eval_seed={evaluator_seed}",
                timeout_sec=self.timeout_sec,
            )
        finally:
            STRATEGY_FILE.write_text(snapshot)

        wall = time.time() - t0
        score = _score_from_session(session_id, wall_seconds=wall)
        if prompt.id is not None:
            self.registry.attach_to_session(session_id, prompt.id)
            self.registry.update_fitness(prompt.id, score.to_dict())
        return score


def _score_from_session(session_id: int, wall_seconds: float) -> FitnessScore:
    """Aggregate one session's candidates+trials into a single FitnessScore."""
    with runs.connect() as conn:
        cand_rows = conn.execute(
            "SELECT id, decision, decision_payload_json, accepted, created_at "
            "FROM candidates WHERE session_id = ? AND decision IS NOT NULL "
            "ORDER BY id",
            (session_id,),
        ).fetchall()
        trial_rows = conn.execute(
            "SELECT t.candidate_id, t.primary_metric, t.status, t.metrics_json "
            "FROM trials t JOIN candidates c ON c.id = t.candidate_id "
            "WHERE c.session_id = ? AND t.status = 'ok'",
            (session_id,),
        ).fetchall()

    decisions = [r for r in cand_rows if r["decision"] != "baseline"]
    n_decided = len(decisions)
    n_kept = sum(1 for r in decisions if r["accepted"] == 1)
    n_inconclusive = sum(1 for r in decisions if r["decision"] == "inconclusive")
    n_crashes = sum(
        1 for r in decisions
        if r["decision"] == "discard" and r["decision_payload_json"]
        and "crashed" in (json.loads(r["decision_payload_json"]).get("reason") or "")
    )

    hit_rate = (n_kept / n_decided) if n_decided else 0.0

    kept_deltas = []
    for r in decisions:
        if r["accepted"] == 1 and r["decision_payload_json"]:
            payload = json.loads(r["decision_payload_json"])
            delta = payload.get("primary_delta")
            if delta is not None and math.isfinite(delta):
                kept_deltas.append(delta)
    mean_kept_delta = statistics.mean(kept_deltas) if kept_deltas else 0.0

    # Terminal candidate = last accepted (or baseline if no keeps)
    with runs.connect() as conn:
        last_acc = conn.execute(
            "SELECT id FROM candidates WHERE session_id = ? AND accepted = 1 "
            "ORDER BY id DESC LIMIT 1", (session_id,),
        ).fetchone()

    terminal_metric = 0.0
    oos_is_sharpe_gap = 0.0
    max_drawdown = 0.0
    turnover_annualized = 0.0
    stability_score = 0.0

    if last_acc:
        # Pull the per-seed trials for the terminal candidate
        terminal_trials = [
            r for r in trial_rows
            if r["candidate_id"] == last_acc["id"] and r["primary_metric"] is not None
            and math.isfinite(r["primary_metric"])
        ]
        if terminal_trials:
            per_seed_oos = [r["primary_metric"] for r in terminal_trials]
            terminal_metric = statistics.mean(per_seed_oos)

            # Stability: 1 - stddev / (|mean| + eps). Bounded to [0, 1].
            if len(per_seed_oos) >= 2:
                sd = statistics.stdev(per_seed_oos)
                stability_score = max(0.0, min(1.0, 1.0 - sd / (abs(terminal_metric) + 0.1)))
            else:
                stability_score = 1.0  # single seed = no variance to measure

            # Extract richer dims from metrics_json (BacktestResult.to_dict shape)
            in_sample_sharpes, oos_max_dds, oos_turnovers = [], [], []
            for r in terminal_trials:
                if not r["metrics_json"]:
                    continue
                m = json.loads(r["metrics_json"])
                if isinstance(m, dict) and "in_sample" in m and "oos" in m:
                    is_s = m["in_sample"].get("sharpe")
                    if is_s is not None and math.isfinite(is_s):
                        in_sample_sharpes.append(is_s)
                    dd = m["oos"].get("max_drawdown")
                    if dd is not None and math.isfinite(dd):
                        oos_max_dds.append(dd)
                    to = m["oos"].get("turnover")
                    if to is not None and math.isfinite(to):
                        oos_turnovers.append(to)

            if in_sample_sharpes:
                oos_is_sharpe_gap = terminal_metric - statistics.mean(in_sample_sharpes)
            if oos_max_dds:
                max_drawdown = statistics.mean(oos_max_dds)
            if oos_turnovers:
                turnover_annualized = statistics.mean(oos_turnovers)

    # Time-to-first-keep = ordinal position of first 'keep' decision (1-indexed over decisions)
    time_to_first_keep = None
    for i, r in enumerate(decisions, start=1):
        if r["accepted"] == 1:
            time_to_first_keep = i
            break

    return FitnessScore(
        hit_rate=hit_rate, mean_kept_delta=mean_kept_delta,
        n_kept=n_kept, n_decided=n_decided,
        n_inconclusive=n_inconclusive, n_crashes=n_crashes,
        terminal_metric=terminal_metric, time_to_first_keep=time_to_first_keep,
        wall_seconds=wall_seconds,
        oos_is_sharpe_gap=oos_is_sharpe_gap, max_drawdown=max_drawdown,
        turnover_annualized=turnover_annualized, stability_score=stability_score,
    )


# --------------------------- synthetic evaluator ---------------------------

class SyntheticFitnessEvaluator:
    """
    Deterministic fake fitness from prompt features. No backtests, no runs.db.

    Used for testing GEPA loop mechanics + the paired-CI gate without spending
    minutes per evaluation. The synthetic score is keyed by (prompt_hash, evaluator_seed)
    so paired comparisons make sense (same evaluator_seed across two prompts gives
    matched random draws).
    """

    def __init__(self, registry: PromptRegistry | None = None, noise_scale: float = 0.05):
        self.registry = registry
        self.noise_scale = noise_scale

    def evaluate(self, prompt: Prompt, evaluator_seed: int) -> FitnessScore:
        # Three synthetic "signals" wrapped in a hashed noise term:
        # - keyword presence boosts hit_rate (mirrors what a real fitness would do
        #   if the prompt actually steered the proposer better)
        # - longer prompts get a small bonus (proxy for "more context")
        # - per-seed noise term so paired CI has variance to chew on
        good_keywords = ["INCONCLUSIVE", "noise floor", "small steps", "structurally different"]
        keyword_bonus = sum(0.05 for kw in good_keywords if kw in prompt.content)
        length_bonus = min(0.10, len(prompt.content) / 100_000)
        base = 0.05 + keyword_bonus + length_bonus

        # Deterministic noise from (prompt_hash, evaluator_seed)
        h = hashlib.sha256(f"{prompt.content_hash}|{evaluator_seed}".encode()).digest()
        noise_unit = (int.from_bytes(h[:4], "big") / 2**32) - 0.5  # [-0.5, +0.5]
        noise = noise_unit * 2 * self.noise_scale

        hit_rate = max(0.0, min(1.0, base + noise))
        n_decided = 20
        n_kept = round(hit_rate * n_decided)
        mean_kept_delta = 0.10 + 0.5 * keyword_bonus + noise * 0.5
        terminal = 0.0 + n_kept * 0.05
        ttfk = max(1, round((1.0 - hit_rate) * 15)) if n_kept > 0 else None

        # Synthetic fillers for the multi-dim fields so Pareto code has something to chew on
        score = FitnessScore(
            hit_rate=hit_rate, mean_kept_delta=mean_kept_delta,
            n_kept=n_kept, n_decided=n_decided,
            n_inconclusive=n_decided - n_kept - 1, n_crashes=0,
            terminal_metric=terminal, time_to_first_keep=ttfk,
            wall_seconds=0.0,
            oos_is_sharpe_gap=0.5 * keyword_bonus,           # keywords reduce overfit
            max_drawdown=max(0.05, 0.35 - 0.5 * hit_rate),   # better strategies have lower DD
            turnover_annualized=2.0 + abs(noise) * 4.0,      # noise inflates turnover
            stability_score=0.9 - abs(noise),                # more noise = less stable
        )
        if self.registry is not None and prompt.id is not None:
            self.registry.update_fitness(prompt.id, score.to_dict())
        return score
