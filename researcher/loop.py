"""
Outer autonomous loop: propose -> replicate -> decide -> log.

Linear hill-climb on a paired-CI gate. On 'keep', the candidate becomes the new
parent and strategy.py reflects it on disk. On 'discard' or 'inconclusive', the
parent strategy is restored.

Live hit-rate readout after every decision so the operator (or future agent)
can watch reasoning quality drift in real time.
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Callable

from researcher import runner, runs
from researcher.decide import TrialSummary, decide
from researcher.domain import FINANCE, Domain
from researcher.proposer import Proposer, ProposerContext

REPO_ROOT = Path(__file__).parent.parent
STRATEGY_FILE = REPO_ROOT / "strategy.py"  # back-compat; domain-aware code uses domain.strategy_file


def _mean_ok_metric(trials: list[TrialSummary]) -> float | None:
    ok = [t.primary_metric for t in trials if t.status == "ok" and t.primary_metric is not None and math.isfinite(t.primary_metric)]
    return statistics.mean(ok) if ok else None


def _print(msg: str) -> None:
    print(msg, flush=True)


def run_loop(
    proposer: Proposer,
    n_iterations: int,
    seeds: list[int | None],
    session_notes: str | None = None,
    timeout_sec: int = 120,
    on_iteration: Callable[[int, str], None] | None = None,
    attribute_accepts: bool = False,
    domain: Domain | None = None,
) -> int:
    """
    Drive the research loop for n_iterations. Returns session_id.

    seeds: list of replicate seeds; len(seeds) determines the paired-CI sample size.
           For paired CI use [1, 2, 3] (or any three distinct ints). Use [None] only
           for smoke-tests where you want a single deterministic full-universe run.
    """
    domain = domain or FINANCE
    session_id = runs.start_session(
        proposer=proposer.name,
        proposer_config={"seeds": seeds, "domain": domain.name},
        n_iterations=n_iterations,
        replicate_seeds=len(seeds),
        notes=session_notes,
    )
    _print(f"\n=== session {session_id} started ({proposer.name} on {domain.name}, {len(seeds)} seeds, {n_iterations} iterations) ===\n")

    # Baseline
    _print(f"[baseline] running current {domain.strategy_file.name} across seeds={seeds} ...")
    baseline = runner.baseline_trials(session_id, seeds, timeout=timeout_sec, domain=domain)
    parent_id = baseline.candidate_id
    parent_trials = baseline.trials
    parent_metric = _mean_ok_metric(parent_trials)
    if parent_metric is None:
        _print(f"[baseline] FAILED — no usable trials. Aborting.")
        for t in parent_trials:
            _print(f"  seed={t.seed} status={t.status} metric={t.primary_metric}")
        runs.end_session(session_id)
        return session_id
    metric_fmt = "{:" + domain.primary_metric_format + "}"
    _print(f"[baseline] mean {domain.primary_metric_name} = " + metric_fmt.format(parent_metric)
           + f" across {sum(1 for t in parent_trials if t.status=='ok')}/{len(parent_trials)} seeds\n")

    parent_src = domain.strategy_file.read_text()

    for i in range(1, n_iterations + 1):
        _print(f"--- iteration {i}/{n_iterations} ---")

        # 1. Propose
        ctx = ProposerContext(
            current_strategy_src=parent_src,
            recent_decided=runs.recent_candidates(session_id, limit=15),
            iteration=i,
            parent_metric=parent_metric,
            session_id=session_id,
            parent_candidate_id=parent_id,
        )
        try:
            output = proposer(ctx)
        except Exception as e:
            _print(f"  proposer error: {type(e).__name__}: {e}")
            if on_iteration:
                on_iteration(i, f"proposer-error: {e}")
            continue

        _print(f"  proposed: {output.rationale[:120]}")

        # 2. Replicate-execute (with adaptive replication: parent_trials enables early screen)
        candidate_run = runner.run_candidate(
            session_id=session_id, parent_id=parent_id,
            parent_src=parent_src, candidate_src=output.new_strategy_src,
            rationale=output.rationale, seeds=seeds, timeout=timeout_sec,
            parent_trials=parent_trials, domain=domain,
        )
        for t in candidate_run.trials:
            metric_str = (("{:" + domain.primary_metric_format + "}").format(t.primary_metric)
                          if t.primary_metric is not None else "n/a")
            _print(f"  seed={t.seed} status={t.status} {domain.primary_metric_name}={metric_str}")

        # 3. Decide — curriculum early-screen bypasses the gate when triggered
        if candidate_run.early_decision is not None:
            decision = candidate_run.early_decision
            _print(f"  [curriculum] early-screened after seed-1, skipped remaining seeds")
        else:
            decision = decide(parent_trials, candidate_run.trials)
        accepted = (decision.action == "keep")
        runs.record_decision(
            candidate_id=candidate_run.candidate_id,
            decision=decision.action, payload=decision.to_dict(), accepted=accepted,
        )

        if accepted:
            runner.commit_parent(domain=domain)
            parent_id = candidate_run.candidate_id
            parent_trials = candidate_run.trials
            parent_metric = _mean_ok_metric(parent_trials)
            parent_src = domain.strategy_file.read_text()
            metric_fmt = "{:" + domain.primary_metric_format + "}"
            _print(f"  KEEP — {decision.reason}. New parent {domain.primary_metric_name} = "
                   + metric_fmt.format(parent_metric))

            # Per-signal attribution: optional post-accept analysis to feed the next proposer
            if attribute_accepts:
                try:
                    from researcher.attribution import attribute_candidate
                    # Extract enabled signals + weights from the kept params (last OK trial has them)
                    params_dict = None
                    for t in candidate_run.trials:
                        if t.status == "ok":
                            with runs.connect() as conn:
                                row = conn.execute(
                                    "SELECT params_json FROM trials WHERE candidate_id = ? AND seed = ? LIMIT 1",
                                    (candidate_run.candidate_id, t.seed),
                                ).fetchone()
                            if row and row["params_json"]:
                                import json as _json
                                params_dict = _json.loads(row["params_json"])
                                break
                    if params_dict:
                        signals = list(params_dict.get("enabled_signals") or [])
                        weights = dict(params_dict.get("weights") or {})
                        if signals:
                            _print(f"  [attribution] computing standalone {domain.primary_metric_name} for {len(signals)} signals...")
                            attrs = attribute_candidate(
                                candidate_id=candidate_run.candidate_id,
                                enabled_signals=signals, weights=weights,
                                seed=seeds[0] if seeds else 1, timeout=timeout_sec,
                                domain=domain,
                            )
                            metric_fmt2 = "{:" + domain.primary_metric_format + "}"
                            for a in attrs:
                                s = metric_fmt2.format(a.standalone_oos_sharpe) if a.standalone_oos_sharpe is not None else "n/a"
                                _print(f"    {a.signal_name} (w={a.weight_in_candidate:.2f}): standalone={s}")
                except Exception as e:
                    _print(f"  [attribution] failed: {type(e).__name__}: {e}")
        else:
            runner.revert_to_parent(domain=domain)
            _print(f"  {decision.action.upper()} — {decision.reason}")

        # 4. Live hit rate
        hits, decided_n, ratio = runs.hit_rate(session_id)
        _print(f"  session hit rate: {hits}/{decided_n} = {ratio:.1%}\n")

        if on_iteration:
            on_iteration(i, decision.action)

    runs.end_session(session_id)
    _print(f"=== session {session_id} done ===")
    return session_id
