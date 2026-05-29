"""
Execute one candidate (= one strategy proposal) across N replicate seeds.

Mechanics:
  1. Back up current strategy.py to strategy.py.parent_backup.
  2. Write the candidate source to strategy.py.
  3. For each seed, spawn a subprocess running _backtest_worker — full isolation
     so a buggy candidate can't corrupt sibling trials or the parent process.
  4. Persist every trial to runs.db (status, metrics, log).
  5. Return the per-seed TrialSummary list. The CALLER decides keep vs discard
     (via decide.py) and calls revert_to_parent() if not keeping.
"""

from __future__ import annotations

import difflib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from researcher import runs
from researcher.decide import Decision, TrialSummary
from researcher.domain import FINANCE, Domain

REPO_ROOT = Path(__file__).parent.parent
# Legacy module-level constants — back-compat. Domain-aware code paths use domain.strategy_file etc.
STRATEGY_FILE = REPO_ROOT / "strategy.py"
PARENT_BACKUP = REPO_ROOT / "strategy.py.parent_backup"

DEFAULT_TIMEOUT_SEC = 120
LOG_TRUNCATE_BYTES = 4 * 1024
DEFAULT_SCREEN_GAP = -0.20   # candidate seed-1 must be within this much of parent seed-1 to continue


@dataclass(frozen=True)
class CandidateRun:
    candidate_id: int
    trials: list[TrialSummary]
    early_decision: Decision | None = None   # set when curriculum short-circuits


def make_diff(parent_src: str, candidate_src: str) -> str:
    return "".join(difflib.unified_diff(
        parent_src.splitlines(keepends=True),
        candidate_src.splitlines(keepends=True),
        fromfile="parent/strategy.py",
        tofile="candidate/strategy.py",
        n=3,
    ))


def _run_one_seed(
    seed: int | None, timeout: int, domain: Domain | None = None,
) -> tuple[str, float | None, dict | None, dict | None, str]:
    """Returns (status, primary_metric, metrics_dict, params_dict, log)."""
    domain = domain or FINANCE
    seed_arg = "none" if seed is None else str(seed)
    try:
        proc = subprocess.run(
            [*domain.worker_command, seed_arg],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return "timeout", None, None, None, f"TIMEOUT after {timeout}s\n{e.stderr or ''}"

    log = (proc.stdout + "\n" + proc.stderr)[-LOG_TRUNCATE_BYTES:]

    if proc.returncode != 0:
        return "crashed", None, None, None, log

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return "crashed", None, None, None, log

    if not payload.get("ok"):
        return "crashed", None, None, None, log

    return (
        "ok",
        float(payload["primary_metric"]),
        payload["result"],
        payload["params"],
        log[-512:],  # store less for successful runs
    )


def run_candidate(
    session_id: int,
    parent_id: int | None,
    parent_src: str,
    candidate_src: str,
    rationale: str,
    seeds: list[int | None],
    timeout: int = DEFAULT_TIMEOUT_SEC,
    parent_trials: list[TrialSummary] | None = None,
    screen_gap: float = DEFAULT_SCREEN_GAP,
    domain: Domain | None = None,
) -> CandidateRun:
    """
    Write the candidate, replicate-execute it, persist everything. Caller reverts on reject.

    domain (default FINANCE) drives which strategy_file is written, which parent_backup_file
    is created, and which worker subprocess is invoked.

    Adaptive replication (curriculum): when parent_trials is provided AND the candidate's
    seed-1 metric is more than screen_gap worse than parent's seed-1 metric, skip the
    remaining seeds and return early with a discard decision.
    """
    domain = domain or FINANCE
    domain.parent_backup_file.write_text(parent_src)
    domain.strategy_file.write_text(candidate_src)

    diff = make_diff(parent_src, candidate_src)
    candidate_id = runs.insert_candidate(
        session_id=session_id, parent_id=parent_id, diff=diff, rationale=rationale,
    )

    parent_by_seed = {t.seed: t for t in (parent_trials or [])}
    trials: list[TrialSummary] = []
    early_decision: Decision | None = None

    for idx, seed in enumerate(seeds):
        status, metric, metrics_d, params_d, log = _run_one_seed(seed, timeout, domain=domain)
        runs.insert_trial(
            candidate_id=candidate_id, seed=seed, status=status, primary_metric=metric,
            metrics=metrics_d, params=params_d, log=log,
        )
        trials.append(TrialSummary(seed=seed, status=status, primary_metric=metric))

        # Adaptive screen after the FIRST seed only. Skip if no parent_trials, if
        # this seed crashed (let normal gate handle), or if no comparable parent metric.
        if idx == 0 and status == "ok" and metric is not None and parent_trials:
            parent_t = parent_by_seed.get(seed)
            if parent_t and parent_t.status == "ok" and parent_t.primary_metric is not None:
                gap = metric - parent_t.primary_metric
                if gap < screen_gap:
                    # Clearly worse — record remaining seeds as 'skipped', emit early discard
                    skip_log = (
                        f"early-screen: seed-{seed} candidate={metric:.4f} vs parent={parent_t.primary_metric:.4f} "
                        f"(gap={gap:.4f} < threshold={screen_gap})"
                    )
                    for remaining_seed in seeds[idx + 1:]:
                        runs.insert_trial(
                            candidate_id=candidate_id, seed=remaining_seed, status="skipped",
                            primary_metric=None, metrics=None, params=None, log=skip_log,
                        )
                        trials.append(TrialSummary(seed=remaining_seed, status="skipped", primary_metric=None))
                    early_decision = Decision(
                        action="discard", reason=f"early-screen on seed-{seed}: " + skip_log.split(': ', 1)[1],
                        primary_delta=gap, ci_low=None, ci_high=None, p_value=None, n_pairs=1,
                    )
                    break

    return CandidateRun(candidate_id=candidate_id, trials=trials, early_decision=early_decision)


def commit_parent(domain: Domain | None = None) -> None:
    """Decision was 'keep' — strategy file stays as written, drop the backup."""
    domain = domain or FINANCE
    if domain.parent_backup_file.exists():
        domain.parent_backup_file.unlink()


def revert_to_parent(domain: Domain | None = None) -> None:
    """Decision was 'discard' or 'inconclusive' — restore the parent strategy."""
    domain = domain or FINANCE
    if domain.parent_backup_file.exists():
        domain.strategy_file.write_text(domain.parent_backup_file.read_text())
        domain.parent_backup_file.unlink()


def baseline_trials(
    session_id: int, seeds: list[int | None], timeout: int = DEFAULT_TIMEOUT_SEC,
    domain: Domain | None = None,
) -> CandidateRun:
    """Run the current strategy file as the initial baseline candidate (no diff, no parent)."""
    domain = domain or FINANCE
    candidate_id = runs.insert_candidate(
        session_id=session_id, parent_id=None,
        diff=f"(baseline — initial {domain.strategy_file.name})",
        rationale="baseline / seed strategy",
    )
    trials: list[TrialSummary] = []
    for seed in seeds:
        status, metric, metrics_d, params_d, log = _run_one_seed(seed, timeout, domain=domain)
        runs.insert_trial(
            candidate_id=candidate_id, seed=seed, status=status, primary_metric=metric,
            metrics=metrics_d, params=params_d, log=log,
        )
        trials.append(TrialSummary(seed=seed, status=status, primary_metric=metric))
    # Baseline is always 'accepted' as the starting parent
    runs.record_decision(
        candidate_id=candidate_id, decision="baseline",
        payload={"primary_delta": None, "reason": "initial baseline"},
        accepted=True,
    )
    return CandidateRun(candidate_id=candidate_id, trials=trials)
