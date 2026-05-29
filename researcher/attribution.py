"""
Per-signal performance attribution.

When a candidate is accepted, run N additional standalone backtests — one per
enabled signal with that signal's weight set to 1.0 and all other weights set
to 0. Records each signal's standalone OOS Sharpe in signal_attribution table.

The proposer can then read "momentum_12_1 standalone=+0.45; reversion_5d standalone=-0.10"
to reason about which signals to keep weighting up or zero out.

Cost: N additional backtests per accepted candidate (single-seed). For N=4 signals,
~4 extra backtests per KEEP, which is amortized cheap given accepts are rare.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from researcher import runs
from researcher.domain import FINANCE, Domain

REPO_ROOT = Path(__file__).parent.parent
ATTRIBUTION_TIMEOUT_SEC = 120


@dataclass(frozen=True)
class SignalAttribution:
    signal_name: str
    standalone_oos_sharpe: float | None
    weight_in_candidate: float


def _run_attribution_seed(
    overrides_json: str, seed: int | None, timeout: int,
    domain: Domain | None = None,
) -> tuple[str, float | None]:
    """
    Spawn a single-seed eval with hyperparameter overrides applied to PARAMS.
    Uses the supplied domain's worker subprocess (defaults to FINANCE for back-compat).
    Returns (status, primary_metric).
    """
    domain = domain or FINANCE
    seed_arg = "none" if seed is None else str(seed)
    try:
        proc = subprocess.run(
            [*domain.worker_command, seed_arg, "--params-overrides", overrides_json],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout", None

    if proc.returncode != 0:
        return "crashed", None
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return "crashed", None
    if not payload.get("ok"):
        return "crashed", None
    return "ok", float(payload["primary_metric"])


def attribute_candidate(
    candidate_id: int, enabled_signals: list[str], weights: dict[str, float],
    seed: int | None = 1, timeout: int = ATTRIBUTION_TIMEOUT_SEC,
    domain: Domain | None = None,
) -> list[SignalAttribution]:
    """
    For each signal in enabled_signals, run an eval with weights overridden to
    isolate just that signal (weight=1.0 for it, 0 for all others, kept enabled).
    Stores results in signal_attribution table. Returns the SignalAttribution list.

    Currently finance-shaped (overrides 'weights' dict). Caller is responsible for
    only invoking this on domains whose strategy params actually have a 'weights' field.
    The `domain` arg drives which worker subprocess runs.
    """
    out: list[SignalAttribution] = []
    for sig in enabled_signals:
        single_weights = {s: (1.0 if s == sig else 0.0) for s in enabled_signals}
        overrides = json.dumps({"weights": single_weights})
        status, sharpe = _run_attribution_seed(overrides, seed=seed, timeout=timeout, domain=domain)
        attribution = SignalAttribution(
            signal_name=sig,
            standalone_oos_sharpe=sharpe if status == "ok" else None,
            weight_in_candidate=weights.get(sig, 0.0),
        )
        out.append(attribution)
        with runs.connect() as conn:
            conn.execute(
                "INSERT INTO signal_attribution(candidate_id, signal_name, "
                "  standalone_oos_sharpe, weight_in_candidate) VALUES (?, ?, ?, ?)",
                (candidate_id, sig, attribution.standalone_oos_sharpe, attribution.weight_in_candidate),
            )
    return out


def get_attribution(candidate_id: int) -> list[SignalAttribution]:
    """Retrieve attribution rows for a candidate, sorted by standalone_oos_sharpe desc."""
    with runs.connect() as conn:
        rows = conn.execute(
            "SELECT signal_name, standalone_oos_sharpe, weight_in_candidate "
            "FROM signal_attribution WHERE candidate_id = ? "
            "ORDER BY standalone_oos_sharpe DESC NULLS LAST",
            (candidate_id,),
        ).fetchall()
    return [
        SignalAttribution(
            signal_name=r["signal_name"],
            standalone_oos_sharpe=r["standalone_oos_sharpe"],
            weight_in_candidate=r["weight_in_candidate"],
        )
        for r in rows
    ]


def format_attribution(attribution: list[SignalAttribution]) -> str:
    """Format for inclusion in proposer prompts."""
    if not attribution:
        return "(no per-signal attribution available)"
    lines = []
    for a in attribution:
        sharpe = f"{a.standalone_oos_sharpe:+.4f}" if a.standalone_oos_sharpe is not None else "n/a"
        # Flag draggers — signals with weight > 0 but negative standalone
        marker = ""
        if a.standalone_oos_sharpe is not None and a.weight_in_candidate > 0 and a.standalone_oos_sharpe < 0:
            marker = " ← drag (weighted but negative)"
        elif a.standalone_oos_sharpe is not None and a.weight_in_candidate == 0:
            marker = " (not enabled)"
        lines.append(
            f"  {a.signal_name} (w={a.weight_in_candidate:.2f}) standalone={sharpe}{marker}"
        )
    return "\n".join(lines)
