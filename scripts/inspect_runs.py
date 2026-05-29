"""
Quick readout of session state from runs.db.

Examples:
    uv run python -m scripts.inspect_runs                   # all sessions
    uv run python -m scripts.inspect_runs --session 3       # session 3 detail
    uv run python -m scripts.inspect_runs --session 3 --leaderboard 20
"""

import argparse
import json
import sys

from researcher import runs


def list_sessions() -> None:
    with runs.connect() as conn:
        rows = conn.execute(
            "SELECT s.id, s.started_at, s.ended_at, s.proposer, s.n_iterations_requested, "
            "  (SELECT COUNT(*) FROM candidates c WHERE c.session_id = s.id) AS n_candidates, "
            "  (SELECT COUNT(*) FROM candidates c WHERE c.session_id = s.id AND c.accepted = 1) AS n_accepted "
            "FROM sessions s ORDER BY s.id"
        ).fetchall()
    if not rows:
        print("(no sessions in runs.db)")
        return
    print(f"{'id':>3}  {'started':<20}  {'proposer':<10}  {'iter':>5}  {'cand':>5}  {'kept':>5}  hit%")
    for r in rows:
        hit_pct = (r["n_accepted"] / r["n_candidates"] * 100) if r["n_candidates"] else 0.0
        print(f"{r['id']:>3}  {r['started_at']:<20}  {r['proposer']:<10}  "
              f"{r['n_iterations_requested']:>5}  {r['n_candidates']:>5}  {r['n_accepted']:>5}  {hit_pct:5.1f}")


def detail(session_id: int, leaderboard_n: int) -> None:
    from researcher.reporting.queries import session_metric_label
    metric_label, _ = session_metric_label(session_id)
    hits, decided, ratio = runs.hit_rate(session_id)
    print(f"\n=== session {session_id} ===")
    print(f"hit rate: {hits}/{decided} = {ratio:.1%}")

    print(f"\n--- top {leaderboard_n} accepted candidates by mean {metric_label} ---")
    lb = runs.leaderboard(session_id, limit=leaderboard_n)
    if not lb:
        print("(none yet)")
    _, mfmt = session_metric_label(session_id)
    metric_fmt = "{:" + mfmt + "}"
    for row in lb:
        rat = (row["rationale"] or "").replace("\n", " ")[:80]
        print(f"  #{row['id']:>4}  {metric_label}={metric_fmt.format(row['mean_metric'])}  n={row['n_trials']}  {rat}")

    print(f"\n--- last 15 decisions ---")
    recent = runs.recent_candidates(session_id, limit=15, only_decided=True)
    for r in recent:
        delta = f"{r.primary_delta:+.4f}" if r.primary_delta is not None else "  n/a   "
        rat = (r.rationale or "").replace("\n", " ")[:70]
        flag = " *" if r.accepted else "  "
        print(f"  #{r.id:>4}{flag} [{r.decision:<13}] Δ={delta}  {rat}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=int, default=None, help="Show detail for one session")
    p.add_argument("--leaderboard", type=int, default=10)
    args = p.parse_args()

    if args.session is None:
        list_sessions()
    else:
        detail(args.session, args.leaderboard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
