"""
Live TUI dashboard for runs.db.

Refreshes every N seconds, showing session list + focused-session detail.
Read-only — safe to run alongside an active research session.

Examples:
    uv run python -m scripts.dashboard                  # newest session, 5s refresh
    uv run python -m scripts.dashboard --session 2
    uv run python -m scripts.dashboard --refresh 2

Quit with Ctrl-C.
"""

import argparse
import sys
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from researcher.reporting.queries import (
    candidates_for_session, leaderboard, list_sessions, session_metric_label,
)


def _sessions_table(focused_id: int | None) -> Table:
    t = Table(title="Sessions", expand=True, show_lines=False)
    t.add_column("id", justify="right", style="cyan", no_wrap=True)
    t.add_column("started", no_wrap=True)
    t.add_column("proposer", no_wrap=True)
    t.add_column("iter", justify="right")
    t.add_column("cand", justify="right")
    t.add_column("kept", justify="right", style="green")
    t.add_column("inc", justify="right", style="yellow")
    t.add_column("disc", justify="right", style="red")
    t.add_column("hit%", justify="right")
    t.add_column("alive")

    for s in list_sessions():
        marker = "●" if s.id == focused_id else " "
        alive = "[green]LIVE[/green]" if s.alive else "[dim]ended[/dim]"
        t.add_row(
            f"{marker} {s.id}",
            s.started_at[:19], s.proposer,
            str(s.n_iterations_requested or "?"),
            str(s.n_candidates), str(s.n_kept),
            str(s.n_inconclusive), str(s.n_discard),
            f"{s.hit_rate * 100:5.1f}",
            alive,
        )
    return t


def _decisions_table(session_id: int) -> Table:
    t = Table(title=f"Recent decisions — session {session_id}", expand=True)
    t.add_column("#", justify="right", style="cyan", no_wrap=True)
    t.add_column("decision", no_wrap=True)
    t.add_column("Δ", justify="right")
    t.add_column("metric", justify="right")
    t.add_column("rationale", overflow="ellipsis", no_wrap=True)

    cands = candidates_for_session(session_id, limit=10)
    for c in cands:
        if c.decision is None:
            decision_text = "[dim]pending[/dim]"
        elif c.accepted:
            decision_text = "[bold green]KEEP[/bold green]"
        elif c.decision == "inconclusive":
            decision_text = "[yellow]inconc[/yellow]"
        elif c.decision == "baseline":
            decision_text = "[cyan]baseline[/cyan]"
        else:
            decision_text = "[red]discard[/red]"

        delta = f"{c.primary_delta:+.4f}" if c.primary_delta is not None else "  n/a   "
        metric = f"{c.mean_metric:+.4f}" if c.mean_metric is not None else "  n/a   "
        rationale = (c.rationale or "").replace("\n", " ")[:120]
        t.add_row(str(c.id), decision_text, delta, metric, rationale)
    return t


def _leaderboard_table(session_id: int) -> Table:
    metric_label, metric_format = session_metric_label(session_id)
    metric_fmt = "{:" + metric_format + "}"
    t = Table(title=f"Leaderboard — session {session_id} (accepted)", expand=True)
    t.add_column("#", justify="right", style="cyan", no_wrap=True)
    t.add_column(metric_label, justify="right", style="green")
    t.add_column("n", justify="right", style="dim")
    t.add_column("source / rationale", overflow="ellipsis", no_wrap=True)
    for c in leaderboard(session_id, limit=8):
        metric = metric_fmt.format(c.mean_metric) if c.mean_metric is not None else "n/a"
        rationale = (c.rationale or "(baseline)").replace("\n", " ")[:120]
        t.add_row(str(c.id), metric, str(c.n_trials), rationale)
    return t


def _build_layout(focused: int | None, refresh: float) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(_sessions_table(focused), name="top", size=20),
        Layout(name="bottom"),
    )
    if focused is not None:
        layout["bottom"].split_row(
            Layout(_decisions_table(focused), name="decisions"),
            Layout(_leaderboard_table(focused), name="leaderboard"),
        )
    else:
        layout["bottom"].update(
            Panel(Text("No session focused. Use --session N.", style="dim"), title="detail")
        )
    return layout


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=int, default=None, help="Focus a specific session")
    p.add_argument("--refresh", type=float, default=5.0, help="Refresh interval in seconds")
    args = p.parse_args()

    console = Console()

    # If no session specified, focus the latest
    focused = args.session
    if focused is None:
        sessions = list_sessions()
        if sessions:
            focused = sessions[-1].id

    try:
        with Live(_build_layout(focused, args.refresh), console=console, refresh_per_second=2, screen=False) as live:
            while True:
                time.sleep(args.refresh)
                live.update(_build_layout(focused, args.refresh))
    except KeyboardInterrupt:
        console.print("\n[dim]dashboard exited[/dim]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
