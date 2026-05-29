"""
HTML report generator for a research session.

Produces a single self-contained HTML file (no external CSS/JS) with:
  - Session metadata header + headline metric
  - Decision timeline (color-coded)
  - Accepted-candidate leaderboard with full rationales
  - Lineage tree (parent-child indent)
  - Per-candidate collapsible diff viewer (uses <details> — pure HTML)

Example:
    uv run python -m scripts.report --session 2
    uv run python -m scripts.report --session 2 --out reports/baseline_v2.html
"""

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

from researcher.reporting.queries import (
    candidates_for_session, leaderboard, lineage_tree, list_sessions,
    session_metric_label, trials_for_candidate,
)

REPORTS_DIR = Path(__file__).parent.parent / "reports"

_CSS = """
:root {
  --bg: #fafafa; --fg: #222; --muted: #777; --border: #e0e0e0;
  --kept: #2e7d32; --discard: #c62828; --inconc: #f9a825; --baseline: #1565c0;
  --code-bg: #f5f5f5;
}
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: var(--bg); color: var(--fg); margin: 0; padding: 24px; }
h1, h2, h3 { margin: 24px 0 12px; }
h1 { border-bottom: 2px solid var(--border); padding-bottom: 8px; }
.meta { color: var(--muted); font-size: 13px; }
.headline { background: white; padding: 16px; border: 1px solid var(--border);
            border-radius: 6px; margin: 16px 0; }
.headline .big { font-size: 28px; font-weight: 600; color: var(--kept); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #efefef; text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
td { padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px;
         font-weight: 600; color: white; }
.badge.kept { background: var(--kept); }
.badge.discard { background: var(--discard); }
.badge.inconc { background: var(--inconc); }
.badge.baseline { background: var(--baseline); }
.delta-pos { color: var(--kept); font-weight: 500; }
.delta-neg { color: var(--discard); }
.lineage { font-family: ui-monospace, "Cascadia Code", Menlo, monospace;
           font-size: 12px; background: white; border: 1px solid var(--border);
           padding: 12px; border-radius: 6px; white-space: pre; overflow-x: auto; }
details { background: white; border: 1px solid var(--border); border-radius: 4px;
          margin: 6px 0; padding: 8px 12px; }
details summary { cursor: pointer; font-weight: 500; outline: none; }
pre { background: var(--code-bg); padding: 10px; border-radius: 4px;
      overflow-x: auto; font: 11px/1.4 ui-monospace, Menlo, monospace; }
.rationale { color: #333; padding-top: 8px; font-style: italic; }
"""


def _fmt_delta(d: float | None) -> str:
    if d is None:
        return '<span class="meta">n/a</span>'
    cls = "delta-pos" if d > 0 else "delta-neg"
    return f'<span class="{cls}">{d:+.4f}</span>'


def _badge(decision: str | None, accepted: bool) -> str:
    if decision is None:
        return '<span class="meta">pending</span>'
    if accepted and decision == "baseline":
        return '<span class="badge baseline">BASELINE</span>'
    if accepted:
        return '<span class="badge kept">KEEP</span>'
    if decision == "inconclusive":
        return '<span class="badge inconc">INCONC</span>'
    return '<span class="badge discard">DISCARD</span>'


def _render_lineage(session_id: int) -> str:
    tree = lineage_tree(session_id)
    cands_by_id = {c.id: c for c in candidates_for_session(session_id)}
    lines: list[str] = []

    def walk(node_id: int | None, depth: int) -> None:
        for child_id in tree.get(node_id, []):
            c = cands_by_id.get(child_id)
            if c is None:
                continue
            indent = "  " * depth + ("└─ " if depth > 0 else "")
            marker = "★" if c.accepted else " "
            delta = f"Δ={c.primary_delta:+.4f}" if c.primary_delta is not None else "Δ=  n/a "
            metric = f"oos={c.mean_metric:+.4f}" if c.mean_metric is not None else "oos= n/a "
            line = f"{indent}{marker} #{c.id:>4}  {(c.decision or 'pending'):<13}  {delta}  {metric}"
            lines.append(html.escape(line))
            walk(child_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines) or "(no candidates yet)"


def render_html(session_id: int) -> str:
    sessions = {s.id: s for s in list_sessions()}
    if session_id not in sessions:
        raise SystemExit(f"Session {session_id} not found in runs.db")
    s = sessions[session_id]
    cands = candidates_for_session(session_id)
    lb = leaderboard(session_id, limit=10)
    metric_label, metric_format = session_metric_label(session_id)
    metric_fmt = "{:" + metric_format + "}"
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Research session {session_id} report</title>
<style>{_CSS}</style>
</head><body>""")

    parts.append(f"""<h1>Research session {session_id}</h1>
<div class="meta">
proposer: <b>{html.escape(s.proposer)}</b> ·
started: {html.escape(s.started_at)} ·
ended: {html.escape(s.ended_at or '(running)')} ·
seeds: {s.replicate_seeds or '?'} ·
iter requested: {s.n_iterations_requested or '?'} ·
generated: {generated}
</div>""")
    if s.notes:
        parts.append(f'<div class="meta">notes: {html.escape(s.notes)}</div>')

    parts.append(f"""<div class="headline">
<div>hit rate</div>
<div class="big">{s.n_kept}/{s.n_decided} = {s.hit_rate * 100:.1f}%</div>
<div class="meta">{s.n_kept} kept · {s.n_inconclusive} inconclusive · {s.n_discard} discard · {s.n_candidates} total</div>
</div>""")

    parts.append("<h2>Decision timeline</h2><table>")
    parts.append(f"<thead><tr><th>#</th><th>decision</th><th>Δ</th><th>{metric_label} mean</th><th>rationale</th></tr></thead><tbody>")
    for c in cands:
        rationale = html.escape((c.rationale or "").replace("\n", " ")[:200])
        metric_str = metric_fmt.format(c.mean_metric) if c.mean_metric is not None else '<span class="meta">n/a</span>'
        parts.append(
            f"<tr><td>{c.id}</td><td>{_badge(c.decision, c.accepted)}</td>"
            f"<td>{_fmt_delta(c.primary_delta)}</td><td>{metric_str}</td>"
            f"<td>{rationale}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h2>Leaderboard (accepted only)</h2><table>")
    parts.append(f"<thead><tr><th>#</th><th>{metric_label}</th><th>n trials</th><th>rationale</th></tr></thead><tbody>")
    for c in lb:
        rationale = html.escape((c.rationale or "(baseline)").replace("\n", " ")[:200])
        metric_str = metric_fmt.format(c.mean_metric) if c.mean_metric is not None else "n/a"
        parts.append(
            f"<tr><td>{c.id}</td><td>{metric_str}</td><td>{c.n_trials}</td>"
            f"<td>{rationale}</td></tr>"
        )
    if not lb:
        parts.append('<tr><td colspan="4" class="meta">(no accepted candidates yet)</td></tr>')
    parts.append("</tbody></table>")

    parts.append("<h2>Lineage tree</h2>")
    parts.append(f'<div class="lineage">{_render_lineage(session_id)}</div>')

    parts.append("<h2>Per-candidate detail (click to expand)</h2>")
    for c in cands:
        rationale = html.escape(c.rationale or "")
        diff_text = html.escape(c.diff or "")
        trials = trials_for_candidate(c.id)
        trial_lines = []
        for t in trials:
            m = metric_fmt.format(t.primary_metric) if t.primary_metric is not None else "n/a"
            trial_lines.append(f"  seed={t.seed} status={t.status} {metric_label}={m}")
        trial_block = html.escape("\n".join(trial_lines)) if trial_lines else "(no trials)"

        summary_label = f"#{c.id} — {c.decision or 'pending'}"
        if c.primary_delta is not None:
            summary_label += f" (Δ={c.primary_delta:+.4f})"
        parts.append(f"""<details>
<summary>{html.escape(summary_label)}</summary>
<div class="rationale">{rationale or '<span class="meta">(no rationale)</span>'}</div>
<h3>Trials</h3><pre>{trial_block}</pre>
<h3>Diff</h3><pre>{diff_text or '(no diff)'}</pre>
</details>""")

    parts.append("</body></html>")
    return "".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: reports/session_N.html)")
    args = p.parse_args()

    out = args.out or (REPORTS_DIR / f"session_{args.session}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_html(args.session)
    out.write_text(html_text)
    print(f"Wrote {out} ({len(html_text):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
