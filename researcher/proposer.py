"""
Proposers: produce the next strategy mutation given history.

A Proposer is a callable: ProposerContext -> ProposerOutput.

Two implementations:
  - AnthropicProposer: real reasoning via Claude. The component whose quality
    we're actually trying to measure with hit-rate over time.
  - StubProposer: deterministic seeded random hyperparameter perturbation. Used
    for smoke-testing the loop without API calls and as a noise-floor baseline
    that any real proposer must beat to justify its cost.

Output is the full new contents of strategy.py (NOT a diff). Diff is computed
by the runner — LLM-generated unified diffs are too fragile to parse.
"""

from __future__ import annotations

import os
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from researcher.runs import CandidateRecord

REPO_ROOT = Path(__file__).parent.parent
HARNESS_FILES = [
    REPO_ROOT / "harness" / "backtest.py",
    REPO_ROOT / "harness" / "metrics.py",
    REPO_ROOT / "harness" / "data.py",
    REPO_ROOT / "harness" / "signals.py",   # signal library — the typed move surface
]
STRATEGY_FILE = REPO_ROOT / "strategy.py"
KEY_FILE_FALLBACK = Path.home() / ".anthropic_key"

DEFAULT_MODEL = "claude-sonnet-4-5-20251022"


def _resolve_api_key() -> str | None:
    """Env var wins; else fall back to ~/.anthropic_key (first line, stripped)."""
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    if KEY_FILE_FALLBACK.exists():
        return KEY_FILE_FALLBACK.read_text().splitlines()[0].strip() or None
    return None


@dataclass(frozen=True)
class ProposerContext:
    current_strategy_src: str
    recent_decided: list[CandidateRecord]          # most recent first
    iteration: int
    parent_metric: float | None                    # parent's mean OOS Sharpe across seeds
    session_id: int | None = None                  # for cross-session memory exclusion
    parent_candidate_id: int | None = None         # for per-signal attribution lookup


@dataclass(frozen=True)
class ProposerOutput:
    new_strategy_src: str
    rationale: str


class Proposer(Protocol):
    name: str

    def __call__(self, ctx: ProposerContext) -> ProposerOutput: ...


# --------------------------- system prompt ---------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a quantitative-research agent iterating on a multi-signal equity strategy.

The objective is a measured improvement in OOS (2022-2024) annualized Sharpe, gated by a
paired-CI decision rule across {n_seeds} replicate sub-universes. Marginal improvements that
fall inside noise will be classified 'inconclusive' and discarded — propose mutations large
enough to move the needle, but not so large they crash.

## The LOCKED harness (do not try to change — these files are read-only)

You are reasoning inside this evaluation. Understanding it is critical to proposing useful moves.

{harness_files}

## What you may change

A single file: strategy.py. Its module-level `PARAMS` and `generate_signals()` are the contract.
You may NOT modify the harness files (everything under harness/) nor import new third-party packages.

## The typed mutation surface

StrategyParams is the explicit knob set. Prefer structured moves over freeform code edits:

- `enabled_signals: tuple[str, ...]` — subset of `harness/signals.py::SIGNAL_LIBRARY` keys.
  ADD a signal from the library by appending its name. REMOVE by dropping it.
- `weights: dict[str, float]` — per-signal blend coefficient. Missing keys default to 1.0.
- `signal_kwargs: dict[str, dict]` — per-signal hyperparameter overrides
  (e.g. `{"momentum_12_1": {"lookback": 189}}`). Read the signal's signature in `signals.py`.
- `combine_mode: Literal["linear", "sign_vote", "sharpe_weighted"]` — how z-scored signals merge.
- `long_pct`, `short_pct`, `rebalance_days`, `gross_leverage` — portfolio knobs.

Available signals (see `harness/signals.py` for definitions): momentum_12_1, momentum_3m,
reversion_5d, reversion_21d, lowvol_20d, lowvol_60d, trend_consistency, acceleration,
vol_adjusted_momentum.

## Mutation categories

- Structural: add/remove a signal; swap `combine_mode`; toggle a signal's lookback.
- Compositional: rebalance the `weights` across enabled signals.
- Portfolio: long/short widths, rebalance cadence, leverage.
- Beyond the typed surface: refactor `generate_signals` to add risk overlays (vol target,
  drawdown cap, beta neutralization), but prefer typed moves when they cover the idea —
  they're easier for the gate to interpret.

## Anti-repetition discipline (critical)

Before you propose your mutation, READ the "Recent attempts" and "Cross-session memory"
sections of the user message and apply these rules STRICTLY:

1. **Do not repeat a discarded move.** If a prior attempt with negative Δ already tried
   adding the same signal, switching to the same `combine_mode`, or tweaking the same
   parameter to a similar value — DO NOT propose that move again. Pick a structurally
   different change.

2. **Move classes to rotate through:** (a) different signal additions (try each unused
   signal from SIGNAL_LIBRARY); (b) different signal removals; (c) the three combine_modes;
   (d) portfolio dimensions (long_pct, short_pct, rebalance_days, gross_leverage); (e)
   per-signal hyperparameter overrides via `signal_kwargs`; (f) risk overlays in
   `generate_signals` body.

3. **If you find yourself rationalizing the same idea** ("the 3-signal blend is too narrow"),
   commit to a SPECIFIC different move that's unrepresented in recent attempts. Diversity
   of attempts beats depth on any single idea — the gate is noisy enough that you need to
   explore breadth before doubling down.

4. **Cross-session memory has higher signal than your priors.** If prior sessions tried
   something and it failed with large negative Δ, that's strong evidence — don't override it
   with first-principles reasoning unless the failure mode is plausibly fixed.

## Output format

Reply with EXACTLY this structure and nothing else:

<rationale>
2-4 sentences. State the move, why you expect it to improve OOS Sharpe over the parent,
and what failure mode you're guarding against.
</rationale>

<strategy_py>
[The complete new contents of strategy.py. Must be valid Python. Must export PARAMS and
generate_signals(prices, params). Do not include the ``` fences — just the file content.]
</strategy_py>
"""


def _format_recent(recent: list[CandidateRecord], limit: int = 15) -> str:
    if not recent:
        return "(no prior attempts in this session)"
    lines = []
    for r in recent[:limit]:
        delta = f"{r.primary_delta:+.4f}" if r.primary_delta is not None else "n/a"
        lines.append(f"#{r.id} [{r.decision} Δ={delta}] {r.rationale or '(no rationale)'}".strip())
    return "\n".join(lines)


def _fetch_failure_traces(
    recent: list[CandidateRecord],
    n_failures: int = 5,
    bytes_per_trace: int = 800,
) -> str:
    """
    Improvement #2 (trace-augmented prompts): include the actual subprocess log
    for the N most recent NON-KEEP candidates (crashed or discard-with-negative-CI).
    The agent can then reason about *why* a strategy failed, not just *that* it did.

    Successful (kept) candidates intentionally omitted — their logs are full of
    normal-looking metrics and consume context without adding insight.
    """
    from researcher import runs

    failures = [r for r in recent if not r.accepted][:n_failures]
    if not failures:
        return "(no failure traces yet — all attempts kept or no attempts made)"

    chunks = []
    for cand in failures:
        trials = runs.trials_for_candidate(cand.id)
        crashed = next((t for t in trials if t["status"] != "ok"), None)
        target = crashed if crashed is not None else (trials[0] if trials else None)
        if target is None or not target["log"]:
            continue
        log_snippet = str(target["log"])[-bytes_per_trace:]
        chunks.append(
            f"### candidate #{cand.id} [{cand.decision}] seed={target['seed']} status={target['status']}\n"
            f"rationale: {cand.rationale or '(none)'}\n"
            f"log tail:\n```\n{log_snippet}\n```"
        )

    return "\n\n".join(chunks) if chunks else "(failures recorded but no log content available)"


_MINIMAL_CONTRACT = """\
### Locked harness contract (you cannot modify this; it defines the eval)

harness/backtest.py exposes `run_backtest(strategy_fn, params, seed)`:
  - strategy_fn must be `generate_signals(prices: PriceData, params: StrategyParams) -> pd.DataFrame`
  - returns target weights, index = trading dates, columns = tickers, values in [-1, +1] range
  - One-day lag applied before fills; survivorship-bias-free via point-in-time S&P 500 membership.
  - Walk-forward split: in-sample 2010-2021, OOS 2022-2024. OOS Sharpe is the gating metric.

harness/data.py provides `PriceData` with fields:
  - .close, .volume, .returns: wide DataFrames (date x ticker), NaN outside membership window
  - .membership: bool DataFrame (date x ticker), True where ticker was an S&P 500 member that day
  - .tickers, .dates accessors

harness/metrics.py provides: sharpe, calmar, max_drawdown, hit_rate, turnover (annualized).

Each backtest seed samples a random 80% sub-universe — same seed gives same sample, enabling
paired comparisons. The gating decision uses paired-CI across 3 seeds: 'keep' only when the
mean OOS-Sharpe delta is strictly positive at 95% CI.
"""


def _read_harness_files(
    harness_files: "tuple[Path, ...] | list[Path] | None" = None,
    concise: bool = False,
    minimal: bool = False,
) -> str:
    """
    Render the locked-harness section of the system prompt for ONE domain.

    harness_files: the domain's locked files (`domain.harness_files`). Defaults to the
                   finance HARNESS_FILES for back-compat when no domain is threaded in.
    full      = all harness files with full source (~4.5K tokens). For Claude.
    concise   = docstrings + signatures only (~2.3K tokens). For mid-size models.
    minimal   = smallest representation for small local models. For finance this is a
                hand-written contract paragraph + signals.py (~1.5K tokens). For other
                domains it falls back to signatures-only over that domain's own files.
    """
    files = list(harness_files) if harness_files is not None else list(HARNESS_FILES)
    signals_path = REPO_ROOT / "harness" / "signals.py"

    # Finance-specific minimal path: a hand-tuned contract + the signal library only.
    # Keyed on the presence of signals.py so non-finance domains never get finance text.
    if minimal and signals_path in files:
        signals_body = _extract_signatures(signals_path.read_text())
        return (
            f"{_MINIMAL_CONTRACT}\n\n"
            f"### harness/signals.py (the signal library you compose from)\n"
            f"```python\n{signals_body}\n```"
        )

    use_concise = concise or minimal  # minimal on a non-finance domain → signatures only
    parts = []
    for path in files:
        body = path.read_text()
        if use_concise:
            body = _extract_signatures(body)
        parts.append(f"### {path.relative_to(REPO_ROOT)}\n```python\n{body}\n```")
    return "\n\n".join(parts)


def _extract_signatures(src: str) -> str:
    """
    Token-budget-conscious harness representation: docstrings + def/class signatures
    only. Drops function bodies. Used by small local LLMs where full implementations
    would consume too much context.
    """
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src  # fall back to full source on parse failure

    out_lines: list[str] = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        out_lines.append(f'"""{module_doc}"""')
        out_lines.append("")

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out_lines.append(ast.unparse(node))
        elif isinstance(node, ast.Assign):
            # Top-level constants — keep these (may include locked schema strings)
            try:
                target = ast.unparse(node)
                if len(target) < 200:
                    out_lines.append(target)
            except Exception:
                pass
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _function_signature(node)
            doc = ast.get_docstring(node)
            out_lines.append(sig)
            if doc:
                out_lines.append(f'    """{doc}"""')
            out_lines.append("    ...")
            out_lines.append("")
        elif isinstance(node, ast.ClassDef):
            out_lines.append(f"class {node.name}{_class_bases(node)}:")
            cdoc = ast.get_docstring(node)
            if cdoc:
                out_lines.append(f'    """{cdoc}"""')
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out_lines.append("    " + _function_signature(item))
                    idoc = ast.get_docstring(item)
                    if idoc:
                        out_lines.append(f'        """{idoc}"""')
                    out_lines.append("        ...")
                elif isinstance(item, ast.AnnAssign):
                    try:
                        out_lines.append("    " + ast.unparse(item))
                    except Exception:
                        pass
            out_lines.append("")

    return "\n".join(out_lines)


def _function_signature(node) -> str:
    import ast
    args = ast.unparse(node.args) if node.args else ""
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"def {node.name}({args}){returns}:"


def _class_bases(node) -> str:
    import ast
    bases = [ast.unparse(b) for b in node.bases]
    return f"({', '.join(bases)})" if bases else ""


def _parse_response(text: str) -> ProposerOutput:
    rationale_match = re.search(r"<rationale>(.*?)</rationale>", text, re.DOTALL)
    strategy_match = re.search(r"<strategy_py>(.*?)</strategy_py>", text, re.DOTALL)
    if not rationale_match or not strategy_match:
        raise ValueError(
            f"Proposer response missing <rationale> or <strategy_py> tags. Got first 500 chars:\n{text[:500]}"
        )
    return ProposerOutput(
        new_strategy_src=strategy_match.group(1).strip() + "\n",
        rationale=rationale_match.group(1).strip(),
    )


# --------------------------- Anthropic implementation ---------------------------

class AnthropicProposer:
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        n_seeds: int = 3,
        max_tokens: int = 8000,
        system_prompt_template: str | None = None,
        domain: "Domain | None" = None,
    ):
        # Lazy import so the stub proposer works in environments without the SDK
        import anthropic
        key = _resolve_api_key()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Set it in env, or write the key to "
                f"{KEY_FILE_FALLBACK} (chmod 600)."
            )
        from researcher.domain import FINANCE as _FINANCE
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._n_seeds = n_seeds
        self._max_tokens = max_tokens
        self._domain = domain or _FINANCE
        # GEPA hook: callers can inject an evolved system-prompt template.
        # Falls back to the domain's default system prompt.
        self._system_prompt_template = system_prompt_template or self._domain.system_prompt

    def __call__(self, ctx: ProposerContext) -> ProposerOutput:

        # Use .replace() rather than .format() — harness source contains Python dict
        # literals (curly braces) that would confuse str.format's placeholder parser.
        system = (
            self._system_prompt_template
            .replace("{n_seeds}", str(self._n_seeds))
            .replace("{harness_files}", _read_harness_files(self._domain.harness_files))
        )

        from researcher.cross_session import find_similar_attempts, format_cross_session
        from researcher.attribution import format_attribution, get_attribution
        cross = find_similar_attempts(
            ctx.current_strategy_src, exclude_session_id=ctx.session_id, limit=8,
        )
        attribution_block = ""
        if ctx.parent_candidate_id is not None:
            attrs = get_attribution(ctx.parent_candidate_id)
            if attrs:
                attribution_block = (
                    "\n## Per-signal attribution on parent (which signals are pulling weight?)\n"
                    + format_attribution(attrs)
                )
        metric_label = self._domain.primary_metric_name
        metric_fmt = "{:" + self._domain.primary_metric_format + "}"
        parent_metric_str = (
            metric_fmt.format(ctx.parent_metric) if ctx.parent_metric is not None
            else "(seed strategy, no parent)"
        )
        user_parts = [
            f"## Iteration {ctx.iteration}",
            f"\nParent mean {metric_label} across {self._n_seeds} seeds: {parent_metric_str}",
            "\n## Recent attempts (most recent first)",
            _format_recent(ctx.recent_decided),
            "\n## Recent failure traces (subprocess logs from crashed/discarded attempts)",
            _fetch_failure_traces(ctx.recent_decided),
            "\n## Cross-session memory (similar attempts from prior sessions)",
            format_cross_session(cross),
            attribution_block,
            f"\n## Current {self._domain.strategy_file.name}",
            f"```python\n{ctx.current_strategy_src}\n```",
            "\nPropose the next mutation. Output ONLY the two tagged blocks specified.",
        ]
        user = "\n".join(user_parts)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},  # harness is stable across calls
                },
            ],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_response(text)


# --------------------------- Stub implementation ---------------------------

class StubProposer:
    """
    Deterministic-RNG hyperparameter perturbation. Picks one field of StrategyParams
    and jitters it by a small amount. Used as a baseline + for offline loop tests.
    """
    name = "stub"

    _PERTURBATIONS: list[tuple[str, Callable[[random.Random, str], str]]] = []  # populated below

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def __call__(self, ctx: ProposerContext) -> ProposerOutput:
        src = ctx.current_strategy_src
        # Surface aligned with strategy.py post-#7 refactor: typed scalar fields only.
        # The richer move types (add/remove signal, change combine_mode, edit weights dict)
        # require structured edits that real LLM proposers do — out of scope for the stub.
        choices = [
            ("long_pct", lambda r: round(r.uniform(0.1, 0.3), 2)),
            ("short_pct", lambda r: round(r.uniform(0.1, 0.3), 2)),
            ("rebalance_days", lambda r: r.choice([5, 10, 21, 63])),
            ("gross_leverage", lambda r: round(r.uniform(0.5, 1.5), 2)),
        ]
        field, sample = self._rng.choice(choices)
        new_value = sample(self._rng)

        # Replace `field: type = old` with `field: type = new`. Strict regex over
        # the StrategyParams block keeps the perturbation surgical.
        pattern = re.compile(rf"^(\s*{field}\s*:[^=]+=\s*)([^\s#]+)", re.MULTILINE)
        match = pattern.search(src)
        if not match:
            raise RuntimeError(f"StubProposer could not locate field {field} in strategy.py")
        new_src = pattern.sub(rf"\g<1>{new_value!r}", src, count=1)

        rationale = f"[stub] perturb {field} -> {new_value!r}"
        return ProposerOutput(new_strategy_src=new_src, rationale=rationale)


def build_proposer(name: str, **kwargs) -> Proposer:
    if name == "anthropic":
        return AnthropicProposer(**kwargs)
    if name == "stub":
        return StubProposer(**kwargs)
    raise ValueError(f"Unknown proposer: {name!r}")
