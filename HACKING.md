# Hacking — extending quant-researcher

Patterns for adding to the system. Read `ARCHITECTURE.md` first for the layer
model.

## Add a new proposer

Implement the `Proposer` protocol from `researcher/proposer.py`:

```python
from researcher.proposer import ProposerContext, ProposerOutput

class MyProposer:
    name = "my_proposer"

    def __init__(self, **config):
        # Construct your inference client / load resources
        ...

    def __call__(self, ctx: ProposerContext) -> ProposerOutput:
        # ctx.current_strategy_src        — the current parent's strategy.py
        # ctx.recent_decided              — last 15 decisions in this session
        # ctx.iteration                   — 1-indexed
        # ctx.parent_metric               — parent's mean primary metric (domain-specific)
        # ctx.session_id                  — for cross-session-memory exclusion
        # ctx.parent_candidate_id         — for attribution lookup

        # Your job: produce (new_strategy_src, rationale)
        return ProposerOutput(
            new_strategy_src="...full new strategy.py contents...",
            rationale="...short justification...",
        )
```

Wire to the CLI by adding a `--proposer my_proposer` branch in
`scripts/run_research.py:main`.

### Things proposer prompts should include

If your proposer is LLM-backed, the user message construction in
`researcher/proposer.py:AnthropicProposer.__call__` is the reference. It
already includes (in order):

1. Iteration index + parent metric
2. Recent attempts (decisions + rationales)
3. Recent failure traces (subprocess logs from crashed/discarded trials)
4. Cross-session memory (similar attempts from prior sessions via FTS5)
5. Per-signal attribution on the parent (if available)
6. The full current strategy.py

Reuse the helpers in `researcher.proposer` and `researcher.cross_session` and
`researcher.attribution`.

### What an LLM proposer MUST output

Strict format the parser expects:

```
<rationale>
2-4 sentences. State the move, why, and what failure mode you're guarding against.
</rationale>

<strategy_py>
[The complete new contents of strategy.py. Must be valid Python. Must export
PARAMS and generate_signals(prices, params).]
</strategy_py>
```

`AnthropicProposer` parses both tags. `LocalProposer` tolerates Markdown fence
wrapping and a missing close tag (small models often truncate), and additionally
checks `domain.required_symbols`. The example above is the finance contract; the
required exports differ per domain (finance needs `generate_signals`; toy and
shakespeare need `build_model` / `build_optimizer`).

## Add a new domain

The framework ships three domains (`finance`, `toy_sklearn`, `shakespeare`). The
`Domain` dataclass in `researcher/domain.py` is the canonical example — read
`FINANCE`, `TOY_SKLEARN`, and `SHAKESPEARE` to see the pattern across three
concrete instances. (`SHAKESPEARE` is the most instructive: its "harness" is a
neural-net trainer and its metric is lower-better, negated to fit the gate.)

To add another domain:

1. **Create a `LOCKED` harness package** that mirrors `harness/` or `toy_harness/`:
   - `your_domain/data.py` — load whatever data the domain needs
   - `your_domain/metrics.py` — pure metric functions
   - `your_domain/evaluator.py` — produces a `Result` dataclass with a `primary_metric`
     property. Must support a `seed` parameter for paired-CI replication.

2. **Create an editable surface** (`your_strategy.py`):
   - Export `PARAMS` (frozen dataclass) and a builder/generator function.
   - The proposer mutates this file.

3. **Create a worker subprocess** (`researcher/_your_worker.py`):
   - Subprocess entrypoint that imports your strategy + harness, runs one eval,
     prints one JSON line. Must accept `<seed>` and optional `--params-overrides <json>`.
   - Patterns: `researcher/_backtest_worker.py` (finance) or `researcher/_sklearn_worker.py` (toy).

4. **Register the Domain** in `researcher/domain.py`:
   - Define `YOUR_SYSTEM_PROMPT` with `{n_seeds}` and `{harness_files}` placeholders.
     The `{harness_files}` slot is filled at runtime from the `harness_files` tuple below —
     list exactly the LOCKED files you want the proposer to read.
   - Define `YOUR_DOMAIN = Domain(name="your_name", strategy_file=..., parent_backup_file=...,
     harness_files=(...), worker_command=(...), primary_metric_name=..., primary_metric_format=...,
     system_prompt=YOUR_SYSTEM_PROMPT, required_symbols=("class HyperParams", "def build_model"))`
   - `required_symbols` are substrings every candidate must contain; the local-proposer
     parser rejects output missing them — a cheap guard against malformed mutations.
   - If your metric is **lower-better**, return `primary_metric = -metric` from the worker so
     the higher-better gate works unchanged (this is exactly how shakespeare handles `val_bpb`).
   - Add to the `DOMAINS` dict: `"your_name": YOUR_DOMAIN`

5. **Test it.** The `--domain your_name` CLI flag works automatically (it iterates `DOMAINS.keys()`).

6. **Reuse everything else as-is**: runner, loop, decide, runs, proposer, cross_session,
   attribution, gepa/*, reporting/*. All domain-agnostic now that the protocol is extracted.

## Swap the decision gate

`decide.decide(parent_trials, candidate_trials)` is the contract. Replace it
with anything that returns a `Decision` dataclass. Examples of alternative
gates worth exploring:

- **Adaptive alpha** — wider CI tolerance early in a session, tighter as
  iterations accumulate
- **Pareto gate** — keep iff Pareto-dominant in (Sharpe, max-DD, turnover),
  not just Sharpe-paired-better
- **Bayesian** — Beta-Binomial on win-rate, decision threshold on posterior
  probability of improvement
- **Crash-tolerant** — current gate hard-discards on any crash; could allow
  if N-1 seeds are clear keeps

`loop.py` is the only caller. The early-decision branch from `runner.py`
already short-circuits the gate for curriculum.

## Add a fitness dimension

`researcher/gepa/fitness.py:FitnessScore` is a frozen dataclass. To add a new
dim:

1. Add the field with a default of `0.0` (or appropriate zero for back-compat).
2. Add a `(value, higher_is_better)` entry in `pareto_components()`.
3. Populate the field in both `_score_from_session` (real evaluator) and
   `SyntheticFitnessEvaluator.evaluate` (synthetic evaluator).
4. If the dim should influence Pareto scalarization weights, update the
   default in `pareto._scalarize._WEIGHTS`.

Pareto selector + dominance code consume `pareto_components()` so they
automatically pick up the new dim. No other changes required.

## Swap the GEPA selector

Pass `selector_mode='pareto_scalarized' | 'pareto_random' | 'pareto_crowding'`
to `run_gepa()`. To add a new mode:

1. Implement it in `researcher/gepa/pareto.py:select_from_frontier`.
2. Add the branch in `researcher/gepa/loop.py` that recognizes the mode prefix.

## Add a real GEPA reflective mutator

Currently `DeterministicMockMutator` does text-level operations on the prompt.
Real GEPA reads execution traces and proposes targeted mutations via an LLM:

```python
# researcher/gepa/mutate.py — placeholder swap point

class GEPAReflectiveMutator:
    name = "gepa_reflective"

    def __init__(self, model: str, registry: PromptRegistry):
        self.model = model
        self.registry = registry

    def mutate(self, parent: Prompt, n: int) -> list[Prompt]:
        # 1. Pull execution traces from runs.db for sessions driven by `parent`
        # 2. Identify failure modes (high crash rate? many inconclusive? specific
        #    mutation classes consistently fail?)
        # 3. Generate N candidate prompt revisions via LLM, prompted with the
        #    parent prompt + failure summary + revision instructions
        # 4. registry.register each child with source='gepa:reflective'
        ...
```

Same interface as the mock. Drop-in replacement.

## Inspect / debug a session

```bash
# Snapshot
uv run python -m scripts.inspect_runs --session N

# Live TUI
uv run python -m scripts.dashboard --session N --refresh 2

# Self-contained HTML report (browse offline)
uv run python -m scripts.report --session N --out reports/session_N.html
```

For ad-hoc queries, `researcher/reporting/queries.py` is the cleanest entry
point — it returns dataclasses, not raw sqlite rows.

## Don't extract Domain prematurely

Two concrete instances are needed before abstracting. The temptation to write
a `Domain` protocol from one (finance) is real but the result will be wrong —
the second domain's needs will reveal what the protocol *actually* requires,
and you'll either back-patch the abstraction or build around it.

Rules of thumb for when to extract:
- You're about to copy-paste a ~50-line block from `researcher/loop.py` for
  the second domain → extract.
- You're about to add a third domain → extract before that.
- Two domains share <50% of their code → don't extract yet; the abstraction
  isn't there.

## House rules

- `harness/` and any `LOCKED` harness for a new domain are never modified.
  Touching them invalidates the prior runs in `runs.db`.
- Tests are run with `uv run python -m pytest tests/ -q`. All must pass before
  merging a change.
- New deepenings should add a test file + a section to `ARCHITECTURE.md`.
- `runs.db` is in `.gitignore` and should stay there. The schema is the
  source of truth; data is ephemeral.
