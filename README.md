# quant-researcher

A local-first **finance Researcher specialist** with an instrumented
self-improvement loop. An agent (local LLM or Anthropic) proposes mutations to
a multi-signal equity strategy, the loop runs paired-CI gated backtests against
a locked walk-forward harness, and accepted mutations become the new parent for
the next iteration. Everything is provenance-logged so you can compare
proposers, prompts, and selectors by measured hit-rate.

The point isn't to discover an alpha — the point is to **measure reasoning
quality on a fixed benchmark**.

## At a glance

| | |
|---|---|
| Source LOC | ~6,700 |
| Tests | 92 (all passing) |
| Domains | **3** — `finance` (S&P 500 multi-signal blend), `toy_sklearn` (GradientBoostingClassifier on digits), `shakespeare` (tiny GPT character-level pretraining on tiny-shakespeare). Domain protocol validated across three independent specialist instances. |
| Validated harness (finance) | S&P 500 daily 2010-2024, point-in-time membership, vectorized walk-forward backtest, paired-CI gate |
| Validated harness (toy) | sklearn `digits` dataset, stratified K-fold CV, fast eval cycle (~2-3s/seed) |
| Validated harness (shakespeare) | tiny-shakespeare 1MB char corpus, vanilla pytorch GPT, val_bpb metric (negated for higher-is-better convention), fast eval cycle (~3-5s/seed on RTX 4070) |
| Proposers | Stub (random), Anthropic (Claude), Local (any OpenAI-compatible HTTP, e.g. LM Studio / Ollama) |
| Meta-loop | GEPA-shaped scaffold (prompt registry, Pareto selector, multi-dim fitness) with deterministic mocks for the LLM-mediated bits |
| Observability | Live TUI dashboard + self-contained HTML reports |
| Open question | Has any proposer been observed producing real KEEPs on this domain? Not yet. See `STATUS.md`. |

## Quick start

```bash
uv sync --extra dev

# One-time (finance only): download S&P 500 universe + bars + point-in-time membership
uv run python -m data.prepare
uv run python -m data.prepare_membership

# toy_sklearn domain — no data download needed, sklearn ships with digits
# Defaults to --domain finance; pass --domain toy_sklearn or --domain shakespeare to switch.
uv run python -m scripts.run_research --domain toy_sklearn --proposer stub --iterations 10 --seeds 1,2,3

# shakespeare domain — downloads tiny-shakespeare on first run (~1MB)
uv run python -m scripts.run_research --domain shakespeare --proposer stub --iterations 10 --seeds 1,2,3

# Finance, stub proposer (no LLM, deterministic random perturbations)
uv run python -m scripts.run_research --domain finance --proposer stub --iterations 20 --seeds 1,2,3

# Local LLM via LM Studio / Ollama (OpenAI-compatible HTTP)
uv run python -m scripts.run_research --domain finance --proposer local \
    --base-url http://localhost:1234/v1 \
    --model qwopus3.5-9b-coder-mtp \
    --iterations 100 --seeds 1,2,3

# Anthropic API (set ANTHROPIC_API_KEY or write to ~/.anthropic_key)
uv run python -m scripts.run_research --domain finance --proposer anthropic \
    --iterations 100 --seeds 1,2,3

# Observe live
uv run python -m scripts.dashboard               # TUI, 5s refresh
uv run python -m scripts.inspect_runs            # snapshot
uv run python -m scripts.report --session 1      # self-contained HTML report

# Evolve the proposer's system prompt (GEPA scaffold; uses synthetic fitness by default)
uv run python -m scripts.evolve_prompt --fitness synthetic --generations 3 --children 3
```

## What's in here

```
researcher/domain.py    Domain protocol — selects which harness/strategy/worker drives a run

# === Finance specialist ===
harness/                LOCKED — defines the question (finance)
  backtest.py           vectorized walk-forward backtester
  metrics.py            Sharpe, Calmar, max-DD, turnover, hit rate
  data.py               panel loader + point-in-time S&P 500 membership
  signals.py            typed signal library (momentum, reversion, lowvol, ...)

strategy.py             EDITABLE — finance editable surface
                        Composes signals from harness/signals.py; typed surface

data/
  prepare.py            S&P 500 universe + per-ticker bar download (yfinance)
  prepare_membership.py historical membership changes (Wikipedia)

# === Toy sklearn specialist ===
toy_harness/            LOCKED — defines the question (toy_sklearn)
  evaluator.py          stratified K-fold CV on sklearn digits
  data.py               dataset loader (sklearn.datasets.load_digits)
  metrics.py            accuracy + f1_macro

toy_strategy.py         EDITABLE — toy editable surface
                        HyperParams dataclass + build_model factory

# === Shared infrastructure (domain-agnostic) ===
researcher/             The loop infrastructure
  runs.py               sqlite + FTS5 provenance (candidates, trials, sessions, prompts, attribution)
  decide.py             paired-CI gate (per-seed t-test)
  proposer.py           AnthropicProposer + StubProposer + ProposerContext
  local_proposer.py     OpenAI-compatible HTTP wrapper (LM Studio / Ollama)
  runner.py             subprocess-isolated backtest execution + adaptive replication
  loop.py               outer orchestrator
  cross_session.py      FTS5 retrieval across sessions for proposer prompts
  attribution.py        per-signal standalone Sharpe (post-accept diagnostic)
  gepa/                 GEPA-shaped scaffold for evolving proposer prompts
    prompts.py          Prompt artifact + registry
    fitness.py          multi-dim FitnessScore (hit_rate, stability, DD gap, ...)
    pareto.py           dominance + frontier + selectors
    mutate.py           DeterministicMockMutator (real GEPA hook in docstring)
    compare.py          paired-CI prompt-vs-prompt comparison
    loop.py             GEPA outer evolution loop
    _mock_proposer.py   prompt-hash-keyed deterministic proposer for testing
  reporting/queries.py  shared read-only DB queries (TUI + HTML)

scripts/
  run_research.py       main research-loop entrypoint
  inspect_runs.py       runs.db snapshot CLI
  dashboard.py          live TUI (rich)
  report.py             standalone HTML report generator
  evolve_prompt.py      GEPA-scaffold entrypoint

tests/                  72 tests
```

## Design principles

- **Locked harness, editable strategy.** The agent never touches the eval — that
  guarantees runs are comparable. Same lock pattern applies to any future
  domain.
- **Paired-CI decision gating.** Same seed = same sub-universe. A candidate must
  beat its parent with a CI that strictly excludes zero — otherwise it's
  `inconclusive` (logged but not accepted). Stops the loop from drifting on
  noise.
- **Provenance over abstraction.** Every trial, every decision, every prompt
  variant goes into `runs.db`. Reasoning improvements are measured by
  hit-rate-against-prior-versions, not vibes.
- **Domain abstraction extracted from two concrete instances.** `researcher/domain.py`
  defines the `Domain` dataclass bundling strategy file, harness files, worker command,
  metric format, and system prompt. Both finance and toy_sklearn use the same loop +
  decision gate + GEPA scaffold; only the Domain differs. See `HACKING.md` for the
  "add a third domain" walkthrough.

## Documentation

- `ARCHITECTURE.md` — full design walkthrough, every layer, every deepening
- `HACKING.md` — extension guide: add a proposer, add a domain, swap the gate
- `STATUS.md` — honest current state, what's validated, open questions

## License

MIT (or whatever the workspace defaults to; not formally licensed yet).
