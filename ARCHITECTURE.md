# Architecture

How quant-researcher is put together, layer by layer, and the design decisions
behind each piece.

## The shape

```
                      ┌───────────────────────────────────────┐
                      │  Meta-loop (GEPA scaffold)            │
                      │   evolves the proposer's system prompt │
                      └────────────────────┬──────────────────┘
                                           │ swaps prompt
                                           ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Outer research loop (researcher/loop.py)                    │
   │                                                              │
   │   for i in iterations:                                       │
   │     1. Proposer reads parent + recent + cross-session +     │
   │        traces + attribution    →  candidate strategy diff   │
   │     2. Runner executes candidate × N seeds                  │
   │        (with adaptive replication / curriculum)             │
   │     3. Decision gate (paired-CI): keep / discard / inconc   │
   │     4. On KEEP: optional per-signal attribution             │
   │     5. Provenance to runs.db                                │
   └──────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Locked harness (harness/)                                    │
   │   data ↔ signals ↔ backtest ↔ metrics                        │
   │   NEVER MODIFIED — defines the question; lock = comparability │
   └──────────────────────────────────────────────────────────────┘
```

## Layer 0 — Substrate

`runs.db` is sqlite + WAL + FTS5. Single source of truth. Tables:

| Table | Purpose |
|---|---|
| `sessions` | one row per research session (proposer, started, ended, notes) |
| `candidates` | one row per strategy proposal (diff, rationale, decision, accepted) |
| `trials` | one row per backtest execution (candidate × seed); status, metrics_json |
| `candidates_fts` | FTS5 over candidate diff + rationale for cross-session retrieval |
| `prompts` | GEPA-evolved system-prompt variants (content, hash, fitness) |
| `session_prompts` | join: which prompt drove which session |
| `gepa_generations` | GEPA evolution trace (run × generation × prompt × fitness) |
| `signal_attribution` | per-signal standalone Sharpe for accepted candidates |

WAL means observability tools can read concurrently with a running session.
Foreign keys are enforced.

Subprocess isolation: every backtest runs as a fresh Python process
(`researcher/_backtest_worker.py`). A buggy candidate can't corrupt sibling
trials or the parent process. This was load-bearing in early sessions where
LM-generated code occasionally hung or NaN'd.

## Layer 1 — Locked harness

`harness/` is the question. The agent never modifies anything in here.

- **`data.py`** — Panel loader. Reads cached parquet bars into wide DataFrames.
  Reconstructs **point-in-time S&P 500 membership** (`load_panel(apply_membership=True)`)
  from `data/sp500_universe.txt` (current pinned constituents) + `data/sp500_changes.parquet`
  (historical add/remove events from Wikipedia). Survivorship-bias-free.

- **`signals.py`** — Typed signal library. Each entry is a pure function
  `(PriceData) → wide DataFrame`. The agent composes from this library; it doesn't
  invent new signals from scratch unless it edits `strategy.py` body directly.

- **`backtest.py`** — Vectorized walk-forward. In-sample 2010-2021 (visible to
  the agent for context), OOS 2022-2024 (the gating window). One-day signal lag
  prevents look-ahead. Transaction cost = `cost_bps × |Δweight|/2`. Optional
  seeded sub-universe (`80%` of tickers) for paired-CI replication.

- **`metrics.py`** — Sharpe, Calmar, max-DD, annualized return/vol, hit rate,
  turnover. Pure functions of return series. `primary_metric = oos.sharpe`.

The walk-forward split + sub-universe sampling is the entire substrate for
paired-CI: same seed → same sub-universe → directly comparable parent vs candidate.

## Layer 2 — Strategy (editable surface)

`strategy.py` is what proposers mutate. After the **typed signal library refactor (#7)**:

```python
@dataclass(frozen=True)
class StrategyParams:
    enabled_signals: tuple[str, ...]    # subset of SIGNAL_LIBRARY keys
    weights: dict[str, float]           # weight per enabled signal
    signal_kwargs: dict[str, dict]      # per-signal hyperparameter overrides
    combine_mode: str                   # 'linear' | 'sign_vote' | 'sharpe_weighted'
    long_pct: float
    short_pct: float
    rebalance_days: int
    gross_leverage: float
```

Typed mutation surface means a proposer's structural move ("add `acceleration`
to enabled_signals with weight 0.4") is just an edit to the dataclass — no
free-form code change needed. The proposer can still rewrite `generate_signals`
when needed (e.g., to add a risk overlay), but typed moves are the common path.

## Layer 3 — Researcher loop

`researcher/loop.py` orchestrates one session:

1. **Baseline** — `runner.baseline_trials()` runs the current `strategy.py` as the initial parent.
2. **Iterate**:
   - **Propose** — `ProposerContext` carries current strategy, recent decisions,
     cross-session matches, trace excerpts, per-signal attribution. Proposer
     returns `(new_strategy_src, rationale)`.
   - **Replicate-execute** — `runner.run_candidate()` writes the new strategy,
     runs N seeds in isolated subprocesses. With `parent_trials` provided,
     **adaptive replication / curriculum** (#5) kicks in: if seed-1 is clearly
     worse than parent's seed-1 (gap < -0.20), remaining seeds are skipped and
     an early `discard` decision is emitted.
   - **Decide** — `decide.decide(parent_trials, candidate_trials)` does the
     paired-CI math. Same-seed pairs only. Returns `keep` / `discard` / `inconclusive`
     with CI bounds.
   - **Persist + advance** — On `keep`: commit the new `strategy.py`, optionally
     run per-signal attribution. Else: revert `strategy.py` to parent backup.
     Either way: persist the candidate + trials + decision to `runs.db`.

The proposer interface (`Proposer` protocol):

```python
class Proposer(Protocol):
    name: str
    def __call__(self, ctx: ProposerContext) -> ProposerOutput: ...
```

Three implementations:

- **`StubProposer`** — seeded random hyperparameter perturbation. Noise floor.
- **`AnthropicProposer`** — `anthropic.Anthropic.messages.create` with system
  prompt cache. XML-tagged output (`<rationale>...</rationale><strategy_py>...</strategy_py>`).
- **`LocalProposer`** — POST to any OpenAI-compatible `/v1/chat/completions`
  endpoint (LM Studio, Ollama, vllm, llama.cpp server). Tolerates Markdown
  fence wrapping, missing close tags, leading prose. Retries once with a
  stricter format reminder on parse failure.

## Layer 4 — Decision gate

`researcher/decide.py` is the heart of the reasoning-quality measurement.
Paired t-test on `candidate_seed_metric - parent_seed_metric` across N seeds.
Decision:

- `ci_low > 0` → **keep**
- `ci_high < 0` → **discard**
- otherwise → **inconclusive** (logged but not accepted)
- any candidate seed crashed → **discard** (don't keep buggy strategies)

The strictness of `inconclusive` is deliberate. Without it, a hill-climber
would accept noise and drift into local pseudo-optima. Across 300 stub
iterations in session 1, the gate correctly rejected 99%+ of moves.

## Layer 5 — Cross-session memory (#3)

`researcher/cross_session.py` queries `candidates_fts` for prior attempts
matching the current strategy's enabled-signals tokens. Excludes the current
session by default. Ranks by `(accepted desc, |Δ| desc)` — kept-with-big-gains
first, then big-failures.

Surfaces in proposer prompts as:

```
## Cross-session memory (similar attempts from prior sessions)
  [KEEP    Δ=+0.1500] s3 #42: added vol_adjusted_momentum w=0.4 ...
  [DISCARD Δ=-0.2473] s1 #16: Adding `acceleration` ...
  [DISCARD Δ=-0.1547] s1 #11: Adding `acceleration` ...
```

The proposer can use this to avoid repeating known failures (anti-repetition
discipline is also baked into the system prompt) and to copy known winners.

## Layer 6 — GEPA-shaped scaffold (`researcher/gepa/`)

Genetic-Pareto Prompt Evolution. The infrastructure to evolve the proposer's
system prompt against measured hit-rate. Currently scaffolded with
deterministic mocks for the LLM-mediated parts; real GEPA plugs in by swapping
two classes.

- **`prompts.py`** — `Prompt` artifact (content + parent + source + fitness)
  with content-hash dedupe. `PromptRegistry` persists to `runs.db` (new
  `prompts` + `session_prompts` + `gepa_generations` tables — purely additive
  to the base schema).

- **`fitness.py`** — `FitnessScore` dataclass with 13 dimensions: primary
  `hit_rate`, plus `mean_kept_delta`, `n_kept`, `n_decided`, `n_inconclusive`,
  `n_crashes`, `terminal_metric`, `time_to_first_keep`, `wall_seconds`,
  **`oos_is_sharpe_gap`** (overfitting proxy), **`max_drawdown`**, **`turnover_annualized`**,
  **`stability_score`** (1 - normalized stddev across seeds). `pareto_components()`
  declares per-dim direction (higher-better vs lower-better).

  Two evaluators: `FitnessEvaluator` runs the actual research loop with the
  given prompt as the proposer's system prompt (real backtests); `SyntheticFitnessEvaluator`
  computes a deterministic fitness from prompt features (no backtests; for
  testing GEPA mechanics).

- **`pareto.py`** — `dominates()`, `frontier()`, `select_from_frontier(mode=...)`.
  Four selector modes: `first` (deterministic, for tests), `random` (exploration),
  `crowding` (NSGA-II diversity preservation), `scalarized` (weighted-sum
  default).

- **`mutate.py`** — `Mutator` protocol + `DeterministicMockMutator` with three
  text operators (keyword inject, section swap, length trim). Real GEPA hook
  documented as `class GEPAReflectiveMutator: ...`.

- **`compare.py`** — `compare_prompts(a, b, evaluator, seeds)` — paired-CI at
  the prompt grain. Same statistical machinery as the candidate-level gate.

- **`loop.py`** — `run_gepa(seed_prompts, mutator, evaluator, registry, ...)`
  with `selector_mode` choosing between single-best and Pareto-aware. Logs
  per-generation results to `gepa_generations`.

- **`_mock_proposer.py`** — Hashes `(prompt.content_hash, evaluator_seed,
  iteration)` to a stable RNG seed, picks a hyperparameter to perturb.
  Different prompts produce different mutation sequences. **Differentiates
  prompts by hash, not semantics** — the honest signal that real GEPA needs a
  real LLM to make prompt semantics matter.

## Layer 7 — Adaptive replication / curriculum (#5)

`researcher/runner.py` parametrized with `parent_trials` enables a cheap screen
after the first seed. If candidate seed-1 is more than 0.20 worse than parent
seed-1, remaining seeds are recorded as `skipped` and the runner returns an
early discard decision. Saves ~2/3 of backtest cost on obvious losers.

## Layer 8 — Per-signal attribution (post-keep)

`researcher/attribution.py` runs N standalone backtests after a KEEP — one per
enabled signal with all other weights zeroed. Records each signal's standalone
OOS Sharpe in `signal_attribution`. The next proposer's prompt then includes:

```
## Per-signal attribution on parent (which signals are pulling weight?)
  momentum_12_1 (w=1.00) standalone=+0.4500
  reversion_5d (w=0.50) standalone=-0.1000 ← drag (weighted but negative)
  lowvol_20d (w=0.30) standalone=+0.2000
```

Off by default (`--attribute` CLI flag) because each KEEP costs N additional
backtests when enabled.

## Layer 9 — Observability

- **`scripts/inspect_runs.py`** — snapshot CLI: list sessions, focus one for
  leaderboard + recent decisions.
- **`scripts/dashboard.py`** — live TUI using `rich`. Auto-refresh every 5s.
  Sessions table + focused-session detail (recent decisions + leaderboard).
  Read-only.
- **`scripts/report.py`** — generates a self-contained HTML file per session.
  Includes leaderboard, decision timeline (color-coded), lineage tree
  (parent→child indent), and a `<details>` collapsible per candidate showing
  rationale + per-seed trials + full diff. No external CSS/JS dependencies.

All three read through `researcher/reporting/queries.py` — single layer that
funnels DB-fetch logic and dataclasses for both TUI and HTML.

## Cross-cutting decisions

### Why subprocess isolation per backtest?

LLM-generated code can monkey-patch globals, NaN-poison numpy state, or fork
threads it doesn't join. A fresh Python process per backtest means the parent
loop is bulletproof. Cost: ~1s startup × 3 seeds × N iterations = ~5-10% of
total wall time. Worth it.

### Why paired-CI instead of bootstrap?

Paired t-test on N=3 seeds is conservative (wide CIs → many inconclusive
decisions) but unbiased given the locked harness's deterministic backtest +
seeded sub-universe. Bootstrap would need many seeds to converge; we want
tight cycle time.

### Why local-first with API as fallback?

The user has commodity hardware (single 8 GB GPU laptop). Local LLMs (4B-class
at Q3/Q4) fit and work but have limited reasoning depth. Anthropic API is the
escalation when local can't make progress — never the default.

### Why content-hash dedupe in PromptRegistry?

GEPA mutations can produce identical content under different operators
(e.g., `length_trim` is a no-op on prompts already trim). Dedupe ensures
`runs.db` has one canonical row per distinct prompt; provenance still
preserves the chain of mutation attempts.

### Why `inconclusive` as a separate outcome?

The 0% real-proposer hit rate observed in early sessions includes many
attempts whose Δ was tiny (within noise). Without the `inconclusive` bucket,
a single-scalar hill-climber would accept ~half of them — drifting parent
strategy on noise. Logging them separately keeps hit-rate honest AND retains
the data for cross-session memory.

## See also

- `HACKING.md` — extension guide
- `STATUS.md` — what's validated vs not
