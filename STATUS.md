# Project status

Honest read on what's been built, what's been validated, and what hasn't.
Last updated 2026-05-29.

## Update 2026-05-29 — proposer domain-leak bug fixed

Code review found a real bug: both proposers (`AnthropicProposer`, `LocalProposer`)
injected the **finance** harness into the `{harness_files}` slot of the system prompt
regardless of `--domain`. `domain.harness_files` existed and was tested for file
existence, but was never threaded into the prompt builder (`_read_harness_files()` read
a module-level finance constant). Effect: on `toy_sklearn` and `shakespeare` runs the
model got the correct task description but **finance** harness code as its "locked
harness" reference.

Implication: the 0-KEEP results on **session 4 (toy_sklearn)** and **session 8
(shakespeare)** were run with this confound and should not be treated as clean evidence
about those domains. Finance sessions (1, 2) are unaffected — they got the right harness.
Fix threads `domain.harness_files` through both proposers; `signals.py` was added to
`FINANCE.harness_files` so the finance proposer still sees the signal library. Decision
reasons in `decide.py` were also de-finance-ified ("primary-metric delta" instead of
"OOS-Sharpe delta"). All 95 tests still pass. A re-run of toy_sklearn/shakespeare with a
real proposer is now warranted before drawing model-capability conclusions on those domains.

## What's built (validated end-to-end)

| Subsystem | LOC | Tests | Status |
|---|---:|---:|---|
| Locked harness (data + signals + backtest + metrics) | ~480 | covered indirectly via end-to-end | ✅ Working — 498/503 S&P 500 tickers cached, point-in-time membership reconstructed, vectorized backtest runs in ~2s |
| Editable strategy (typed signal-library composition) | ~135 | indirect | ✅ Working — composes signals from library, supports linear/sign_vote/sharpe_weighted combine modes |
| Decision gate (paired-CI) | 128 | 7 | ✅ Working — verified on 9-run synthetic test cases + 39 real candidates across two sessions |
| Provenance store (runs.db + FTS5) | ~290 | indirect | ✅ Working — 8 tables, foreign keys enforced, WAL mode validated under concurrent reads |
| Stub proposer | 60 | indirect | ✅ Working — produced 7 keeps / 301 attempts (2.3% noise-floor) in early session |
| AnthropicProposer | ~80 | compile-only | ⚠️ Code present but **never exercised against the API** in any session |
| LocalProposer (LM Studio / Ollama / vllm) | ~150 | indirect | ✅ Working — verified end-to-end with `opus4.7-gods.ghost.codex-4b.gguf` via LM Studio; two real sessions ran the full loop |
| Runner (subprocess isolation + adaptive replication) | ~165 | 4 | ✅ Working — curriculum tested on 4 synthetic-RNG cases |
| Outer loop | ~170 | indirect | ✅ Working — drove sessions 1 and 2 end-to-end |
| Cross-session memory | ~145 | 9 | ✅ Working — verified surfacing session-1 discards to session-2 proposer |
| Per-signal attribution | ~130 | 5 | ✅ Working — DB persistence + format helpers tested; **no real KEEPs yet so never run in production** |
| GEPA scaffold (prompts + mutate + compare + fitness + loop) | ~785 | 17 | ⚠️ Scaffold complete with deterministic mocks. Synthetic-fitness end-to-end works. **Never run against a real LLM** (would need API access). |
| Pareto selector | ~145 | 10 | ✅ Working — 10 tests cover dominance, frontier, 4 selector modes |
| Multi-dim FitnessScore | (extends fitness.py) | 4 | ✅ Working — extracts 4 new dims (oos/is gap, max-DD, turnover, stability) from trials.metrics_json |
| Observability (queries + TUI + HTML report) | ~530 | 6 | ✅ Working — generated reports for sessions 1 & 2, TUI smoke-rendered |
| **Domain protocol** | ~150 | 7 | ✅ Working — extracted from finance + toy_sklearn + shakespeare (three concrete instances). `Domain` dataclass bundles strategy file, harness, worker command, metric format, system prompt, required_symbols. |
| **Toy sklearn specialist** | ~190 | 6 | ✅ Working — GradientBoostingClassifier on digits, stratified K-fold CV, paired-CI substrate (different seeds → different splits). Baseline produces 96.14% CV accuracy in session 3. |
| **Shakespeare specialist** | ~620 | 7 | ✅ Working — tiny GPT (vanilla pytorch, ~80 LOC), char-level tokenizer over tiny-shakespeare, val_bpb metric. Default config (n_layer=4, n_embd=128, 500 iters) trains in 3-4s on RTX 4070. Baseline produces val_bpb=3.52 in session 5. Demonstrates framework on actual neural-net training (not just hyperparameter tuning). |

**Totals:** ~6,700 source LOC, 92 tests, all passing.

## What's NOT validated

**The big open question — does any proposer beat baseline on this loop?**

We've never seen a real KEEP. Two sessions with the 4B local model:

| Session | Domain | Proposer | Iter completed | Real proposer keeps | Notes |
|---|---|---|---:|---:|---|
| 1 | finance | `local` (opus4.7-gods.ghost.codex-4b.gguf) | 29 | **0** | Baseline at +0.19 OOS Sharpe; every mutation regressed |
| 2 | finance | same | 10 | **0** | Cross-session memory + anti-repetition discipline added; still 0 |
| 3 | toy_sklearn | `stub` (0 iter) | 0 | (baseline only) | Validates Domain wiring end-to-end. 96.14% CV accuracy baseline. |
| 4 | toy_sklearn | `local` (opus4.7 4B) | 28 | **0** | Domain-leak bug exposed during run (now fixed); 12 inconclusive, 15 discard, 7 mid-run crashes. Bug-fix cluster cleaned 7 hardcoded 'OOS Sharpe' strings across proposers/dashboard/report. |
| 5 | shakespeare | `stub` (0 iter) | 0 | (baseline only) | First training-domain session through framework. val_bpb=3.522 baseline. Cycle time 3-4s/seed on RTX 4070 — framework now runs across hyperparameter-tuning, equity-research, AND neural-net-pretraining tasks. |

Plus the 300-iter stub baseline (session 0) which produced 6 lucky keeps via
random perturbation — but that's noise-floor behavior, not reasoning.

This means three things are *plausibly* still bottlenecks, in priority order:

1. **The 4B model genuinely can't reason about strategies.** Most likely
   explanation. Smaller models struggle with structured-output tasks that
   require domain reasoning. The rationales they produce are coherent but the
   actual mutations consistently regress.

2. **The seed strategy is near-optimal on this universe.** Plausible. The
   multi-signal blend with default weights gets +0.19 OOS Sharpe with
   point-in-time membership — that's decent for a long-short equity strategy
   on a single universe.

3. **The eval/gate is too strict.** Less likely — the gate is doing what it
   should (rejecting noise). But alpha=0.05 with N=3 seeds is genuinely tight;
   a wider CI tolerance might surface signal.

We **have not** disambiguated which. Two ways to disambiguate:

- **Sandbag the seed** — replace strategy.py defaults with a deliberately weak
  starting point. If even the 4B finds keeps from there, infrastructure is
  validated and the seed was indeed the issue. Cheap.
- **Anthropic baseline** — pay $15-25 to run a Claude session against the
  same seed. If Claude finds keeps where local can't, model capability is the
  bottleneck. Definitive.

Both have been on the table for several sessions; both have been deferred in
favor of more infrastructure work. That choice is the user's to make.

## Specific things never exercised in production

- **AnthropicProposer** — code compiles, prompt format identical to LocalProposer's
  validated path, but never actually called.
- **Per-signal attribution** — code + DB schema ready, but only triggers on
  KEEPs (we have 0).
- **GEPA loop** — synthetic mode end-to-end works; real mode (with LLM
  proposer/mutator) requires API access.
- **Pareto selector** — synthetic-fitness path tested; never used in a real
  evolution run.
- **Multi-dim fitness** — collected on every trial, never used by a selector
  in production.

## Inventory at a glance

```
$ find . -name '*.py' -not -path './.venv/*' -not -path '*/__pycache__/*' | wc -l
46 Python files

$ pytest tests/ -q
72 passed

$ ls reports/
session_1.html  session_2.html

$ sqlite3 runs.db '.schema' | grep -c 'CREATE TABLE\|CREATE VIRTUAL'
8 tables
```

## Path forward (when ready)

In rough order of unlock value vs effort:

1. **Run toy_sklearn with a real proposer** — 1 hour wall. The toy domain is
   designed to be tractable for a 4B model (hyperparameter tuning is well
   within their training distribution). If LocalProposer produces KEEPs here,
   infrastructure is fully validated and the open question becomes "is finance
   intrinsically harder, or is the seed near-optimal?"
2. **Sandbag the finance seed + relaunch local** — 30 min setup, ~1 hour wall.
   Disambiguation for finance specifically.
3. **Anthropic baseline on finance** — needs API key reachable from subprocesses
   (`~/.anthropic_key` works with current code). $15-25, ~2-3 hours wall.
   Definitive answer for the strong-seed-vs-weak-model question.
4. **Codegraph retrieval specialist** — re-scoped to MVP eval wrapper (~200 LOC).
   Adds a third specialist using an existing eval harness. Domain protocol is
   already extracted; adding a third just instantiates it.
5. **Real GEPA reflective mutator** — only worth doing once a proposer has
   demonstrated keeps. Otherwise we're optimizing prompts for a 0% outcome.

Each of these is unblocked from here — the infrastructure is ready.
