"""
Domain protocol — the abstraction that lets one Researcher framework drive
multiple concrete specialists.

A `Domain` bundles the domain-specific paths, subprocess command, and the
system-prompt template that the framework needs to drive a research loop.
Everything above this layer (runs.db, decision gate, GEPA scaffold, cross-session
memory, attribution, observability) is domain-agnostic and consumes a Domain
instance as a parameter.

This protocol was extracted from TWO concrete instances (finance + toy_sklearn).
Don't extract from one — see HACKING.md "Don't extract Domain prematurely."
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class Domain:
    """
    One concrete specialist. The framework selects between Domain instances via
    the `--domain` CLI flag; everything downstream is parameterized.
    """
    name: str                            # e.g. 'finance', 'toy_sklearn'
    strategy_file: Path                  # the editable file the agent mutates
    parent_backup_file: Path             # where runner stashes the parent for revert
    harness_files: tuple[Path, ...]      # LOCKED files included in proposer system prompt
    worker_command: tuple[str, ...]      # base argv for subprocess (e.g. ['python','-m','...'])
    primary_metric_name: str             # display label for logs, e.g. 'OOS Sharpe'
    primary_metric_format: str           # python format spec, e.g. '+.4f'
    system_prompt: str                   # domain-specific system prompt template
    required_symbols: tuple[str, ...] = ()  # substrings that MUST appear in a candidate strategy


# ---------------------------------------------------------------------------
# Finance domain (the original)
# ---------------------------------------------------------------------------

FINANCE_SYSTEM_PROMPT = """You are a quantitative-research agent iterating on a multi-signal equity strategy.

The objective is a measured improvement in OOS (2022-2024) annualized Sharpe, gated by a
paired-CI decision rule across {n_seeds} replicate sub-universes. Marginal improvements that
fall inside noise will be classified 'inconclusive' and discarded — propose mutations large
enough to move the needle, but not so large they crash.

## The LOCKED harness (do not try to change — these files are read-only)

You are reasoning inside this evaluation. Understanding it is critical to proposing useful moves.

{harness_files}

## What you may change

A single file: strategy.py. Its module-level `PARAMS` and `generate_signals()` are the contract
that the harness depends on. You may add fields to StrategyParams, add new signals, change the
combination, change portfolio construction. Don't import new third-party packages.

## Mutation categories

- Structural: add/remove a signal; swap `combine_mode`; toggle a signal's lookback.
- Compositional: rebalance the `weights` across enabled signals.
- Portfolio: long/short widths, rebalance cadence, leverage.
- Beyond the typed surface: refactor `generate_signals` to add risk overlays (vol target,
  drawdown cap, beta neutralization), but prefer typed moves when they cover the idea.

## Anti-repetition discipline (critical)

Before proposing your mutation, READ "Recent attempts" and "Cross-session memory" and STRICTLY:
1. Do not repeat a discarded move. If a prior attempt with negative Δ tried the same change,
   pick a structurally different one.
2. Rotate move classes: different signal additions, different signal removals, the three
   combine_modes, portfolio dimensions, per-signal hyperparameter overrides, risk overlays.
3. Cross-session memory beats first-principles reasoning when prior data is decisive.

## Output format

Reply with EXACTLY this structure and nothing else:

<rationale>
2-4 sentences. State the move, why you expect improvement, and what failure mode you're guarding against.
</rationale>

<strategy_py>
[The complete new contents of strategy.py. Must be valid Python. Must export PARAMS and
generate_signals(prices, params).]
</strategy_py>
"""

FINANCE = Domain(
    name="finance",
    strategy_file=REPO_ROOT / "strategy.py",
    parent_backup_file=REPO_ROOT / "strategy.py.parent_backup",
    harness_files=(
        REPO_ROOT / "harness" / "backtest.py",
        REPO_ROOT / "harness" / "metrics.py",
        REPO_ROOT / "harness" / "data.py",
        REPO_ROOT / "harness" / "signals.py",   # signal library — the typed move surface
    ),
    worker_command=(sys.executable, "-m", "researcher._backtest_worker"),
    primary_metric_name="OOS Sharpe",
    primary_metric_format="+.4f",
    system_prompt=FINANCE_SYSTEM_PROMPT,
    required_symbols=("class StrategyParams", "def generate_signals"),
)


# ---------------------------------------------------------------------------
# Toy domain: sklearn hyperparameter tuning on `digits`
# ---------------------------------------------------------------------------

TOY_SYSTEM_PROMPT = """You are a hyperparameter-tuning agent for a sklearn GradientBoostingClassifier on the digits dataset.

The objective is a measured improvement in mean cross-validation accuracy, gated by a paired-CI
decision rule across {n_seeds} replicate seeds. The eval is fast (~3-5s per seed), so iterate
freely — but inconclusive mutations are discarded the same as in any other domain.

## The LOCKED harness (do not try to change — these files are read-only)

{harness_files}

## What you may change

A single file: toy_strategy.py. Its module-level `PARAMS` is a HyperParams dataclass; its
`build_model()` returns the configured estimator. You may tune any HyperParams field, or
refactor `build_model()` to use a different sklearn estimator entirely (still must be a
classifier with `fit`/`predict`).

## Mutation categories

- Tune hyperparameters: n_estimators, learning_rate, max_depth, min_samples_split,
  min_samples_leaf, subsample, max_features.
- Swap estimator: GradientBoostingClassifier → HistGradientBoostingClassifier, RandomForestClassifier,
  ExtraTreesClassifier, etc.
- Add preprocessing: StandardScaler, PCA via sklearn.pipeline.Pipeline.
- Feature engineering: polynomial features (sparingly), kernel approximations.

## Anti-repetition discipline (critical)

Before proposing your mutation, READ "Recent attempts" and "Cross-session memory" and STRICTLY:
1. Do not repeat a discarded move. If a prior attempt with negative Δ tried the same change,
   pick a structurally different one.
2. Rotate move classes: different hyperparameter fields, different estimators, different
   preprocessing pipelines.
3. Cross-session memory beats first-principles reasoning when prior data is decisive.

## Output format

Reply with EXACTLY this structure and nothing else:

<rationale>
2-4 sentences. State the move, why you expect improvement, and what failure mode you're guarding against.
</rationale>

<strategy_py>
[The complete new contents of toy_strategy.py. Must be valid Python. Must export PARAMS and
build_model(params).]
</strategy_py>
"""

TOY_SKLEARN = Domain(
    name="toy_sklearn",
    strategy_file=REPO_ROOT / "toy_strategy.py",
    parent_backup_file=REPO_ROOT / "toy_strategy.py.parent_backup",
    harness_files=(
        REPO_ROOT / "toy_harness" / "evaluator.py",
        REPO_ROOT / "toy_harness" / "data.py",
        REPO_ROOT / "toy_harness" / "metrics.py",
    ),
    worker_command=(sys.executable, "-m", "researcher._sklearn_worker"),
    primary_metric_name="CV accuracy",
    primary_metric_format=".4f",
    system_prompt=TOY_SYSTEM_PROMPT,
    required_symbols=("class HyperParams", "def build_model"),
)


# ---------------------------------------------------------------------------
# Shakespeare domain: tiny GPT character-level pretraining on tiny shakespeare
# ---------------------------------------------------------------------------

SHAKESPEARE_SYSTEM_PROMPT = """You are an optimizer-and-architecture-tuning agent for a tiny GPT
trained on tiny shakespeare with a character-level tokenizer.

The objective is a measured improvement in val_bpb (bits-per-byte on the held-out validation
split). val_bpb is LOWER-better, but the framework's decision gate is HIGHER-better — the
trainer reports primary_metric = -val_bpb so that a smaller bpb produces a larger primary_metric.
Paired-CI across {n_seeds} replicate seeds gates accept/discard the same as in other domains.

A single training cycle is 30-90 seconds (CPU or GPU). Iterate freely.

## The LOCKED harness (do not try to change — these files are read-only)

{harness_files}

## What you may change

A single file: shakespeare_strategy.py. Its module-level `PARAMS` is a HyperParams dataclass;
`build_model(vocab_size, params)` returns a model; `build_optimizer(model, params)` returns
an optimizer.

## Mutation categories

- Model architecture: n_layer, n_head, n_embd, block_size, dropout.
- Training schedule: learning_rate, warmup_iters, max_iters, grad_clip, weight_decay.
- Optimizer choice: 'adamw' or 'hyperball'.
- For hyperball: hyperball_lr, hyperball_beta (state retention), hyperball_update_rule
  ('retract' | 'slerp'), hyperball_matrix_ulmo ('gram_ns' | 'sign' | 'colnorm' |
  'frobenius'). Matrix params get the selected ULMO; vector params (LayerNorm)
  always use FrobeniusULMO (only ULMO safe on 1D tensors).
- Architecture refactor: swap TinyGPT for a different model (custom transformer variant,
  RWKV, S4, etc.) — must still produce (logits, loss) on forward(idx, targets).

## Anti-repetition discipline (critical)

Before proposing your mutation, READ "Recent attempts" and "Cross-session memory" and STRICTLY:
1. Do not repeat a discarded move. If a prior attempt with negative Δ tried the same change,
   pick a structurally different one.
2. Rotate move classes: different optimizer arms, different architecture knobs, different
   schedules — diversity of attempts beats depth on one idea.
3. Cross-session memory beats first-principles reasoning when prior data is decisive.

## Output format

Reply with EXACTLY this structure and nothing else:

<rationale>
2-4 sentences. State the move, why you expect val_bpb to drop, and what failure mode you're guarding against (NaN, OOM, divergence).
</rationale>

<strategy_py>
[The complete new contents of shakespeare_strategy.py. Must be valid Python. Must export
PARAMS, build_model(vocab_size, params), and build_optimizer(model, params).]
</strategy_py>
"""

SHAKESPEARE = Domain(
    name="shakespeare",
    strategy_file=REPO_ROOT / "shakespeare_strategy.py",
    parent_backup_file=REPO_ROOT / "shakespeare_strategy.py.parent_backup",
    harness_files=(
        REPO_ROOT / "shakespeare_harness" / "trainer.py",
        REPO_ROOT / "shakespeare_harness" / "model.py",
        REPO_ROOT / "shakespeare_harness" / "data.py",
        REPO_ROOT / "shakespeare_harness" / "metrics.py",
    ),
    worker_command=(sys.executable, "-m", "researcher._shakespeare_worker"),
    primary_metric_name="−val_bpb",         # negated bpb; higher-better convention
    primary_metric_format="+.4f",
    system_prompt=SHAKESPEARE_SYSTEM_PROMPT,
    required_symbols=("class HyperParams", "def build_model", "def build_optimizer"),
)


DOMAINS: dict[str, Domain] = {
    "finance": FINANCE,
    "toy_sklearn": TOY_SKLEARN,
    "shakespeare": SHAKESPEARE,
}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise ValueError(
            f"Unknown domain {name!r}. Available: {sorted(DOMAINS.keys())}"
        )
    return DOMAINS[name]
