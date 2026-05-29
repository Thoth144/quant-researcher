"""
GEPALoop — outer evolution loop.

v1 = single-parent mutation only (no crossover). Each generation:
  1. From the current population, evaluate any unscored prompts.
  2. Pick the best by primary_metric as the next parent.
  3. Mutate it to create N children -> new population for next generation.
  4. Log everything to runs.db (gepa_generations table).

Swap-in for real GEPA: use a Pareto-aware selector instead of single-best.
The single-best selector is intentionally crude so the v1 scaffold stays
small and the test signal stays interpretable.

The 'gepa_run_id' returned can be used with PromptRegistry.generations() to
reconstruct the full evolution trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from researcher.gepa.fitness import Evaluator, FitnessScore
from researcher.gepa.mutate import Mutator
from researcher.gepa.pareto import select_from_frontier
from researcher.gepa.prompts import Prompt, PromptRegistry


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    prompts: list[Prompt]
    scores: dict[int, FitnessScore]   # prompt_id -> score
    parent_id: int                    # selected best for next gen


@dataclass(frozen=True)
class GEPARunResult:
    run_id: int
    best_prompt: Prompt
    best_score: FitnessScore
    generations: list[GenerationResult]
    wall_seconds: float


def _print(msg: str) -> None:
    print(msg, flush=True)


def _next_run_id(registry: PromptRegistry) -> int:
    """Get the next gepa run id by querying max(run_id)+1 from the generations table."""
    from researcher.gepa.prompts import connect
    with connect(registry.db_path) as conn:
        row = conn.execute("SELECT COALESCE(MAX(run_id), 0) AS m FROM gepa_generations").fetchone()
    return int(row["m"]) + 1


def run_gepa(
    seed_prompts: list[Prompt],
    mutator: Mutator,
    evaluator: Evaluator,
    registry: PromptRegistry,
    n_generations: int = 3,
    children_per_parent: int = 3,
    evaluator_seeds: list[int] | None = None,
    primary_metric: str = "hit_rate",
    selector_mode: str = "single_best",   # 'single_best' | 'pareto_scalarized' | 'pareto_random' | 'pareto_crowding'
    selector_weights: dict[str, float] | None = None,
) -> GEPARunResult:
    """
    Drive `n_generations` of single-parent mutation evolution.

    selector_mode:
      'single_best'        — max by primary_metric (default; back-compat)
      'pareto_scalarized'  — Pareto frontier, pick max weighted-sum (recommended for multi-dim)
      'pareto_crowding'    — Pareto frontier, pick most-isolated entry (max diversity)
      'pareto_random'      — Pareto frontier, uniform random pick (exploration)

    Total evaluations ≈ (len(seed_prompts) + n_generations * children_per_parent) * len(evaluator_seeds).
    """
    if not seed_prompts:
        raise ValueError("Need at least one seed prompt")
    if evaluator_seeds is None:
        evaluator_seeds = [1, 2, 3]

    run_id = _next_run_id(registry)
    t0 = time.time()
    _print(f"\n=== gepa run {run_id} ({mutator.name} mutator, {n_generations} generations, "
           f"{children_per_parent} children/gen, evaluator_seeds={evaluator_seeds}, "
           f"selector={selector_mode}) ===\n")

    population = list(seed_prompts)
    scores: dict[int, FitnessScore] = {}
    history: list[GenerationResult] = []

    for gen in range(n_generations + 1):  # gen 0 = seed pop, then n_generations rounds of mutation
        _print(f"--- generation {gen}: evaluating {len(population)} prompts ---")
        for p in population:
            if p.id is None:
                raise RuntimeError("Prompt must be registered (have an id) before evaluation")
            if p.id in scores:
                continue  # already scored in a prior generation
            agg = _aggregate_score(p, evaluator, evaluator_seeds, primary_metric)
            scores[p.id] = agg
            registry.log_generation(run_id, gen, p.id, agg.to_dict(), is_parent=False)
            _print(f"  prompt #{p.id} [{p.source}] {primary_metric}={getattr(agg, primary_metric):.4f}")

        # Parent selection: single_best vs Pareto-aware
        pop_scores = {p.id: scores[p.id] for p in population}
        if selector_mode == "single_best":
            best_id = max(population, key=lambda x: getattr(scores[x.id], primary_metric)).id
        elif selector_mode.startswith("pareto_"):
            mode = selector_mode.split("_", 1)[1]   # 'scalarized' | 'crowding' | 'random'
            best_id = select_from_frontier(pop_scores, mode=mode, weights=selector_weights)
        else:
            raise ValueError(f"Unknown selector_mode: {selector_mode!r}")

        # Update the parent marker for this generation (overwrites the is_parent=0 row)
        registry.log_generation(run_id, gen, best_id, scores[best_id].to_dict(), is_parent=True)
        parent = registry.get(best_id)
        history.append(GenerationResult(
            generation=gen, prompts=list(population),
            scores=pop_scores,
            parent_id=best_id,
        ))
        _print(f"  generation {gen} parent: #{best_id} ({primary_metric}={getattr(scores[best_id], primary_metric):.4f})")

        # Last generation: don't mutate further
        if gen == n_generations:
            break

        # Mutate the chosen parent to form next population (parent + children for elitism)
        children = mutator.mutate(parent, n=children_per_parent)
        population = [parent] + children
        _print(f"  mutated {len(children)} children for generation {gen + 1}\n")

    # Overall best across all evaluated
    best_id = max(scores.keys(), key=lambda pid: getattr(scores[pid], primary_metric))
    best_prompt = registry.get(best_id)
    wall = time.time() - t0
    _print(f"\n=== gepa run {run_id} done in {wall:.1f}s ===")
    _print(f"best prompt: #{best_id} [{best_prompt.source}] {primary_metric}={getattr(scores[best_id], primary_metric):.4f}")

    return GEPARunResult(
        run_id=run_id, best_prompt=best_prompt, best_score=scores[best_id],
        generations=history, wall_seconds=wall,
    )


def _aggregate_score(
    prompt: Prompt, evaluator: Evaluator, evaluator_seeds: list[int], primary_metric: str,
) -> FitnessScore:
    """Run evaluator across all seeds, return mean fitness (per-seed scores cached on disk via update_fitness)."""
    from statistics import mean

    per_seed = [evaluator.evaluate(prompt, evaluator_seed=s) for s in evaluator_seeds]
    return FitnessScore(
        hit_rate=mean(s.hit_rate for s in per_seed),
        mean_kept_delta=mean(s.mean_kept_delta for s in per_seed),
        n_kept=sum(s.n_kept for s in per_seed),
        n_decided=sum(s.n_decided for s in per_seed),
        n_inconclusive=sum(s.n_inconclusive for s in per_seed),
        n_crashes=sum(s.n_crashes for s in per_seed),
        terminal_metric=mean(s.terminal_metric for s in per_seed),
        time_to_first_keep=(
            int(mean(s.time_to_first_keep for s in per_seed if s.time_to_first_keep is not None))
            if any(s.time_to_first_keep is not None for s in per_seed) else None
        ),
        wall_seconds=sum(s.wall_seconds for s in per_seed),
        # Multi-dim aggregation: mean across seeds (Pareto comparisons happen on aggregated scores)
        oos_is_sharpe_gap=mean(s.oos_is_sharpe_gap for s in per_seed),
        max_drawdown=mean(s.max_drawdown for s in per_seed),
        turnover_annualized=mean(s.turnover_annualized for s in per_seed),
        stability_score=mean(s.stability_score for s in per_seed),
    )
