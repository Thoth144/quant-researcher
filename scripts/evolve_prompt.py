"""
CLI entrypoint for the GEPA-shaped prompt evolution scaffold.

Examples:
    # Synthetic fitness (no backtests, fast verification of the loop mechanics)
    uv run python -m scripts.evolve_prompt --fitness synthetic --generations 3 --children 3

    # Real fitness (runs the actual research loop for each prompt eval)
    uv run python -m scripts.evolve_prompt --fitness real --generations 2 --children 2 --eval-iterations 5

    # Inspect outcomes
    uv run python -m scripts.evolve_prompt --leaderboard
"""

import argparse
import sys
from pathlib import Path

from researcher.gepa.fitness import FitnessEvaluator, SyntheticFitnessEvaluator
from researcher.gepa.loop import run_gepa
from researcher.gepa.mutate import DeterministicMockMutator
from researcher.gepa.prompts import PromptRegistry
from researcher.proposer import SYSTEM_PROMPT_TEMPLATE

REPO_ROOT = Path(__file__).parent.parent
STRATEGY_FILE = REPO_ROOT / "strategy.py"


def main() -> int:
    p = argparse.ArgumentParser(description="Run the GEPA-shaped prompt evolution loop.")
    p.add_argument("--fitness", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--children", type=int, default=3, help="Children per parent per generation")
    p.add_argument("--evaluator-seeds", type=str, default="1,2,3",
                   help="Comma-separated seeds for fitness evaluation pairing")
    p.add_argument("--eval-iterations", type=int, default=5,
                   help="(real fitness only) iterations of the research loop per prompt evaluation")
    p.add_argument("--backtest-seeds", type=str, default="1,2,3",
                   help="(real fitness only) seeds passed into the research loop")
    p.add_argument("--mutator-seed", type=int, default=0)
    p.add_argument("--leaderboard", action="store_true", help="Just print leaderboard and exit")
    p.add_argument("--primary-metric", default="hit_rate")
    args = p.parse_args()

    registry = PromptRegistry()

    if args.leaderboard:
        rows = registry.leaderboard(limit=20, primary_metric=args.primary_metric)
        if not rows:
            print("(no prompts evaluated yet)")
            return 0
        print(f"{'id':>4}  {'metric':>8}  {'n_eval':>6}  source")
        for r in rows:
            print(f"{r['id']:>4}  {r['metric']:>8.4f}  {r['n_evaluations']:>6}  {r['source']}")
        return 0

    evaluator_seeds = [int(s) for s in args.evaluator_seeds.split(",") if s.strip()]

    # Seed population: the current canonical system prompt + 2 trivial variants for diversity
    seed_a = registry.register(SYSTEM_PROMPT_TEMPLATE, source="seed:canonical")
    seed_b = registry.register(
        SYSTEM_PROMPT_TEMPLATE + "\n\nLook for unexplored combinations of weights.\n",
        source="seed:hint_combinations",
    )
    seed_c = registry.register(
        SYSTEM_PROMPT_TEMPLATE + "\n\nThe noise floor is real: small steps tend to be INCONCLUSIVE.\n",
        source="seed:hint_noise_floor",
    )

    if args.fitness == "synthetic":
        evaluator = SyntheticFitnessEvaluator(registry=registry)
    else:
        seed_strategy_src = STRATEGY_FILE.read_text()
        backtest_seeds = [int(s) for s in args.backtest_seeds.split(",") if s.strip()]
        evaluator = FitnessEvaluator(
            seed_strategy_src=seed_strategy_src, registry=registry,
            n_iterations=args.eval_iterations, backtest_seeds=backtest_seeds,
        )

    mutator = DeterministicMockMutator(registry=registry, seed=args.mutator_seed)

    result = run_gepa(
        seed_prompts=[seed_a, seed_b, seed_c], mutator=mutator,
        evaluator=evaluator, registry=registry,
        n_generations=args.generations, children_per_parent=args.children,
        evaluator_seeds=evaluator_seeds, primary_metric=args.primary_metric,
    )
    print(f"\nDone. gepa_run_id={result.run_id}, best_prompt_id={result.best_prompt.id}")
    print(f"Inspect with: uv run python -m scripts.evolve_prompt --leaderboard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
