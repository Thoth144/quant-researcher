"""
CLI entrypoint for the research loop.

Examples:
    # Smoke test, no API or local LLM needed
    uv run python -m scripts.run_research --proposer stub --iterations 3 --seeds 1,2,3

    # Local LLM via Ollama (preferred — no API spend)
    uv run python -m scripts.run_research --proposer ollama --iterations 100 --seeds 1,2,3

    # Anthropic API
    ANTHROPIC_API_KEY=... uv run python -m scripts.run_research \\
        --proposer anthropic --iterations 50 --seeds 1,2,3
"""

import argparse
import sys

from researcher.domain import DOMAINS, get_domain
from researcher.loop import run_loop
from researcher.proposer import build_proposer


def parse_seeds(s: str) -> list[int | None]:
    if s.strip() == "none":
        return [None]
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Run the autonomous research loop.")
    p.add_argument("--domain", choices=sorted(DOMAINS.keys()), default="finance",
                   help="Which Researcher domain (specialist) to drive. Selects locked harness + editable file + worker.")
    p.add_argument("--proposer", choices=["stub", "anthropic", "local"], required=True)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seeds", type=parse_seeds, default=[1, 2, 3],
                   help="Comma-separated replicate seeds (e.g. '1,2,3'), or 'none' for single deterministic full-universe run")
    p.add_argument("--timeout", type=int, default=120, help="Per-backtest timeout in seconds")
    p.add_argument("--proposer-seed", type=int, default=0, help="(stub only) RNG seed for the stub proposer")
    p.add_argument("--model", default=None,
                   help="(anthropic/local only) model id. Defaults to claude-sonnet-4-5 / qwopus3.5-9b-coder-mtp.")
    p.add_argument("--base-url", default=None,
                   help="(local only) OpenAI-compatible base URL (e.g. http://192.168.50.223:1234/v1)")
    p.add_argument("--max-tokens", type=int, default=6000,
                   help="(local only) response token cap")
    p.add_argument("--attribute", action="store_true",
                   help="After each KEEP, run per-signal attribution (N extra single-seed backtests per accept)")
    p.add_argument("--notes", default=None)
    args = p.parse_args()

    domain = get_domain(args.domain)

    if args.proposer == "stub":
        proposer = build_proposer("stub", seed=args.proposer_seed)
    elif args.proposer == "anthropic":
        kwargs = {"n_seeds": len(args.seeds), "domain": domain}
        if args.model:
            kwargs["model"] = args.model
        proposer = build_proposer("anthropic", **kwargs)
    elif args.proposer == "local":
        from researcher.local_proposer import LocalProposer
        ok = {"n_seeds": len(args.seeds), "max_tokens": args.max_tokens, "domain": domain}
        if args.model:
            ok["model"] = args.model
        if args.base_url:
            ok["base_url"] = args.base_url
        proposer = LocalProposer(**ok)
    else:
        raise ValueError(f"unknown proposer {args.proposer}")

    session_id = run_loop(
        proposer=proposer, n_iterations=args.iterations, seeds=args.seeds,
        session_notes=args.notes, timeout_sec=args.timeout,
        attribute_accepts=args.attribute, domain=domain,
    )
    print(f"\nDone. session_id={session_id}")
    print(f"Inspect with: uv run python -m scripts.inspect_runs --session {session_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
