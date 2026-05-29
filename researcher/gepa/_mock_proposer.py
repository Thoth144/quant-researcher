"""
MockProposer — deterministic stand-in for AnthropicProposer during GEPA scaffolding.

Real proposer = LLM reads system prompt and reasons about a mutation.
Mock proposer = hashes (prompt.content_hash, iteration) to seed an RNG, picks
a hyperparameter to perturb. Different prompts produce different mutation
sequences -> different hit rates -> different fitness scores.

This means fitness DIFFERENTIATES prompts via their content hash, NOT their
semantic content. That is the honest signal: the scaffold's pipes work; real
GEPA needs a real LLM to make the prompt's semantics matter.

Swap-in for real GEPA: replace MockProposer with AnthropicProposer(system_prompt_template=prompt.content)
in fitness.py. Same Proposer protocol, no other changes required.
"""

from __future__ import annotations

import hashlib
import random
import re

from researcher.gepa.prompts import Prompt
from researcher.proposer import ProposerContext, ProposerOutput

# Aligned with strategy.py post-#7 refactor. The mock perturbs only typed scalar
# fields; richer moves (add/remove signal, change combine_mode, edit weights dict)
# require structured edits that real LLM proposers handle.
_PERTURBATIONS = [
    ("long_pct", lambda r: round(r.uniform(0.1, 0.3), 2)),
    ("short_pct", lambda r: round(r.uniform(0.1, 0.3), 2)),
    ("rebalance_days", lambda r: r.choice([5, 10, 21, 63])),
    ("gross_leverage", lambda r: round(r.uniform(0.5, 1.5), 2)),
]


def _seed_from(prompt_hash: str, evaluator_seed: int, iteration: int) -> int:
    """Combine prompt identity + run seed + iteration into a stable RNG seed."""
    digest = hashlib.sha256(f"{prompt_hash}|{evaluator_seed}|{iteration}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class MockProposer:
    """Proposer-protocol-compatible. Behavior keyed by (prompt, evaluator_seed, iteration)."""
    name = "mock"

    def __init__(self, prompt: Prompt, evaluator_seed: int = 0):
        self._prompt = prompt
        self._evaluator_seed = evaluator_seed

    def __call__(self, ctx: ProposerContext) -> ProposerOutput:
        rng = random.Random(_seed_from(self._prompt.content_hash, self._evaluator_seed, ctx.iteration))
        field, sample = rng.choice(_PERTURBATIONS)
        new_value = sample(rng)

        pattern = re.compile(rf"^(\s*{field}\s*:[^=]+=\s*)([^\s#]+)", re.MULTILINE)
        match = pattern.search(ctx.current_strategy_src)
        if not match:
            raise RuntimeError(
                f"MockProposer could not locate field {field!r} in strategy.py "
                f"(prompt id={self._prompt.id}, hash={self._prompt.content_hash})"
            )
        new_src = pattern.sub(rf"\g<1>{new_value!r}", ctx.current_strategy_src, count=1)

        # Rationale carries provenance so runs.db retains prompt linkage in candidates.rationale
        rationale = (
            f"[mock prompt={self._prompt.content_hash[:8]} eval_seed={self._evaluator_seed}] "
            f"perturb {field} -> {new_value!r}"
        )
        return ProposerOutput(new_strategy_src=new_src, rationale=rationale)
