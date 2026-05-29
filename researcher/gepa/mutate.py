"""
Mutator — produces child Prompts from a parent.

DeterministicMockMutator does seeded text-level operations on the prompt's
content (insert known-good keyword, reorder sections, perturb length). Each
mutation is tagged with its operator name in the child's `source` field for
provenance — so a leaderboard can attribute fitness gains to operator class.

SWAP POINT for real GEPA:
    class GEPAReflectiveMutator:
        '''Read execution traces from runs.db, propose targeted mutations via LLM.'''
        def __init__(self, model: str, hermes_path: Path | None = None): ...
        def mutate(self, parent: Prompt, n: int) -> list[Prompt]: ...

The Mutator protocol below is what real GEPA must satisfy.
"""

from __future__ import annotations

import random
import re
from typing import Protocol

from researcher.gepa.prompts import Prompt, PromptRegistry


class Mutator(Protocol):
    name: str
    def mutate(self, parent: Prompt, n: int) -> list[Prompt]: ...


# Known-good phrases sprinkled into prompts as a synthetic "good-mutation" source.
# Replace with real GEPA reflective output once API access exists.
_KEYWORD_INJECTIONS = [
    "When in doubt about a hyperparameter, prefer the previously-accepted range.",
    "Crashes are unrecoverable — propose small steps when the parent already works.",
    "Inspect the recent INCONCLUSIVE attempts: those define the noise floor; cross it.",
    "If repeated mutations on one field fail, switch fields rather than persist.",
    "When `Δ` is tiny across attempts, propose a structurally different change, not a tune.",
    "Look for unexplored combinations: signal weights that haven't been varied together.",
]


def _swap_two_paragraphs(text: str, rng: random.Random) -> str:
    """Reorder two paragraphs (\\n\\n-separated). Idempotent on prompts with <2 paragraphs."""
    paras = text.split("\n\n")
    if len(paras) < 2:
        return text
    i = rng.randrange(0, len(paras) - 1)
    j = rng.randrange(i + 1, len(paras))
    paras[i], paras[j] = paras[j], paras[i]
    return "\n\n".join(paras)


def _inject_keyword(text: str, rng: random.Random) -> str:
    """Append a known-good phrase to the system prompt body."""
    kw = rng.choice(_KEYWORD_INJECTIONS)
    # Insert before the final '## Output format' section if present; else append.
    marker = "## Output format"
    if marker in text:
        head, tail = text.split(marker, 1)
        return f"{head.rstrip()}\n\n{kw}\n\n{marker}{tail}"
    return f"{text.rstrip()}\n\n{kw}\n"


def _trim_excess_blank_lines(text: str, rng: random.Random) -> str:
    """Collapse runs of 3+ blank lines down to 2. Tiny edit; rarely improves anything."""
    return re.sub(r"\n{3,}", "\n\n", text)


_OPERATORS = [
    ("keyword_inject", _inject_keyword),
    ("section_swap", _swap_two_paragraphs),
    ("length_trim", _trim_excess_blank_lines),
]


class DeterministicMockMutator:
    """Seed-able text mutations on prompt content. Reproducible given (parent.content_hash, seed)."""
    name = "mock"

    def __init__(self, registry: PromptRegistry, seed: int = 0):
        self._registry = registry
        self._seed = seed

    def mutate(self, parent: Prompt, n: int = 1) -> list[Prompt]:
        # RNG keyed by parent content + global seed so the SAME parent always gets the
        # same family of children under the same seed — true GEPA reproducibility.
        rng = random.Random(hash((parent.content_hash, self._seed)))
        children: list[Prompt] = []
        for k in range(n):
            # Shuffle operators and try each in order — first one that actually produces
            # a different content wins. Silent dedupe to the parent would break
            # GEPA's "n children" contract.
            ops = list(_OPERATORS)
            rng.shuffle(ops)
            child_content = parent.content
            op_name: str | None = None
            for candidate_name, candidate_fn in ops:
                attempt = candidate_fn(parent.content, rng)
                if attempt != parent.content:
                    child_content = attempt
                    op_name = candidate_name
                    break
            if op_name is None:
                raise RuntimeError(
                    f"DeterministicMockMutator: no operator could mutate prompt "
                    f"#{parent.id} (hash={parent.content_hash}). Prompt is too small "
                    f"or structurally too simple for any registered operator."
                )
            child = self._registry.register(
                content=child_content,
                source=f"mutation:{op_name}",
                parent_id=parent.id,
            )
            children.append(child)
        return children
