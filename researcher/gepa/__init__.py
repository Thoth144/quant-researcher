"""
GEPA-shaped scaffolding for prompt evolution.

This package implements the SHAPE of a Genetic-Pareto Prompt Evolution loop —
fitness signature, candidate registry, paired-CI prompt-vs-prompt gate, runs.db
extension — with DETERMINISTIC MOCKS for the two LLM-mediated components:

  - `_mock_proposer.MockProposer`        replaces AnthropicProposer for fitness eval
  - `mutate.DeterministicMockMutator`    replaces a real GEPA reflective mutator

Real GEPA plugs in by swapping those two mocks for LLM-driven implementations
(see the docstrings of each module). Everything else — registry, fitness shape,
paired-CI comparison, generational loop, CLI — is reusable as-is.
"""
