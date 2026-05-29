"""
Local-LLM proposer via an OpenAI-compatible HTTP API.

Works with LM Studio (default), vLLM, sglang, llama.cpp's openai-server mode,
or any server that exposes /v1/chat/completions. Same Proposer protocol as
AnthropicProposer / StubProposer — drop-in swap.

Small-model reliability budget is lower than Claude's, so this implementation:
  - tolerates Markdown fence wrapping around the structured blocks
  - tolerates leading/trailing prose outside the tags
  - retries once with a stricter format reminder if first parse fails
  - hard-fails (rather than silently retrying forever) on persistent failure

Configure via env vars OR constructor kwargs:
  LOCAL_LLM_BASE_URL   default http://localhost:1234/v1   (LM Studio default)
  LOCAL_LLM_MODEL      default qwopus3.5-9b-coder-mtp     (user's library)
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from researcher.domain import FINANCE, Domain
from researcher.proposer import (
    ProposerContext,
    ProposerOutput,
    _fetch_failure_traces,
    _format_recent,
    _read_harness_files,
)

DEFAULT_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwopus3.5-9b-coder-mtp")
DEFAULT_TIMEOUT_SEC = 900            # reasoning models on a laptop GPU are slow
DEFAULT_MAX_TOKENS = 12000           # reasoning_content + content combined; 6K starves output, 20K+ OOMs
DEFAULT_DISABLE_THINKING = False     # Qwen 3.5 community fine-tunes vary in /no_think support


class LocalProposer:
    """Proposer-protocol-compatible. Calls any OpenAI-compatible chat-completions endpoint."""
    name = "local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        n_seeds: int = 3,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt_template: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        disable_thinking: bool = DEFAULT_DISABLE_THINKING,
        domain: Domain | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._n_seeds = n_seeds
        self._timeout = timeout_sec
        self._max_tokens = max_tokens
        self._domain = domain or FINANCE
        self._system_prompt_template = system_prompt_template or self._domain.system_prompt
        self._temperature = temperature
        self._top_p = top_p
        self._disable_thinking = disable_thinking

    def __call__(self, ctx: ProposerContext) -> ProposerOutput:
        # Minimal harness (hand-written contract + signals.py only) ~1.5K tokens.
        # Small local models with small context windows need the budget for reasoning + output.
        # Use .replace() rather than .format() — harness sources contain Python dict
        # literals (curly braces) that confuse str.format's placeholder parser.
        system = (
            self._system_prompt_template
            .replace("{n_seeds}", str(self._n_seeds))
            .replace("{harness_files}", _read_harness_files(self._domain.harness_files, minimal=True))
        )
        user = _build_user_message(ctx, domain=self._domain)
        if self._disable_thinking:
            # Qwen 3.5 family: /no_think suffix instructs the model to skip the long
            # internal-reasoning chain and emit visible content directly. Massively
            # reduces token consumption per call.
            user = user + "\n\n/no_think"

        text = self._chat(system=system, user=user)
        try:
            return _parse_response(text, required_symbols=self._domain.required_symbols)
        except ValueError as first_err:
            stricter_user = (
                user
                + "\n\nIMPORTANT: Your previous response could not be parsed. "
                "Reply with EXACTLY two XML blocks and NOTHING else (no markdown fences, "
                "no commentary outside the tags): <rationale>...</rationale><strategy_py>...</strategy_py>"
            )
            text2 = self._chat(system=system, user=stricter_user)
            try:
                return _parse_response(text2, required_symbols=self._domain.required_symbols)
            except ValueError as second_err:
                raise RuntimeError(
                    f"LocalProposer failed to produce parseable output after retry.\n"
                    f"First error: {first_err}\nRetry error: {second_err}\n"
                    f"First response head: {text[:400]!r}\nRetry response head: {text2[:400]!r}"
                )

    def _chat(self, system: str, user: str) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Local LLM HTTP {e.code} at {url}: {err_body[:400]}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Local LLM connect failed at {url}: {e}. Is the server running and a model loaded?"
            )

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"Local LLM returned no choices: {body!r}")
        return choices[0].get("message", {}).get("content", "")


# Back-compat alias so existing imports keep working
OllamaProposer = LocalProposer


def _build_user_message(ctx: ProposerContext, domain: Domain | None = None) -> str:
    """Same user-message shape as AnthropicProposer for parity (incl. trace + cross-session + attribution)."""
    from researcher.cross_session import find_similar_attempts, format_cross_session
    from researcher.attribution import format_attribution, get_attribution
    domain = domain or FINANCE
    cross = find_similar_attempts(
        ctx.current_strategy_src, exclude_session_id=ctx.session_id, limit=6,
    )
    attribution_block = ""
    if ctx.parent_candidate_id is not None:
        attrs = get_attribution(ctx.parent_candidate_id)
        if attrs:
            attribution_block = (
                "\n## Per-signal attribution on parent (which signals are pulling weight?)\n"
                + format_attribution(attrs)
            )
    metric_label = domain.primary_metric_name
    metric_fmt = "{:" + domain.primary_metric_format + "}"
    parent_metric_str = (
        metric_fmt.format(ctx.parent_metric) if ctx.parent_metric is not None
        else "(seed strategy, no parent)"
    )
    parts = [
        f"## Iteration {ctx.iteration}",
        f"\nParent mean {metric_label} across replicate seeds: {parent_metric_str}",
        "\n## Recent attempts (most recent first)",
        _format_recent(ctx.recent_decided),
        "\n## Recent failure traces (subprocess logs from crashed/discarded attempts)",
        _fetch_failure_traces(ctx.recent_decided),
        "\n## Cross-session memory (similar attempts from prior sessions)",
        format_cross_session(cross),
        attribution_block,
        f"\n## Current {domain.strategy_file.name}",
        f"```python\n{ctx.current_strategy_src}\n```",
        "\nPropose the next mutation. Output ONLY the two tagged blocks specified.",
    ]
    return "\n".join(parts)


# Match either <strategy_py>...</strategy_py> OR an unclosed start tag through end-of-string
# (small models sometimes truncate the closing tag).
_RATIONALE_RX = re.compile(r"<rationale>(.*?)</rationale>", re.DOTALL)
_STRATEGY_RX = re.compile(r"<strategy_py>(.*?)(?:</strategy_py>|\Z)", re.DOTALL)
_FENCE_RX = re.compile(r"^```(?:python)?\s*\n?|```\s*$", re.MULTILINE)


def _parse_response(text: str, required_symbols: tuple[str, ...] = ()) -> ProposerOutput:
    """Tolerant of common small-model quirks: fence wrapping, missing close tag, leading prose.

    required_symbols: domain-specific substrings that must appear in the strategy block
    (e.g. ('class HyperParams', 'def build_model') for toy_sklearn).
    """
    rationale_m = _RATIONALE_RX.search(text)
    strategy_m = _STRATEGY_RX.search(text)
    if not rationale_m or not strategy_m:
        raise ValueError(
            f"Missing <rationale> or <strategy_py> tag. "
            f"Found rationale={bool(rationale_m)}, strategy={bool(strategy_m)}"
        )
    rationale = rationale_m.group(1).strip()
    strategy = strategy_m.group(1).strip()
    strategy = _FENCE_RX.sub("", strategy).strip()
    if not strategy:
        raise ValueError("strategy_py block parsed but is empty after fence removal")
    for sym in required_symbols:
        if sym not in strategy:
            raise ValueError(f"strategy_py missing required symbol: {sym!r}")
    return ProposerOutput(new_strategy_src=strategy + "\n", rationale=rationale)
