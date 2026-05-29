"""Domain protocol + per-domain configuration tests."""

import pytest

from researcher.domain import DOMAINS, FINANCE, TOY_SKLEARN, get_domain


def test_finance_domain_paths_resolve():
    assert FINANCE.name == "finance"
    assert FINANCE.strategy_file.name == "strategy.py"
    assert FINANCE.strategy_file.exists()
    for h in FINANCE.harness_files:
        assert h.exists(), f"missing harness file: {h}"


def test_toy_domain_paths_resolve():
    assert TOY_SKLEARN.name == "toy_sklearn"
    assert TOY_SKLEARN.strategy_file.name == "toy_strategy.py"
    assert TOY_SKLEARN.strategy_file.exists()
    for h in TOY_SKLEARN.harness_files:
        assert h.exists(), f"missing toy harness file: {h}"


def test_get_domain_lookup():
    assert get_domain("finance") is FINANCE
    assert get_domain("toy_sklearn") is TOY_SKLEARN
    with pytest.raises(ValueError, match="Unknown domain"):
        get_domain("not_a_domain")


def test_all_domains_registered():
    assert "finance" in DOMAINS
    assert "toy_sklearn" in DOMAINS
    assert "shakespeare" in DOMAINS
    assert len(DOMAINS) >= 3


def test_system_prompts_include_placeholders():
    """Both system prompts must have {n_seeds} and {harness_files} format slots."""
    for d in DOMAINS.values():
        assert "{n_seeds}" in d.system_prompt, f"{d.name} prompt missing {{n_seeds}}"
        assert "{harness_files}" in d.system_prompt, f"{d.name} prompt missing {{harness_files}}"


def test_domains_have_distinct_worker_commands():
    """Different domains must spawn different subprocesses."""
    cmds = {d.name: tuple(d.worker_command) for d in DOMAINS.values()}
    assert cmds["finance"] != cmds["toy_sklearn"]


def test_metric_format_strings_are_valid():
    """primary_metric_format must be a usable Python format spec."""
    for d in DOMAINS.values():
        formatted = ("{:" + d.primary_metric_format + "}").format(0.1234)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
