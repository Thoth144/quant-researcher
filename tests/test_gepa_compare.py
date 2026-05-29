"""Paired-CI prompt comparison tests. Uses cached-scores path to avoid running fitness."""

import pytest

from researcher.gepa.compare import compare_prompts_from_cached_scores as cmp_cached


def test_clear_b_better():
    d = cmp_cached([0.1, 0.1, 0.1], [0.3, 0.3, 0.3])
    assert d.action == "b_better"
    assert d.primary_delta == pytest.approx(0.2)
    assert d.ci_low > 0


def test_clear_a_better():
    d = cmp_cached([0.5, 0.6, 0.5], [0.1, 0.2, 0.1])
    assert d.action == "a_better"
    assert d.primary_delta < 0
    assert d.ci_high < 0


def test_inconclusive_when_ci_straddles_zero():
    # Same mean, small noise both directions
    d = cmp_cached([0.30, 0.32, 0.28], [0.32, 0.28, 0.30])
    assert d.action == "inconclusive"
    assert d.ci_low < 0 < d.ci_high


def test_single_pair_is_inconclusive():
    d = cmp_cached([0.1], [0.9])
    assert d.action == "inconclusive"
    assert d.n_pairs == 1


def test_zero_variance_strict_better():
    d = cmp_cached([0.2, 0.2, 0.2], [0.3, 0.3, 0.3])
    assert d.action == "b_better"
    assert d.primary_delta == pytest.approx(0.1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cmp_cached([0.1, 0.2], [0.3, 0.4, 0.5])
