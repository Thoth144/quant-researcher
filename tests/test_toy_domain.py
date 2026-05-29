"""Toy-domain evaluator + worker tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from toy_harness.evaluator import run_eval
from toy_strategy import PARAMS, HyperParams, build_model

REPO_ROOT = Path(__file__).parent.parent


def test_toy_baseline_eval_runs():
    """The seed HyperParams should produce a usable accuracy on digits."""
    result = run_eval(build_model, PARAMS, seed=1, n_folds=3)
    assert 0.5 < result.mean_accuracy <= 1.0
    assert 0.5 < result.mean_f1_macro <= 1.0
    assert len(result.per_fold) == 3
    for f in result.per_fold:
        assert f.fit_seconds > 0
        assert 0.0 <= f.accuracy <= 1.0


def test_toy_eval_is_seed_reproducible():
    """Same seed → identical primary_metric (deterministic)."""
    r1 = run_eval(build_model, PARAMS, seed=42, n_folds=3)
    r2 = run_eval(build_model, PARAMS, seed=42, n_folds=3)
    assert r1.primary_metric == r2.primary_metric


def test_toy_eval_differs_across_seeds():
    """Different seeds → different splits → different per-fold scores (paired-CI substrate)."""
    r1 = run_eval(build_model, PARAMS, seed=1, n_folds=3)
    r2 = run_eval(build_model, PARAMS, seed=2, n_folds=3)
    # Per-fold accuracies should differ (means can occasionally coincide by chance with N=3 folds)
    fold_accs_1 = [f.accuracy for f in r1.per_fold]
    fold_accs_2 = [f.accuracy for f in r2.per_fold]
    assert fold_accs_1 != fold_accs_2


def test_toy_hyperparams_change_score():
    """A weaker model should produce a measurably different (usually lower) score."""
    weak = HyperParams(n_estimators=10, learning_rate=0.05, max_depth=2)
    strong = HyperParams(n_estimators=100, learning_rate=0.1, max_depth=3)
    r_weak = run_eval(build_model, weak, seed=1, n_folds=3)
    r_strong = run_eval(build_model, strong, seed=1, n_folds=3)
    # Strong should beat weak — but the magnitude varies; just check they differ
    assert r_weak.primary_metric != r_strong.primary_metric


def test_sklearn_worker_subprocess():
    """The subprocess entrypoint produces parseable JSON with ok=True."""
    proc = subprocess.run(
        [sys.executable, "-m", "researcher._sklearn_worker", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert "primary_metric" in payload
    assert payload["primary_metric"] > 0.5


def test_sklearn_worker_with_overrides():
    """--params-overrides should apply via dataclasses.replace."""
    overrides = json.dumps({"n_estimators": 20})
    proc = subprocess.run(
        [sys.executable, "-m", "researcher._sklearn_worker", "1",
         "--params-overrides", overrides],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["params"]["n_estimators"] == 20
