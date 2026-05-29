"""Shakespeare domain: trainer + tokenizer + worker subprocess tests.

These tests download tiny-shakespeare on first run (small, ~1MB) and run real
training on tiny GPTs. Each test takes 2-10 seconds.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # skip if torch isn't installed; bind name for use below

from shakespeare_harness.tokenizer import get_tokenizer
from shakespeare_harness.trainer import run_training
from shakespeare_strategy import PARAMS, HyperParams, build_model, build_optimizer

REPO_ROOT = Path(__file__).parent.parent


def _fast_params() -> HyperParams:
    """Tiny config that trains in 1-2s."""
    return dataclasses.replace(
        PARAMS, n_layer=2, n_head=2, n_embd=64, block_size=64, max_iters=100, eval_every=50,
    )


def test_tokenizer_roundtrip():
    tok = get_tokenizer()
    text = "Hello, world!"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
    assert tok.vocab_size > 50  # tiny shakespeare has ~65 unique chars


def test_baseline_training_runs():
    params = _fast_params()
    result = run_training(build_model, build_optimizer, params, seed=1)
    assert result.val_bpb > 0
    assert result.val_bpb < 10  # ln(vocab~65)/ln(2) ≈ 6.0; even random is bounded
    assert result.n_params > 0
    assert result.primary_metric == -result.val_bpb  # negated convention
    assert result.steps_completed == params.max_iters


def test_training_is_seed_reproducible():
    params = _fast_params()
    r1 = run_training(build_model, build_optimizer, params, seed=42)
    r2 = run_training(build_model, build_optimizer, params, seed=42)
    # Same seed → same primary_metric within tight tolerance (model + data shuffle deterministic)
    assert abs(r1.val_bpb - r2.val_bpb) < 0.01


def test_training_differs_across_seeds():
    params = _fast_params()
    r1 = run_training(build_model, build_optimizer, params, seed=1)
    r2 = run_training(build_model, build_optimizer, params, seed=2)
    assert r1.val_bpb != r2.val_bpb  # different shuffles → different result


def test_more_params_trains_better():
    """Sanity: a bigger model trained the same number of steps should reach lower val_bpb."""
    small = dataclasses.replace(PARAMS, n_layer=1, n_head=2, n_embd=32, block_size=64, max_iters=200, eval_every=100)
    bigger = dataclasses.replace(PARAMS, n_layer=2, n_head=2, n_embd=128, block_size=64, max_iters=200, eval_every=100)
    r_small = run_training(build_model, build_optimizer, small, seed=1)
    r_big = run_training(build_model, build_optimizer, bigger, seed=1)
    assert r_big.val_bpb < r_small.val_bpb


def test_worker_subprocess():
    """Force CPU so this test is reliable when LM Studio is hogging VRAM on dev machines."""
    import os
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-m", "researcher._shakespeare_worker", "1",
         "--params-overrides", '{"n_layer":2,"n_embd":64,"max_iters":100,"eval_every":50}'],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    # primary_metric is negated bpb — should be negative
    assert payload["primary_metric"] < 0
    assert payload["result"]["val_bpb"] > 0


def test_unknown_optimizer_raises():
    bad_params = dataclasses.replace(_fast_params(), optimizer="not_a_thing")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        m = build_model(get_tokenizer().vocab_size, bad_params)
        build_optimizer(m, bad_params)


def test_hyperball_optimizer_imports_and_constructs():
    """Verify Hyperball can be built from scionh when tinyshakespeare-gpt is sibling-installed."""
    import os
    from pathlib import Path
    sibling = Path(__file__).parent.parent.parent / "tinyshakespeare-gpt"
    if not sibling.exists():
        pytest.skip(f"tinyshakespeare-gpt not at {sibling}; set TINYSHAKESPEARE_GPT_PATH")
    hb_params = dataclasses.replace(
        _fast_params(),
        optimizer="hyperball", hyperball_lr=0.05, hyperball_beta=0.9,
        hyperball_matrix_ulmo="gram_ns",
    )
    m = build_model(get_tokenizer().vocab_size, hb_params)
    opt = build_optimizer(m, hb_params)
    # Hyperball is a torch.optim.Optimizer subclass
    assert isinstance(opt, torch.optim.Optimizer)
    # Two param groups: matrix params + vector params
    assert 1 <= len(opt.param_groups) <= 2


def test_hyperball_training_runs_to_completion():
    """End-to-end: Hyperball can drive a real training cycle without crashing."""
    from pathlib import Path
    sibling = Path(__file__).parent.parent.parent / "tinyshakespeare-gpt"
    if not sibling.exists():
        pytest.skip(f"tinyshakespeare-gpt not at {sibling}")
    hb_params = dataclasses.replace(
        _fast_params(),
        max_iters=50, eval_every=25,
        optimizer="hyperball", hyperball_lr=0.05, hyperball_beta=0.9,
        hyperball_matrix_ulmo="sign",  # cheap ULMO for fast test
    )
    result = run_training(build_model, build_optimizer, hb_params, seed=1)
    assert result.val_bpb > 0
    assert result.steps_completed == 50


def test_unknown_matrix_ulmo_raises():
    bad_params = dataclasses.replace(
        _fast_params(),
        optimizer="hyperball", hyperball_matrix_ulmo="not_a_real_ulmo",
    )
    from pathlib import Path
    sibling = Path(__file__).parent.parent.parent / "tinyshakespeare-gpt"
    if not sibling.exists():
        pytest.skip(f"tinyshakespeare-gpt not at {sibling}")
    with pytest.raises(ValueError, match="Unknown hyperball_matrix_ulmo"):
        m = build_model(get_tokenizer().vocab_size, bad_params)
        build_optimizer(m, bad_params)
