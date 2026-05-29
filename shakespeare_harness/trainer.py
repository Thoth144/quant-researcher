"""
Locked training loop for the shakespeare domain.

LOCKED — do not modify. The eval definition is the comparability contract.

Contract:
    run_training(model_builder, optimizer_builder, params, seed=None) -> TrainingResult

    model_builder must be: (vocab_size, params) -> nn.Module producing (logits, loss)
    optimizer_builder must be: (model, params) -> torch.optim.Optimizer

    The seed has the same role as in finance/sklearn harnesses: same seed →
    same data shuffle + init → directly comparable parent vs candidate for
    paired-CI replication.

Default trainer chooses CPU if no CUDA; the model is small enough.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from shakespeare_harness.data import load
from shakespeare_harness.metrics import loss_to_bpb
from shakespeare_harness.tokenizer import get_tokenizer

EVAL_BATCH_SIZE = 8
EVAL_ITERS = 50


@dataclass(frozen=True)
class StepMetrics:
    step: int
    train_loss: float
    val_loss: float


@dataclass(frozen=True)
class TrainingResult:
    val_bpb: float
    val_loss: float
    train_loss_final: float
    n_params: int
    wall_seconds: float
    steps_completed: int
    history: list[StepMetrics]
    seed: int | None
    device: str

    def to_dict(self) -> dict:
        return {
            "val_bpb": self.val_bpb,
            "val_loss": self.val_loss,
            "train_loss_final": self.train_loss_final,
            "n_params": self.n_params,
            "wall_seconds": self.wall_seconds,
            "steps_completed": self.steps_completed,
            "history": [asdict(h) for h in self.history],
            "seed": self.seed,
            "device": self.device,
        }

    @property
    def primary_metric(self) -> float:
        """
        Returns -val_bpb so the framework's higher-is-better decision gate works
        natively. val_bpb itself (lower-better) is preserved in to_dict() for
        human display. The Domain's primary_metric_name communicates this convention.
        """
        return -self.val_bpb


def _get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences."""
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_val_loss(model, data: torch.Tensor, batch_size: int, block_size: int, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = _get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def run_training(
    model_builder: Callable[[int, Any], torch.nn.Module],
    optimizer_builder: Callable[[torch.nn.Module, Any], torch.optim.Optimizer],
    params: Any,
    seed: int | None = None,
) -> TrainingResult:
    """
    Train a tiny GPT on tiny shakespeare for params.max_iters steps.
    Returns TrainingResult with primary_metric=val_bpb (lower is better).
    """
    effective_seed = 0 if seed is None else seed
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer()
    ds = load()

    train_ids = torch.tensor(tok.encode(ds.train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(ds.val_text), dtype=torch.long)

    block_size = int(params.block_size)
    batch_size = int(params.batch_size)
    max_iters = int(params.max_iters)
    eval_every = max(1, int(params.eval_every))

    model = model_builder(tok.vocab_size, params).to(device)
    optimizer = optimizer_builder(model, params)
    n_params = sum(p.numel() for p in model.parameters())

    history: list[StepMetrics] = []
    t0 = time.time()
    train_loss_recent = 0.0
    model.train()

    for step in range(1, max_iters + 1):
        # Linear warmup over warmup_iters
        if params.warmup_iters > 0 and step <= params.warmup_iters:
            lr_scale = step / params.warmup_iters
            for g in optimizer.param_groups:
                g["lr"] = float(params.learning_rate) * lr_scale

        x, y = _get_batch(train_ids, batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if params.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
        optimizer.step()
        train_loss_recent = loss.item()

        if math.isnan(train_loss_recent) or train_loss_recent > 100:
            # Loss exploded — abort early with current val measurement
            break

        if step % eval_every == 0 or step == max_iters:
            val_loss = _estimate_val_loss(model, val_ids, batch_size, block_size, device)
            history.append(StepMetrics(step=step, train_loss=train_loss_recent, val_loss=val_loss))

    final_val_loss = _estimate_val_loss(model, val_ids, batch_size, block_size, device)
    val_bpb = loss_to_bpb(final_val_loss, bytes_per_token=1.0)  # char-level: 1 byte per token

    return TrainingResult(
        val_bpb=val_bpb,
        val_loss=final_val_loss,
        train_loss_final=train_loss_recent,
        n_params=n_params,
        wall_seconds=time.time() - t0,
        steps_completed=step if 'step' in locals() else 0,
        history=history,
        seed=seed,
        device=device,
    )
