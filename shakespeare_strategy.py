"""
The editable surface for the shakespeare domain. The agent mutates THIS file.

Seed model: small GPT trained on tiny shakespeare with AdamW. The agent can tune
hyperparameters, swap optimizer, change model architecture, add scheduling, etc.

Mutation surface:
  - Model: n_layer, n_head, n_embd, block_size, dropout
  - Training: batch_size, max_iters, learning_rate, weight_decay, warmup_iters, grad_clip
  - Optimizer: 'adamw' (default) or 'hyperball' (Hyperball + ULMO from the user's
    tinyshakespeare-gpt research repo — a sibling project)
  - For hyperball: matrix-ULMO arm (gram_ns / sign / colnorm / frobenius), vector-ULMO
    arm (frobenius — only ULMO that works on 1D LayerNorm params), state retention β,
    update rule (retract / slerp).

Contract that must NOT break (the harness depends on it):
  - PARAMS is a module-level HyperParams instance
  - build_model(vocab_size, params) -> nn.Module producing (logits, loss)
  - build_optimizer(model, params) -> torch.optim.Optimizer
"""

import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from shakespeare_harness.model import TinyGPT

# Sibling-project path. The user keeps tinyshakespeare-gpt next to quant-researcher
# in ~/Documents/GitHub/. Override with $TINYSHAKESPEARE_GPT_PATH for other layouts.
TINYSHAKESPEARE_GPT_PATH = os.environ.get(
    "TINYSHAKESPEARE_GPT_PATH",
    str(Path(__file__).parent.parent / "tinyshakespeare-gpt"),
)


@dataclass(frozen=True)
class HyperParams:
    # Model architecture
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0

    # Training schedule
    batch_size: int = 16
    max_iters: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_iters: int = 50
    grad_clip: float = 1.0
    eval_every: int = 100

    # Optimizer choice. 'adamw' (default) | 'hyperball' (imports from tinyshakespeare-gpt).
    optimizer: str = "adamw"

    # AdamW-specific
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # Hyperball-specific (only used when optimizer='hyperball')
    hyperball_lr: float = 0.01            # pre-retraction Euclidean step
    hyperball_beta: float = 0.95           # state retention (momentum memory)
    hyperball_update_rule: str = "retract" # 'retract' | 'slerp'
    # Matrix params (≥2D): tunable ULMO arm
    hyperball_matrix_ulmo: str = "gram_ns" # 'gram_ns' | 'sign' | 'colnorm' | 'frobenius'
    # Vector/1D params (LayerNorm weights/biases): Frobenius is the only safe ULMO
    # because Sign/ColNorm assume 2D input.

    def to_dict(self) -> dict:
        return asdict(self)


PARAMS = HyperParams()


def build_model(vocab_size: int, params: HyperParams) -> torch.nn.Module:
    """Construct the model. Must produce (logits, loss) tuple on forward(idx, targets)."""
    return TinyGPT(
        vocab_size=vocab_size,
        n_layer=params.n_layer,
        n_head=params.n_head,
        n_embd=params.n_embd,
        block_size=params.block_size,
        dropout=params.dropout,
    )


def _build_hyperball(model: torch.nn.Module, params: HyperParams) -> torch.optim.Optimizer:
    """
    Construct a Hyperball optimizer using the actual scionh package from tinyshakespeare-gpt.

    Param groups:
      - matrix_params (≥2D weights, e.g. Linear, Embedding) → params.hyperball_matrix_ulmo
      - vector_params (<2D, e.g. LayerNorm weight/bias)     → FrobeniusULMO (only safe choice)

    Raises ImportError with a clear message if scionh isn't reachable.
    """
    if TINYSHAKESPEARE_GPT_PATH not in sys.path:
        sys.path.insert(0, TINYSHAKESPEARE_GPT_PATH)

    try:
        from scionh.optim.scion import Hyperball
        from scionh.ulmos.core import (
            ColNormULMO, FrobeniusULMO, GramNewtonSchulzULMO, SignULMO,
        )
    except ImportError as e:
        raise ImportError(
            f"Hyperball requires tinyshakespeare-gpt's scionh package. "
            f"Expected at {TINYSHAKESPEARE_GPT_PATH} or via $TINYSHAKESPEARE_GPT_PATH. "
            f"Underlying: {e}"
        )

    work_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    matrix_ulmo_factories = {
        "gram_ns": lambda: GramNewtonSchulzULMO(work_dtype=work_dtype),
        "sign": lambda: SignULMO(),
        "colnorm": lambda: ColNormULMO(),
        "frobenius": lambda: FrobeniusULMO(),
    }
    if params.hyperball_matrix_ulmo not in matrix_ulmo_factories:
        raise ValueError(
            f"Unknown hyperball_matrix_ulmo {params.hyperball_matrix_ulmo!r}; "
            f"supported: {sorted(matrix_ulmo_factories)}"
        )
    matrix_ulmo = matrix_ulmo_factories[params.hyperball_matrix_ulmo]()
    vector_ulmo = FrobeniusULMO()  # only ULMO that handles 1D params safely

    matrix_params = [p for p in model.parameters() if p.dim() >= 2]
    vector_params = [p for p in model.parameters() if p.dim() < 2]

    groups = []
    if matrix_params:
        groups.append({"params": matrix_params, "ulmo": matrix_ulmo})
    if vector_params:
        groups.append({"params": vector_params, "ulmo": vector_ulmo})

    return Hyperball(
        groups,
        lr=params.hyperball_lr,
        beta=params.hyperball_beta,
        update_rule=params.hyperball_update_rule,
    )


def build_optimizer(model: torch.nn.Module, params: HyperParams) -> torch.optim.Optimizer:
    """Construct the optimizer. Agent can swap this implementation."""
    if params.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=params.learning_rate,
            betas=(params.adam_beta1, params.adam_beta2),
            weight_decay=params.weight_decay,
        )
    if params.optimizer == "hyperball":
        return _build_hyperball(model, params)
    raise ValueError(
        f"Unknown optimizer {params.optimizer!r}; supported: 'adamw', 'hyperball'"
    )
