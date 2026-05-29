"""
Toy-domain metrics.

LOCKED — do not modify. Definitions are the comparability contract.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def stratified_mean(scores: list[float]) -> float:
    """Plain arithmetic mean — kept as a named function for substitution later."""
    if not scores:
        return 0.0
    return float(np.mean(scores))
