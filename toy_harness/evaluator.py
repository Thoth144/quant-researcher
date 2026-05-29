"""
Locked toy evaluator: train + score a sklearn classifier on the `digits` dataset
with a seeded stratified train/test split, repeated across multiple CV folds.

LOCKED — do not modify. This is the comparability contract for the toy_sklearn domain.

Contract:
    run_eval(model_builder, hyperparams, seed=None) -> EvalResult

    model_builder must be:
        (params: Any) -> sklearn-compatible estimator (.fit(X, y), .predict(X))

The seed has the same role as in finance/harness/backtest.py: same seed → same
stratified split → directly comparable parent vs candidate across paired CI.
"""

from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from toy_harness import data as _data
from toy_harness import metrics as M

DEFAULT_TEST_SIZE = 0.25
DEFAULT_N_FOLDS = 5


@dataclass(frozen=True)
class FoldMetrics:
    accuracy: float
    f1_macro: float
    fit_seconds: float


@dataclass(frozen=True)
class EvalResult:
    mean_accuracy: float                 # primary metric
    mean_f1_macro: float
    per_fold: list[FoldMetrics]
    seed: int | None
    n_classes: int
    n_features: int

    def to_dict(self) -> dict:
        return {
            "mean_accuracy": self.mean_accuracy,
            "mean_f1_macro": self.mean_f1_macro,
            "per_fold": [asdict(f) for f in self.per_fold],
            "seed": self.seed,
            "n_classes": self.n_classes,
            "n_features": self.n_features,
        }

    @property
    def primary_metric(self) -> float:
        return self.mean_accuracy


def run_eval(
    model_builder: Callable[[Any], Any],
    params: Any,
    seed: int | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
) -> EvalResult:
    """
    Run stratified K-fold CV on the digits dataset with the supplied hyperparams.

    seed=None    -> deterministic split (no seed influence)
    seed=int     -> stratified split with that random_state
    """
    import time

    ds = _data.load()
    effective_seed = 0 if seed is None else seed

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=effective_seed)
    per_fold: list[FoldMetrics] = []

    for train_idx, test_idx in skf.split(ds.X, ds.y):
        X_train, X_test = ds.X[train_idx], ds.X[test_idx]
        y_train, y_test = ds.y[train_idx], ds.y[test_idx]

        t0 = time.time()
        model = model_builder(params)
        model.fit(X_train, y_train)
        dt = time.time() - t0

        y_pred = model.predict(X_test)
        per_fold.append(FoldMetrics(
            accuracy=M.accuracy(y_test, y_pred),
            f1_macro=M.f1_macro(y_test, y_pred),
            fit_seconds=dt,
        ))

    return EvalResult(
        mean_accuracy=M.stratified_mean([f.accuracy for f in per_fold]),
        mean_f1_macro=M.stratified_mean([f.f1_macro for f in per_fold]),
        per_fold=per_fold,
        seed=seed,
        n_classes=ds.n_classes,
        n_features=ds.n_features,
    )
