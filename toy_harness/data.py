"""
Dataset loading for the toy_sklearn domain.

LOCKED — do not modify. Pinned to sklearn's bundled `digits` dataset (1797
samples, 64 features, 10 classes). No external downloads, fully reproducible.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.datasets import load_digits


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray                 # (n_samples, n_features)
    y: np.ndarray                 # (n_samples,)
    n_classes: int

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


@lru_cache(maxsize=1)
def load() -> Dataset:
    digits = load_digits()
    return Dataset(name="digits", X=digits.data, y=digits.target, n_classes=10)
