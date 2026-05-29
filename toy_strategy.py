"""
The editable surface for the toy_sklearn domain. The agent mutates THIS file.

Seed model: sklearn GradientBoostingClassifier on the digits dataset with
moderate defaults. The agent can tune hyperparameters, swap the estimator,
add preprocessing, etc.

Contract that must NOT break (the harness depends on it):
  - PARAMS is a module-level HyperParams instance
  - build_model(params) returns an object with `.fit(X, y)` and `.predict(X)`
"""

from dataclasses import dataclass, asdict

from sklearn.ensemble import GradientBoostingClassifier


@dataclass(frozen=True)
class HyperParams:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    subsample: float = 1.0
    max_features: str | None = None    # 'sqrt' | 'log2' | None
    random_state: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


PARAMS = HyperParams()


def build_model(params: HyperParams):
    """Construct the estimator. Must return a sklearn-compatible classifier."""
    return GradientBoostingClassifier(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        subsample=params.subsample,
        max_features=params.max_features,
        random_state=params.random_state,
    )
