# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class BaseSurrogate(ABC):
    """Unified interface for all scikit-learn and PyTorch surrogate models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSurrogate":
        pass

    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean_predictions, std_uncertainty) each of shape (N,)."""
        pass


class RandomForestSurrogate(BaseSurrogate):
    """Random Forest using inter-tree variance as an uncertainty estimate."""

    def __init__(self, n_estimators: int = 100, n_jobs: int = 1, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSurrogate":
        self.model.fit(X, y)
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tree_preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
        mean = np.mean(tree_preds, axis=0)
        std = np.std(tree_preds, axis=0)
        return mean.astype(np.float32), std.astype(np.float32)


class GaussianProcessSurrogate(BaseSurrogate):
    """Exact Bayesian Gaussian Process baseline with analytical posterior std."""

    def __init__(self, random_state: int = 42):
        kernel = Matern(nu=2.5) + WhiteKernel(
            noise_level=0.1, noise_level_bounds=(1e-7, 1e1)
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        self.model.fit(X, y)
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, std = self.model.predict(X, return_std=True)
        return mean.astype(np.float32), std.astype(np.float32)


class BaggedSVRSurrogate(BaseSurrogate):
    """Bootstrap ensemble of RBF-kernel SVRs to quantify prediction uncertainty."""

    def __init__(
        self, n_estimators: int = 10, C: float = 1.0, n_jobs: int = 1, random_state: int = 42
    ):
        base_svr = SVR(kernel="rbf", C=C)
        self.model = BaggingRegressor(
            estimator=base_svr,
            n_estimators=n_estimators,
            max_samples=0.8,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedSVRSurrogate":
        self.model.fit(X, y)
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        member_preds = np.stack([est.predict(X) for est in self.model.estimators_], axis=0)
        mean = np.mean(member_preds, axis=0)
        std = np.std(member_preds, axis=0)
        return mean.astype(np.float32), std.astype(np.float32)


class TanimotoKNNSurrogate(BaseSurrogate):
    """
    k-Nearest Neighbours using Jaccard distance (1 - Tanimoto similarity),
    estimating uncertainty from neighbour target spread.
    """

    def __init__(self, n_neighbors: int = 5, n_jobs: int = 1):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            metric="jaccard",
            weights="distance",
            n_jobs=n_jobs,
        )
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TanimotoKNNSurrogate":
        X_bool = X.astype(bool)
        self.model.fit(X_bool, y)
        self._y_train = y
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_bool = X.astype(bool)
        k = min(self.n_neighbors, len(self._y_train))
        self.model.n_neighbors = k

        mean = self.model.predict(X_bool)
        _, neighbor_indices = self.model.kneighbors(X_bool, n_neighbors=k)
        neighbor_targets = self._y_train[neighbor_indices]  # shape (N, k)
        std = np.std(neighbor_targets, axis=1)
        return mean.astype(np.float32), std.astype(np.float32)


def get_baseline_model(
    name: str, n_jobs: int = 1, random_state: int = 42
) -> BaseSurrogate:
    """Helper function to instantiate baseline models by name."""
    registry = {
        "rf": lambda: RandomForestSurrogate(
            n_estimators=100, n_jobs=n_jobs, random_state=random_state
        ),
        "gp": lambda: GaussianProcessSurrogate(random_state=random_state),
        "svr": lambda: BaggedSVRSurrogate(
            n_estimators=10, n_jobs=n_jobs, random_state=random_state
        ),
        "knn": lambda: TanimotoKNNSurrogate(n_neighbors=5, n_jobs=n_jobs),
    }
    if name not in registry:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(registry.keys())}")
    return registry[name]()
