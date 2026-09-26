"""Shared representation correctness probe.

hidden representation -> StandardScaler -> TruncatedSVD (<= 128) -> class-balanced
logistic regression.

One probe scores every historical candidate. It is fit on the flattened
(question, candidate) pairs of the current training split only. No checkpoint
identity feature is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SVD_COMPONENTS = 128
LR_MAX_ITER = 400


def safe_n_components(n_samples: int, n_features: int, requested: int = SVD_COMPONENTS) -> Optional[int]:
    if n_features <= 1 or n_samples <= 1:
        return None
    n = min(int(requested), int(n_features) - 1, int(n_samples) - 1 if n_samples > 1 else 1)
    n = min(n, 128)
    return int(n) if n >= 1 else None


@dataclass
class RepresentationProbe:
    svd_components: int = SVD_COMPONENTS
    lr_max_iter: int = LR_MAX_ITER
    scaler: Optional[StandardScaler] = None
    svd: Optional[TruncatedSVD] = None
    clf: Optional[LogisticRegression] = None
    n_components: Optional[int] = None
    class_counts: dict = field(default_factory=dict)
    constant_prob: Optional[float] = None
    fitted_n_samples: int = 0
    random_state: int = 0

    def _transform(self, h: np.ndarray) -> np.ndarray:
        x = np.asarray(h, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"hidden states must be 2D, got {x.shape}")
        if self.scaler is None:
            raise RuntimeError("RepresentationProbe scaler was not fit")
        z = self.scaler.transform(x)
        if self.svd is not None:
            z = self.svd.transform(z)
        return z

    def fit(self, h: np.ndarray, y: np.ndarray, random_state: int = 0) -> "RepresentationProbe":
        self.random_state = int(random_state)
        x = np.asarray(h, dtype=np.float32)
        y = np.asarray(y, dtype=int).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("hidden states and labels have different lengths")
        self.fitted_n_samples = int(x.shape[0])
        uniq, counts = np.unique(y, return_counts=True)
        self.class_counts = {int(k): int(v) for k, v in zip(uniq, counts)}
        if len(uniq) < 2:
            self.constant_prob = float(uniq[0])
            self.scaler = None
            self.svd = None
            self.clf = None
            self.n_components = None
            return self
        self.constant_prob = None
        self.scaler = StandardScaler()
        z = self.scaler.fit_transform(x)
        n_comp = safe_n_components(z.shape[0], z.shape[1], self.svd_components)
        self.n_components = n_comp
        if n_comp is not None and n_comp >= 1:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            z = self.svd.fit_transform(z)
        else:
            self.svd = None
        self.clf = LogisticRegression(
            max_iter=self.lr_max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=self.random_state,
        )
        self.clf.fit(z, y)
        return self

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        x = np.asarray(h, dtype=np.float32)
        n = x.shape[0]
        if self.constant_prob is not None:
            return np.full(n, self.constant_prob, dtype=np.float64)
        if self.scaler is None or self.clf is None:
            raise RuntimeError("RepresentationProbe is not fit")
        z = self._transform(x)
        proba = self.clf.predict_proba(z)
        classes = list(self.clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float64)
        return np.zeros(n, dtype=np.float64)

    def score_groups(self, hidden_nkd: np.ndarray) -> np.ndarray:
        """hidden (N, K, D) -> correctness scores (N, K). Shared probe."""
        h = np.asarray(hidden_nkd, dtype=np.float32)
        if h.ndim != 3:
            raise ValueError(f"expected (N, K, D), got {h.shape}")
        n, k, d = h.shape
        return self.predict_proba(h.reshape(n * k, d)).reshape(n, k)
