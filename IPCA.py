"""Incremental/Batch PCA strategies for feature aggregation."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import IncrementalPCA  # type: ignore[import-untyped]

class StreamingIncrementalPCAReductionStrategy():
   
    def __init__(self, variance_ratio: float) -> None:
        self._variance_ratio = float(variance_ratio)
        self._pca = IncrementalPCA(n_components=None)
        self._seen_batches = 0
        self._finalized = False
    
    def update(self, batch: np.ndarray) -> None:
        self._pca.partial_fit(batch)
        self._seen_batches += 1

    def finalize(self) -> None:
        if self._seen_batches <= 0:
            raise RuntimeError("StreamingIncrementalPCAReductionStrategy received no batches.")
        cumulative_variance = np.cumsum(self._pca.explained_variance_ratio_)
        n_components = int(np.argmax(cumulative_variance >= self._variance_ratio) + 1)
        self._pca.n_components = n_components
        self._pca.components_ = self._pca.components_[:n_components]
        self._pca.n_components_ = n_components
        self._finalized = True

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.update(features)
        self.finalize()
        return self.transform(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self._finalized:
            raise RuntimeError(
                "StreamingIncrementalPCAReductionStrategy must be finalized before transform()."
            )
        return np.asarray(self._pca.transform(features))
