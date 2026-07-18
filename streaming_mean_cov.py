"""Plugin-local streaming covariance estimator for transform fitting."""

from __future__ import annotations

import torch


class StreamingCovariance:
    """Streaming covariance accumulator using stable merge updates."""

    def __init__(
        self,
        *,
        num_features: int = None,
        dtype: torch.dtype,
        device: torch.device,
        shrinkage : float = 0.7,
        eigen_floor_ratio : float = 1e-8,
        min_jitter : float = 1e-12,
        max_jitter : float = 1,
        jitter_multiplier : float = 10

    ) -> None:
        self._dtype = dtype
        self._device = device
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.num_features = num_features
        self.shrinkage = shrinkage
        self.eigen_floor_ratio = eigen_floor_ratio
        self.min_jitter = min_jitter
        self.max_jitter = max_jitter
        self.jitter_multiplier = jitter_multiplier
        self.factor : torch.Tensor | None = None
        self.initialize()
        
    def initialize(self, num_features = None):
        if num_features is not None:
            self.num_features = num_features
        if self.num_features is not None:
            self.num_features = int(self.num_features)
            self.mean = torch.zeros(self.num_features, dtype=self._dtype, device=self._device)
            self.m2 = torch.zeros(
                (self.num_features, self.num_features),
                dtype=self._dtype,
                device=self._device
            )

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim < 2:
            raise ValueError("batch must have shape [batch, features...]")
        prepared = batch.to(device=self._device, dtype=self._dtype)
        if prepared.ndim > 2:
            prepared = prepared.reshape(prepared.shape[0], -1)
        if int(prepared.shape[1]) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {int(prepared.shape[1])}."
            )
        return prepared

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        prepared = self._prepare_batch(batch)
        batch_size = int(prepared.shape[0])
        if batch_size == 0:
            return
        batch_size_f = torch.tensor(
            float(batch_size),
            dtype=self._dtype,
            device=self._device,
        )
        batch_mean = prepared.mean(dim=0)
        centered = prepared - batch_mean
        batch_m2 = centered.t().matmul(centered)
        delta = batch_mean - self.mean
        new_count = self.count + batch_size_f
        mean = self.mean + delta * (batch_size_f / new_count)
        m2 = self.m2 + batch_m2 + torch.outer(delta, delta) * (
            self.count * batch_size_f / new_count
        )
        self.mean.copy_(mean)
        self.m2.copy_(m2)
        self.count.copy_(new_count)

    def covariance(self, *, unbiased: bool = True) -> torch.Tensor:
        if float(self.count.item()) < 2.0:
            return torch.zeros(
                (self.num_features, self.num_features),
                dtype=self._dtype,
                device=self._device,
            )
        denom = self.count - 1.0 if unbiased else self.count
        return self.m2 / denom
    
    def regularize_covariance(self) -> torch.Tensor:
        covariance = self.covariance()
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(
                f"Covariance must be a 2D square matrix, got shape={tuple(covariance.shape)}"
            )
        
        cov = 0.5 * (covariance + covariance.T)

        finite_mask = torch.isfinite(cov)
        if not bool(finite_mask.all().item()):
            non_finite_count = int((~finite_mask).sum().item())
            raise ValueError(
                f"Covariance contains non-finite values"
                f"non_finite_count={non_finite_count}"
            )

        dim = int(cov.shape[0])
        mu = float(torch.trace(cov).item()) / max(dim, 1)
        identity = torch.eye(dim, device=self._device, dtype=cov.dtype)

        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * mu * identity

        if self.eigen_floor_ratio > 0.0:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            floor_value = self.eigen_floor_ratio * max(abs(mu), 1.0e-12)
            eigvals = torch.clamp(eigvals, min=floor_value)
            cov = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            cov = 0.5 * (cov + cov.T)
        
        return cov
    
    def compute_stable_cholesky(self) -> torch.Tensor:
        covariance = self.regularize_covariance()
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(
                f"Covariance must be a 2D square matrix, got shape={tuple(covariance.shape)}"
            )

        eye = torch.eye(covariance.shape[0], dtype=covariance.dtype)

        jitter = 0.0
        attempts = 0
        while True:
            attempts += 1
            candidate = covariance if jitter == 0.0 else covariance + eye * jitter
            success = False
            try:
                if hasattr(torch.linalg, "cholesky_ex"):
                    factor, info = torch.linalg.cholesky_ex(
                        candidate,
                        check_errors=False,
                    )
                    success = int(info.max().item()) == 0 and bool(
                        torch.isfinite(factor).all().item()
                    )
                else:
                    factor = torch.linalg.cholesky(candidate)
                    success = bool(torch.isfinite(factor).all().item())
            except RuntimeError:
                success = False

            if success:
                if jitter > 0.0:
                    print(f"jitter {jitter} | attempts {attempts}")
                else:
                    print(f"Computed stable Cholesky without jitter , attempts {attempts}")

                self.factor = factor
                return factor

            if jitter == 0.0:
                jitter = self.min_jitter
            else:
                jitter *= self.jitter_multiplier

            if jitter > self.max_jitter:
                break


        raise RuntimeError(
            "Failed to compute a stable Cholesky factor. "
            )
    
    def apply_mahalanobis_transform(self, features: torch.Tensor) -> torch.Tensor:

        transformed = torch.linalg.solve_triangular(
            self.factor,
            (features - self.mean).T,
            upper=False).T

        return transformed
