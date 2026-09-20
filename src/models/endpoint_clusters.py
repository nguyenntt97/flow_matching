"""Durable endpoint-cluster buffers, shared by every head that keeps upstream's B=8 latent.

The mechanism and its rationale are documented at length in
``src/models/crowdes_parity.py`` -- upstream stores ``endpoint_cluster_centers``
as a bare tensor attribute, so nothing checkpoints it and a resumed run would
refit KMeans and silently relabel the discrete behaviour states mid-training.

``ParityCrowdESSimulator`` deliberately does NOT use this mixin: its behaviour is
pinned bit-for-bit by ``tests/parity/test_cluster_buffer.py`` and
``tests/parity/test_model_parity.py``, and the parity baseline must not drift
because a later head wanted to share code. This is a faithful copy for new heads,
cross-referenced so the two cannot diverge unnoticed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EndpointClusterMixin(nn.Module):
    """Adds ``endpoint_cluster_centers`` / ``centers_fitted`` buffers to a wrapper.

    Expects ``self.net`` to be a ``CrowdESSimulatorModel`` (it owns the KMeans
    routine and the rotation/scale normalisation). Call ``_init_cluster_buffers``
    from ``__init__`` after ``self.net`` is assigned.
    """

    needs_endpoint_clusters = True

    def _init_cluster_buffers(self, latent_dim: int) -> None:
        self.register_buffer("endpoint_cluster_centers", torch.zeros(latent_dim, 2), persistent=True)
        self.register_buffer("centers_fitted", torch.zeros((), dtype=torch.bool), persistent=True)

    @torch.no_grad()
    def fit_endpoint_clusters(self, endpoint: torch.Tensor, control: torch.Tensor) -> None:
        """Fit the k-means centres by calling upstream's own routine.

        ``control.clone()`` is mandatory: ``normalize_rotation_scale`` aliases its
        ``control`` argument (``dir = control``) and writes ``(1, 0)`` into
        near-zero rows *in place*. Upstream escapes this only because it happens
        to pass a freshly concatenated tensor.
        """
        device = self.endpoint_cluster_centers.device
        self.net.endpoint_cluster_center_generation(
            endpoint.to(device),
            control.clone().to(device),
        )
        self.endpoint_cluster_centers.copy_(self.net.endpoint_cluster_centers.to(device))
        self.centers_fitted.fill_(True)

    def _bind_centers(self) -> None:
        """Upstream reads the centres off the module it is executing on."""
        self.net.endpoint_cluster_centers = self.endpoint_cluster_centers
