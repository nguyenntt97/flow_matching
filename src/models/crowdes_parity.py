"""Upstream simulator, wrapped so the endpoint cluster centres survive.

Upstream stores ``endpoint_cluster_centers`` as a bare tensor attribute -- not
a ``Parameter``, not a ``register_buffer``. Consequences:

* it is absent from ``state_dict()``, so nothing checkpoints it;
* ``from_pretrained`` leaves it ``None``;
* the *training* branch dereferences it (``self.endpoint_cluster_centers.detach()``).

Upstream papers over this by refitting KMeans before every run. That is fine
for a script and wrong for a resumable trainer: a resumed run would refit
against whatever data it happens to see and silently relabel the discrete
behaviour states mid-training.

So the centres live in a buffer on *this wrapper*. They ride along in the
Lightning ``.ckpt`` like any other buffer, while ``self.net.save_pretrained()``
still emits exactly upstream's key set -- no extra keys, no warnings when
``CrowdESFramework`` loads it, and ``endpoint_cluster_centers is None`` there,
which is precisely upstream's status quo (its inference path never reads them).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src._upstream import CrowdESSimulatorConfig, CrowdESSimulatorModel


class ParityCrowdESSimulator(nn.Module):
    needs_endpoint_clusters = True

    def __init__(
        self,
        history_length: int = 10,
        future_length: int = 10,
        env_size=(64, 64),
        env_dim: int = 8,
        neighbor_max_num: int = 4,
        latent_dim: int = 8,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        # Only the arguments upstream's trainer passes. Everything else keeps
        # CrowdESSimulatorConfig's defaults, so the architecture is identical.
        hf_config = CrowdESSimulatorConfig(
            history_length=history_length,
            future_length=future_length,
            env_size=list(env_size),
            env_dim=env_dim,
            neighbor_max_num=neighbor_max_num,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.net = CrowdESSimulatorModel(config=hf_config)

        self.register_buffer("endpoint_cluster_centers", torch.zeros(latent_dim, 2), persistent=True)
        self.register_buffer("centers_fitted", torch.zeros((), dtype=torch.bool), persistent=True)

    # ------------------------------------------------------------- clusters
    @torch.no_grad()
    def fit_endpoint_clusters(self, endpoint: torch.Tensor, control: torch.Tensor) -> None:
        """Fit the k-means centres by calling upstream's own routine.

        ``control.clone()`` is mandatory: ``normalize_rotation_scale`` writes
        ``(1, 0)`` into near-zero rows *in place*. Upstream escapes this only
        because it happens to pass a freshly concatenated tensor.
        """
        device = self.endpoint_cluster_centers.device
        self.net.endpoint_cluster_center_generation(
            endpoint.to(device),
            control.clone().to(device),
        )
        self.endpoint_cluster_centers.copy_(self.net.endpoint_cluster_centers.to(device))
        self.centers_fitted.fill_(True)

    # -------------------------------------------------------------- forward
    def forward(self, traj_hist, traj_fut=None, *args, **kwargs):
        # Upstream reads the centres off the module it is executing on, so bind
        # the buffer across before delegating. Positional pass-through keeps
        # upstream's argument order authoritative.
        self.net.endpoint_cluster_centers = self.endpoint_cluster_centers
        return self.net(traj_hist, traj_fut, *args, **kwargs)

    # --------------------------------------------------------------- export
    def export_pretrained(self, out_dir: str) -> None:
        # safe_serialization=False is required: upstream's from_pretrained call
        # site passes no use_safetensors, so it looks for pytorch_model.bin.
        self.net.save_pretrained(out_dir, safe_serialization=False)
