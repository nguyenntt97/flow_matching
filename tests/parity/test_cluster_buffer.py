"""The endpoint cluster centres must checkpoint, restore, and not refit.

Upstream keeps them as a bare tensor attribute, so they vanish from every
state dict and ``from_pretrained`` hands back ``None``. Our wrapper puts them
in a buffer -- on the wrapper, not on ``net``, so ``save_pretrained`` still
emits exactly upstream's key set.

The one thing worth pinning against the installed Lightning is the ordering
question: are buffers restored before ``on_fit_start``? ``test_buffers_restore``
answers it directly rather than trusting the docs.
"""

from __future__ import annotations

import torch

from src._upstream import CrowdESSimulatorConfig, CrowdESSimulatorModel
from src.models.crowdes_parity import ParityCrowdESSimulator

LATENT_DIM = 8


def build(latent_dim=LATENT_DIM, future_length=10):
    return ParityCrowdESSimulator(
        history_length=10,
        future_length=future_length,
        env_size=(64, 64),
        env_dim=8,
        neighbor_max_num=4,
        latent_dim=latent_dim,
        hidden_dim=64,  # small, for speed; irrelevant to this test
    )


def test_centres_start_unfitted():
    model = build()
    assert bool(model.centers_fitted) is False
    assert torch.equal(model.endpoint_cluster_centers, torch.zeros(LATENT_DIM, 2))


def test_fit_populates_the_buffer():
    torch.manual_seed(0)
    model = build()
    endpoint = torch.randn(256, 2)
    control = torch.randn(256, 2)
    model.fit_endpoint_clusters(endpoint, control)

    assert bool(model.centers_fitted) is True
    assert model.endpoint_cluster_centers.shape == (LATENT_DIM, 2)
    assert not torch.equal(model.endpoint_cluster_centers, torch.zeros(LATENT_DIM, 2))


def test_fit_does_not_mutate_the_control_argument():
    """normalize_rotation_scale writes (1, 0) into near-zero rows in place."""
    torch.manual_seed(0)
    model = build()
    control = torch.randn(64, 2)
    control[:8] = 0.0  # rows that trigger the in-place write
    before = control.clone()
    model.fit_endpoint_clusters(torch.randn(64, 2), control)
    assert torch.equal(control, before)


def test_buffers_survive_a_state_dict_round_trip():
    torch.manual_seed(0)
    model = build()
    model.fit_endpoint_clusters(torch.randn(256, 2), torch.randn(256, 2))

    restored = build()
    restored.load_state_dict(model.state_dict())

    assert bool(restored.centers_fitted) is True
    assert torch.equal(restored.endpoint_cluster_centers, model.endpoint_cluster_centers)


def test_buffers_are_in_the_wrappers_state_dict_not_the_nets():
    model = build()
    keys = set(model.state_dict())
    assert "endpoint_cluster_centers" in keys
    assert "centers_fitted" in keys
    assert not any(k.startswith("net.endpoint_cluster_centers") for k in keys)


def test_export_emits_exactly_upstreams_key_set(tmp_path):
    """No extra keys means no missing/unexpected warnings when
    CrowdESFramework calls from_pretrained."""
    model = build()
    model.export_pretrained(str(tmp_path))

    reloaded = CrowdESSimulatorModel.from_pretrained(str(tmp_path))
    assert set(reloaded.state_dict()) == set(model.net.state_dict())
    # Exactly upstream's status quo: the inference path never reads these.
    assert reloaded.endpoint_cluster_centers is None


def test_reference_model_has_no_cluster_centres_in_its_state_dict():
    """Documents the upstream behaviour this wrapper exists to fix."""
    config = CrowdESSimulatorConfig(env_dim=8, latent_dim=LATENT_DIM, hidden_dim=64)
    net = CrowdESSimulatorModel(config=config)
    assert net.endpoint_cluster_centers is None
    assert not any("endpoint_cluster_centers" in k for k in net.state_dict())


def test_buffers_restore_before_on_fit_start(tmp_path):
    """Pins the Lightning hook ordering our refit guard depends on."""
    import pytorch_lightning as pl

    seen = {}

    class Probe(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.register_buffer("centers_fitted", torch.zeros((), dtype=torch.bool))
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, batch_idx):
            return self.layer(batch).sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

        def on_fit_start(self):
            seen["fitted_at_fit_start"] = bool(self.centers_fitted)

    loader = torch.utils.data.DataLoader(torch.ones(4, 1), batch_size=2)

    first = Probe()
    first.centers_fitted.fill_(True)
    trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                         accelerator="cpu", enable_progress_bar=False)
    trainer.fit(first, loader)
    ckpt = tmp_path / "probe.ckpt"
    trainer.save_checkpoint(str(ckpt))

    seen.clear()
    second = Probe()
    trainer2 = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                          accelerator="cpu", enable_progress_bar=False)
    trainer2.fit(second, loader, ckpt_path=str(ckpt))

    assert seen["fitted_at_fit_start"] is True, (
        "Lightning restored module state AFTER on_fit_start; the refit guard in "
        "SimulatorLitModule.on_fit_start must rely on _restored_from_ckpt instead "
        "of centers_fitted"
    )
