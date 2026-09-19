"""The wrapper must be upstream's model, structurally and numerically."""

from __future__ import annotations

import torch

from src._upstream import CrowdESSimulatorConfig, CrowdESSimulatorModel
from src.models.crowdes_parity import ParityCrowdESSimulator

KWARGS = dict(
    history_length=10,
    future_length=10,
    env_size=[64, 64],
    env_dim=8,
    neighbor_max_num=4,
    latent_dim=8,
    hidden_dim=256,
)


def make_pair(seed: int = 0):
    torch.manual_seed(seed)
    upstream = CrowdESSimulatorModel(config=CrowdESSimulatorConfig(**KWARGS))
    torch.manual_seed(seed)
    ours = ParityCrowdESSimulator(**KWARGS)
    return upstream, ours


def make_batch(batch_size: int = 8, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    return {
        "traj_hist": randn(batch_size, 10, 2),
        "traj_fut": randn(batch_size, 10, 2),
        "goal": randn(batch_size, 2),
        "attr": randn(batch_size, 2),
        "control": randn(batch_size, 2),
        "neighbor": randn(batch_size, 4, 10, 2),
        "environment": torch.rand(8, 64, 64, generator=generator).expand(batch_size, 8, 64, 64).clone(),
    }


def test_same_parameter_count():
    upstream, ours = make_pair()
    assert sum(p.numel() for p in upstream.parameters()) == sum(p.numel() for p in ours.parameters())


def test_state_dict_keys_differ_only_by_the_net_prefix_and_two_buffers():
    upstream, ours = make_pair()
    theirs = set(upstream.state_dict())
    mine = set(ours.state_dict())
    assert {f"net.{k}" for k in theirs} | {"endpoint_cluster_centers", "centers_fitted"} == mine


def test_weights_match_under_the_same_seed():
    upstream, ours = make_pair()
    ours.net.load_state_dict(upstream.state_dict())  # must be accepted verbatim
    for (name, a), b in zip(upstream.state_dict().items(), ours.net.state_dict().values()):
        assert torch.equal(a, b), name


def test_training_loss_is_bit_identical():
    """Same module, same weights, same centres -- so the numbers must agree
    exactly. Anything else means the wrapper altered the forward path."""
    upstream, ours = make_pair()
    ours.net.load_state_dict(upstream.state_dict())

    torch.manual_seed(7)
    centers = torch.randn(KWARGS["latent_dim"], 2)
    upstream.endpoint_cluster_centers = centers.clone()
    ours.endpoint_cluster_centers.copy_(centers)
    ours.centers_fitted.fill_(True)

    upstream.eval()
    ours.eval()
    batch = make_batch()
    order = ("traj_hist", "traj_fut", "goal", "attr", "control", "neighbor", "environment")

    with torch.no_grad():
        theirs = upstream(*[batch[k].clone() for k in order])
        mine = ours(*[batch[k].clone() for k in order])

    assert torch.equal(theirs.loss, mine.loss)
    assert torch.equal(theirs.preds, mine.preds)


def test_traj_fut_none_selects_the_inference_branch():
    """Upstream switches on ``traj_fut is not None``, not ``self.training``."""
    _, ours = make_pair()
    ours.endpoint_cluster_centers.copy_(torch.randn(KWARGS["latent_dim"], 2))
    ours.centers_fitted.fill_(True)
    batch = make_batch()

    ours.train()  # training mode, but no traj_fut
    with torch.no_grad():
        output = ours(
            batch["traj_hist"], None, batch["goal"], batch["attr"],
            batch["control"], batch["neighbor"], batch["environment"], sampling=False,
        )
    assert output.loss is None
    assert output.preds.shape == (8, KWARGS["future_length"], 2)
