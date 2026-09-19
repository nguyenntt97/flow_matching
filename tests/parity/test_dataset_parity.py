"""The highest-value test: our dataset must be byte-identical to upstream's.

``SimulatorDataset.__getitem__`` crops the environment by offsetting an
``(h, w)``-ordered pointer grid with ``startpoint[[1, 0]]`` and pushing it
through ``world2image``, which reads axis 0 as ``x``. The ordering is odd, but
``CrowdES/inference_model.py::process_simulator`` does exactly the same thing,
so train and inference agree and upstream's numbers depend on it. If a
refactor ever "fixes" the crop, this test is what catches it.

Slow: builds the gcs cache on first run.
"""

from __future__ import annotations

import pytest
import torch

from src._upstream import SimulatorDataset
from src.data.simulator_dataset import ParitySimulatorDataset
from src.util.fingerprint import build_fingerprint

pytestmark = pytest.mark.slow

EXPECTED_KEYS = {
    "traj_hist",
    "traj_fut",
    "goal",
    "attr",
    "control",
    "neighbor",
    "environment",
    "startpoint",
}


@pytest.fixture(scope="module")
def ours(crowdes_cfg, preprocessed_available):
    import os

    fingerprint = build_fingerprint(crowdes_cfg["crowd_simulator"]["simulator"])
    cache = os.path.join(
        crowdes_cfg["dataset"]["dataset_preprocessed_path"],
        "cache",
        f"train_simulator.{fingerprint}.pickle",
    )
    return ParitySimulatorDataset(crowdes_cfg, "train", cache_path=cache)


@pytest.fixture(scope="module")
def theirs(crowdes_cfg, preprocessed_available):
    return SimulatorDataset(crowdes_cfg, "train")


def test_same_length(ours, theirs):
    assert len(ours) == len(theirs)


def test_same_scene_list_and_index_map(ours, theirs):
    assert ours.scene_list == theirs.scene_list
    assert ours.idx2data == theirs.idx2data


def test_same_env_dim(ours, theirs):
    assert ours.env_dim == theirs.env_dim == 8


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_items_are_identical(ours, theirs, position):
    index = {"first": 0, "middle": len(ours) // 2, "last": len(ours) - 1}[position]
    mine, upstream = ours[index], theirs[index]

    assert set(mine) == set(upstream) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert mine[key].dtype == upstream[key].dtype == torch.float32, key
        assert mine[key].shape == upstream[key].shape, key
        assert torch.equal(mine[key], upstream[key]), key


def test_shapes_match_the_config(ours, crowdes_cfg):
    simulator = crowdes_cfg["crowd_simulator"]["simulator"]
    hist = simulator["history_length"]
    fut = simulator["future_length"]
    neighbors = simulator["interaction_max_num_agents"]
    env_h, env_w = simulator["environment_size"]

    sample = ours[0]
    assert sample["traj_hist"].shape == (hist, 2)
    assert sample["traj_fut"].shape == (fut, 2)
    assert sample["goal"].shape == (2,)
    assert sample["attr"].shape == (2,)
    assert sample["control"].shape == (2,)
    assert sample["neighbor"].shape == (neighbors, hist, 2)
    assert sample["environment"].shape == (ours.env_dim, env_h, env_w)
    assert sample["startpoint"].shape == (2,)


def test_clustering_inputs_are_present(ours):
    """on_fit_start reads these off the dataset to fit the k-means centres."""
    for scene in ours.scene_list:
        assert ours.traj_pred_norm_scene[scene].ndim == 3
        assert ours.control_point_scene[scene].shape[1] == 2


def test_test_split_requires_opt_in(crowdes_cfg):
    with pytest.raises(ValueError, match="allow_test_split=True"):
        ParitySimulatorDataset(crowdes_cfg, "test")


def test_fingerprint_changes_with_the_build_config(crowdes_cfg):
    simulator = dict(crowdes_cfg["crowd_simulator"]["simulator"])
    before = build_fingerprint(simulator)
    simulator["history_length"] = simulator["history_length"] + 1
    assert build_fingerprint(simulator) != before
