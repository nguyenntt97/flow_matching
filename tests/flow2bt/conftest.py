"""Shared fixtures. Mirrors tests/parity/conftest.py's approach."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

import src._upstream  # noqa: F401,E402
from src.util.paths import register_resolvers  # noqa: E402

register_resolvers()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "eth"


@pytest.fixture(scope="session")
def crowdes_cfg():
    from hydra import compose, initialize_config_dir

    from src.data.dotdict_bridge import to_crowdes_config

    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "src" / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train_simulator", overrides=[f"data={DATASET}"])
    return to_crowdes_config(cfg.data)


@pytest.fixture(scope="session")
def real_batch(crowdes_cfg):
    """A real batch out of the built cache, or skip. Needs the eth cache."""
    import numpy as np
    import torch

    from src.data.simulator_dataset import ParitySimulatorDataset
    from src.util.fingerprint import build_fingerprint

    simulator_cfg = crowdes_cfg["crowd_simulator"]["simulator"]
    fingerprint = build_fingerprint(simulator_cfg)
    cache = (
        Path(crowdes_cfg["dataset"]["dataset_preprocessed_path"])
        / "cache"
        / f"train_simulator.{fingerprint}.pickle"
    )
    if not cache.is_file():
        pytest.skip(f"no dataset cache at {cache}; run src.train build_cache_only=true")

    dataset = ParitySimulatorDataset(crowdes_cfg, "train", cache_path=str(cache))
    indices = np.random.default_rng(0).choice(len(dataset), size=2048, replace=False)
    keys = ("traj_hist", "traj_fut", "neighbor", "control", "goal", "attr", "environment")
    return {
        key: torch.stack([dataset[int(i)][key] for i in indices]).numpy() for key in keys
    }
