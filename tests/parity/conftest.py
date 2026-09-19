"""Shared fixtures for the CrowdES parity tests.

These are integration tests against the vendored submodule and the
preprocessed dataset on disk. They need the ``flowbt`` environment (torch,
lightning, hydra, transformers, sklearn, pathfinder, opencv) and, for the
dataset tests, the ``datasets/preprocessed/`` tree.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="parity tests need the flowbt environment")

from src.util.paths import REPO_ROOT, register_resolvers  # noqa: E402

register_resolvers()

#: gcs has 2 train scenes, so building its cache takes minutes rather than
#: tens of minutes. It is also the one dataset with the longer 50-step horizon,
#: which makes it a decent shape check too.
SMOKE_DATASET = "gcs"


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_cfg():
    from hydra import compose, initialize_config_dir

    config_dir = str((REPO_ROOT / "src" / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_simulator", overrides=[f"data={SMOKE_DATASET}"])
    return cfg.data


@pytest.fixture(scope="session")
def crowdes_cfg(data_cfg):
    from src.data.dotdict_bridge import to_crowdes_config

    return to_crowdes_config(data_cfg)


@pytest.fixture(scope="session")
def preprocessed_available(crowdes_cfg):
    import os

    path = os.path.join(crowdes_cfg["dataset"]["dataset_preprocessed_path"], "train", "image")
    if not os.path.isdir(path):
        pytest.skip(f"preprocessed dataset not found at {path}")
    return True
