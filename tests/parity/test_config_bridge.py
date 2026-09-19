"""The DotDict handed to upstream must be shaped exactly as upstream expects."""

from __future__ import annotations

import pickle

import pytest

from src._upstream import DotDict
from src.data.dotdict_bridge import to_crowdes_config, validate_required


def test_is_a_dotdict_with_attribute_access(crowdes_cfg):
    assert isinstance(crowdes_cfg, DotDict)
    assert crowdes_cfg.dataset.dataset_name == "gcs"
    assert crowdes_cfg.crowd_simulator.simulator.history_length == 10


def test_lists_are_plain_lists(crowdes_cfg):
    """Upstream does ``tuple(environment_size)`` and iterates
    ``environment_types``; a ListConfig would leak OmegaConf into the cache."""
    simulator = crowdes_cfg.crowd_simulator.simulator
    assert type(simulator.environment_size) is list
    assert type(simulator.environment_types) is list
    assert tuple(simulator.environment_size) == (64, 64)


def test_no_unresolved_interpolations(crowdes_cfg):
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, str):
            yield node

    unresolved = [s for s in walk(crowdes_cfg) if "${" in s]
    assert not unresolved, f"unresolved interpolations reached upstream: {unresolved}"


def test_paths_are_absolute(crowdes_cfg):
    assert crowdes_cfg.dataset.dataset_preprocessed_path.startswith("/")
    assert crowdes_cfg.dataset.dataset_path.startswith("/")


def test_segmentation_classes_join_resolves(crowdes_cfg):
    """BaseDataset builds this path textually from the *raw* dataset_path, which
    need not exist -- only the joined file does."""
    import os

    path = os.path.join(crowdes_cfg.dataset.dataset_path, "..", "segmentation_classes.json")
    assert os.path.isfile(path), path


def test_checkpoint_dir_keeps_the_format_placeholder(crowdes_cfg):
    template = crowdes_cfg.crowd_simulator.simulator.checkpoint_dir
    assert "{}" in template
    assert template.format("gcs").endswith("/gcs/dynamics/")


def test_picklable(crowdes_cfg):
    """SimulatorDataset pickles self.__dict__, and self.config is in it."""
    restored = pickle.loads(pickle.dumps(crowdes_cfg))
    assert restored.dataset.dataset_name == "gcs"


def test_missing_required_key_raises():
    """DotDict returns None for missing keys instead of raising, so the bridge
    has to catch it rather than letting ``tuple(None)`` fail three frames on."""
    broken = DotDict({"dataset": DotDict({"dataset_name": "x"})})
    with pytest.raises(KeyError, match="missing required keys"):
        validate_required(broken)


def test_loader_block_does_not_cross_the_bridge(data_cfg):
    config = to_crowdes_config(data_cfg)
    assert "loader" not in config
