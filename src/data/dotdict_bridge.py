"""Convert a Hydra ``DictConfig`` into the ``DotDict`` upstream expects.

``BaseDataset``, ``SimulatorDataset`` and ``CrowdESFramework`` all do plain
attribute access on the merged config object upstream's ``get_config`` builds.
Three properties of that object drive this module:

1. It has no interpolation machinery, so every ``${...}`` must already be
   resolved -- an unresolved string would reach ``os.path.join`` verbatim.
2. It holds plain ``list`` objects: upstream calls ``tuple(environment_size)``
   and iterates ``environment_types`` with ``getattr(self, it)``.
3. ``SimulatorDataset.__init__`` ends with ``pickle.dump(self.__dict__)``, and
   ``self.config`` is in there. ``DotDict`` defines ``__getstate__``/
   ``__setstate__`` for exactly this; a ``DictConfig`` in the cache would be a
   portability hazard.

And one hazard: ``DotDict.__getattr__`` is ``dict.get``, so a missing or
misspelled key yields ``None`` silently and blows up several frames away
(``tuple(None)``). Hence the explicit required-key check below.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from src._upstream import DotDict

#: Subtrees handed to upstream verbatim. Everything else in the data config
#: (the ``loader`` block) is ours and must not cross the bridge.
UPSTREAM_SUBTREES = ("dataset", "crowd_simulator", "crowd_emitter")

#: Keys upstream dereferences during dataset construction and model config.
REQUIRED_KEYS = (
    "dataset.dataset_name",
    "dataset.dataset_path",
    "dataset.dataset_preprocessed_path",
    "dataset.dataset_fps",
    "crowd_simulator.type",
    "crowd_simulator.simulator.environment_types",
    "crowd_simulator.simulator.environment_size",
    "crowd_simulator.simulator.environment_pixel_meter",
    "crowd_simulator.simulator.history_length",
    "crowd_simulator.simulator.future_length",
    "crowd_simulator.simulator.window_stride",
    "crowd_simulator.simulator.interaction_range",
    "crowd_simulator.simulator.interaction_max_num_agents",
    "crowd_simulator.simulator.control_time_offset",
    "crowd_simulator.simulator.latent_dim",
    "crowd_simulator.simulator.hidden_dim",
    "crowd_simulator.simulator.checkpoint_dir",
)


def _dotify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _dotify(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return [_dotify(v) for v in obj]
    return obj


def _lookup(config: Any, dotted: str) -> Any:
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
        if node is None:
            return None
    return node


def validate_required(config: DotDict, keys: tuple[str, ...] = REQUIRED_KEYS) -> None:
    missing = [k for k in keys if _lookup(config, k) is None]
    if missing:
        raise KeyError(
            "upstream config is missing required keys (DotDict returns None for these "
            f"rather than raising, so catch them here): {missing}"
        )


def validate_paths(config: DotDict) -> None:
    """Fail early, and in terms of our config, on paths upstream will open.

    Without this the first failure is a bare FileNotFoundError raised inside
    ``BaseDataset.__init__``, several frames deep and naming a path with a
    ``..`` in it that nobody wrote.
    """
    import os

    preprocessed = config.dataset.dataset_preprocessed_path
    if not os.path.isdir(preprocessed):
        raise FileNotFoundError(
            f"dataset.dataset_preprocessed_path does not exist: {preprocessed}"
        )

    # BaseDataset does open(join(dataset_path, '..', 'segmentation_classes.json')).
    # open() walks '..' through the filesystem, so dataset_path itself has to be
    # a real, traversable directory -- a textually correct path is not enough.
    classes = os.path.join(config.dataset.dataset_path, "..", "segmentation_classes.json")
    if not os.path.isfile(classes):
        raise FileNotFoundError(
            f"BaseDataset will fail to load segmentation classes from {classes!r}.\n"
            f"dataset.dataset_path is {config.dataset.dataset_path!r}; it must be an "
            "EXISTING directory whose parent holds segmentation_classes.json, because "
            "open() resolves '..' through the filesystem. See the dataset_path comment "
            "in src/configs/data/base.yaml."
        )


def to_crowdes_config(data_cfg: DictConfig, validate: bool = True) -> DotDict:
    """Build the merged ``DotDict`` upstream's dataset and model classes expect."""
    plain = OmegaConf.to_container(data_cfg, resolve=True, throw_on_missing=True)
    if not isinstance(plain, dict):
        raise TypeError(f"expected a mapping at the data config root, got {type(plain)!r}")

    config = _dotify({k: plain[k] for k in UPSTREAM_SUBTREES if k in plain})
    if validate:
        validate_required(config)
        validate_paths(config)
    return config
