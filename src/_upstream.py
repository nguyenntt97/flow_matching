"""Bind the CrowdES submodule's packages and re-export the API ``src/`` depends on.

Import this before any bare ``utils.`` / ``CrowdES.`` import. Idempotent.

Upstream ships no ``__init__.py`` anywhere, so ``utils`` and ``CrowdES`` are
PEP-420 namespace packages. Putting ``third_party/crowdes`` on ``sys.path``
would therefore be doubly unsafe: a regular ``utils`` module anywhere else on
the path beats the namespace portion, and the submodule's sibling directories
(``datasets``, ``configs``, ``checkpoints``, ``scripts``, ``img``) would become
importable top-level names that could shadow real packages -- ``import
datasets`` resolving to a data folder, for instance.

So we do not touch ``sys.path`` at all. We bind exactly the two packages we
need straight into ``sys.modules`` with an explicit search location. Submodule
resolution then walks ``__path__``, ``sys.modules`` is consulted before any
finder, and nothing else in the submodule becomes importable.

Not covered, deliberately: ``utils/preprocessor/*.py`` does ``from homography
import ...``, which only resolves when ``utils/`` itself is on ``sys.path``.
Those modules are for regenerating the preprocessed dataset, which we never do
(``datasets/preprocessed/`` is already populated), so the broken import is out
of scope.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CROWDES_ROOT = REPO_ROOT / "third_party" / "crowdes"

#: Top-level names bound out of the submodule. Nothing else from it is importable.
_BOUND = ("utils", "CrowdES")


def _origin(module) -> str:
    file = getattr(module, "__file__", None)
    if file:
        return str(Path(file).resolve())
    search = list(getattr(module, "__path__", []) or [])
    return str(Path(search[0]).resolve()) if search else "<unknown>"


def _bind(name: str) -> None:
    """Register ``name`` in sys.modules as a package rooted in the submodule."""
    existing = sys.modules.get(name)
    if existing is not None:
        origin = _origin(existing)
        if not origin.startswith(str(CROWDES_ROOT)):
            raise ImportError(
                f"top-level module {name!r} is already imported from {origin!r}, "
                f"which shadows the CrowdES submodule at {CROWDES_ROOT}. "
                "Import src._upstream before anything else, or remove the conflict."
            )
        return

    location = CROWDES_ROOT / name
    if not location.is_dir():
        raise RuntimeError(f"expected package directory {location} in the CrowdES submodule")

    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(location)]
    sys.modules[name] = importlib.util.module_from_spec(spec)


def install() -> None:
    if not (CROWDES_ROOT / "utils" / "config.py").is_file():
        raise RuntimeError(
            f"CrowdES submodule missing or empty at {CROWDES_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    for name in _BOUND:
        _bind(name)


install()

# Curated re-exports. Every upstream symbol src/ uses is listed here, so the
# dependency surface on the submodule is one auditable block.
from utils.config import DotDict, get_config  # noqa: E402
from utils.dataloader.base_dataloader import BaseDataset  # noqa: E402
from utils.dataloader.evaluation_dataloader import EvaluationDataset  # noqa: E402
from utils.dataloader.simulator_dataloader import SimulatorDataset  # noqa: E402
from utils.homography import image2world, world2image  # noqa: E402
from utils.utils import reproducibility_settings  # noqa: E402
from CrowdES.simulator.simulator_config import CrowdESSimulatorConfig  # noqa: E402
from CrowdES.simulator.simulator_model import (  # noqa: E402
    CrowdESSimulatorModel,
    CrowdESSimulatorModelOutput,
)

# Intentionally NOT re-exported -- they pull POT, dtaidistance, diffusers and
# pathfinder.pyrvo, costing seconds of import time. src/evaluate_scene.py
# imports them lazily:
#   utils.metrics, utils.visualization, CrowdES.inference_model

__all__ = [
    "REPO_ROOT",
    "CROWDES_ROOT",
    "DotDict",
    "get_config",
    "BaseDataset",
    "EvaluationDataset",
    "SimulatorDataset",
    "image2world",
    "world2image",
    "reproducibility_settings",
    "CrowdESSimulatorConfig",
    "CrowdESSimulatorModel",
    "CrowdESSimulatorModelOutput",
]
