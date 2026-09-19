"""The shim must bind upstream's packages and bind nothing else."""

from __future__ import annotations

import sys

import pytest


def test_bound_packages_resolve_into_the_submodule():
    import src._upstream as up

    import utils.config
    import CrowdES.simulator.simulator_model

    assert str(up.CROWDES_ROOT) in utils.config.__file__
    assert str(up.CROWDES_ROOT) in CrowdES.simulator.simulator_model.__file__


def test_sibling_directories_are_not_importable():
    """The submodule sits next to ``datasets/``, ``configs/``, ``checkpoints/``,
    ``scripts/`` and ``img/``. Putting it on sys.path would make those
    importable as top-level packages -- ``import datasets`` silently resolving
    to a data folder is a genuine hazard when HF datasets is not installed.
    Binding only what we need avoids it."""
    import src._upstream as up

    for name in ("configs", "scripts", "img", "checkpoints"):
        module = sys.modules.get(name)
        if module is None:
            continue
        origin = getattr(module, "__file__", None) or str(
            list(getattr(module, "__path__", []) or ["<none>"])[0]
        )
        assert str(up.CROWDES_ROOT) not in origin, f"{name} leaked from the submodule"


def test_sys_path_is_untouched():
    import src._upstream as up

    assert str(up.CROWDES_ROOT) not in sys.path


def test_reimport_is_idempotent():
    import importlib

    import src._upstream as up

    importlib.reload(up)
    import utils.config

    assert str(up.CROWDES_ROOT) in utils.config.__file__


def test_conflicting_top_level_module_is_rejected(monkeypatch):
    import types

    import src._upstream as up

    fake = types.ModuleType("utils")
    fake.__file__ = "/somewhere/else/utils.py"
    monkeypatch.setitem(sys.modules, "utils", fake)
    with pytest.raises(ImportError, match="shadows the CrowdES submodule"):
        up.install()
