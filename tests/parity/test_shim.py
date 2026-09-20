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


def test_submodule_root_is_on_sys_path_for_workers():
    """The sys.modules binding is process-local. joblib/loky workers build the
    dataset and must import utils.dataloader.simulator_dataloader themselves,
    resolving it through the sys.path that loky copies from the parent."""
    import src._upstream as up

    assert str(up.CROWDES_ROOT) in sys.path


def test_a_worker_process_can_import_upstream():
    """Regression test for `BrokenProcessPool: A task has failed to
    un-serialize`, which is what a worker that cannot import `utils` looks
    like from the parent."""
    from joblib import Parallel, delayed

    import src._upstream as up

    def where():
        import utils.config

        return utils.config.__file__

    (origin,) = Parallel(n_jobs=1, backend="loky")([delayed(where)()])
    assert str(up.CROWDES_ROOT) in origin


def test_the_config_survives_a_round_trip_through_a_worker():
    """process_scene receives the DotDict as a task argument, so the worker
    has to be able to unpickle it -- which needs utils.config importable
    there."""
    from joblib import Parallel, delayed

    from src._upstream import DotDict

    payload = DotDict({"dataset": DotDict({"dataset_name": "gcs"})})
    (result,) = Parallel(n_jobs=1, backend="loky")(
        [delayed(lambda c: c.dataset.dataset_name)(payload)]
    )
    assert result == "gcs"


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
