"""Derived simulator dataset: upstream's builder, two behavioural overrides.

Everything about sample construction is inherited. In particular ``__getitem__``
is NOT overridden: its environment crop indexes ``env_base_pointer`` with an
``(h, w)`` ordering while offsetting by ``startpoint[[1, 0]]`` and passing the
result through ``world2image``, which treats axis 0 as ``x``. Whatever the
intent, the identical code runs in ``CrowdES/inference_model.py::process_simulator``,
so training and inference see the same distribution. Reproducing upstream's
numbers means reproducing that, so the crop stays exactly as inherited.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import shutil

import utils.dataloader.simulator_dataloader as upstream_module
from src._upstream import BaseDataset, SimulatorDataset
from src.util.compat import check_pandas

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def capped_build_workers(n_jobs: int | None):
    """Temporarily cap the worker count upstream's dataset builder uses.

    ``SimulatorDataset.__init__`` hardcodes ``Parallel(n_jobs=256)``. An
    explicit ``n_jobs`` beats ``joblib.parallel_backend(...)``, so the usual
    context manager does nothing here -- verified. On a 32-core box that means
    256 processes, each unpickling its own copy of the scene DataFrames.

    Results are unaffected: ``process_scene`` is independent per scene, so the
    worker count cannot change what is built. This rebinds ``Parallel`` inside
    the upstream module for the duration of the build rather than editing the
    submodule.
    """
    if n_jobs is None:
        yield
        return

    original = upstream_module.Parallel

    class _CappedParallel(original):
        def __init__(self, *args, **kwargs):
            kwargs["n_jobs"] = n_jobs
            super().__init__(*args, **kwargs)

    upstream_module.Parallel = _CappedParallel
    try:
        yield
    finally:
        upstream_module.Parallel = original


class ParitySimulatorDataset(SimulatorDataset):
    """``SimulatorDataset`` with an auditable test-split guard and a versioned cache.

    Two changes, both about safety rather than behaviour:

    1. ``BaseDataset._check_phase`` allows ``phase='test'`` only when
       ``self.__class__.__name__`` is literally ``'EvaluationDataset'`` or
       ``'SyntheticDataset'`` -- a name check no subclass can satisfy. We keep
       the guard's intent (the test split stays blind during training) but make
       the exemption an explicit, greppable argument.
    2. Upstream's cache path is keyed only on the phase, so a config change
       silently reloads stale tensors. We key it on a hash of the
       build-affecting config as well.
    """

    def __init__(self, config, phase, allow_test_split: bool = False, cache_path: str | None = None):
        check_pandas()

        # Consumed by _check_phase, which super().__init__ calls before
        # returning, so it has to be set first.
        self._allow_test_split = bool(allow_test_split)

        if cache_path is None:
            super().__init__(config, phase)
            return

        if os.path.exists(cache_path):
            # Mirrors upstream's cache-hit branch: BaseDataset first (it sets
            # scene_list, dataset_path, label2id), then overwrite from the
            # pickle. Skips the joblib build entirely.
            BaseDataset.__init__(self, config, phase)
            with open(cache_path, "rb") as handle:
                for key, value in pickle.load(handle).items():
                    setattr(self, key, value)
            logger.info("loaded %s dataset from cache %s (%d samples)", phase, cache_path, len(self))
            return

        upstream_cache = os.path.join(
            config.dataset.dataset_preprocessed_path, "cache", f"{phase}_simulator.pickle"
        )
        pre_existing = os.path.exists(upstream_cache)

        # Let upstream build and write its own cache file, untouched, then take
        # a fingerprinted copy. Reimplementing the ~120-line builder here would
        # be a fork that silently drifts from the submodule.
        super().__init__(config, phase)

        if not os.path.exists(upstream_cache):
            logger.warning("upstream wrote no cache at %s; nothing to fingerprint", upstream_cache)
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if pre_existing:
            # An upstream-built cache was already there and super().__init__
            # just loaded from it. Copy rather than move, so a plain upstream
            # run still finds its own file.
            shutil.copy2(upstream_cache, cache_path)
        else:
            os.replace(upstream_cache, cache_path)
        logger.info("cached %s dataset at %s (%d samples)", phase, cache_path, len(self))

    def _check_phase(self):
        if self.phase == "test" and not self._allow_test_split:
            raise ValueError(
                "phase='test' requires allow_test_split=True. The test split must stay "
                "blind during training -- upstream validates on the train split and never "
                "touches test. Only src/evaluate_agent.py may opt in."
            )
