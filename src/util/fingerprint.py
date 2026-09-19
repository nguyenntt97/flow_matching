"""Content fingerprint for the dataset cache filename.

Upstream caches the built dataset at ``<preprocessed>/cache/{phase}_simulator.pickle``
and never invalidates it: change ``history_length`` or ``interaction_range`` and
you silently reload tensors built under the old settings. We hash the subset of
the config that actually affects the build into the filename instead.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

#: Keys under ``crowd_simulator.simulator`` consumed by ``process_scene`` and the
#: environment-raster construction. Anything here changes the built tensors.
BUILD_KEYS = (
    "environment_types",
    "environment_size",
    "environment_pixel_meter",
    "history_length",
    "future_length",
    "window_stride",
    "interaction_range",
    "interaction_max_num_agents",
    "control_time_offset",
)


def build_fingerprint(simulator_cfg: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> str:
    """Return a short stable hash of the build-affecting config subset."""
    payload: dict[str, Any] = {k: simulator_cfg.get(k) for k in BUILD_KEYS}
    missing = [k for k, v in payload.items() if v is None]
    if missing:
        raise KeyError(f"cannot fingerprint the dataset cache; missing config keys: {missing}")
    if extra:
        payload["_extra"] = dict(extra)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]
