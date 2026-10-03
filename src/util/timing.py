"""Optional wall-clock timing of locomotion inference, for comparing rates.

The comparison this exists for is "how often can each locomotion model make a
decision for the whole crowd": one batched CrowdES / flow-teacher forward that
commits a 2 s chunk, against one Behavior Tree tick (features + tree + DMP step)
that commits ``tick_dt``. Every measurement therefore records three things:

* wall time of the call, with ``torch.cuda.synchronize`` on both sides when the
  call runs on the GPU -- otherwise an asynchronous launch is timed, not the work;
* how many agents the call served;
* how much simulated time the call commits the crowd to.

``summary()`` turns those into per-call latency, the decision rate the call
could sustain for the whole crowd (``1000 / mean_ms``), and the real-time
factor (simulated seconds per wall second). The first ``warmup`` calls of each
label are dropped: they carry CUDA context setup and allocator growth.

Timing is off unless a ``timing=true`` override asks for it, and nothing here
changes what is computed.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

import numpy as np


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except ImportError:
        pass


class InferenceTimer:
    def __init__(self, warmup: int = 3):
        self.warmup = int(warmup)
        self._records: dict[str, list[tuple[float, int, float]]] = defaultdict(list)

    @contextmanager
    def measure(self, label: str, agents: int, sim_seconds: float, cuda: bool = False):
        """Time the enclosed block as one call of ``label``."""
        if cuda:
            _cuda_sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                _cuda_sync()
            self._records[label].append(
                (time.perf_counter() - start, int(agents), float(sim_seconds))
            )

    def summary(self) -> dict:
        out = {}
        for label, records in self._records.items():
            kept = records[self.warmup:] if len(records) > self.warmup else records
            wall = np.array([r[0] for r in kept])
            agents = np.array([r[1] for r in kept], dtype=float)
            sim = np.array([r[2] for r in kept])
            mean_ms = float(wall.mean() * 1e3)
            out[label] = {
                "calls": len(kept),
                "warmup_dropped": len(records) - len(kept),
                "mean_ms": mean_ms,
                "p50_ms": float(np.percentile(wall, 50) * 1e3),
                "p95_ms": float(np.percentile(wall, 95) * 1e3),
                "max_ms": float(wall.max() * 1e3),
                "mean_agents": float(agents.mean()),
                "us_per_agent": float((wall / np.maximum(agents, 1)).mean() * 1e6),
                "sim_seconds_per_call": float(sim.mean()),
                # How often this call could run back to back for the whole crowd.
                "max_rate_hz": 1e3 / mean_ms if mean_ms > 0 else float("inf"),
                # Simulated seconds produced per wall second spent in this call.
                "realtime_factor": float(sim.sum() / wall.sum()) if wall.sum() > 0 else float("inf"),
            }
        return out

    def report(self, title: str = "INFERENCE TIMING") -> str:
        rows = self.summary()
        if not rows:
            return f"{title}: nothing timed"
        lines = [
            title,
            f"  {'label':<28} {'calls':>7} {'agents':>7} {'mean ms':>9} {'p95 ms':>9} "
            f"{'us/agent':>9} {'max Hz':>9} {'sim s/call':>10} {'RT factor':>10}",
        ]
        for label, r in sorted(rows.items()):
            lines.append(
                f"  {label:<28} {r['calls']:>7d} {r['mean_agents']:>7.1f} {r['mean_ms']:>9.3f} "
                f"{r['p95_ms']:>9.3f} {r['us_per_agent']:>9.2f} {r['max_rate_hz']:>9.1f} "
                f"{r['sim_seconds_per_call']:>10.3f} {r['realtime_factor']:>10.1f}"
            )
        lines.append(
            "  max Hz = 1000 / mean ms: how often the call could run back to back "
            "for the whole crowd. RT factor = simulated s per wall s."
        )
        return "\n".join(lines)


class TimedSimulator:
    """Wrap a simulator-protocol model so every forward is timed.

    Attribute access falls through to the wrapped model, then to ``config_source``
    -- ``CrowdESFramework`` reads ``.config.latent_dim`` off its simulator, which
    our Lightning-loaded models do not carry.
    """

    def __init__(self, model, timer: Optional[InferenceTimer], prefix: str, sim_seconds: float,
                 config_source=None):
        self._model = model
        self._timer = timer
        self._prefix = prefix
        self._sim_seconds = float(sim_seconds)
        self._config_source = config_source

    def __call__(self, traj_hist, *args, **kwargs):
        if self._timer is None:                     # adapter only (``.config``)
            return self._model(traj_hist, *args, **kwargs)
        label = f"{self._prefix}/{'sampled' if kwargs.get('sampling', True) else 'deterministic'}"
        cuda = getattr(traj_hist, "is_cuda", False)
        with self._timer.measure(label, traj_hist.shape[0], self._sim_seconds, cuda=cuda):
            return self._model(traj_hist, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return getattr(self._model, name)
        except AttributeError:
            if self._config_source is not None:
                return getattr(self._config_source, name)
            raise


def time_method(obj, method: str, timer: InferenceTimer, label: str,
                agents_fn, sim_seconds_fn, cuda: bool = False) -> None:
    """Replace ``obj.method`` in place with a timed version of itself."""
    original = getattr(obj, method)

    def timed(*args, **kwargs):
        with timer.measure(label, agents_fn(obj), sim_seconds_fn(obj), cuda=cuda):
            return original(*args, **kwargs)

    setattr(obj, method, timed)


def maybe_timer(enabled: bool, warmup: int = 3) -> Optional[InferenceTimer]:
    return InferenceTimer(warmup) if enabled else None
