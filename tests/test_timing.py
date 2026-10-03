import time

import numpy as np
import pytest

from src.util.timing import InferenceTimer, TimedSimulator, time_method


class _Model:
    def __call__(self, traj_hist, *args, **kwargs):
        time.sleep(0.002)
        return "out"


class _Released:
    class config:
        latent_dim = 8


def test_warmup_calls_are_dropped_per_label():
    timer = InferenceTimer(warmup=2)
    for _ in range(5):
        with timer.measure("a", agents=4, sim_seconds=2.0):
            pass
    with timer.measure("b", agents=4, sim_seconds=2.0):
        pass
    summary = timer.summary()
    assert summary["a"]["calls"] == 3 and summary["a"]["warmup_dropped"] == 2
    assert summary["b"]["calls"] == 1, "a label with fewer calls than warmup keeps them"


def test_rates_follow_from_wall_time_and_simulated_time():
    timer = InferenceTimer(warmup=0)
    for _ in range(3):
        with timer.measure("tick", agents=10, sim_seconds=0.05):
            time.sleep(0.01)
    row = timer.summary()["tick"]
    assert row["mean_ms"] == pytest.approx(10.0, rel=0.5)
    assert row["max_rate_hz"] == pytest.approx(1e3 / row["mean_ms"])
    assert row["realtime_factor"] == pytest.approx(0.05 / (row["mean_ms"] / 1e3), rel=1e-6)
    assert row["us_per_agent"] == pytest.approx(row["mean_ms"] * 1e3 / 10, rel=0.05)


def test_timed_simulator_labels_by_sampling_and_passes_through():
    timer = InferenceTimer(warmup=0)
    model = TimedSimulator(_Model(), timer, "scene", 2.0, config_source=_Released())
    batch = np.zeros((16, 10, 2))
    assert model(batch, sampling=True) == "out"
    model(batch, sampling=False)
    assert set(timer.summary()) == {"scene/sampled", "scene/deterministic"}
    assert timer.summary()["scene/sampled"]["mean_agents"] == 16
    assert model.config.latent_dim == 8, "framework reads .config off its simulator"


def test_timed_simulator_without_timer_is_a_plain_adapter():
    model = TimedSimulator(_Model(), None, "scene", 2.0, config_source=_Released())
    assert model(np.zeros((2, 10, 2))) == "out"


def test_time_method_wraps_in_place():
    class Thing:
        count = 3

        def step(self):
            return 42

    timer = InferenceTimer(warmup=0)
    thing = Thing()
    time_method(thing, "step", timer, "t", lambda o: o.count, lambda o: 0.05)
    assert thing.step() == 42
    assert timer.summary()["t"]["mean_agents"] == 3
