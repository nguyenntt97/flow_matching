"""Subsystem 7b, the part that runs today: bounded falsification.

Design doc: ``survey/design/07_safety_cbf_formal_verification.md`` sec 1.2.

The design specifies compiling the tree to BehaVerify DSL and model-checking it
with nuXmv. Both are external and nuXmv's distribution is licence-gated, so no
milestone depends on them; ``src/flow2bt/verification.py`` emits the SMV text so
the path stays open.

What runs here instead is a bounded falsifier: search initial conditions for
counterexamples to the same properties, against **the code that actually runs**
rather than a hand-written model of it. That is a weaker guarantee -- absence of
a counterexample is not a proof -- and a stronger form of evidence about the
implementation, because a model checker verifies the model, and the gap between
a transition system and 400 lines of numpy is exactly where this project has
already found several bugs.

Properties, following design 07 sec 1.2:

``G !collision``      separation never drops below ``d_min``
``F at_goal``         every agent reaches its destination within the horizon
``G F moving``        no agent is stalled indefinitely short of its goal
                      (the deadlock the design predicts for two agents that
                      both choose Yield)

Counterexamples are returned as initial conditions, which is what design 07
sec 1.2.3 wants fed back into the guard thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.flow2bt.cbf import CBFConfig, CBFFilter
from src.flow2bt.features import Geometry, build_features
from src.flow2bt.primitives import integrate_step
from src.runtime.controllers import NominalController, TickState


@dataclass
class Episode:
    """One rollout's verdict against each property."""

    positions: np.ndarray           # (T, N, 2)
    min_separation: float
    reached: np.ndarray             # (N,) bool
    stalled: np.ndarray             # (N,) bool
    relaxed_ticks: int

    def violations(self, d_min: float, tolerance: float = 1e-3) -> list[str]:
        """Which properties this episode falsifies.

        ``tolerance`` exists because a barrier that is working converges to
        *exactly* ``d_min`` -- measured 0.4499 against a d_min of 0.45 -- and
        floating point then puts it a hair under. Reporting that as a collision
        would drown the real counterexamples in boundary noise.
        """
        out = []
        if self.min_separation < d_min - tolerance:
            out.append("G !collision")
        if not self.reached.all():
            out.append("F at_goal")
        if self.stalled.any():
            out.append("G F moving")
        return out


@dataclass
class Counterexample:
    property_name: str
    start: np.ndarray
    goal: np.ndarray
    preferred_speed: np.ndarray
    episode: Episode = field(repr=False)


def run_episode(
    start: np.ndarray,
    goal: np.ndarray,
    preferred_speed: np.ndarray,
    controller: NominalController,
    cbf: Optional[CBFFilter] = None,
    horizon: float = 30.0,
    dt: float = 0.05,
    arrival: float = 0.5,
    stall_speed: float = 0.05,
    stall_seconds: float = 4.0,
    feature_fn: Optional[Callable] = None,
    max_neighbors: int = 4,
    interaction_range: float = 4.0,
) -> Episode:
    """Drive the real controller + filter over one scenario. No scene, no emitter.

    Deliberately standalone: the properties are about the locomotion stack, and
    running it without ``CrowdESFramework`` keeps a falsification sweep to
    milliseconds per episode instead of minutes.
    """
    position = np.array(start, dtype=float, copy=True)
    count = len(position)
    direction = goal - position
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    velocity = np.divide(direction, np.maximum(norm, 1e-9)) * preferred_speed[:, None]

    controller.reset(count)
    steps = int(horizon / dt)
    trace = np.empty((steps + 1, count, 2))
    trace[0] = position

    reached = np.zeros(count, dtype=bool)
    slow_for = np.zeros(count)
    min_separation = np.inf
    relaxed = 0

    for step in range(steps):
        distance = np.linalg.norm(position[None, :, :] - position[:, None, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        if count > 1:
            min_separation = min(min_separation, float(distance.min()))

        take = min(max_neighbors, max(count - 1, 0))
        neighbor_position = np.zeros((count, max_neighbors, 2))
        neighbor_velocity = np.zeros((count, max_neighbors, 2))
        mask = np.zeros((count, max_neighbors), dtype=bool)
        if take:
            order = np.argsort(distance, axis=1)[:, :take]
            rows = np.arange(count)[:, None]
            neighbor_position[:, :take] = position[order]
            neighbor_velocity[:, :take] = velocity[order]
            mask[:, :take] = distance[rows, order] < interaction_range

        # Straight-line guidance: no navmesh here, so any failure is the
        # controller's or the filter's, not the planner's.
        toward = goal - position
        toward_norm = np.linalg.norm(toward, axis=1, keepdims=True)
        waypoint = position + np.divide(toward, np.maximum(toward_norm, 1e-9)) * np.minimum(
            toward_norm, (preferred_speed * 2.0)[:, None]
        )

        state = TickState(
            position=position, velocity=velocity, waypoint=waypoint, goal=goal,
            preferred_speed=preferred_speed,
            neighbor_position=neighbor_position, neighbor_velocity=neighbor_velocity,
            neighbor_mask=mask, dt=dt,
        )
        if feature_fn is not None:
            state.features = feature_fn(state)

        command = controller.acceleration(state)
        if cbf is not None:
            command, info = cbf.filter(
                u_nominal=command, position=position, velocity=velocity,
                neighbor_position=neighbor_position, neighbor_velocity=neighbor_velocity,
                neighbor_mask=mask,
            )
            relaxed += int(info["relaxed"].sum())

        position, velocity = integrate_step(position, velocity, command, dt)
        trace[step + 1] = position

        reached |= np.linalg.norm(position - goal, axis=1) < arrival
        speed = np.linalg.norm(velocity, axis=1)
        slow_for = np.where((speed < stall_speed) & ~reached, slow_for + dt, 0.0)

    return Episode(
        positions=trace,
        min_separation=min_separation,
        reached=reached,
        stalled=(slow_for >= stall_seconds) & ~reached,
        relaxed_ticks=relaxed,
    )


def head_on_scenario(rng, separation: float, offset: float, speed: float):
    """Two agents crossing, parameterised by gap and lateral offset."""
    start = np.array([[-separation / 2, 0.0], [separation / 2, offset]])
    goal = np.array([[separation / 2, offset], [-separation / 2, 0.0]])
    return start, goal, np.full(2, speed)


def corridor_scenario(rng, count: int, width: float, speed: float, min_spawn_gap: float = 0.6):
    """Counter-flow: half the agents each way down a corridor of given width.

    Two corrections, both about not blaming the controller for the generator:

    * Lanes are spaced rather than sampled independently. Drawing ``y``
      uniformly put agents on the same side millimetres apart in 128 of 200
      episodes, and a barrier cannot recover a state that starts unsafe.
    * ``count`` is then capped by what the corridor can physically hold. Four
      pedestrians abreast at 0.6 m spacing need 2.4 m; asking for that in a
      1.5 m corridor is an infeasible request, not a controller failure, and it
      still left 26 of 200 episodes starting in violation.

    The returned count may therefore be smaller than requested.
    """
    per_lane = max(1, int(width // min_spawn_gap))
    count = max(2, min(count, 2 * per_lane))
    half = count // 2
    per_side = [half, count - half]
    ys = []
    for side_count in per_side:
        usable = max(width - min_spawn_gap, 1e-6)
        lanes = np.linspace(-usable / 2, usable / 2, max(side_count, 1))
        jitter = rng.uniform(-min_spawn_gap / 4, min_spawn_gap / 4, max(side_count, 1))
        ys.append((lanes + jitter)[:side_count])
    y = np.concatenate(ys)

    start = np.stack([np.where(np.arange(count) < half, -6.0, 6.0), y], axis=1)
    goal = np.stack([np.where(np.arange(count) < half, 6.0, -6.0), y], axis=1)
    return start, goal, np.full(count, speed)


def falsify(
    controller_factory: Callable[[], NominalController],
    cbf_config: Optional[CBFConfig] = None,
    episodes: int = 200,
    seed: int = 0,
    feature_fn: Optional[Callable] = None,
    scenario: str = "corridor",
    horizon: float = 30.0,
    **episode_kwargs,
) -> dict:
    """Search initial conditions for counterexamples. Returns a summary.

    ``controller_factory`` is called per episode because a controller carries
    per-agent phase state; reusing one across scenarios with different agent
    counts would silently mix them.
    """
    rng = np.random.default_rng(seed)
    config = cbf_config or CBFConfig()
    cbf = CBFFilter(config)

    counterexamples: list[Counterexample] = []
    worst_separation = np.inf
    arrival_rate = []

    for _ in range(episodes):
        if scenario == "head_on":
            start, goal, speed = head_on_scenario(
                rng, rng.uniform(4.0, 12.0), rng.uniform(-0.6, 0.6), rng.uniform(0.8, 1.8)
            )
        elif scenario == "corridor":
            start, goal, speed = corridor_scenario(
                rng, int(rng.integers(2, 9)), rng.uniform(1.5, 4.0), rng.uniform(0.8, 1.8)
            )
        else:
            raise ValueError(f"unknown scenario {scenario!r}")

        episode = run_episode(
            start, goal, speed, controller_factory(), cbf,
            horizon=horizon, feature_fn=feature_fn, **episode_kwargs,
        )
        worst_separation = min(worst_separation, episode.min_separation)
        arrival_rate.append(float(episode.reached.mean()))

        for name in episode.violations(config.d_min):
            counterexamples.append(
                Counterexample(name, start.copy(), goal.copy(), speed.copy(), episode)
            )

    by_property: dict[str, int] = {}
    for item in counterexamples:
        by_property[item.property_name] = by_property.get(item.property_name, 0) + 1

    return {
        "episodes": episodes,
        "scenario": scenario,
        "d_min": config.d_min,
        "violations": by_property,
        "worst_separation": float(worst_separation),
        "mean_arrival_rate": float(np.mean(arrival_rate)),
        "counterexamples": counterexamples,
    }


def report(summary: dict) -> str:
    lines = [
        f"bounded falsification: {summary['episodes']} episodes, "
        f"scenario={summary['scenario']}, d_min={summary['d_min']}",
        f"  worst separation seen : {summary['worst_separation']:.4f} m",
        f"  mean arrival rate     : {summary['mean_arrival_rate']:.3f}",
    ]
    if not summary["violations"]:
        lines.append("  no counterexamples found (NOT a proof -- this is bounded search)")
    else:
        for name, count in sorted(summary["violations"].items()):
            lines.append(f"  {name:16s} violated in {count} episode(s)")
    return "\n".join(lines)
