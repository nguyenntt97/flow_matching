"""Nominal controllers -- what the CBF filter is handed before it intervenes.

Keeping this pluggable is what makes the ablation possible. "Flow2BT beats
CrowdES on collisions" is not a useful claim if the safety filter alone
accounts for all of it, so the runtime must be able to run:

* ``WaypointController`` -- follow the A* waypoint at preferred pace, no
  avoidance at all. The floor: whatever this plus the CBF achieves is what the
  safety layer contributes on its own, and anything the behavior tree claims has
  to be measured against it, not against CrowdES.
* ``BTController`` -- the induced Behavior Tree selecting DMP leaves.

Both emit an acceleration, so the CBF sees the same kind of input either way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TickState:
    """Everything a controller can see on one tick. Metres, world frame."""

    position: np.ndarray            # (N, 2)
    velocity: np.ndarray            # (N, 2)
    waypoint: np.ndarray            # (N, 2) absolute c_nav
    goal: np.ndarray                # (N, 2)
    preferred_speed: np.ndarray     # (N,)
    neighbor_position: np.ndarray   # (N, K, 2)
    neighbor_velocity: np.ndarray   # (N, K, 2)
    neighbor_mask: np.ndarray       # (N, K) bool
    features: Optional[np.ndarray] = None   # (N, d), built lazily by the runtime
    dt: float = 0.05

    def __len__(self) -> int:
        return int(self.position.shape[0])


class NominalController(ABC):
    name = "nominal"

    def reset(self, num_agents: int) -> None:
        """Called when the active-agent set changes. Default: stateless."""

    @abstractmethod
    def acceleration(self, state: TickState) -> np.ndarray:
        """(N, 2) commanded acceleration."""

    def active_leaf(self, state: TickState) -> Optional[np.ndarray]:
        """(N,) index of the primitive each agent is running, when meaningful."""
        return None


class WaypointController(NominalController):
    """Proportional velocity tracking toward the navmesh waypoint.

    The deliberate floor described in the module docstring: it has no notion of
    other agents. ``gain`` is how hard it corrects velocity error, in 1/s.
    """

    name = "waypoint"

    def __init__(self, gain: float = 3.0):
        self.gain = float(gain)

    def acceleration(self, state: TickState) -> np.ndarray:
        direction = state.waypoint - state.position
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.divide(direction, np.maximum(norm, 1e-9))
        desired = direction * state.preferred_speed[:, None]
        return self.gain * (desired - state.velocity)


class BTController(NominalController):
    """The induced Behavior Tree, selecting a DMP leaf per agent per tick.

    Phase handling is the subtle part. A DMP's canonical clock only means
    anything relative to when its primitive was *entered*, so an agent that the
    tree preempts onto a different leaf must restart that leaf's phase -- the
    design's whole reactivity story is agents switching mid-movement, and
    carrying a stale phase across a switch would replay the wrong part of the
    new shape. ``_leaf`` tracks what each agent ran last tick so switches can be
    detected and only the switching agents reset.

    The action lifecycle is the other half, and leaving it out is not a subtle
    failure. A DMP is a *finite* movement: design 05 sec 1.3 has the leaf return
    SUCCESS once ``||x - g|| <= eps``, at which point the tree re-ticks it and a
    fresh execution begins from the live waypoint. Re-aiming only on a *switch*
    means an agent that keeps selecting the same leaf reaches that leaf's goal
    once and then sits on it forever, because the spring has nothing left to
    pull against. Measured consequence on eth: 71 of 406 agents ever reached
    their destination and the travel-time EMD was 11.9 against CrowdES's 0.60.
    So a primitive is re-entered when it completes as well as when it switches.
    """

    name = "bt"

    def __init__(
        self,
        tree,
        bank,
        tau_scale: float = 1.0,
        rotate_to_nav: bool = True,
        arrival_tolerance: float = 0.25,
        phase_floor: float = 0.05,
    ):
        self.tree = tree
        self.bank = bank
        self.tau_scale = float(tau_scale)
        self.rotate_to_nav = bool(rotate_to_nav)
        #: A primitive counts as complete when it is within this of its goal, or
        #: when its phase clock has run out -- whichever happens first.
        self.arrival_tolerance = float(arrival_tolerance)
        self.phase_floor = float(phase_floor)
        self.completions = 0
        self._leaf: Optional[np.ndarray] = None
        self._phase: Optional[np.ndarray] = None
        self._entry: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None
        #: The displacement each primitive covers, in its fitted frame.
        self._displacement = np.stack([d.displacement for d in bank.dmps])

    def reset(self, num_agents: int) -> None:
        self._leaf = np.full(num_agents, -1)
        self._phase = np.ones(num_agents)
        self._entry = None
        self._goal = None

    def _ensure(self, state: TickState) -> None:
        if self._leaf is None or len(self._leaf) != len(state):
            self.reset(len(state))
        if self._entry is None or len(self._entry) != len(state):
            self._entry = state.position.copy()
        if self._goal is None or len(self._goal) != len(state):
            self._goal = state.waypoint.copy()

    def _aim(self, leaf: np.ndarray, state: TickState) -> np.ndarray:
        """Fixed goal for this execution: entry + the leaf's own displacement.

        The primitives are fitted in a frame rotated onto the navmesh direction,
        so replaying one means rotating its displacement back onto the direction
        the agent is currently being guided along. Without the rotation a leaf
        fitted from agents walking one way would drag agents walking the other
        way backwards.
        """
        displacement = self._displacement[leaf]
        if not self.rotate_to_nav:
            return state.position + displacement

        direction = state.waypoint - state.position
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.where(
            norm > 1e-6, np.divide(direction, np.maximum(norm, 1e-12)),
            np.tile([1.0, 0.0], (len(state), 1)),
        )
        cos, sin = direction[:, 0], direction[:, 1]
        rotated = np.stack([
            cos * displacement[:, 0] - sin * displacement[:, 1],
            sin * displacement[:, 0] + cos * displacement[:, 1],
        ], axis=1)
        return state.position + rotated

    def active_leaf(self, state: TickState) -> np.ndarray:
        self._ensure(state)
        return self.tree.tick(state.features).action

    def acceleration(self, state: TickState) -> np.ndarray:
        self._ensure(state)
        leaf = self.tree.tick(state.features).action
        leaf = np.where(leaf < 0, 0, leaf)          # totality guard; see assembly.py

        switched = leaf != self._leaf
        # SUCCESS / timeout, per design 05 sec 1.3 -- re-enter the primitive.
        reached = np.linalg.norm(self._goal - state.position, axis=1) < self.arrival_tolerance
        expired = self._phase < self.phase_floor
        completed = (reached | expired) & ~switched
        self.completions += int(completed.sum())

        restart = switched | completed
        self._phase = np.where(restart, 1.0, self._phase)
        self._entry = np.where(restart[:, None], state.position, self._entry)
        # The goal is fixed for the duration of one execution, which is what
        # makes the spring term relax as the agent advances instead of pulling
        # at full strength forever; a new execution takes a fresh one.
        self._goal = np.where(restart[:, None], self._aim(leaf, state), self._goal)
        self._leaf = leaf

        tau = self.bank.tau[leaf] * self.tau_scale
        acceleration = self.bank.acceleration(
            leaf, state.position, state.velocity, self._goal, self._entry, self._phase, tau
        )
        self._phase = self.bank.step_phase(self._phase, state.dt, tau)
        return acceleration
