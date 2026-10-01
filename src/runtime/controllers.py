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
    #: (N,) stable identity per row. Without it a stateful controller can only
    #: key its per-agent state by row, and has to drop all of it whenever the
    #: crowd changes size.
    agent_ids: Optional[np.ndarray] = None

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


#: Timers accumulate ``dt`` per tick, and 10 x 0.05 is 0.49999999999999994.
_CLOCK_EPS = 1e-9


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
        rotate_forcing: bool = True,
        finite_stop: bool = True,
        stop_duration: Optional[float] = None,
        stop_refractory: Optional[float] = None,
    ):
        self.tree = tree
        self.bank = bank
        self.tau_scale = float(tau_scale)
        self.rotate_to_nav = bool(rotate_to_nav)
        #: A primitive counts as complete when it is within this of its goal, or
        #: when its phase clock has run out -- whichever happens first.
        self.arrival_tolerance = float(arrival_tolerance)
        self.phase_floor = float(phase_floor)
        #: Evaluate the forcing term in the fitted (navmesh) frame and rotate it
        #: out, instead of applying fitted-frame weights to world axes. See
        #: ``DMPBank.acceleration``. ``False`` reproduces the runs made before
        #: the fix and exists only for that comparison.
        self.rotate_forcing = bool(rotate_forcing)
        self.completions = 0
        #: The displacement each primitive covers, in its fitted frame.
        self._displacement = np.stack([d.displacement for d in bank.dmps])

        # ---- Finite-duration stationary leaves.
        #
        # Every induced guard is dominated by the agent's own speed, and at speed
        # ~0 every guard on the path to the stop leaf passes. So a stopped agent
        # is routed to ``stop``, which holds it at ~0, which routes it to ``stop``
        # again. Nothing in the tree says "resume". Measured on eth: once the CBF
        # brakes an agent to rest in a crowd it never moves again, so fewer
        # agents finish and Travel Time EMD reaches 1.9-4.6 against CrowdES's 0.60.
        #
        # A leaf whose goal lies within ``arrival_tolerance`` of its entry cannot
        # complete by arrival: it "arrives" on the tick it starts. Such a leaf
        # completes by *time* instead. After ``stop_duration`` it returns FAILURE,
        # and for ``stop_refractory`` more it keeps failing. In BT semantics its
        # guarded Sequence then fails, and the enclosing Fallback moves on to its
        # next child: the nearest walking subtree in the dendrogram. Both
        # durations default to the leaf's own ``tau``, i.e. one demonstration's
        # length of standing, then long enough walking for the speed-dominated
        # guards to see a moving agent again. An agent the CBF still holds at rest
        # after the refractory period re-enters ``stop`` legitimately.
        self.finite_stop = bool(finite_stop)
        self.stationary = np.linalg.norm(self._displacement, axis=1) < self.arrival_tolerance
        self.stop_duration = (
            bank.tau * self.tau_scale if stop_duration is None
            else np.full(len(bank), float(stop_duration))
        )
        self.stop_refractory = (
            bank.tau * self.tau_scale if stop_refractory is None
            else np.full(len(bank), float(stop_refractory))
        )
        self.stop_yields = 0
        self._stop_enabled = True
        if self.finite_stop:
            self._install_stop_terminators()

        self._ids: Optional[np.ndarray] = None
        self._leaf: Optional[np.ndarray] = None
        self._phase: Optional[np.ndarray] = None
        self._entry: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None
        self._rotation: Optional[np.ndarray] = None
        self._elapsed: Optional[np.ndarray] = None
        self._refractory: Optional[np.ndarray] = None

    # ------------------------------------------------------------ per-agent state
    def reset(self, num_agents: int) -> None:
        self._ids = None
        self._leaf = np.full(num_agents, -1)
        self._phase = np.ones(num_agents)
        self._entry = None
        self._goal = None
        self._rotation = None
        self._elapsed = np.zeros(num_agents)
        self._refractory = np.zeros(num_agents)

    def _fresh(self, state: TickState, rows: np.ndarray) -> dict:
        count = len(rows)
        return {
            "_leaf": np.full(count, -1),
            "_phase": np.ones(count),
            "_entry": state.position[rows].copy(),
            "_goal": state.waypoint[rows].copy(),
            "_rotation": np.tile(np.eye(2), (count, 1, 1)),
            "_elapsed": np.zeros(count),
            "_refractory": np.zeros(count),
        }

    def _ensure(self, state: TickState) -> None:
        """Line per-agent state up with this tick's rows.

        With ``agent_ids`` the state follows each agent across admissions and
        retirements. Keyed by row alone, every change in crowd size wiped every
        agent's phase, goal and stop timer -- and the emitter admits or retires
        someone on most frames, so a stop timer would rarely get to expire.
        """
        count = len(state)
        ids = state.agent_ids
        if ids is None:
            if self._leaf is None or len(self._leaf) != count or self._goal is None:
                self.reset(count)
                for key, value in self._fresh(state, np.arange(count)).items():
                    setattr(self, key, value)
            return

        ids = np.asarray(ids)
        if self._ids is not None and len(self._ids) == count and np.array_equal(self._ids, ids):
            return

        fresh = self._fresh(state, np.arange(count))
        if self._ids is not None:
            previous = {agent: row for row, agent in enumerate(self._ids.tolist())}
            for row, agent in enumerate(ids.tolist()):
                old = previous.get(agent)
                if old is None:
                    continue
                for key in fresh:
                    fresh[key][row] = getattr(self, key)[old]
        for key, value in fresh.items():
            setattr(self, key, value)
        self._ids = ids.copy()

    # ------------------------------------------------------------- stop leaves
    def _install_stop_terminators(self) -> None:
        """Give every stationary action leaf a terminator reading this controller's timers.

        The tree is the one ``BTController`` owns for the run; the terminator is
        the hook ``bt.Action`` provides for exactly this (RUNNING until the
        action ends). Trees without a walkable root (test doubles) are left alone.
        """
        from src.flow2bt.bt import FAILURE, RUNNING, action_leaves

        root = getattr(self.tree, "root", None)
        if root is None:
            return
        for action in action_leaves(root):
            if not self.stationary[action.leaf_index]:
                continue

            def terminator(features, _leaf=action.leaf_index):
                if not self._stop_enabled or self._refractory is None:
                    return np.full(features.shape[0], RUNNING)
                exhausted = (self._leaf == _leaf) & (self._elapsed >= self.stop_duration[_leaf] - _CLOCK_EPS)
                blocked = (self._refractory > _CLOCK_EPS) | exhausted
                return np.where(blocked, FAILURE, RUNNING)

            action.terminator = terminator

    def _select(self, features: np.ndarray) -> np.ndarray:
        """Tick the tree; an agent with nothing left to run keeps its stop leaf.

        If an exhausted stop leaf's failure propagates to the root, no walking
        subtree claimed the agent. Rather than hand it to leaf 0 by default,
        re-tick those agents with stop leaves allowed: a finite leaf yields only
        when something else can take over.
        """
        leaf = self.tree.tick(features).action
        missing = leaf < 0
        if missing.any() and self.finite_stop:
            self._stop_enabled = False
            try:
                leaf = np.where(missing, self.tree.tick(features).action, leaf)
            finally:
                self._stop_enabled = True
        return np.where(leaf < 0, 0, leaf)          # totality guard; see assembly.py

    def _nav_rotation(self, state: TickState) -> np.ndarray:
        """``(N, 2, 2)`` rotation from the fitted frame (+x = navmesh direction) to world."""
        if not self.rotate_to_nav:
            return np.tile(np.eye(2), (len(state), 1, 1))
        direction = state.waypoint - state.position
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.where(
            norm > 1e-6, np.divide(direction, np.maximum(norm, 1e-12)),
            np.tile([1.0, 0.0], (len(state), 1)),
        )
        cos, sin = direction[:, 0], direction[:, 1]
        return np.stack([np.stack([cos, -sin], axis=1), np.stack([sin, cos], axis=1)], axis=1)

    def _aim(self, leaf: np.ndarray, state: TickState, rotation: np.ndarray) -> np.ndarray:
        """Fixed goal for this execution: entry + the leaf's own displacement.

        The primitives are fitted in a frame rotated onto the navmesh direction,
        so replaying one means rotating its displacement back onto the direction
        the agent is currently being guided along. Without the rotation a leaf
        fitted from agents walking one way would drag agents walking the other
        way backwards.
        """
        return state.position + np.einsum("nij,nj->ni", rotation, self._displacement[leaf])

    def active_leaf(self, state: TickState) -> np.ndarray:
        self._ensure(state)
        return self._select(state.features)

    def acceleration(self, state: TickState) -> np.ndarray:
        self._ensure(state)
        leaf = self._select(state.features)

        switched = leaf != self._leaf
        if self.finite_stop:
            # Leaving a stationary leaf after its full duration opens the
            # refractory window; being preempted out of it early does not.
            was_stationary = (self._leaf >= 0) & self.stationary[np.maximum(self._leaf, 0)]
            prev = np.maximum(self._leaf, 0)
            yielded = was_stationary & switched & (self._elapsed >= self.stop_duration[prev] - _CLOCK_EPS)
            self._refractory = np.where(yielded, self.stop_refractory[prev], self._refractory)
            self.stop_yields += int(yielded.sum())
            in_stop = self.stationary[leaf]
            self._elapsed = np.where(in_stop, np.where(switched, 0.0, self._elapsed) + state.dt, 0.0)
            self._refractory = np.maximum(self._refractory - state.dt, 0.0)

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
        rotation = self._nav_rotation(state)
        self._goal = np.where(restart[:, None], self._aim(leaf, state, rotation), self._goal)
        # Fixed per execution, like the goal it was used to aim.
        self._rotation = np.where(restart[:, None, None], rotation, self._rotation)
        self._leaf = leaf

        tau = self.bank.tau[leaf] * self.tau_scale
        acceleration = self.bank.acceleration(
            leaf, state.position, state.velocity, self._goal, self._entry, self._phase, tau,
            rotation=self._rotation if self.rotate_forcing else None,
        )
        self._phase = self.bank.step_phase(self._phase, state.dt, tau)
        return acceleration
