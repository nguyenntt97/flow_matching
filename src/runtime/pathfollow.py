"""Cached navmesh path following -- the fix for the runtime's worst hotspot.

``get_control_point`` calls ``shortest_pathfinder``
(``utils/navmesh.py:193``), which constructs a **fresh ``PathFinderNew`` -- full
navmesh plus BVH -- on every call, and the A* underneath uses a linear-scan open
list (``pathfinder/navmesh/navmesh_graph.py:90-97``). Upstream pays that once
per agent per 2-second chunk, which is tolerable. A receding-horizon loop
re-planning every tick would pay it 40x more often, and that alone would swamp
any latency claim the design makes.

So the A* runs **once per agent**, and each subsequent tick projects the agent
onto its cached polyline and reads the lookahead point off it. Re-planning
happens only when the agent has drifted off its own path (it was pushed aside)
or its goal changed.

This means any speed-up measured later has to be attributed carefully: caching
the path helps *any* controller, including the baseline, so it is not a
contribution of the behavior tree.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _project_onto_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[float, float]:
    """Return ``(arc_length_at_closest_point, distance_to_path)``."""
    starts, ends = polyline[:-1], polyline[1:]
    segments = ends - starts
    lengths_sq = np.maximum((segments**2).sum(-1), 1e-12)

    t = np.clip(((point - starts) * segments).sum(-1) / lengths_sq, 0.0, 1.0)
    closest = starts + t[:, None] * segments
    distances = np.linalg.norm(closest - point, axis=1)

    index = int(np.argmin(distances))
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(segments, axis=1))])
    arc = cumulative[index] + t[index] * np.sqrt(lengths_sq[index])
    return float(arc), float(distances[index])


def _point_at_arc(polyline: np.ndarray, arc: float) -> np.ndarray:
    """Point at ``arc`` along the polyline, extrapolating past the end.

    Extrapolation matches ``locate_point_on_path`` (``utils/navmesh.py:257-271``),
    which does the same "because agents too slow to reach the goal at the end of
    the path" -- so a lookahead longer than what is left does not pin the
    waypoint onto the goal and stall the agent.
    """
    segments = polyline[1:] - polyline[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

    if arc <= 0.0:
        return polyline[0].copy()
    if arc >= cumulative[-1]:
        overshoot = arc - cumulative[-1]
        direction = segments[-1] / max(lengths[-1], 1e-9)
        return polyline[-1] + direction * overshoot

    index = int(np.searchsorted(cumulative, arc) - 1)
    index = max(0, min(index, len(segments) - 1))
    remaining = arc - cumulative[index]
    return polyline[index] + segments[index] * (remaining / max(lengths[index], 1e-9))


class PathFollower:
    """Per-agent cached polylines in metres, with lookahead and lazy re-planning."""

    def __init__(self, plan_fn, replan_distance: float = 2.0):
        """``plan_fn(starts, goals) -> list of (P, 2) polylines in metres``."""
        self.plan_fn = plan_fn
        self.replan_distance = float(replan_distance)
        self.paths: dict[int, np.ndarray] = {}
        self.goals: dict[int, np.ndarray] = {}
        self.stats = {"planned": 0, "replanned": 0, "followed": 0}

    def forget(self, agent_id: int) -> None:
        self.paths.pop(agent_id, None)
        self.goals.pop(agent_id, None)

    def ensure(self, agent_ids, positions: np.ndarray, goals: np.ndarray) -> None:
        """Plan for any agent that has no usable path. One batched A* call."""
        needed, starts, wanted = [], [], []
        for index, agent_id in enumerate(agent_ids):
            cached = self.paths.get(agent_id)
            stale = cached is None or not np.allclose(self.goals[agent_id], goals[index])
            if not stale:
                _, distance = _project_onto_polyline(positions[index], cached)
                stale = distance > self.replan_distance
                if stale:
                    self.stats["replanned"] += 1
            if stale:
                needed.append(agent_id)
                starts.append(positions[index])
                wanted.append(goals[index])

        if not needed:
            return
        for agent_id, polyline, goal in zip(needed, self.plan_fn(starts, wanted), wanted):
            polyline = np.asarray(polyline, dtype=float)
            if len(polyline) < 2:
                polyline = np.stack([polyline.reshape(-1, 2)[0], goal])
            self.paths[agent_id] = polyline
            self.goals[agent_id] = np.asarray(goal, dtype=float)
        self.stats["planned"] += len(needed)

    def waypoints(self, agent_ids, positions: np.ndarray, lookahead: np.ndarray) -> np.ndarray:
        """Point ``lookahead`` metres further along each agent's own path."""
        out = np.empty((len(agent_ids), 2))
        for index, agent_id in enumerate(agent_ids):
            polyline = self.paths[agent_id]
            arc, _ = _project_onto_polyline(positions[index], polyline)
            out[index] = _point_at_arc(polyline, arc + lookahead[index])
        self.stats["followed"] += len(agent_ids)
        return out
