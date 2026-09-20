"""Subsystem 7/8: the reactive runtime, as a subclass of ``CrowdESFramework``.

Design docs: ``survey/design/06`` (reactive execution) and ``08`` (synthesis).

Why a subclass and not a fork
-----------------------------
Everything except locomotion should be *identical* to the baseline or the
comparison means nothing: the same emitter, the same scene setup, the same
post-processing, the same metric call. So this inherits all of it and overrides
exactly one method, ``process_simulator``. The submodule stays untouched, as
``src/README.md`` requires.

What changes, precisely
-----------------------
Upstream commits a whole 2-second chunk per call and only refreshes neighbour
state at chunk boundaries (``inference_model.py:333``), so nothing that happens
inside a chunk can affect it. Here the loop steps at ``1 / (simulator_fps *
substeps)`` with neighbours, waypoints and guards recomputed every tick.
``process_orca_simulator`` (``inference_model.py:464``) already demonstrates the
per-frame shape by setting ``traj_fut_frame = 1``; this goes finer.

Two bugs inherited from upstream, fixed here rather than in the submodule
------------------------------------------------------------------------
* ``CrowdESFramework.__init__`` builds the ORCA pathfinder from ``self.navmesh``
  and ``self.H``, which ``initialize_scene`` only assigns afterwards -- so
  constructing with ``crowd_simulator.type = 'ORCA'`` raises ``AttributeError``
  before it can do anything. The ORCA setup is deferred to ``initialize_scene``,
  which makes ORCA usable as the third anchor in the comparison.
* ``densify_scenario`` linearly interpolates 5 fps -> 25 fps, and the collision
  metric is computed on those interpolated rows. ``dense_output`` emits the
  simulated path instead.

  **Measured, and it changed the default.** The worry was that re-interpolating
  a sub-sampled simulation would put back collisions the CBF prevented. It does
  not: the raw collision rate is 0.00000 either way. What emitting directly
  *does* do is bypass ``postprocess_trajectory``'s Kalman smoothing, and that
  smoothing is worth a 2x improvement in the Kinematics metric (0.451 with it,
  0.919 without). So ``dense_output`` defaults to **False** -- it costs realism
  and buys nothing. It stays available because the comparison is worth being
  able to re-run.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.flow2bt.cbf import CBFConfig, CBFFilter
from src.flow2bt.features import Geometry, build_features
from src.runtime.controllers import NominalController, TickState, WaypointController
from src.runtime.pathfollow import PathFollower

logger = logging.getLogger(__name__)


def make_framework_class():
    """Build the subclass lazily -- importing upstream pulls diffusers and pyrvo."""
    from CrowdES.inference_model import CrowdESFramework
    from utils.homography import image2world, world2image
    from utils.navmesh import PathFinderNew, filter_collinear_polygons
    from utils.trajectory import batched_nearest_nonzero_idx_kdtree

    class Flow2BTFramework(CrowdESFramework):
        def __init__(
            self,
            config,
            controller: Optional[NominalController] = None,
            cbf_config: Optional[CBFConfig] = None,
            use_cbf: bool = True,
            substeps: int = 4,
            dense_output: bool = False,
            agent_radius: float = 0.2,
            feature_fn=None,
            snap_to_walkable: bool = True,
            kalman_smooth: bool = True,
        ):
            # Upstream's ORCA branch would explode here; neutralise it and set
            # ORCA up in initialize_scene instead, where navmesh/H exist.
            requested_type = config.crowd_simulator.type
            config.crowd_simulator.type = "CrowdES"
            try:
                super().__init__(config)
            finally:
                config.crowd_simulator.type = requested_type
            self._orca_requested = requested_type == "ORCA"

            self.controller = controller or WaypointController()
            self.cbf = CBFFilter(cbf_config or CBFConfig()) if use_cbf else None
            self.substeps = max(1, int(substeps))
            self.dense_output = bool(dense_output)
            self.agent_radius = float(agent_radius)
            self.feature_fn = feature_fn
            # Three post-processing steps can move an agent after the filter has
            # finished with it. Each is switchable so its contribution to the
            # reported collision rate can be measured rather than argued about.
            self.snap_to_walkable = bool(snap_to_walkable)
            self.kalman_smooth = bool(kalman_smooth)

            self.tick_dt = 1.0 / (self.simulator_fps * self.substeps)
            self.diagnostics = {
                "ticks": 0, "agent_ticks": 0, "relaxed_agent_ticks": 0,
                "min_separation": np.inf, "cbf_interventions": 0,
                # A barrier only guarantees forward invariance from a state that
                # is already safe. The emitter places agents without regard to
                # separation, so an agent can be *born* inside d_min -- and no
                # filter can undo that. Counting it separately is the difference
                # between "the CBF failed" and "the CBF was never applicable".
                "spawned_in_violation": 0,
                "spawn_min_separation": np.inf,
                # Separation among pairs that were actually constrained. Upstream
                # caps neighbours at interaction_max_num_agents (4), so a fifth
                # agent closing in is simply not in anyone's QP.
                "min_separation_constrained": np.inf,
                "untracked_close_pairs": 0,
            }
            # Raw simulated positions in metres, before the walkable snap, the
            # Kalman smoother and the 5->25 fps interpolation. The collision
            # metric scores the *end* of that pipeline, so keeping the input lets
            # us say whether a residual collision came from the controller or
            # from the post-processing.
            self._raw_frames: list[tuple[int, np.ndarray]] = []

        # ------------------------------------------------------------ scene
        def initialize_scene(self, img, seg, walkable, navmesh, H, *args, **kwargs):
            super().initialize_scene(img, seg, walkable, navmesh, H, *args, **kwargs)

            # One PathFinder per scene rather than one per query -- see pathfollow.py.
            polygons = filter_collinear_polygons(navmesh["vertices"], navmesh["polygons"])
            vertices_meter = image2world(np.array(navmesh["vertices"])[..., [0, 1]], H)
            self._pathfinder = PathFinderNew(
                [[float(x), 0.0, float(y)] for x, y in vertices_meter], polygons
            )
            self.follower = PathFollower(self._plan_paths)

            self._position: dict[int, np.ndarray] = {}
            self._velocity: dict[int, np.ndarray] = {}
            self._dense: dict[int, list] = {}
            self._raw_frames = []
            self._frame_counter = 0

            if self._orca_requested:
                self.pf = self._pathfinder
                self.sim_agent_id_to_agent_id = {}

        def _plan_paths(self, starts, goals):
            paths = self._pathfinder.search_shortest_path(
                [(float(p[0]), 0.0, float(p[1])) for p in starts],
                [(float(g[0]), 0.0, float(g[1])) for g in goals],
            )
            out = []
            for path, start, goal in zip(paths, starts, goals):
                if len(path) == 0:
                    out.append(np.array([start, goal], dtype=float))
                else:
                    out.append(np.array([[x, y] for x, _, y in path], dtype=float))
            return out

        # --------------------------------------------------------- the loop
        def process_simulator(self):
            interaction_range = self.config.crowd_simulator.simulator.interaction_range
            max_neighbors = self.config.crowd_simulator.simulator.interaction_max_num_agents
            control_offset = self.config.crowd_simulator.simulator.control_time_offset
            before = len(self.agent_ids_in_current_scene)

            for frame in range(self.window_frame):
                self._admit_agents(frame)
                if not self.agent_ids_in_current_scene:
                    continue

                for _ in range(self.substeps):
                    self._step(interaction_range, max_neighbors, control_offset)
                    if not self.agent_ids_in_current_scene:
                        break
                self._record_frame()
                self._retire_agents()

            after = len(self.agent_ids_in_current_scene)
            self.statistics_dropped = self.statistics_added + before - after

        def _admit_agents(self, frame: int) -> None:
            for agent_id in list(self.new_agent_ids):
                parameters = self.agent_parameter[agent_id]
                if frame <= parameters["frame_origin"] < frame + 1:
                    origin = image2world(np.array(parameters["origin_xy"], dtype=float), self.H)
                    goal = image2world(np.array(parameters["goal_xy"], dtype=float), self.H)
                    direction = goal - origin
                    norm = np.linalg.norm(direction)
                    heading = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])

                    if self.agent_ids_in_current_scene:
                        existing = np.stack([
                            self._position[i] for i in self.agent_ids_in_current_scene
                        ])
                        gap = float(np.linalg.norm(existing - origin, axis=1).min())
                        self.diagnostics["spawn_min_separation"] = min(
                            self.diagnostics["spawn_min_separation"], gap
                        )
                        if self.cbf is not None and gap < self.cbf.config.d_min:
                            self.diagnostics["spawned_in_violation"] += 1

                    self._position[agent_id] = origin.astype(float)
                    self._velocity[agent_id] = heading * parameters["preferred_speed"]
                    self._dense[agent_id] = []
                    self.agent_ids_in_current_scene.append(agent_id)
                    self.new_agent_ids.remove(agent_id)
                    if agent_id not in self.agent_trajectory:
                        self.agent_trajectory[agent_id] = np.array(
                            parameters["origin_xy"], dtype=float
                        )[None, :]

        def _gather(self):
            ids = list(self.agent_ids_in_current_scene)
            position = np.stack([self._position[i] for i in ids])
            velocity = np.stack([self._velocity[i] for i in ids])
            goal = np.stack([
                image2world(np.array(self.agent_parameter[i]["goal_xy"], dtype=float), self.H)
                for i in ids
            ])
            speed = np.array([self.agent_parameter[i]["preferred_speed"] for i in ids])
            return ids, position, velocity, goal, speed

        @staticmethod
        def _neighbors(position, velocity, max_neighbors, interaction_range):
            """Upstream's rule, recomputed every tick instead of every chunk."""
            count = len(position)
            distance = np.linalg.norm(position[None, :, :] - position[:, None, :], axis=2)
            np.fill_diagonal(distance, np.inf)
            take = min(max_neighbors, max(count - 1, 0))

            neighbor_position = np.zeros((count, max_neighbors, 2))
            neighbor_velocity = np.zeros((count, max_neighbors, 2))
            mask = np.zeros((count, max_neighbors), dtype=bool)
            if take == 0:
                return neighbor_position, neighbor_velocity, mask, distance

            order = np.argsort(distance, axis=1)[:, :take]
            rows = np.arange(count)[:, None]
            within = distance[rows, order] < interaction_range
            neighbor_position[:, :take] = position[order]
            neighbor_velocity[:, :take] = velocity[order]
            mask[:, :take] = within
            return neighbor_position, neighbor_velocity, mask, distance

        def _step(self, interaction_range, max_neighbors, control_offset) -> None:
            ids, position, velocity, goal, speed = self._gather()
            neighbor_position, neighbor_velocity, mask, distance = self._neighbors(
                position, velocity, max_neighbors, interaction_range
            )

            self.follower.ensure(ids, position, goal)
            waypoint = self.follower.waypoints(ids, position, speed * control_offset)

            state = TickState(
                position=position, velocity=velocity, waypoint=waypoint, goal=goal,
                preferred_speed=speed,
                neighbor_position=neighbor_position, neighbor_velocity=neighbor_velocity,
                neighbor_mask=mask, dt=self.tick_dt,
            )
            if self.feature_fn is not None:
                state.features = self.feature_fn(state)

            command = self.controller.acceleration(state)

            if self.cbf is not None:
                command, info = self.cbf.filter(
                    u_nominal=command,
                    position=position, velocity=velocity,
                    neighbor_position=neighbor_position,
                    neighbor_velocity=neighbor_velocity,
                    neighbor_mask=mask,
                    agent_radius=self.agent_radius,
                )
                self.diagnostics["relaxed_agent_ticks"] += int(info["relaxed"].sum())
                self.diagnostics["cbf_interventions"] += int((info["intervention"] > 1e-6).sum())

            # Semi-implicit Euler, same integrator the primitives were fitted against.
            velocity = velocity + command * self.tick_dt
            position = position + velocity * self.tick_dt

            self.diagnostics["ticks"] += 1
            self.diagnostics["agent_ticks"] += len(ids)
            if len(ids) > 1:
                self.diagnostics["min_separation"] = min(
                    self.diagnostics["min_separation"], float(distance.min())
                )
                # Which close pairs were actually in each other's constraint set?
                constrained = np.zeros_like(distance, dtype=bool)
                take = mask.shape[1]
                order = np.argsort(distance, axis=1)[:, :take]
                rows = np.arange(len(ids))[:, None]
                constrained[rows, order] = mask[:, : order.shape[1]]
                constrained = constrained | constrained.T
                if constrained.any():
                    self.diagnostics["min_separation_constrained"] = min(
                        self.diagnostics["min_separation_constrained"],
                        float(distance[constrained].min()),
                    )
                if self.cbf is not None:
                    close = distance < self.cbf.config.d_min
                    self.diagnostics["untracked_close_pairs"] += int(
                        np.triu(close & ~constrained, 1).sum()
                    )

            for index, agent_id in enumerate(ids):
                self._position[agent_id] = position[index]
                self._velocity[agent_id] = velocity[index]

        def _record_frame(self) -> None:
            """Append one simulator-rate sample, snapped into the walkable area.

            The snap is upstream's (``inference_model.py:422``) and is applied at
            the same cadence, so the two paths are comparable.
            """
            ids = list(self.agent_ids_in_current_scene)
            if not ids:
                return
            position = np.stack([self._position[i] for i in ids])
            self._raw_frames.append((self._frame_counter, position.copy()))
            self._frame_counter += 1

            pixel = world2image(position, self.H)
            if self.snap_to_walkable:
                pixel = batched_nearest_nonzero_idx_kdtree(self.kdtree, pixel)

            for index, agent_id in enumerate(ids):
                self.agent_trajectory[agent_id] = np.concatenate(
                    [self.agent_trajectory[agent_id], pixel[index][None, :]], axis=0
                )
                if self.dense_output:
                    self._dense[agent_id].append(pixel[index].copy())

        def _retire_agents(self) -> None:
            upsample = self.dataset_fps // self.simulator_fps
            for agent_id in list(self.agent_ids_in_current_scene):
                position = self._position[agent_id]
                goal = image2world(
                    np.array(self.agent_parameter[agent_id]["goal_xy"], dtype=float), self.H
                )
                pixel = world2image(position, self.H)
                out_of_scene = bool(
                    (pixel <= 0).any()
                    or (pixel >= np.array(self.scene_size)[[1, 0]] - 1).any()
                )
                # Upstream's own thresholds (inference_model.py:453-456).
                if out_of_scene or np.linalg.norm(position - goal) < 0.5:
                    self.agent_ids_in_current_scene.remove(agent_id)
                    self.follower.forget(agent_id)

        def raw_collision_rate(self, threshold: float = 0.2) -> dict:
            """Collision rate on the simulated state, before any post-processing.

            ``utils/metrics.py`` scores the densified, snapped, Kalman-smoothed
            trajectory. Three of those steps can move an agent:

            * ``batched_nearest_nonzero_idx_kdtree`` snaps to the nearest walkable
              pixel, and two agents can snap onto the same one;
            * ``postprocess_trajectory`` Kalman-smooths at 5 fps;
            * ``densify_scenario`` interpolates 5 fps -> 25 fps.

            None of that is under the safety filter's control, so the difference
            between this number and the reported Collision is the part of the
            metric the filter cannot reach.
            """
            collisions = total = 0
            worst = np.inf
            for _, position in self._raw_frames:
                if len(position) < 2:
                    total += len(position)
                    continue
                distance = np.linalg.norm(position[:, None] - position[None, :], axis=-1)
                np.fill_diagonal(distance, np.inf)
                collisions += int(np.count_nonzero((distance < threshold).any(axis=0)))
                total += len(position)
                worst = min(worst, float(distance.min()))
            return {
                "threshold": threshold,
                "rate": collisions / max(total, 1),
                "rows": total,
                "min_separation": None if worst == np.inf else worst,
            }

        def postprocess_trajectory(self):
            """Kalman smoothing, made optional.

            Upstream smooths every trajectory at 5 fps with a constant-
            acceleration Kalman model (``inference_model.py:552-585``). That is a
            realism step, and it moves agents after the barrier has certified
            them.
            """
            if self.kalman_smooth:
                return super().postprocess_trajectory()
            return None

        # ------------------------------------------------------------ output
        def densify_scenario(self, interpolation_method="linear"):
            """Emit the simulated path instead of interpolating a sub-sampled one.

            See the module docstring: upstream interpolates 5 fps -> 25 fps and
            the collision metric scores the interpolated rows, which can
            reintroduce collisions the filter prevented. With ``dense_output``
            off this falls straight through to upstream's behaviour so the two
            can be compared.
            """
            if not self.dense_output:
                return super().densify_scenario(interpolation_method)

            upsample = self.dataset_fps // self.simulator_fps
            for agent_id, samples in self._dense.items():
                if len(samples) < 2:
                    continue
                agent_type = self.agent_parameter[agent_id]["agent_type"]
                origin = self.agent_parameter[agent_id]["frame_global"]
                track = np.asarray(samples, dtype=float)

                # The simulation advanced `substeps` times per simulator frame,
                # so upsampling to dataset_fps is still an interpolation -- but
                # of a path that was actually integrated, not of chunk endpoints.
                frames = np.arange(origin, origin + len(track)) * upsample
                target = np.arange(origin * upsample, (origin + len(track)) * upsample)
                x = np.interp(target, frames, track[:, 0])
                y = np.interp(target, frames, track[:, 1])
                for frame, px, py in zip(target, x, y):
                    if 0 <= frame < self.scenario_len:
                        self.scenario.append([agent_id, agent_type, frame, px, py])

    return Flow2BTFramework
