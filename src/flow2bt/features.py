"""Subsystem 4 (first half): the physical feature space the guards are fitted in.

Design doc: ``survey/design/04_condition_node_induction.md`` sec 1.1.

Every feature the design asks for is computable from the batch upstream already
builds (``utils/dataloader/simulator_dataloader.py:465-497``), so nothing new
has to be extracted from the dataset:

========================  ===================================================
Design feature            Source
========================  ===================================================
``c_nav - c_t``           ``control`` (B, 2), already relative to the agent
``v_rel``                 ``neighbor`` (B, N, T, 2), last two frames x fps
``TTC``                   closed form from the above, with a contact radius
``d_lat``                 neighbour offset perpendicular to heading
``SDF``, clearance        distance transform of the ``environment`` crop
========================  ===================================================

Frames and units
----------------
Everything in the batch is metres, translated so the agent's *last observed
position* is the origin (``target_startpoint = target_traj[history_length-1]``).
So ``control`` is already ``c_nav - c_t``, and a neighbour's last history entry
is already its position relative to the agent. No re-centring is needed, and
adding one would be a silent bug.

The zero-padding trap
---------------------
Unused neighbour slots are left as **zeros**, not marked absent
(``simulator_dataloader.py:296-299``). A zeroed slot's last position is
``(0, 0)`` -- which is exactly the agent's own position, i.e. a neighbour at
zero distance. Read naively, every agent looks permanently in collision with
four phantoms. ``neighbor_mask`` distinguishes them by whether the whole
history is identically zero; a real neighbour cannot be, because it would have
to be exactly coincident with the agent for all ten frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

#: Order matters: it is the column order of the feature matrix, the order
#: ``Condition.describe`` renders, and the order the SMT export emits.
FEATURE_NAMES = [
    "speed",          # m/s, own
    "nav_dist",       # m, distance to the navmesh waypoint
    "nav_heading",    # rad, signed heading error to the waypoint
    "goal_dist",      # m
    "ttc",            # s, to the most threatening neighbour (capped)
    "d_long",         # m, that neighbour's along-track offset (+ = ahead)
    "d_lat",          # m, ditto across-track (+ = left)
    "closing_speed",  # m/s, + = approaching
    "neighbor_count",
    "clearance",      # m, distance to the nearest non-traversable cell
]

#: TTC is capped rather than infinite so it can sit in a linear model; anything
#: beyond this is "no threat" and the exact value carries no information.
TTC_CAP = 10.0

#: Segmentation channel indices that count as traversable, matching
#: ``crowd_simulator.locomotion.walkable_class: [tree, sidewalk, road]`` against
#: datasets/segmentation_classes.json order
#: (building, structure, bush, grass, tree, sidewalk, road).
WALKABLE_CHANNELS = (4, 5, 6)


@dataclass
class Geometry:
    """Raw per-agent geometry. Feeds the CBF filter, which needs vectors not features."""

    position: np.ndarray            # (N, 2) -- zeros by construction
    velocity: np.ndarray            # (N, 2)
    neighbor_position: np.ndarray   # (N, K, 2)
    neighbor_velocity: np.ndarray   # (N, K, 2)
    neighbor_mask: np.ndarray       # (N, K) bool
    heading: np.ndarray             # (N, 2) unit
    control: np.ndarray             # (N, 2)
    sdf: Optional[np.ndarray] = None            # (N,)
    sdf_gradient: Optional[np.ndarray] = None   # (N, 2) unit


def _unit(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, vectors / np.maximum(norm, 1e-12), fallback)


def time_to_collision(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    contact_radius: float,
    cap: float = TTC_CAP,
) -> np.ndarray:
    r"""Time until two discs of combined radius ``contact_radius`` touch.

    Solves :math:`\|p + v t\| = r` for the smaller positive root. Returns
    ``cap`` when the pair never closes to contact, and ``0`` when already
    overlapping -- the design's ``||p|| / ||v_rel||`` is only the special case of
    a head-on approach with zero radius, and it reports a finite, alarming TTC
    for pairs that will comfortably pass each other.
    """
    a = (relative_velocity**2).sum(-1)
    b = 2.0 * (relative_position * relative_velocity).sum(-1)
    c = (relative_position**2).sum(-1) - contact_radius**2

    ttc = np.full(a.shape, cap, dtype=float)
    np.copyto(ttc, 0.0, where=c <= 0.0)                      # already in contact

    discriminant = b**2 - 4.0 * a * c
    solvable = (a > 1e-9) & (discriminant >= 0.0) & (c > 0.0)
    root = np.zeros_like(ttc)
    np.divide(
        -b - np.sqrt(np.where(solvable, discriminant, 0.0)),
        2.0 * np.where(solvable, a, 1.0),
        out=root,
        where=solvable,
    )
    hit = solvable & (root >= 0.0)
    return np.where(hit, np.minimum(root, cap), ttc)


def extract_geometry(
    traj_hist: np.ndarray,
    neighbor: np.ndarray,
    control: np.ndarray,
    environment: Optional[np.ndarray] = None,
    fps: float = 5.0,
    pixel_meter: float = 0.25,
) -> Geometry:
    """Pull vectors out of a CrowdES batch. All inputs are numpy, batch-first."""
    velocity = (traj_hist[:, -1] - traj_hist[:, -2]) * fps
    position = traj_hist[:, -1]                       # zeros by construction

    # See "The zero-padding trap" in the module docstring.
    neighbor_mask = np.abs(neighbor).sum(axis=(-1, -2)) > 0.0
    neighbor_position = neighbor[:, :, -1, :]
    neighbor_velocity = (neighbor[:, :, -1, :] - neighbor[:, :, -2, :]) * fps

    # Heading: direction of travel, falling back to the navmesh waypoint for a
    # stationary agent, and to +x if that is degenerate too (upstream's own
    # convention in normalize_rotation_scale).
    default = np.tile(np.array([1.0, 0.0]), (traj_hist.shape[0], 1))
    heading = _unit(velocity, _unit(control, default))

    sdf = sdf_gradient = None
    if environment is not None:
        sdf, sdf_gradient = clearance_from_environment(environment, pixel_meter)

    return Geometry(
        position=position, velocity=velocity,
        neighbor_position=neighbor_position, neighbor_velocity=neighbor_velocity,
        neighbor_mask=neighbor_mask, heading=heading, control=control,
        sdf=sdf, sdf_gradient=sdf_gradient,
    )


def clearance_from_environment(
    environment: np.ndarray,
    pixel_meter: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the agent to the nearest non-traversable cell, and its gradient.

    The crop is agent-centred by construction, so the agent sits at the middle
    pixel. A crop that is entirely traversable has no zero to measure from;
    ``distance_transform_edt`` returns the array diagonal there, which is the
    honest answer ("at least this far") given a 16 m window.
    """
    num_agents, _, height, width = environment.shape
    walkable = environment[:, WALKABLE_CHANNELS, :, :].sum(axis=1) > 0.5

    centre_row, centre_col = height // 2, width // 2
    sdf = np.empty(num_agents)
    gradient = np.zeros((num_agents, 2))

    for index in range(num_agents):
        distance = distance_transform_edt(walkable[index]) * pixel_meter
        sdf[index] = distance[centre_row, centre_col]
        # Uphill points away from the obstacle, which is the direction safety lies.
        d_row = distance[min(centre_row + 1, height - 1), centre_col] - distance[
            max(centre_row - 1, 0), centre_col
        ]
        d_col = distance[centre_row, min(centre_col + 1, width - 1)] - distance[
            centre_row, max(centre_col - 1, 0)
        ]
        gradient[index] = (d_col, d_row)

    default = np.tile(np.array([1.0, 0.0]), (num_agents, 1))
    return sdf, _unit(gradient, default)


def build_features(
    geometry: Geometry,
    goal: np.ndarray,
    attr: Optional[np.ndarray] = None,
    contact_radius: float = 0.4,
) -> np.ndarray:
    """The (N, len(FEATURE_NAMES)) matrix the guards are evaluated on.

    The neighbour features describe the *most threatening* neighbour -- lowest
    TTC, not nearest. A close neighbour walking alongside is not the one a guard
    should fire on; one further away and closing fast is.
    """
    num_agents, num_neighbors, _ = geometry.neighbor_position.shape

    relative_position = geometry.neighbor_position - geometry.position[:, None, :]
    relative_velocity = geometry.neighbor_velocity - geometry.velocity[:, None, :]

    ttc_all = time_to_collision(relative_position, relative_velocity, contact_radius)
    ttc_all = np.where(geometry.neighbor_mask, ttc_all, TTC_CAP)

    threat = np.argmin(ttc_all, axis=1)
    rows = np.arange(num_agents)
    ttc = ttc_all[rows, threat]
    offset = relative_position[rows, threat]
    closing = -(relative_position[rows, threat] * relative_velocity[rows, threat]).sum(-1)
    closing /= np.maximum(np.linalg.norm(offset, axis=-1), 1e-6)

    heading = geometry.heading
    normal = np.stack([-heading[:, 1], heading[:, 0]], axis=1)      # left normal
    d_long = (offset * heading).sum(-1)
    d_lat = (offset * normal).sum(-1)

    # A masked-out threat carries no geometry; zero it so the guard sees
    # "nothing there" rather than a phantom at the origin.
    any_neighbor = geometry.neighbor_mask.any(axis=1)
    d_long = np.where(any_neighbor, d_long, 0.0)
    d_lat = np.where(any_neighbor, d_lat, 0.0)
    closing = np.where(any_neighbor, closing, 0.0)

    nav = geometry.control
    nav_dist = np.linalg.norm(nav, axis=-1)
    nav_unit = _unit(nav, heading)
    nav_heading = np.arctan2(
        (nav_unit * normal).sum(-1), (nav_unit * heading).sum(-1)
    )

    clearance = geometry.sdf if geometry.sdf is not None else np.zeros(num_agents)

    return np.stack(
        [
            np.linalg.norm(geometry.velocity, axis=-1),
            nav_dist,
            nav_heading,
            np.linalg.norm(goal, axis=-1),
            ttc,
            d_long,
            d_lat,
            closing,
            geometry.neighbor_mask.sum(axis=1).astype(float),
            clearance,
        ],
        axis=1,
    )
