"""Subsystem 4, first half: the physical feature space."""

import numpy as np
import pytest

from src.flow2bt.features import (
    FEATURE_NAMES,
    TTC_CAP,
    WALKABLE_CHANNELS,
    build_features,
    clearance_from_environment,
    extract_geometry,
    time_to_collision,
)

FPS = 5.0


def history(velocity, steps=10):
    """A straight-line history ending at the origin, as the batch supplies it."""
    offsets = np.arange(-(steps - 1), 1)[:, None] * np.asarray(velocity)[None, :] / FPS
    return offsets


# ------------------------------------------------------------------- TTC
def test_ttc_of_a_head_on_pair():
    """Closing at 2 m/s from 4.4 m apart with 0.4 m contact -> 2.0 s."""
    ttc = time_to_collision(
        np.array([[4.4, 0.0]]), np.array([[-2.0, 0.0]]), contact_radius=0.4
    )
    assert ttc[0] == pytest.approx(2.0)


def test_ttc_is_capped_when_the_pair_never_touches():
    """A neighbour passing 3 m abeam must not read as a threat.

    The design's ||p|| / ||v_rel|| would report a finite, alarming number here;
    solving for actual contact reports the cap.
    """
    ttc = time_to_collision(
        np.array([[10.0, 3.0]]), np.array([[-2.0, 0.0]]), contact_radius=0.4
    )
    assert ttc[0] == TTC_CAP


def test_ttc_of_a_receding_pair_is_capped():
    ttc = time_to_collision(
        np.array([[2.0, 0.0]]), np.array([[2.0, 0.0]]), contact_radius=0.4
    )
    assert ttc[0] == TTC_CAP


def test_ttc_is_zero_when_already_overlapping():
    ttc = time_to_collision(
        np.array([[0.3, 0.0]]), np.array([[-1.0, 0.0]]), contact_radius=0.4
    )
    assert ttc[0] == 0.0


def test_ttc_of_a_stationary_pair_is_capped():
    ttc = time_to_collision(np.array([[2.0, 0.0]]), np.zeros((1, 2)), contact_radius=0.4)
    assert ttc[0] == TTC_CAP


# -------------------------------------------------------------- geometry
def test_velocity_and_heading_come_out_of_the_history():
    traj = history([1.3, 0.0])[None, ...]
    geometry = extract_geometry(
        traj, np.zeros((1, 4, 10, 2)), np.array([[2.6, 0.0]]), fps=FPS
    )
    assert np.allclose(geometry.velocity[0], [1.3, 0.0])
    assert np.allclose(geometry.heading[0], [1.0, 0.0])
    assert np.allclose(geometry.position[0], [0.0, 0.0])


def test_zero_padded_neighbours_are_masked_out():
    """The trap this module exists to avoid: a zeroed slot is not a neighbour at
    zero distance. If this regresses, every agent reads as permanently colliding
    with four phantoms."""
    neighbor = np.zeros((1, 4, 10, 2))
    neighbor[0, 0] = history([-1.3, 0.0]) + np.array([3.0, 0.0])   # one real
    geometry = extract_geometry(
        history([1.3, 0.0])[None, ...], neighbor, np.array([[2.6, 0.0]]), fps=FPS
    )
    assert list(geometry.neighbor_mask[0]) == [True, False, False, False]


def test_stationary_agent_falls_back_to_the_navmesh_heading():
    traj = np.zeros((1, 10, 2))
    geometry = extract_geometry(
        traj, np.zeros((1, 4, 10, 2)), np.array([[0.0, 2.0]]), fps=FPS
    )
    assert np.allclose(geometry.heading[0], [0.0, 1.0])


def test_fully_degenerate_heading_falls_back_to_x():
    geometry = extract_geometry(
        np.zeros((1, 10, 2)), np.zeros((1, 4, 10, 2)), np.zeros((1, 2)), fps=FPS
    )
    assert np.allclose(geometry.heading[0], [1.0, 0.0])


# ------------------------------------------------------------- clearance
def test_clearance_measures_distance_to_the_nearest_obstacle():
    environment = np.zeros((1, 8, 64, 64))
    environment[0, WALKABLE_CHANNELS[0]] = 1.0          # all traversable
    environment[0, WALKABLE_CHANNELS[0], :, 40:] = 0.0  # wall 8 px right of centre
    sdf, gradient = clearance_from_environment(environment, pixel_meter=0.25)
    assert sdf[0] == pytest.approx(8 * 0.25, abs=0.25)
    # Uphill is away from the wall, i.e. toward -col.
    assert gradient[0, 0] < 0


def test_clearance_in_open_space_is_large():
    environment = np.zeros((1, 8, 64, 64))
    environment[0, WALKABLE_CHANNELS[0]] = 1.0
    sdf, _ = clearance_from_environment(environment, pixel_meter=0.25)
    assert sdf[0] > 4.0


# --------------------------------------------------------------- features
def test_feature_matrix_shape_and_names():
    geometry = extract_geometry(
        history([1.3, 0.0])[None, ...], np.zeros((1, 4, 10, 2)),
        np.array([[2.6, 0.0]]), fps=FPS,
    )
    features = build_features(geometry, goal=np.array([[10.0, 0.0]]))
    assert features.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_head_on_neighbour_produces_the_expected_physics():
    """Hand-computable: oncoming walker 4 m ahead, both at 1.3 m/s."""
    neighbor = np.zeros((1, 4, 10, 2))
    neighbor[0, 0] = history([-1.3, 0.0]) + np.array([4.0, 0.0])
    geometry = extract_geometry(
        history([1.3, 0.0])[None, ...], neighbor, np.array([[2.6, 0.0]]), fps=FPS
    )
    row = dict(zip(FEATURE_NAMES, build_features(geometry, np.array([[10.0, 0.0]]))[0]))

    assert row["speed"] == pytest.approx(1.3)
    assert row["closing_speed"] == pytest.approx(2.6)
    assert row["d_long"] == pytest.approx(4.0)       # dead ahead
    assert row["d_lat"] == pytest.approx(0.0, abs=1e-9)
    assert row["ttc"] == pytest.approx((4.0 - 0.4) / 2.6, abs=1e-6)
    assert row["neighbor_count"] == 1.0
    assert row["nav_heading"] == pytest.approx(0.0, abs=1e-9)


def test_lateral_offset_is_signed_left_positive():
    neighbor = np.zeros((1, 4, 10, 2))
    neighbor[0, 0] = history([0.0, 0.0]) + np.array([3.0, 1.0])   # ahead and left
    geometry = extract_geometry(
        history([1.3, 0.0])[None, ...], neighbor, np.array([[2.6, 0.0]]), fps=FPS
    )
    row = dict(zip(FEATURE_NAMES, build_features(geometry, np.array([[10.0, 0.0]]))[0]))
    assert row["d_lat"] == pytest.approx(1.0)
    assert row["d_long"] == pytest.approx(3.0)


def test_most_threatening_neighbour_wins_not_the_nearest():
    """A close neighbour walking alongside should not mask a closing one further out."""
    neighbor = np.zeros((1, 4, 10, 2))
    neighbor[0, 0] = history([1.3, 0.0]) + np.array([0.0, 1.0])    # alongside, 1 m
    neighbor[0, 1] = history([-1.3, 0.0]) + np.array([6.0, 0.0])   # closing, 6 m
    geometry = extract_geometry(
        history([1.3, 0.0])[None, ...], neighbor, np.array([[2.6, 0.0]]), fps=FPS
    )
    row = dict(zip(FEATURE_NAMES, build_features(geometry, np.array([[10.0, 0.0]]))[0]))
    assert row["d_long"] == pytest.approx(6.0), "picked the nearest, not the threat"
    assert row["ttc"] < TTC_CAP


def test_no_neighbours_gives_neutral_geometry():
    geometry = extract_geometry(
        history([1.3, 0.0])[None, ...], np.zeros((1, 4, 10, 2)),
        np.array([[2.6, 0.0]]), fps=FPS,
    )
    row = dict(zip(FEATURE_NAMES, build_features(geometry, np.array([[10.0, 0.0]]))[0]))
    assert row["ttc"] == TTC_CAP
    assert row["d_long"] == 0.0 and row["d_lat"] == 0.0
    assert row["closing_speed"] == 0.0
    assert row["neighbor_count"] == 0.0


# ------------------------------------------------- against the real dataset
pytestmark_slow = pytest.mark.slow


@pytest.mark.slow
def test_features_are_finite_and_sane_on_real_data(real_batch):
    geometry = extract_geometry(
        real_batch["traj_hist"], real_batch["neighbor"], real_batch["control"],
        environment=real_batch["environment"], fps=5.0, pixel_meter=0.25,
    )
    features = build_features(geometry, real_batch["goal"], real_batch["attr"])
    assert features.shape == (real_batch["traj_hist"].shape[0], len(FEATURE_NAMES))
    assert np.isfinite(features).all()

    row = {name: features[:, i] for i, name in enumerate(FEATURE_NAMES)}
    assert (row["ttc"] >= 0).all() and (row["ttc"] <= TTC_CAP).all()
    assert (row["neighbor_count"] <= real_batch["neighbor"].shape[1]).all()
    assert (row["speed"] >= 0).all()
    assert (row["clearance"] >= 0).all()


@pytest.mark.slow
def test_navmesh_waypoint_is_preferred_speed_times_the_time_offset(real_batch, crowdes_cfg):
    """An end-to-end consistency check on the whole conditioning path.

    ``get_control_point`` places c_nav at ``preferred_speed * control_time_offset``
    along the A* polyline (``simulator_dataloader.py:170``), so ``nav_dist`` must
    track ``attr[:, 1] * 2 s``. If the frame or the units of either drifted, this
    is where it shows up -- measured medians 1.456 m and 0.728 m/s.
    """
    offset = crowdes_cfg["crowd_simulator"]["simulator"]["control_time_offset"]
    geometry = extract_geometry(
        real_batch["traj_hist"], real_batch["neighbor"], real_batch["control"], fps=5.0
    )
    features = build_features(geometry, real_batch["goal"], real_batch["attr"])
    nav_dist = features[:, FEATURE_NAMES.index("nav_dist")]
    expected = real_batch["attr"][:, 1] * offset

    # Not exact per-agent: the polyline is shorter than the lookahead when the
    # goal is close, and locate_point_on_path then extrapolates. Medians match.
    assert np.median(nav_dist) == pytest.approx(np.median(expected), rel=0.02)
    assert (nav_dist <= expected + 1e-3).mean() > 0.95
