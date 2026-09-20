"""Stage 7: path following and nominal controllers (no upstream needed)."""

import numpy as np
import pytest

from src.flow2bt.primitives import DMPBank, fit_dmp
from src.runtime.controllers import BTController, TickState, WaypointController
from src.runtime.pathfollow import PathFollower, _point_at_arc, _project_onto_polyline

DT = 0.05


def straight(n=2):
    return np.array([[0.0, 0.0], [10.0, 0.0]])


def state(position, velocity, waypoint, speed=1.3, dt=DT, features=None):
    position = np.atleast_2d(position)
    count = len(position)
    return TickState(
        position=position,
        velocity=np.atleast_2d(velocity),
        waypoint=np.atleast_2d(waypoint),
        goal=np.tile([10.0, 0.0], (count, 1)),
        preferred_speed=np.full(count, speed),
        neighbor_position=np.zeros((count, 4, 2)),
        neighbor_velocity=np.zeros((count, 4, 2)),
        neighbor_mask=np.zeros((count, 4), dtype=bool),
        features=features,
        dt=dt,
    )


# --------------------------------------------------------------- geometry
def test_projection_finds_arc_length_and_offset():
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
    arc, distance = _project_onto_polyline(np.array([3.0, 2.0]), polyline)
    assert arc == pytest.approx(3.0)
    assert distance == pytest.approx(2.0)


def test_projection_clamps_before_the_start():
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
    arc, distance = _project_onto_polyline(np.array([-5.0, 0.0]), polyline)
    assert arc == pytest.approx(0.0)
    assert distance == pytest.approx(5.0)


def test_point_at_arc_walks_a_corner():
    polyline = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert np.allclose(_point_at_arc(polyline, 0.0), [0.0, 0.0])
    assert np.allclose(_point_at_arc(polyline, 3.0), [3.0, 0.0])
    assert np.allclose(_point_at_arc(polyline, 5.0), [3.0, 2.0])
    assert np.allclose(_point_at_arc(polyline, 7.0), [3.0, 4.0])


def test_point_at_arc_extrapolates_past_the_end():
    """Matches locate_point_on_path, which extrapolates rather than pinning.

    Pinning the lookahead onto the goal would make the waypoint stop moving as
    an agent approaches, and the agent would decelerate into a stall.
    """
    polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert np.allclose(_point_at_arc(polyline, 13.0), [13.0, 0.0])


# ---------------------------------------------------------- path follower
def test_paths_are_planned_once_and_then_reused():
    """The hotspot fix: A* must not run per tick. See pathfollow.py."""
    calls = {"n": 0}

    def plan(starts, goals):
        calls["n"] += 1
        return [np.array([s, g]) for s, g in zip(starts, goals)]

    follower = PathFollower(plan)
    ids = [1, 2, 3]
    position = np.zeros((3, 2))
    goal = np.tile([10.0, 0.0], (3, 1))

    for _ in range(50):
        follower.ensure(ids, position, goal)
        follower.waypoints(ids, position, np.full(3, 2.6))
        position = position + np.array([0.05, 0.0])

    assert calls["n"] == 1, "A* ran more than once for an unchanged plan"
    assert follower.stats["planned"] == 3
    assert follower.stats["followed"] == 150


def test_drifting_off_the_path_triggers_a_replan():
    calls = {"n": 0}

    def plan(starts, goals):
        calls["n"] += 1
        return [np.array([s, g]) for s, g in zip(starts, goals)]

    follower = PathFollower(plan, replan_distance=2.0)
    follower.ensure([1], np.zeros((1, 2)), np.array([[10.0, 0.0]]))
    follower.ensure([1], np.array([[5.0, 0.5]]), np.array([[10.0, 0.0]]))
    assert calls["n"] == 1, "a small offset should not replan"

    follower.ensure([1], np.array([[5.0, 5.0]]), np.array([[10.0, 0.0]]))
    assert calls["n"] == 2 and follower.stats["replanned"] == 1


def test_changing_the_goal_triggers_a_replan():
    calls = {"n": 0}

    def plan(starts, goals):
        calls["n"] += 1
        return [np.array([s, g]) for s, g in zip(starts, goals)]

    follower = PathFollower(plan)
    follower.ensure([1], np.zeros((1, 2)), np.array([[10.0, 0.0]]))
    follower.ensure([1], np.zeros((1, 2)), np.array([[0.0, 10.0]]))
    assert calls["n"] == 2


def test_waypoint_is_the_lookahead_along_the_path():
    follower = PathFollower(lambda s, g: [np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])])
    follower.ensure([1], np.array([[1.0, 0.0]]), np.array([[3.0, 4.0]]))
    waypoint = follower.waypoints([1], np.array([[1.0, 0.0]]), np.array([3.0]))
    assert np.allclose(waypoint[0], [3.0, 1.0])     # 1 m along, +3 m lookahead


def test_forgetting_an_agent_drops_its_path():
    follower = PathFollower(lambda s, g: [np.array([s[0], g[0]])])
    follower.ensure([7], np.zeros((1, 2)), np.array([[10.0, 0.0]]))
    assert 7 in follower.paths
    follower.forget(7)
    assert 7 not in follower.paths and 7 not in follower.goals


# ----------------------------------------------------------- controllers
def test_waypoint_controller_accelerates_toward_the_waypoint():
    command = WaypointController(gain=3.0).acceleration(
        state([0.0, 0.0], [0.0, 0.0], [5.0, 0.0])
    )
    assert command[0, 0] > 0 and command[0, 1] == pytest.approx(0.0)


def test_waypoint_controller_is_at_equilibrium_at_preferred_speed():
    command = WaypointController(gain=3.0).acceleration(
        state([0.0, 0.0], [1.3, 0.0], [5.0, 0.0], speed=1.3)
    )
    assert np.allclose(command, 0.0, atol=1e-9)


def test_waypoint_controller_decelerates_when_too_fast():
    command = WaypointController(gain=3.0).acceleration(
        state([0.0, 0.0], [3.0, 0.0], [5.0, 0.0], speed=1.3)
    )
    assert command[0, 0] < 0


def test_waypoint_controller_ignores_neighbours_by_design():
    """It is the floor the CBF's contribution is measured against."""
    plain = state([0.0, 0.0], [1.3, 0.0], [5.0, 0.0])
    crowded = state([0.0, 0.0], [1.3, 0.0], [5.0, 0.0])
    crowded.neighbor_position[:] = np.array([[0.3, 0.0], [0.0, 0.3], [-0.3, 0.0], [0.0, -0.3]])
    crowded.neighbor_mask[:] = True
    controller = WaypointController()
    assert np.allclose(controller.acceleration(plain), controller.acceleration(crowded))


# ------------------------------------------------------------- BT phase
def swerve(lateral=0.6, steps=11):
    t = np.linspace(0.0, 1.0, steps)
    return np.stack([2.6 * t, -lateral * np.sin(np.pi * t)], axis=1)


class _Tree:
    """Routes on feature column 0: >= 0 -> leaf 0, else leaf 1."""

    def __init__(self):
        self.forced = None

    def tick(self, features):
        from src.flow2bt.bt import TickResult

        action = np.where(features[:, 0] >= 0, 0, 1) if self.forced is None else self.forced
        return TickResult(np.full(len(features), 2), np.asarray(action))


def test_bt_controller_resets_phase_when_the_tree_preempts():
    """The reactivity mechanism: a switched leaf must restart its own clock.

    Carrying a stale phase across a switch replays the middle of the new
    primitive, which is exactly wrong when the switch happened because
    something urgent appeared.
    """
    bank = DMPBank([
        fit_dmp(swerve(0.6), 0.2, num_basis=10, name="right"),
        fit_dmp(swerve(-0.6), 0.2, num_basis=10, name="left"),
    ])
    tree = _Tree()
    controller = BTController(tree, bank)

    features = np.array([[1.0]])
    for _ in range(20):
        controller.acceleration(state([0.0, 0.0], [1.3, 0.0], [3.0, 0.0], features=features))
    assert controller._leaf[0] == 0
    assert controller._phase[0] < 0.9, "phase did not advance"

    # Preempt onto the other leaf.
    controller.acceleration(
        state([0.0, 0.0], [1.3, 0.0], [3.0, 0.0], features=np.array([[-1.0]]))
    )
    assert controller._leaf[0] == 1
    assert controller._phase[0] == pytest.approx(
        bank.step_phase(np.array([1.0]), DT, bank.tau[1:2])[0]
    ), "phase was not restarted on the switch"


def test_bt_controller_records_the_entry_point_on_a_switch():
    """A DMP's spatial scaling is relative to where the primitive was entered."""
    bank = DMPBank([
        fit_dmp(swerve(0.6), 0.2, num_basis=10),
        fit_dmp(swerve(-0.6), 0.2, num_basis=10),
    ])
    controller = BTController(_Tree(), bank)
    controller.acceleration(state([0.0, 0.0], [1.3, 0.0], [3.0, 0.0], features=np.array([[1.0]])))
    assert np.allclose(controller._entry[0], [0.0, 0.0])
    controller.acceleration(state([2.0, 1.0], [1.3, 0.0], [3.0, 0.0], features=np.array([[-1.0]])))
    assert np.allclose(controller._entry[0], [2.0, 1.0])


def test_bt_controller_keeps_phase_while_the_leaf_is_unchanged():
    bank = DMPBank([fit_dmp(swerve(0.6), 0.2, num_basis=10)])
    controller = BTController(_Tree(), bank)
    features = np.array([[1.0]])
    phases = []
    for _ in range(5):
        controller.acceleration(state([0.0, 0.0], [1.3, 0.0], [3.0, 0.0], features=features))
        phases.append(float(controller._phase[0]))
    assert all(b < a for a, b in zip(phases, phases[1:])), phases


def test_bt_controller_handles_a_changing_agent_count():
    """Agents enter and leave every frame; the controller must not carry stale state."""
    bank = DMPBank([fit_dmp(swerve(0.6), 0.2, num_basis=10)])
    controller = BTController(_Tree(), bank)
    for count in (3, 5, 2):
        features = np.ones((count, 1))
        command = controller.acceleration(
            state(np.zeros((count, 2)), np.zeros((count, 2)), np.ones((count, 2)), features=features)
        )
        assert command.shape == (count, 2)
        assert np.isfinite(command).all()


def test_bt_controller_re_enters_a_completed_primitive():
    """Design 05 sec 1.3's action lifecycle, without which agents stall.

    An agent holding one leaf must get a fresh goal once it reaches the old one.
    Measured cost of omitting this: 71 of 406 agents ever finished, travel-time
    EMD 11.9 against CrowdES's 0.60.
    """
    bank = DMPBank([fit_dmp(swerve(0.0), 0.2, num_basis=10, name="march")])
    controller = BTController(bank=bank, tree=_Tree(), arrival_tolerance=0.25)
    features = np.array([[1.0]])

    position = np.array([[0.0, 0.0]])
    controller.acceleration(state(position, [1.3, 0.0], [3.0, 0.0], features=features))
    first_goal = controller._goal.copy()
    assert controller.completions == 0

    # Teleport onto the goal: the primitive is complete.
    controller.acceleration(state(first_goal, [1.3, 0.0], [9.0, 0.0], features=features))
    assert controller.completions == 1
    assert not np.allclose(controller._goal, first_goal), "goal was not refreshed"
    assert controller._phase[0] < 1.0 and controller._phase[0] > 0.9, "phase did not restart"


def test_bt_controller_re_enters_on_an_expired_phase():
    bank = DMPBank([fit_dmp(swerve(0.0), 0.2, num_basis=10)])
    controller = BTController(bank=bank, tree=_Tree(), arrival_tolerance=1e-6, phase_floor=0.5)
    features = np.array([[1.0]])
    for _ in range(40):
        controller.acceleration(state([0.0, 0.0], [0.0, 0.0], [3.0, 0.0], features=features))
    assert controller.completions > 0, "an exhausted phase never triggered re-entry"


def test_goal_is_rotated_onto_the_navmesh_direction():
    """A leaf fitted in the nav frame must aim along wherever the agent is guided."""
    bank = DMPBank([fit_dmp(swerve(0.0), 0.2, num_basis=10, name="march")])
    controller = BTController(bank=bank, tree=_Tree(), rotate_to_nav=True)
    features = np.array([[1.0]])
    reach = float(np.linalg.norm(bank.dmps[0].displacement))

    controller.acceleration(state([0.0, 0.0], [0.0, 0.0], [10.0, 0.0], features=features))
    east = controller._goal.copy()
    controller.reset(1)
    controller.acceleration(state([0.0, 0.0], [0.0, 0.0], [0.0, 10.0], features=features))
    north = controller._goal.copy()

    assert east[0, 0] == pytest.approx(reach, abs=1e-6) and abs(east[0, 1]) < 1e-6
    assert north[0, 1] == pytest.approx(reach, abs=1e-6) and abs(north[0, 0]) < 1e-6
