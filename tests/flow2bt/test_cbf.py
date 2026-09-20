"""Subsystem 7a (CBF-QP safety filter)."""

import numpy as np
import pytest

from src.flow2bt.cbf import (
    CBFConfig,
    CBFFilter,
    acceleration_polytope,
    pairwise_constraints,
    solve_qp_hildreth,
    solve_qp_osqp,
    solve_qp_slack,
)

TICK = 0.01


# ------------------------------------------------------------------- solver
def feasible_problem(rng, num_agents=24, num_constraints=6, scale=2.0):
    """Random half-planes with a guaranteed non-empty interior.

    Drawing A and b independently makes most instances *infeasible*, where
    "the" projection does not exist and comparing two solvers is meaningless.
    Anchoring b on an interior point is what makes the comparison well posed.
    """
    u_nominal = rng.normal(size=(num_agents, 2)) * scale
    A = rng.normal(size=(num_agents, num_constraints, 2))
    interior = rng.normal(size=(num_agents, 2)) * 0.5
    b = np.einsum("nkd,nd->nk", A, interior) + rng.uniform(0.05, 1.5, (num_agents, num_constraints))
    return u_nominal, A, b


def test_hildreth_matches_osqp():
    """Hildreth is the runtime path; OSQP is the reference it must agree with."""
    u_nominal, A, b = feasible_problem(np.random.default_rng(0))
    fast = solve_qp_hildreth(u_nominal, A, b, iterations=2000)
    reference = solve_qp_osqp(u_nominal, A, b)
    assert np.abs(fast - reference).max() < 1e-5


def test_larger_slack_weight_approaches_the_hard_projection():
    """The soft rows really are soft, by an amount that scales as 1/w.

    This is the property the default weight rests on, and it is the opposite of
    what the first pass at this module assumed. See "Slack weight is a safety
    parameter" in cbf.py.
    """
    rng = np.random.default_rng(3)
    u_nominal, A, b = feasible_problem(rng, num_constraints=4)
    hard_A = np.zeros((A.shape[0], 1, 2))
    hard_b = np.full((A.shape[0], 1), np.inf)
    u_hard = solve_qp_hildreth(u_nominal, A, b, iterations=4000)

    # Measured: the gap falls by roughly one decade per decade of weight
    # (3.2, 0.78, 1.0e-2, 1.0e-4 at w = 1, 1e2, 1e4, 1e6).
    gaps = {}
    for weight in (1.0, 1e2, 1e4, 1e6):
        u_slack, _ = solve_qp_slack(
            u_nominal, A, b, hard_A, hard_b, slack_weight=weight, iterations=400
        )
        gaps[weight] = np.abs(u_slack - u_hard).max()

    ordered = [gaps[w] for w in (1.0, 1e2, 1e4, 1e6)]
    assert all(b < a for a, b in zip(ordered, ordered[1:])), gaps
    assert gaps[1e4] < 5e-2, "the default weight no longer tracks the hard projection"
    assert gaps[1e6] < 1e-3


def test_control_is_converged_at_the_default_iteration_count():
    """60 sweeps must give the same *command* as 400; only xi lags.

    If this fails the iteration budget is no longer adequate and the latency
    figure has to be re-derived, not just nudged.
    """
    config = CBFConfig(d_min=0.7)
    neighbours = np.array([[[0.9, 0.1], [-1.2, 0.3]]])
    closing = np.array([[[-1.3, 0.0], [1.1, -0.2]]])
    u_nominal = np.array([[1.0, 0.4]])

    A, b, _ = pairwise_constraints(
        np.zeros((1, 2)), np.array([[1.3, 0.0]]), neighbours, closing,
        np.ones((1, 2), dtype=bool), config,
    )
    hard_A, hard_b = acceleration_polytope(1, config)
    u_60, _ = solve_qp_slack(u_nominal, A, b, hard_A, hard_b, config.slack_weight, iterations=60)
    u_400, _ = solve_qp_slack(u_nominal, A, b, hard_A, hard_b, config.slack_weight, iterations=400)
    assert np.abs(u_60 - u_400).max() < 1e-6


def test_slack_absorbs_an_infeasible_soft_set():
    """Three mutually exclusive demands must degrade, not produce garbage."""
    # x <= -1 and x >= 1 simultaneously: empty.
    A = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
    b = np.array([[-1.0, -1.0]])
    hard_A = np.zeros((1, 1, 2))
    hard_b = np.full((1, 1), np.inf)

    u, xi = solve_qp_slack(
        np.array([[0.0, 0.0]]), A, b, hard_A, hard_b, slack_weight=1e2, iterations=2000
    )
    assert np.isfinite(u).all()
    assert xi[0] > 0.5, f"slack should have absorbed the conflict, got {xi[0]}"
    # Both rows satisfied once relaxed by xi.
    assert np.all(A[0] @ u[0] <= b[0] + xi[0] + 1e-6)


def test_hard_rows_are_never_relaxed():
    """Actuation and walls must hold even when the pairwise set is impossible."""
    config = CBFConfig(a_max=1.5)
    A_soft = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
    b_soft = np.array([[-100.0, -100.0]])          # wildly infeasible
    hard_A, hard_b = acceleration_polytope(1, config)
    u, xi = solve_qp_slack(
        np.array([[0.0, 0.0]]), A_soft, b_soft, hard_A, hard_b,
        slack_weight=1e4, iterations=2000,
    )
    assert np.linalg.norm(u) <= config.a_max + 1e-6
    assert xi[0] > 0


def test_projection_leaves_a_feasible_command_untouched():
    u_nominal = np.array([[0.5, 0.0], [-0.2, 0.3]])
    A = np.zeros((2, 3, 2))
    A[:, 0] = [1.0, 0.0]
    b = np.full((2, 3), 10.0)          # miles from binding
    assert np.allclose(solve_qp_hildreth(u_nominal, A, b), u_nominal)


def test_projection_is_onto_the_feasible_set():
    """A violated half-plane must end up satisfied, and exactly on its face."""
    u_nominal = np.array([[3.0, 0.0]])
    A = np.array([[[1.0, 0.0]]])
    b = np.array([[1.0]])
    u = solve_qp_hildreth(u_nominal, A, b)
    assert u[0, 0] == pytest.approx(1.0)     # projected onto the boundary
    assert u[0, 1] == pytest.approx(0.0)     # orthogonal component untouched


def test_masked_and_infinite_constraints_never_bind():
    u_nominal = np.array([[5.0, 5.0]])
    A = np.array([[[1.0, 0.0], [0.0, 0.0]]])      # second row is a dead slot
    b = np.array([[np.inf, np.inf]])
    assert np.allclose(solve_qp_hildreth(u_nominal, A, b), u_nominal)


# --------------------------------------------------------------- constraints
def test_acceleration_polytope_is_an_inner_approximation():
    """Satisfying the polygon must imply ||u|| <= a_max, not merely approximate it."""
    config = CBFConfig(a_max=3.0, accel_facets=8)
    A, b = acceleration_polytope(1, config)
    rng = np.random.default_rng(1)
    for _ in range(400):
        u = rng.normal(size=(1, 2)) * 5.0
        if np.all((A[0] @ u[0]) <= b[0] + 1e-9):
            assert np.linalg.norm(u) <= config.a_max + 1e-9


def test_acceleration_bound_is_enforced_by_the_filter():
    config = CBFConfig(a_max=2.0)
    filt = CBFFilter(config)
    u_safe, _ = filt.filter(
        u_nominal=np.array([[50.0, 50.0]]),
        position=np.zeros((1, 2)),
        velocity=np.zeros((1, 2)),
        neighbor_position=np.zeros((1, 1, 2)),
        neighbor_velocity=np.zeros((1, 1, 2)),
        neighbor_mask=np.zeros((1, 1), dtype=bool),
    )
    assert np.linalg.norm(u_safe) <= config.a_max + 1e-6


def test_responsibility_sharing_sums_to_the_joint_constraint():
    """The correctness argument for responsibility=0.5, checked rather than asserted.

    Agent i's row plus agent j's row must reproduce the undivided pairwise
    constraint, because every non-u term is symmetric in i and j.
    """
    config = CBFConfig(responsibility=0.5, d_min=0.7)
    position = np.array([[0.0, 0.0], [1.0, 0.2]])
    velocity = np.array([[1.3, 0.0], [-1.3, 0.0]])
    mask = np.ones((2, 1), dtype=bool)

    A, b, h = pairwise_constraints(
        position, velocity,
        position[::-1][:, None, :], velocity[::-1][:, None, :],
        mask, config,
    )
    # rows are negatives of each other (A = -2 dp, and dp flips sign)
    assert np.allclose(A[0, 0], -A[1, 0])
    # the shared halves add back up to the full constant
    full = CBFConfig(responsibility=1.0, d_min=0.7)
    _, b_full, _ = pairwise_constraints(
        position, velocity,
        position[::-1][:, None, :], velocity[::-1][:, None, :],
        mask, full,
    )
    assert b[0, 0] + b[1, 0] == pytest.approx(b_full[0, 0])
    assert np.allclose(h[0], h[1])     # the barrier itself is symmetric


def test_barrier_sign_matches_separation():
    config = CBFConfig(d_min=0.7)
    for distance, expect_safe in ((1.5, True), (0.7, False), (0.3, False)):
        _, _, h = pairwise_constraints(
            np.zeros((1, 2)), np.zeros((1, 2)),
            np.array([[[distance, 0.0]]]), np.zeros((1, 1, 2)),
            np.ones((1, 1), dtype=bool), config,
        )
        assert (h[0, 0] > 0) == expect_safe


# ------------------------------------------------------------ invariance
def head_on(config, responsibility=None, duration=10.0, dt=TICK, speed=1.3):
    """Two agents driving straight through each other; returns the trace."""
    if responsibility is not None:
        config = CBFConfig(**{**config.__dict__, "responsibility": responsibility})
    filt = CBFFilter(config)

    position = np.array([[-5.0, 0.0], [5.0, 0.0]])
    velocity = np.array([[speed, 0.0], [-speed, 0.0]])
    goal = np.array([[5.0, 0.0], [-5.0, 0.0]])

    barriers, separations = [], []
    for _ in range(int(duration / dt)):
        # A deliberately blind nominal controller: drive at the goal, no avoidance.
        direction = goal - position
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        u_nominal = 4.0 * (speed * direction - velocity)

        u_safe, info = filt.filter(
            u_nominal=u_nominal,
            position=position, velocity=velocity,
            neighbor_position=position[::-1][:, None, :],
            neighbor_velocity=velocity[::-1][:, None, :],
            neighbor_mask=np.ones((2, 1), dtype=bool),
        )
        velocity = velocity + u_safe * dt
        position = position + velocity * dt
        separations.append(np.linalg.norm(position[0] - position[1]))
        barriers.append(info["min_barrier"].min())
    return np.array(separations), np.array(barriers), position


def test_head_on_pair_never_breaches_d_min():
    """Forward invariance: the whole point of Subsystem 7a."""
    config = CBFConfig(d_min=0.7, a_max=3.0)
    separations, barriers, _ = head_on(config)
    # Measured breach at the default weight is 4e-5 m; 1 mm is loose enough to
    # absorb integrator noise and tight enough that a weight regression trips it
    # (at w=10 the same run breaches by 8 cm, at w=1 it collides outright).
    assert separations.min() >= config.d_min - 1e-3, f"min separation {separations.min():.4f} m"
    assert barriers.min() > -1e-3


def test_head_on_stays_safe_at_the_benchmark_threshold():
    """d_min = 0.2 m is the threshold utils/metrics.py actually scores."""
    config = CBFConfig(d_min=0.2, a_max=3.0)
    separations, _, _ = head_on(config)
    assert separations.min() >= 0.2 - 1e-3


def test_unshared_responsibility_over_brakes():
    """The measured justification for responsibility=0.5 (see cbf.py).

    With responsibility=1.0 both agents apply the whole correction as if the
    other were passive. If that does NOT cost progress, the default should be
    revisited rather than left in place on argument alone.
    """
    config = CBFConfig(d_min=0.7, a_max=3.0)
    _, _, shared_end = head_on(config, responsibility=0.5)
    _, _, unshared_end = head_on(config, responsibility=1.0)
    shared_progress = abs(shared_end[0, 0] - (-5.0))
    unshared_progress = abs(unshared_end[0, 0] - (-5.0))
    assert shared_progress > unshared_progress, (
        f"sharing made no difference: {shared_progress:.3f} vs {unshared_progress:.3f} m"
    )


def test_filter_reports_its_intervention():
    config = CBFConfig(d_min=0.7)
    filt = CBFFilter(config)
    close = np.array([[0.0, 0.0], [0.75, 0.0]])
    approaching = np.array([[1.3, 0.0], [-1.3, 0.0]])
    _, info = filt.filter(
        u_nominal=np.array([[2.0, 0.0], [-2.0, 0.0]]),
        position=close, velocity=approaching,
        neighbor_position=close[::-1][:, None, :],
        neighbor_velocity=approaching[::-1][:, None, :],
        neighbor_mask=np.ones((2, 1), dtype=bool),
    )
    assert (info["intervention"] > 0).all(), "filter did nothing on an imminent collision"
    assert info["num_soft"] == 1
    assert info["num_hard"] == CBFConfig().accel_facets
    assert not info["relaxed"].any(), "a single pairwise constraint should not need relaxing"


def test_squeezed_agent_reports_relaxation_rather_than_failing():
    """Pinned between two neighbours already inside d_min and both still closing.

    Both rows then demand acceleration in opposite directions (b < 0 on each),
    so no u satisfies them. Measured: d=0.5 m closing at 1.5 m/s gives
    b = -1.23 on both. The honest outcome is a reported relaxation.
    """
    config = CBFConfig(d_min=0.7, a_max=3.0)
    filt = CBFFilter(config)
    neighbours = np.array([[[0.5, 0.0], [-0.5, 0.0]]])
    closing = np.array([[[-1.5, 0.0], [1.5, 0.0]]])

    u_safe, info = filt.filter(
        u_nominal=np.array([[0.0, 0.0]]),
        position=np.zeros((1, 2)), velocity=np.zeros((1, 2)),
        neighbor_position=neighbours, neighbor_velocity=closing,
        neighbor_mask=np.ones((1, 2), dtype=bool),
    )
    assert np.isfinite(u_safe).all()
    assert np.linalg.norm(u_safe) <= config.a_max + 1e-6
    assert info["relaxed"][0], "an impossible geometry should be reported, not hidden"
    assert info["violation"][0] > 1.0, "the residual should show how badly it is unmet"
    assert info["min_barrier"][0] < 0, "this geometry is already in breach"


def test_tight_actuation_also_surfaces_as_relaxation():
    """Soft set feasible in principle, unreachable within a_max -> still reported."""
    config = CBFConfig(d_min=0.7, a_max=0.05)
    filt = CBFFilter(config)
    u_safe, info = filt.filter(
        u_nominal=np.zeros((1, 2)),
        position=np.zeros((1, 2)), velocity=np.zeros((1, 2)),
        neighbor_position=np.array([[[0.5, 0.0], [-0.5, 0.0]]]),
        neighbor_velocity=np.array([[[-2.0, 0.0], [2.0, 0.0]]]),
        neighbor_mask=np.ones((1, 2), dtype=bool),
    )
    assert np.linalg.norm(u_safe) <= config.a_max + 1e-6
    assert info["relaxed"][0]


def test_head_on_needs_no_relaxation_at_the_default_weight():
    """A two-agent encounter is comfortably feasible; nothing should be traded.

    Guards against the default weight silently drifting down: at w=10 this run
    relaxes on 79% of ticks and breaches d_min by 8 cm.
    """
    config = CBFConfig(d_min=0.7, a_max=3.0)
    filt = CBFFilter(config)
    position = np.array([[-5.0, 0.0], [5.0, 0.0]])
    velocity = np.array([[1.3, 0.0], [-1.3, 0.0]])
    goal = np.array([[5.0, 0.0], [-5.0, 0.0]])

    relaxed_ticks = 0
    for _ in range(int(10.0 / TICK)):
        direction = goal - position
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        u_safe, info = filt.filter(
            u_nominal=4.0 * (1.3 * direction - velocity),
            position=position, velocity=velocity,
            neighbor_position=position[::-1][:, None, :],
            neighbor_velocity=velocity[::-1][:, None, :],
            neighbor_mask=np.ones((2, 1), dtype=bool),
        )
        velocity = velocity + u_safe * TICK
        position = position + velocity * TICK
        relaxed_ticks += int(info["relaxed"].any())
    assert relaxed_ticks == 0, f"relaxed on {relaxed_ticks} ticks of a feasible encounter"
