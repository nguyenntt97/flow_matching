"""Subsystem 7a: the online CBF-QP safety filter.

Design doc: ``survey/design/07_safety_cbf_formal_verification.md``.

The behavior tree emits a nominal acceleration; this projects it onto the set of
accelerations that keep the collision-free set forward invariant::

    u* = argmin ||u - u_nominal||^2   s.t.   A u <= b

Agents are double integrators, ``p_dot = v``, ``v_dot = u``, so the pairwise
barrier ``h_ij = ||p_i - p_j||^2 - d_min^2`` has relative degree 2 with respect
to ``u`` and needs the higher-order form (Nagumo / exponential CBF)::

    h_ddot + (a1 + a2) h_dot + a1 a2 h >= 0

Three things the design leaves implicit, decided here
-----------------------------------------------------

**Responsibility sharing.** Design 07 eq. (1.1.1) leaves ``u_j`` on the
right-hand side, i.e. each agent solves as if its neighbour were passive. Run
decentralised with both agents making that assumption and both apply the *full*
correction: they over-brake, and a head-on pair stalls -- the frozen-robot
deadlock the design elsewhere wants to avoid. ``responsibility=0.5`` splits the
burden symmetrically; summing the two agents' constraints recovers the joint
constraint exactly, because the non-``u`` terms are symmetric in ``i`` and ``j``.

**The obstacle barrier's relative degree.** Design 07 eq. (1.1.2) writes
``grad(h)^T v + a h >= 0``, which has no ``u`` in it and so cannot constrain an
acceleration. Under double-integrator dynamics ``h_obs = SDF(p) - r`` is also
relative degree 2, so it gets the same higher-order treatment. The
``v^T Hess(SDF) v`` term is dropped: for polygonal navmesh boundaries the SDF is
piecewise linear and its Hessian is zero almost everywhere.

**The acceleration bound.** ``||u|| <= a_max`` is a second-order cone, which
would stop this being a QP. It is replaced by a circumscribed-normal polygon
with ``accel_facets`` sides -- an *inner* approximation, so the bound is
enforced conservatively and never violated.

Feasibility is not optional
---------------------------
A crowded agent can easily face pairwise constraints that no acceleration
satisfies -- three neighbours closing at once, inside an ``a_max`` disc. With a
hard QP that set is empty and the "solution" is whatever the solver last held:
in a crowd simulator that surfaces as an agent lurching or leaving the map,
which is exactly the kind of silent failure that quietly invalidates a result.

Design 07 eq. (1.1.3) already carries a slack term, so the QP solved here is::

    min 0.5||u - u_nom||^2 + w xi^2
    s.t.  A_pair u <= b_pair + xi     (soft: relaxable together)
          A_obs  u <= b_obs           (hard: never walk through a wall)
          A_acc  u <= b_acc           (hard: actuation limit)
          xi >= 0

Obstacle and acceleration rows stay hard, and they alone are always feasible
(``u = 0`` satisfies both), so the relaxed problem cannot be infeasible.
``xi > 0`` is reported per agent so a rollout can say how often -- and how
badly -- safety had to be traded away, rather than reporting a collision-free
number that quietly depended on it.

Slack weight is a safety parameter, and it must be large
--------------------------------------------------------
The soft rows really are soft: a binding constraint is relaxed by an amount
that scales as ``1/w``, so ``w`` directly sets how much separation is traded
away. Measured on the two-agent head-on with ``d_min = 0.7 m``:

    w        min separation      relaxation
    1        0.003 m             1.81         <- total failure
    10       0.621 m             0.209
    100      0.693 m             0.019
    1e3      0.699 m             0.0019
    1e4      0.700 m             0.00019      <- exact

So ``w`` must be large. Two things that look like reasons to lower it are not:

* **Convergence of the control.** 60 and 400 sweeps give *identical* separation
  at every weight above. The slack coordinate's own convergence is slow at
  large ``w`` (its Hildreth step scales with ``1/w``), but that shows up in the
  reported ``xi``, not in ``u``.
* **Reporting infeasibility.** Because ``xi`` converges slowly, it is a poor
  infeasibility signal at the weight safety requires. The diagnostic used
  instead is the *residual* ``max(A_soft u - b_soft, 0)`` -- the amount by
  which the barrier constraints are actually unmet. It is in the units of the
  constraint, converges at the same rate as ``u``, and does not depend on how
  the relaxation happens to be parameterised.

Solver
------
Each QP is three variables with a handful of half-planes. Calling OSQP per agent
would be dominated by setup, so the default path is Hildreth's method (dual
coordinate ascent) generalised to a diagonal Hessian and vectorised across the
whole crowd: it converges to the exact optimum and is a fixed number of numpy
ops. ``solve_qp_osqp`` is the reference the tests check it against, not the
runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-12


@dataclass
class CBFConfig:
    """Tunables. ``d_min`` is the one the plan sweeps.

    Note ``d_min`` is *not* the benchmark's collision threshold. Upstream scores
    collisions at centre-to-centre < 0.2 m (``utils/metrics.py:477``); the design
    proposes 0.7 m, which is 3.5x that and will visibly thin the crowd. Sweeping
    it is the point -- see the frontier in the plan.
    """

    d_min: float = 0.45
    alpha_1: float = 2.0
    alpha_2: float = 2.0
    alpha_obs_1: float = 2.0
    alpha_obs_2: float = 2.0
    a_max: float = 3.0
    accel_facets: int = 8
    responsibility: float = 0.5
    #: Penalty on the slack variable -- a safety parameter. See "Slack weight is
    #: a safety parameter" above: at 1e4 the head-on pair holds d_min exactly,
    #: at 10 it breaches by 8 cm and at 1 it collides outright.
    slack_weight: float = 1e4
    #: Residual (in constraint units) above which an agent is reported as having
    #: had its barrier constraints genuinely relaxed. Routine give at the default
    #: weight measures ~2e-4, so this sits an order of magnitude clear of it.
    #: A reporting threshold only -- forward invariance is claimed from the
    #: barrier values, never from this.
    violation_report_threshold: float = 5e-3
    iterations: int = 60


def pairwise_constraints(
    position: np.ndarray,
    velocity: np.ndarray,
    neighbor_position: np.ndarray,
    neighbor_velocity: np.ndarray,
    neighbor_mask: np.ndarray,
    config: CBFConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Higher-order pairwise barrier rows, as ``A u <= b``.

    Shapes: ``position``/``velocity`` are (N, 2); the neighbour arrays are
    (N, K, 2); ``neighbor_mask`` is (N, K) and marks real neighbours.
    Returns ``(A, b, h)`` with A (N, K, 2), b (N, K), h (N, K).
    """
    delta_p = position[:, None, :] - neighbor_position          # (N, K, 2)
    delta_v = velocity[:, None, :] - neighbor_velocity

    h = (delta_p**2).sum(-1) - config.d_min**2                  # (N, K)
    h_dot = 2.0 * (delta_p * delta_v).sum(-1)

    # h_ddot = 2||dv||^2 + 2 dp^T (u_i - u_j); sharing splits the rest.
    constant = 2.0 * (delta_v**2).sum(-1) + (config.alpha_1 + config.alpha_2) * h_dot \
        + config.alpha_1 * config.alpha_2 * h

    A = -2.0 * delta_p                                          # (N, K, 2)
    b = config.responsibility * constant                        # (N, K)

    # Inactive slots must never bind: zero row, +inf slack.
    A = np.where(neighbor_mask[..., None], A, 0.0)
    b = np.where(neighbor_mask, b, np.inf)
    return A, b, np.where(neighbor_mask, h, np.inf)


def obstacle_constraints(
    velocity: np.ndarray,
    sdf: np.ndarray,
    sdf_gradient: np.ndarray,
    agent_radius: float,
    config: CBFConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static-boundary barrier, also higher order. See the module docstring.

    ``sdf`` is (N,) signed distance to the nearest non-traversable boundary,
    ``sdf_gradient`` is (N, 2) and should be unit-norm.
    """
    h = sdf - agent_radius                                      # (N,)
    h_dot = (sdf_gradient * velocity).sum(-1)
    constant = (config.alpha_obs_1 + config.alpha_obs_2) * h_dot \
        + config.alpha_obs_1 * config.alpha_obs_2 * h
    return -sdf_gradient[:, None, :], constant[:, None], h[:, None]


def acceleration_polytope(num_agents: int, config: CBFConfig) -> tuple[np.ndarray, np.ndarray]:
    """Inner polygonal approximation of ``||u|| <= a_max``.

    Facet normals evenly around the circle with offset ``a_max * cos(pi/M)``
    gives the polygon *inscribed* in the disc, so satisfying it implies the norm
    bound rather than merely approximating it.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, config.accel_facets, endpoint=False)
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)       # (M, 2)
    offset = config.a_max * np.cos(np.pi / config.accel_facets)
    A = np.broadcast_to(normals, (num_agents,) + normals.shape).copy()
    b = np.full((num_agents, config.accel_facets), offset)
    return A, b


def solve_qp_hildreth(
    u_nominal: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    iterations: int = 60,
    z_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project ``u_nominal`` onto ``{z : A z <= b}``, for every agent at once.

    Hildreth's method is dual coordinate ascent: with ``z = z_unc - A^T lam``
    and ``lam >= 0``, each multiplier has a closed-form optimum given the rest.
    Sweeping the constraints is Gauss-Seidel over ``lam``, and every agent takes
    the same sweep, so the inner loop is over the (small) constraint index and
    everything else is vectorised.

    Shapes: ``A`` is (N, K, D) and ``b`` is (N, K). ``z_init`` supplies the
    unconstrained optimum when it has more dimensions than ``u_nominal`` -- used
    by ``solve_qp_slack`` to carry the extra slack coordinate. The Hessian is
    assumed to be the identity; a non-identity one must be absorbed by rescaling
    the variables first (see ``solve_qp_slack``).
    """
    z = np.array(u_nominal if z_init is None else z_init, dtype=float, copy=True)
    num_agents, num_constraints, _ = A.shape
    lam = np.zeros((num_agents, num_constraints))

    norms = (A**2).sum(-1)                                   # (N, K)
    live = (norms > EPS) & np.isfinite(b)
    safe_norms = np.where(live, norms, 1.0)

    for _ in range(iterations):
        for k in range(num_constraints):
            a_k = A[:, k, :]                                 # (N, D)
            # Remove constraint k's current contribution, re-optimise it.
            z_without = z + lam[:, k : k + 1] * a_k
            violation = (a_k * z_without).sum(-1) - b[:, k]
            lam_k = np.where(live[:, k], np.maximum(0.0, violation / safe_norms[:, k]), 0.0)
            z = z_without - lam_k[:, None] * a_k
            lam[:, k] = lam_k
    return z


def solve_qp_slack(
    u_nominal: np.ndarray,
    A_soft: np.ndarray,
    b_soft: np.ndarray,
    A_hard: np.ndarray,
    b_hard: np.ndarray,
    slack_weight: float,
    iterations: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Slack-relaxed projection: see "Feasibility is not optional" above.

    Conditioning
    ------------
    Solving directly in ``(u_x, u_y, xi)`` gives a Hessian ``diag(1, 1, 2w)``,
    so the slack coordinate's Hildreth step is scaled by ``1/(2w)`` -- at
    ``w = 1e6`` that is ``5e-7`` per sweep and ``xi`` barely moves. Measured:
    60 iterations reported ``xi = 1.5e-4`` where the converged answer was
    ``7.4e-3``, a 50x under-estimate, which would have made the filter silently
    wrong at runtime rather than merely slow.

    Substituting ``xi = eta / sqrt(2w)`` makes the objective
    ``0.5||u - u_nom||^2 + 0.5 eta^2`` with Hessian ``I``, so every coordinate
    takes an O(1) step and plain Hildreth applies. ``slack_weight`` then only
    scales how much a unit of relaxation costs, not how fast it is found.

    Returns ``(u, xi)``.
    """
    num_agents = u_nominal.shape[0]
    num_soft = A_soft.shape[1]
    num_hard = A_hard.shape[1]
    sigma = 1.0 / np.sqrt(2.0 * slack_weight)

    # Rows in (u_x, u_y, eta): soft rows carry -sigma on eta, hard rows 0,
    # plus one row enforcing eta >= 0.
    G = np.zeros((num_agents, num_soft + num_hard + 1, 3))
    g = np.empty((num_agents, num_soft + num_hard + 1))

    G[:, :num_soft, :2] = A_soft
    G[:, :num_soft, 2] = -sigma
    g[:, :num_soft] = b_soft

    G[:, num_soft : num_soft + num_hard, :2] = A_hard
    g[:, num_soft : num_soft + num_hard] = b_hard

    G[:, -1, 2] = -1.0
    g[:, -1] = 0.0

    z = np.zeros((num_agents, 3))
    z[:, :2] = u_nominal                       # unconstrained optimum, eta = 0
    z_out = solve_qp_hildreth(z[:, :2], G, g, iterations=iterations, z_init=z)
    return z_out[:, :2], z_out[:, 2] * sigma


def solve_qp_osqp(
    u_nominal: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Per-agent OSQP. Reference implementation for the tests -- not the runtime path."""
    import osqp
    import scipy.sparse as sparse

    out = np.array(u_nominal, dtype=float, copy=True)
    for index in range(A.shape[0]):
        rows = np.isfinite(b[index]) & ((A[index] ** 2).sum(-1) > EPS)
        if not rows.any():
            continue
        problem = osqp.OSQP()
        problem.setup(
            P=sparse.csc_matrix(np.eye(2)),
            q=-np.asarray(u_nominal[index], dtype=float),
            A=sparse.csc_matrix(A[index][rows]),
            l=np.full(int(rows.sum()), -np.inf),
            u=b[index][rows],
            verbose=False,
            eps_abs=1e-10,
            eps_rel=1e-10,
            max_iter=20000,
        )
        result = problem.solve()
        if result.info.status_val in (1, 2):  # solved / solved inaccurate
            out[index] = result.x
    return out


class CBFFilter:
    """Assembles the constraint rows and projects. One call per tick."""

    def __init__(self, config: Optional[CBFConfig] = None):
        self.config = config or CBFConfig()

    def filter(
        self,
        u_nominal: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        neighbor_position: np.ndarray,
        neighbor_velocity: np.ndarray,
        neighbor_mask: np.ndarray,
        sdf: Optional[np.ndarray] = None,
        sdf_gradient: Optional[np.ndarray] = None,
        agent_radius: float = 0.2,
        solver: str = "hildreth",
    ) -> tuple[np.ndarray, dict]:
        """Return ``(u_safe, diagnostics)``.

        ``diagnostics`` carries the barrier values and how far the projection
        moved the command, so a rollout can report *why* it was safe rather than
        only that it was.
        """
        config = self.config

        A_pair, b_pair, h_pair = pairwise_constraints(
            position, velocity, neighbor_position, neighbor_velocity, neighbor_mask, config
        )
        barriers = [h_pair]

        hard_A, hard_b = [], []
        if sdf is not None and sdf_gradient is not None:
            A_obs, b_obs, h_obs = obstacle_constraints(
                velocity, sdf, sdf_gradient, agent_radius, config
            )
            hard_A.append(A_obs)
            hard_b.append(b_obs)
            barriers.append(h_obs)

        A_acc, b_acc = acceleration_polytope(position.shape[0], config)
        hard_A.append(A_acc)
        hard_b.append(b_acc)

        A_hard = np.concatenate(hard_A, axis=1)
        b_hard = np.concatenate(hard_b, axis=1)

        if solver == "hildreth":
            u_safe, slack = solve_qp_slack(
                u_nominal, A_pair, b_pair, A_hard, b_hard,
                slack_weight=config.slack_weight, iterations=config.iterations,
            )
        elif solver == "osqp":
            u_safe = solve_qp_osqp(
                u_nominal,
                np.concatenate([A_pair, A_hard], axis=1),
                np.concatenate([b_pair, b_hard], axis=1),
            )
            slack = np.zeros(position.shape[0])
        else:
            raise ValueError(f"unknown solver {solver!r}")

        h_all = np.concatenate(barriers, axis=1)
        finite = np.isfinite(h_all)

        # How far the returned command still misses the barrier constraints.
        # This -- not xi -- is the infeasibility signal; see the module docstring.
        residual = np.einsum("nkd,nd->nk", A_pair, u_safe) - b_pair
        violation = np.maximum(np.where(np.isfinite(b_pair), residual, -np.inf), 0.0).max(axis=1)

        return u_safe, {
            "min_barrier": np.where(finite.any(1), np.where(finite, h_all, np.inf).min(1), np.inf),
            "intervention": np.linalg.norm(u_safe - u_nominal, axis=-1),
            "violation": violation,
            "relaxed": violation > config.violation_report_threshold,
            "slack": slack,  # solver internal; converges slowly at large weight
            "num_soft": int(A_pair.shape[1]),
            "num_hard": int(A_hard.shape[1]),
        }
