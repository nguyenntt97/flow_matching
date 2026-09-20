"""Subsystem 5: Dynamical Movement Primitive action leaves.

Design doc: ``survey/design/05_action_leaf_dmp_compression.md``.

A DMP is a spring-damper pulled toward a goal, plus a learned forcing term that
replays the *shape* of a demonstration. Once a tree leaf's trajectory bundle is
unimodal, one DMP reproduces it, and evaluating it is a handful of flops rather
than a network forward.

Conventions here (Ijspeert's, with ``tau`` as a time-scaling factor)::

    canonical:       tau * s_dot = -alpha_s * s,            s(0) = 1
    transformation:  tau^2 * x_ddot = K (g - x) - D tau x_dot + (g - x0) f(s)
    forcing:         f(s) = [ sum_i psi_i(s) w_i / sum_i psi_i(s) ] * s
                     psi_i(s) = exp(-h_i (s - c_i)^2)

The ``(g - x0)`` factor is what makes a DMP generalise to a new goal, and it is
also its classic failure mode: when the goal sits on the start the factor
vanishes, the fit divides by ~0 and the weights explode. ``_spatial_scale``
clamps it and records where it did, rather than silently returning a primitive
that detonates at replay time.

Integration step size is not a free parameter
---------------------------------------------
Measured here: with ``K = 100`` and ``tau = 2 s``, semi-implicit Euler is past
its stability limit at the simulator's own ``dt = 0.2 s`` (5 fps) and the
rollout diverges to tens of metres within one chunk. The damping term
``D tau v / tau^2 = 2 sqrt(K) v / tau`` alone gives a coefficient of 10 s^-1,
and semi-implicit Euler needs ``dt < 2 / 10 = 0.2 s`` -- exactly the step the
simulator runs at.

So these leaves *cannot* be integrated at 5 fps. ``rollout_dmp`` substeps at
``integration_dt`` (default 10 ms) and reports at the requested ``dt``. This is
an independent, numerical argument for the design's fine tick rate: it is not
only about reacting sooner, it is what makes the primitives integrable at all.

Basis count is tied to sample count
-----------------------------------
A leaf's demonstration at ``simulator_fps = 5`` over a ``future_length = 10``
chunk is **11 points**. Fitting 20 Gaussian bases to 11 points is
under-determined, and ridge at ``1e-6`` does not rescue it: measured replay
error went *up* from 2.2 cm at ``P=10`` to 8.9 cm at ``P=20``. ``fit_dmp``
therefore resamples the demonstration onto the integration grid with a cubic
spline before fitting, which both conditions the regression and gives
``np.gradient`` something better than 0.2 s spacing to differentiate. Without
resampling, ``num_basis`` must satisfy ``P <= T/2`` and ``fit_dmp`` enforces it.

Residual accuracy floor
-----------------------
Even with an exact forcing fit, replay carries ~1.7 cm of error over a 2.6 m
movement. That is semi-implicit Euler being first order, not a bad fit -- the
fit residual keeps falling as ``P`` rises while the replay error plateaus. It is
well inside the 0.2 m the benchmark scores collisions at, so it is accepted
rather than traded for a costlier integrator.

Everything is batched over agents. A per-agent Python ``tick()`` would not hold
up at the tick rates the design asks for; ``DMPBank.acceleration`` advances
every agent in one vectorised call and is the only entry point the runtime uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

#: Below this, ``g - x0`` is treated as degenerate and the spatial scaling is
#: pinned to 1.0 for that dimension (metres).
MIN_SPATIAL_SCALE = 1e-2

#: Default internal integration step. See "Integration step size is not a free
#: parameter" above -- integrating at the simulator's 0.2 s diverges.
DEFAULT_INTEGRATION_DT = 0.01


def phase(t: np.ndarray, tau: float, alpha_s: float) -> np.ndarray:
    """Canonical system, solved in closed form: ``s(t) = exp(-alpha_s t / tau)``."""
    return np.exp(-alpha_s * np.asarray(t, dtype=float) / tau)


def basis_centers(num_basis: int, alpha_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian centres spaced evenly in *time*, hence geometrically in phase.

    Returns ``(centers, widths)``. Even spacing in phase would bunch every basis
    near ``s = 0`` and leave the start of the movement unmodelled.
    """
    centers = np.exp(-alpha_s * np.linspace(0.0, 1.0, num_basis))
    # Width so neighbouring bases overlap at roughly half height.
    widths = np.empty_like(centers)
    widths[:-1] = 1.0 / (np.diff(centers) ** 2 + 1e-12)
    widths[-1] = widths[-2] if num_basis > 1 else 1.0
    return centers, widths


def design_matrix(s: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """``Phi[t, i] = psi_i(s_t) * s_t / sum_j psi_j(s_t)`` so ``f(s_t) = Phi[t] @ w``."""
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    psi = np.exp(-widths[None, :] * (s - centers[None, :]) ** 2)
    denom = psi.sum(axis=1, keepdims=True)
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return psi / denom * s


def _spatial_scale(goal: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``g - x0`` with the degenerate-goal guard. Returns (scale, was_clamped)."""
    raw = goal - start
    degenerate = np.abs(raw) < MIN_SPATIAL_SCALE
    return np.where(degenerate, 1.0, raw), degenerate


@dataclass
class DMP:
    """One fitted primitive: a leaf of the behavior tree.

    ``weights`` is ``(num_basis, dim)``. ``tau`` is the demonstration duration in
    seconds, so replaying with a larger ``tau`` executes the same shape slower --
    which is how the design's "Yield / slow down" mode is expressed.
    """

    weights: np.ndarray
    tau: float
    alpha_s: float
    K: float
    D: float
    goal: np.ndarray
    start: np.ndarray
    #: The demonstration's velocity at t=0. A pedestrian leaf is entered at
    #: walking pace, never from rest, so replaying from zero velocity leaves a
    #: start transient the forcing term cannot absorb (measured: ~8 cm on a 2 s
    #: swerve). The runtime passes the agent's live velocity; this is the
    #: sensible default when it does not.
    start_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    name: str = "unnamed"
    fit_residual: float = float("nan")
    #: ``goal - start`` in the frame the primitive was fitted in. At runtime the
    #: leaf is re-aimed by rotating this onto the live navmesh direction and
    #: adding it to where the leaf was entered -- see ``BTController``. A DMP
    #: goal has to be a *fixed* attractor for the movement to converge to;
    #: design 05 sec 1.1 sets it to ``c_{t,nav}``, which recedes as the agent
    #: advances, so the spring term never relaxes. Measured consequence: with
    #: K=100, tau=1.8 and the median waypoint distance of 1.456 m, the spring
    #: term alone commands 45 m/s^2 against an a_max of 3 -- the primitive
    #: saturates actuation permanently and the safety filter has no authority
    #: left. Hence a per-execution fixed goal.
    degenerate_dims: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=bool))

    @property
    def displacement(self) -> np.ndarray:
        return self.goal - self.start

    @property
    def num_basis(self) -> int:
        return self.weights.shape[0]

    @property
    def dim(self) -> int:
        return self.weights.shape[1]

    def forcing(self, s: np.ndarray) -> np.ndarray:
        centers, widths = basis_centers(self.num_basis, self.alpha_s)
        return design_matrix(np.atleast_1d(s), centers, widths) @ self.weights


def resample_trajectory(trajectory: np.ndarray, dt: float, new_dt: float) -> np.ndarray:
    """Cubic-spline a demonstration onto a finer uniform grid.

    Interpolation is an assumption, so it is a named step rather than something
    buried in the fit: pedestrian paths at 5 fps really are smooth at this
    scale, but a demonstration with a genuine discontinuity would be smoothed.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    duration = (trajectory.shape[0] - 1) * dt
    source_t = np.arange(trajectory.shape[0]) * dt
    target_t = np.linspace(0.0, duration, int(round(duration / new_dt)) + 1)
    return CubicSpline(source_t, trajectory, axis=0)(target_t)


def fit_dmp(
    trajectory: np.ndarray,
    dt: float,
    num_basis: int = 20,
    alpha_s: float = 4.0,
    K: float = 100.0,
    D: Optional[float] = None,
    ridge: float = 1e-6,
    name: str = "unnamed",
    resample_dt: Optional[float] = DEFAULT_INTEGRATION_DT,
) -> DMP:
    """Fit one DMP to one demonstration by closed-form ridge regression.

    ``trajectory`` is ``(T, dim)`` in metres, uniformly sampled at ``dt``.
    ``D = 2 sqrt(K)`` (critical damping) unless given.

    ``resample_dt`` upsamples onto the integration grid first -- see "Basis
    count is tied to sample count" above. Pass ``None`` to fit the raw samples,
    in which case ``num_basis`` is capped at half the sample count.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[0] < 4:
        raise ValueError(f"need a (T>=4, dim) trajectory, got {trajectory.shape}")
    if D is None:
        D = 2.0 * np.sqrt(K)

    if resample_dt is not None and resample_dt < dt:
        trajectory = resample_trajectory(trajectory, dt, resample_dt)
        dt = resample_dt
    elif num_basis > trajectory.shape[0] // 2:
        raise ValueError(
            f"num_basis={num_basis} against {trajectory.shape[0]} samples is "
            f"under-determined; use num_basis <= {trajectory.shape[0] // 2} or "
            "set resample_dt"
        )

    num_steps, dim = trajectory.shape
    tau = (num_steps - 1) * dt
    start, goal = trajectory[0].copy(), trajectory[-1].copy()

    # np.gradient is second-order accurate in the interior and one-sided at the
    # ends, which matters: forward differences bias the forcing term at t=0
    # exactly where the phase weight s is largest.
    velocity = np.gradient(trajectory, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)

    scale, degenerate = _spatial_scale(goal, start)
    f_target = (tau**2 * acceleration + D * tau * velocity - K * (goal - trajectory)) / scale

    s = phase(np.arange(num_steps) * dt, tau, alpha_s)
    centers, widths = basis_centers(num_basis, alpha_s)
    phi = design_matrix(s, centers, widths)

    gram = phi.T @ phi + ridge * np.eye(num_basis)
    weights = np.linalg.solve(gram, phi.T @ f_target)
    residual = float(np.sqrt(np.mean((phi @ weights - f_target) ** 2)))

    return DMP(
        weights=weights, tau=tau, alpha_s=alpha_s, K=K, D=D,
        goal=goal, start=start, start_velocity=velocity[0].copy(), name=name,
        fit_residual=residual, degenerate_dims=degenerate,
    )


def integrate_step(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-implicit (symplectic) Euler: velocity first, then position with it.

    Shared by ``rollout_dmp`` and the runtime so the two can never drift apart.
    Explicit Euler is unconditionally unstable on an oscillator; this is not.
    """
    velocity = velocity + acceleration * dt
    return position + velocity * dt, velocity


def rollout_dmp(
    dmp: DMP,
    dt: float,
    duration: Optional[float] = None,
    start: Optional[np.ndarray] = None,
    goal: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
    start_velocity: Optional[np.ndarray] = None,
    integration_dt: float = DEFAULT_INTEGRATION_DT,
) -> np.ndarray:
    """Integrate one DMP forward, reporting every ``dt``. Returns ``(T, dim)``.

    Passing ``start`` / ``goal`` / ``tau`` is how a leaf is *reused*: the same
    shape, re-aimed at the live navmesh waypoint and re-timed.

    ``integration_dt`` is the *internal* step and is capped at ``dt``; the
    default is fine enough for the stiffnesses this module fits. Raising it to
    ``dt`` reproduces the divergence documented in the module docstring.
    """
    tau = float(dmp.tau if tau is None else tau)
    start = dmp.start if start is None else np.asarray(start, dtype=float)
    goal = dmp.goal if goal is None else np.asarray(goal, dtype=float)
    duration = tau if duration is None else duration

    num_steps = int(round(duration / dt)) + 1
    substeps = max(1, int(np.ceil(dt / min(integration_dt, dt))))
    inner_dt = dt / substeps
    scale, _ = _spatial_scale(goal, start)

    position = start.copy()
    velocity = (
        np.asarray(dmp.start_velocity, dtype=float).copy()
        if start_velocity is None
        else np.asarray(start_velocity, dtype=float).copy()
    )
    elapsed = 0.0
    out = np.empty((num_steps, dmp.dim))
    out[0] = position
    for step in range(1, num_steps):
        for _ in range(substeps):
            elapsed += inner_dt
            s = phase(elapsed, tau, dmp.alpha_s)
            forcing = dmp.forcing(s)[0] * scale
            acceleration = (
                dmp.K * (goal - position) - dmp.D * tau * velocity + forcing
            ) / tau**2
            position, velocity = integrate_step(position, velocity, acceleration, inner_dt)
        out[step] = position
    return out


class DMPBank:
    """Every leaf's DMP, stepped for every agent at once.

    ``weights`` is ``(num_leaves, num_basis, dim)``; an agent selects a leaf with
    an index, and one ``step`` advances the whole crowd. This is the only form
    the runtime uses -- see the module docstring.
    """

    def __init__(self, dmps: list[DMP]):
        if not dmps:
            raise ValueError("DMPBank needs at least one primitive")
        num_basis = {d.num_basis for d in dmps}
        alphas = {round(d.alpha_s, 9) for d in dmps}
        if len(num_basis) != 1 or len(alphas) != 1:
            raise ValueError(
                "every DMP in a bank must share num_basis and alpha_s "
                f"(got {sorted(num_basis)} and {sorted(alphas)})"
            )
        self.dmps = list(dmps)
        self.names = [d.name for d in dmps]
        self.alpha_s = dmps[0].alpha_s
        self.weights = np.stack([d.weights for d in dmps])      # (L, P, dim)
        self.K = np.array([d.K for d in dmps])                  # (L,)
        self.D = np.array([d.D for d in dmps])
        self.tau = np.array([d.tau for d in dmps])
        self.centers, self.widths = basis_centers(dmps[0].num_basis, self.alpha_s)

    def __len__(self) -> int:
        return len(self.dmps)

    def acceleration(
        self,
        leaf: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        goal: np.ndarray,
        start: np.ndarray,
        phase_s: np.ndarray,
        tau: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Commanded acceleration for every agent. All arrays are (N, ...).

        ``leaf`` (N,) int selects each agent's primitive. Returns ``(N, dim)``.
        """
        leaf = np.asarray(leaf, dtype=int)
        tau_a = self.tau[leaf] if tau is None else np.asarray(tau, dtype=float)
        K_a, D_a = self.K[leaf], self.D[leaf]

        s = np.asarray(phase_s, dtype=float).reshape(-1, 1)
        psi = np.exp(-self.widths[None, :] * (s - self.centers[None, :]) ** 2)  # (N, P)
        denom = psi.sum(axis=1, keepdims=True)
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        basis = psi / denom * s                                                  # (N, P)

        # (N, P) x (N, P, dim) -> (N, dim)
        forcing = np.einsum("np,npd->nd", basis, self.weights[leaf])
        scale, _ = _spatial_scale(goal, start)

        return (
            K_a[:, None] * (goal - position)
            - D_a[:, None] * tau_a[:, None] * velocity
            + forcing * scale
        ) / tau_a[:, None] ** 2

    def step_phase(self, phase_s: np.ndarray, dt: float, tau: np.ndarray) -> np.ndarray:
        """Exact update of the canonical system over ``dt`` (no Euler drift)."""
        return np.asarray(phase_s) * np.exp(-self.alpha_s * dt / np.asarray(tau, dtype=float))
