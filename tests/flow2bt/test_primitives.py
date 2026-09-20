"""Subsystem 5 (DMP action leaves)."""

import numpy as np
import pytest

from src.flow2bt.primitives import (
    DEFAULT_INTEGRATION_DT,
    resample_trajectory,
    MIN_SPATIAL_SCALE,
    DMPBank,
    basis_centers,
    fit_dmp,
    integrate_step,
    phase,
    rollout_dmp,
)

DT = 0.2        # simulator_fps = 5 -- the rate the demonstrations are sampled at
TICK = 0.01     # the reactive tick the design asks for


def swerve(num_steps=11, lateral=0.6):
    """A 2 s rightward swerve at ~1.3 m/s -- the design's canonical evasion."""
    t = np.linspace(0.0, 1.0, num_steps)
    return np.stack([2.6 * t, -lateral * np.sin(np.pi * t)], axis=1)


def test_phase_decays_monotonically_from_one():
    s = phase(np.arange(11) * DT, tau=2.0, alpha_s=4.0)
    assert s[0] == pytest.approx(1.0)
    assert np.all(np.diff(s) < 0)
    assert s[-1] < 0.05


def test_basis_centers_span_the_phase_range():
    centers, widths = basis_centers(20, alpha_s=4.0)
    assert centers[0] == pytest.approx(1.0)
    assert centers[-1] == pytest.approx(np.exp(-4.0))
    assert np.all(np.diff(centers) < 0)
    assert np.all(widths > 0)


def test_fitted_dmp_reproduces_its_demonstration():
    demo = swerve()
    dmp = fit_dmp(demo, DT, num_basis=20, name="swerve_right")
    replay = rollout_dmp(dmp, DT)
    assert replay.shape == demo.shape
    error = np.linalg.norm(replay - demo, axis=1)
    # 2.5 cm, not 1 mm. The floor is semi-implicit Euler being first order
    # (measured ~1.7 cm over a 2.6 m movement, and it does not improve as the
    # forcing fit improves). Well inside the 0.2 m the benchmark scores
    # collisions at. See "Residual accuracy floor" in primitives.py.
    assert error.max() < 0.025, f"max reproduction error {error.max():.4f} m"


def test_dmp_converges_to_the_goal_from_a_perturbed_start():
    """The stability claim in design 05 -- the leaf must recover, not diverge."""
    dmp = fit_dmp(swerve(), DT, num_basis=20)
    perturbed = dmp.start + np.array([0.0, 0.8])
    replay = rollout_dmp(
        dmp, DT, duration=6.0, start=perturbed, start_velocity=np.zeros(2)
    )
    assert np.linalg.norm(replay[-1] - dmp.goal) < 0.05
    assert np.isfinite(replay).all()


def test_retargeting_the_goal_moves_the_endpoint():
    dmp = fit_dmp(swerve(), DT, num_basis=20)
    new_goal = dmp.goal + np.array([1.0, 0.5])
    replay = rollout_dmp(dmp, DT, duration=6.0, goal=new_goal)
    assert np.linalg.norm(replay[-1] - new_goal) < 0.05


def test_slowing_tau_stretches_the_same_shape():
    """This is how design 05's 'Yield / slow down' mode is expressed."""
    dmp = fit_dmp(swerve(), DT, num_basis=20)
    fast = rollout_dmp(dmp, DT)
    slow = rollout_dmp(dmp, DT, duration=2.5 * dmp.tau, tau=2.5 * dmp.tau)
    assert np.linalg.norm(slow[-1] - fast[-1]) < 0.05       # same destination
    # At the same wall-clock instant the slowed leaf has covered less ground.
    instant = min(len(slow), len(fast)) // 2
    assert np.linalg.norm(slow[instant] - dmp.start) < np.linalg.norm(fast[instant] - dmp.start)


def test_degenerate_goal_is_clamped_not_exploded():
    """g == x0 makes the (g - x0) factor vanish; the fit must stay finite."""
    stationary = np.zeros((11, 2))
    stationary[:, 0] = 1e-4 * np.arange(11)  # essentially not moving
    dmp = fit_dmp(stationary, DT, num_basis=10)
    assert dmp.degenerate_dims.any(), "guard did not fire on a degenerate goal"
    assert np.isfinite(dmp.weights).all()
    assert np.isfinite(rollout_dmp(dmp, DT)).all()


def test_spatial_scale_guard_threshold():
    demo = swerve()
    demo[:, 1] = 0.0                       # y never moves -> degenerate in y only
    dmp = fit_dmp(demo, DT, num_basis=10)
    assert not dmp.degenerate_dims[0]
    assert dmp.degenerate_dims[1]
    assert abs(dmp.goal[1] - dmp.start[1]) < MIN_SPATIAL_SCALE


# --------------------------------------------------------------------- bank
def test_bank_matches_single_dmp_rollout():
    """The batched path is the one the runtime uses; it must not drift.

    Both sides integrate at TICK so this compares the *maths*, not two
    different discretisations.
    """
    dmps = [
        fit_dmp(swerve(lateral=0.6), DT, num_basis=20, name="right"),
        fit_dmp(swerve(lateral=-0.6), DT, num_basis=20, name="left"),
    ]
    bank = DMPBank(dmps)
    assert len(bank) == 2 and bank.names == ["right", "left"]

    for leaf_index, dmp in enumerate(dmps):
        reference = rollout_dmp(dmp, TICK, integration_dt=TICK)

        leaf = np.array([leaf_index])
        position = dmp.start[None, :].copy()
        velocity = dmp.start_velocity[None, :].copy()
        goal = dmp.goal[None, :].copy()
        start = dmp.start[None, :].copy()
        tau = np.array([dmp.tau])
        phase_s = np.array([1.0])

        traced = [position[0].copy()]
        for _ in range(len(reference) - 1):
            phase_s = bank.step_phase(phase_s, TICK, tau)
            acceleration = bank.acceleration(
                leaf, position, velocity, goal, start, phase_s, tau
            )
            position, velocity = integrate_step(position, velocity, acceleration, TICK)
            traced.append(position[0].copy())

        assert np.allclose(np.array(traced), reference, atol=1e-9)


def test_integrating_at_simulator_rate_diverges():
    """Pins the finding that motivates the fine tick (see the module docstring).

    K=100 / tau=2 s gives a damping coefficient of 10 per second; semi-implicit
    Euler needs dt < 0.2 s, which is exactly the simulator's own step. If this
    ever starts passing, the default stiffness changed and the claim in
    primitives.py must be re-measured rather than left standing.
    """
    dmp = fit_dmp(swerve(), DT, num_basis=20)
    coarse = rollout_dmp(dmp, DT, integration_dt=DT)
    fine = rollout_dmp(dmp, DT, integration_dt=DEFAULT_INTEGRATION_DT)
    assert np.linalg.norm(fine[-1] - dmp.goal) < 0.05
    assert np.linalg.norm(coarse[-1] - dmp.goal) > 1.0


def test_bank_steps_agents_on_different_leaves_independently():
    bank = DMPBank([
        fit_dmp(swerve(lateral=0.6), DT, num_basis=20, name="right"),
        fit_dmp(swerve(lateral=-0.6), DT, num_basis=20, name="left"),
    ])
    leaf = np.array([0, 1, 0])
    position = np.zeros((3, 2))
    velocity = np.zeros((3, 2))
    goal = np.tile(np.array([2.6, 0.0]), (3, 1))
    acceleration = bank.acceleration(
        leaf, position, velocity, goal, position.copy(), np.full(3, 0.9)
    )
    assert acceleration.shape == (3, 2)
    # agents 0 and 2 share a leaf and a state -> identical; agent 1 differs.
    assert np.allclose(acceleration[0], acceleration[2])
    assert not np.allclose(acceleration[0], acceleration[1])


def test_bank_rejects_mismatched_primitives():
    a = fit_dmp(swerve(), DT, num_basis=10)
    b = fit_dmp(swerve(), DT, num_basis=20)
    with pytest.raises(ValueError, match="num_basis"):
        DMPBank([a, b])


def test_step_phase_is_exact_not_euler():
    bank = DMPBank([fit_dmp(swerve(), DT, num_basis=10)])
    tau = np.array([2.0])
    s = np.array([1.0])
    for _ in range(10):
        s = bank.step_phase(s, DT, tau)
    assert s[0] == pytest.approx(phase(10 * DT, 2.0, bank.alpha_s))


# ------------------------------------------------------- conditioning of the fit
def test_more_bases_never_makes_replay_worse_once_resampled():
    """Pins the fix for the measured P=20-on-11-samples overfit.

    Without resampling, replay error rose from 2.2 cm at P=10 to 8.9 cm at
    P=20. With resampling it must decrease monotonically (to the integrator
    floor) -- if this regresses, resampling silently stopped happening.
    """
    demo = swerve()
    errors = []
    for num_basis in (5, 10, 20, 30):
        dmp = fit_dmp(demo, DT, num_basis=num_basis)
        replay = rollout_dmp(dmp, DT)
        errors.append(np.linalg.norm(replay - demo, axis=1).max())
    assert all(b <= a + 1e-6 for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 0.02


def test_unresampled_fit_refuses_an_underdetermined_basis():
    with pytest.raises(ValueError, match="under-determined"):
        fit_dmp(swerve(), DT, num_basis=20, resample_dt=None)


def test_unresampled_fit_allows_a_conditioned_basis():
    dmp = fit_dmp(swerve(), DT, num_basis=5, resample_dt=None)
    assert dmp.num_basis == 5
    assert np.isfinite(dmp.weights).all()


def test_resample_preserves_endpoints_and_refines_the_grid():
    demo = swerve()
    fine = resample_trajectory(demo, DT, 0.01)
    assert fine.shape[0] == int(round((len(demo) - 1) * DT / 0.01)) + 1
    assert np.allclose(fine[0], demo[0])
    assert np.allclose(fine[-1], demo[-1])
    # the resampled path passes through every original sample
    assert np.allclose(fine[::20], demo, atol=1e-9)
