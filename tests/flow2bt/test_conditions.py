"""Subsystem 4, second half: guard induction."""

import numpy as np
import pytest

from src.flow2bt.bt import FAILURE, SUCCESS
from src.flow2bt.clustering import induce
from src.flow2bt.conditions import (
    WEAK_GUARD_ACCURACY,
    fit_guard,
    guard_report,
    induce_guards,
    to_condition,
)

NAMES = ["ttc", "d_lat", "speed"]


def separable(n=200, seed=0):
    """Two classes split by ttc = 2.5, with two nuisance features."""
    rng = np.random.default_rng(seed)
    ttc = np.concatenate([rng.uniform(2.6, 10.0, n), rng.uniform(0.0, 2.4, n)])
    features = np.stack([ttc, rng.normal(0, 1.5, 2 * n), rng.normal(1.3, 0.3, 2 * n)], axis=1)
    labels = np.zeros(2 * n, dtype=bool)
    labels[:n] = True
    return features, labels


# ------------------------------------------------------------------ fitting
def test_guard_recovers_a_separable_threshold():
    features, labels = separable()
    guard = fit_guard(features, labels, NAMES, name="path_clear")
    assert guard.accuracy > 0.98
    assert not guard.is_weak
    assert guard.top_terms(1)[0][0] == "ttc", "did not identify the discriminating feature"
    # The implied threshold should land near 2.5 s.
    threshold = -guard.bias / guard.weights[0]
    assert 2.0 < threshold < 3.0, f"implied ttc threshold {threshold:.2f}"


def test_hyperplane_is_reported_in_physical_units():
    """The exact scaled->unscaled identity the auditability claim rests on.

    If this drifts, guards start reading in standardised units and
    ``describe()`` becomes quietly meaningless.
    """
    features, labels = separable()
    guard = fit_guard(features, labels, NAMES)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    reference = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, loss="squared_hinge", dual=True, class_weight="balanced",
                  max_iter=20000, random_state=0),
    ).fit(features, labels.astype(int))

    assert np.array_equal(guard.decide(features), reference.predict(features).astype(bool))
    margin = features @ guard.weights + guard.bias
    assert np.allclose(margin, reference.decision_function(features))


def test_guard_weights_are_in_per_feature_units():
    """A guard whose discriminating feature is metres must have a per-metre weight."""
    rng = np.random.default_rng(1)
    # Identical data, but d_lat expressed in centimetres instead of metres.
    features, labels = separable()
    rescaled = features.copy()
    rescaled[:, 1] *= 100.0
    a = fit_guard(features, labels, NAMES)
    b = fit_guard(rescaled, labels, NAMES)
    # The physical decision is the same, so the weight must shrink by 100x.
    assert np.isclose(a.weights[1], b.weights[1] * 100.0, rtol=0.2, atol=1e-6)


def test_weak_guard_is_flagged_not_dropped():
    """A bifurcation that observable state cannot predict must be reported."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 3))
    labels = rng.random(400) < 0.5            # pure noise
    guard = fit_guard(features, labels, NAMES, name="undecidable")
    assert guard.is_weak
    assert guard.accuracy < WEAK_GUARD_ACCURACY
    assert "[WEAK]" in guard.describe()


def test_single_class_is_rejected():
    features, _ = separable(n=10)
    with pytest.raises(ValueError, match="both classes"):
        fit_guard(features, np.ones(len(features), dtype=bool), NAMES)


def test_tiny_minority_reports_no_cross_validated_score():
    """One sample on a side: resubstitution would flatter, so report nothing."""
    features, labels = separable(n=50)
    labels[:] = False
    labels[0] = True
    guard = fit_guard(features, labels, NAMES)
    assert guard.accuracy == 0.0 and guard.is_weak
    assert guard.balance < 0.02


def test_l1_penalty_produces_sparser_guards():
    features, labels = separable()
    dense = fit_guard(features, labels, NAMES, penalty="l2")
    sparse = fit_guard(features, labels, NAMES, penalty="l1", C=0.1)
    assert np.count_nonzero(np.abs(sparse.weights) > 1e-6) <= np.count_nonzero(
        np.abs(dense.weights) > 1e-6
    )


# ------------------------------------------------------------- integration
def three_modes(per_mode=60, steps=11, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    bundles, starts = [], []
    for lateral, ttc in ((0.0, 8.0), (1.0, 1.5), (-1.0, 1.4)):
        base = np.stack([2.6 * t, lateral * np.sin(np.pi * t)], axis=1)
        bundles.append(base[None, ...] + rng.normal(0, 0.02, (per_mode, steps, 2)))
        starts.append(np.stack([
            rng.normal(ttc, 0.2, per_mode),
            rng.normal(-lateral, 0.2, per_mode),
            rng.normal(1.3, 0.05, per_mode),
        ], axis=1))
    return np.concatenate(bundles), np.concatenate(starts)


def test_guards_are_induced_for_every_bifurcation():
    trajectories, features = three_modes()
    dendrogram = induce(trajectories, 0.2, num_leaves=3)
    guards = induce_guards(features, dendrogram.bifurcations, NAMES)
    assert len(guards) == len(dendrogram.bifurcations) == 2
    assert all(guard.num_samples > 0 for guard in guards)
    assert [guard.node_id for guard in guards] == [
        b.node_id for b in dendrogram.bifurcations
    ]


def test_induced_guards_separate_a_well_posed_problem():
    """Modes distinguished by their initial state must yield strong guards."""
    trajectories, features = three_modes()
    dendrogram = induce(trajectories, 0.2, num_leaves=3)
    guards = induce_guards(features, dendrogram.bifurcations, NAMES)
    assert all(not guard.is_weak for guard in guards), guard_report(guards)


def test_guard_converts_to_a_tickable_condition():
    features, labels = separable()
    guard = fit_guard(features, labels, NAMES, name="path_clear")
    condition = to_condition(guard)
    status = condition.tick(features).status
    expected = np.where(guard.decide(features), SUCCESS, FAILURE)
    assert np.array_equal(status, expected)
    assert "ttc" in condition.describe(NAMES)


def test_report_surfaces_weak_guards():
    features, labels = separable()
    strong = fit_guard(features, labels, NAMES, name="strong")
    rng = np.random.default_rng(0)
    weak = fit_guard(rng.normal(size=(300, 3)), rng.random(300) < 0.5, NAMES, name="weak")
    text = guard_report([strong, weak])
    assert "1 of 2 do not separate" in text
    assert "not decidable at tick time" in text
