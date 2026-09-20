"""Subsystem 4 (second half): fitting a guard hyperplane at each bifurcation.

Design doc: ``survey/design/04_condition_node_induction.md`` sec 1.2.

Each bifurcation from Subsystem 3 gives two sets of trajectories. Their *initial
observable states* become the positive and negative classes of a soft-margin
SVM, and the resulting hyperplane is the guard a Behavior Tree ticks.

Scaled to fit, unscaled to read
-------------------------------
An SVM's ``C`` only means anything against comparably-scaled features, and this
feature space is not: ``ttc`` spans 0-10 s while ``nav_heading`` spans +/- pi and
``clearance`` runs to 11 m. Fitting raw makes the penalty effectively
per-feature-arbitrary.

But a guard standing in standardised units is not auditable, and auditability is
the entire claim being made for this layer over a neural gate. So the fit is
done on standardised features and the hyperplane is then transformed back::

    z = (x - mu) / sigma
    w_z . z + b_z  ==  (w_z / sigma) . x  +  (b_z - sum_i w_z,i mu_i / sigma_i)

which is an exact identity, not an approximation -- ``test_conditions.py`` pins
that the two forms agree on every sample. The stored guard is in metres,
seconds and radians and reads as physics.

Reporting a guard that does not separate
----------------------------------------
Not every split in a trajectory dendrogram is decidable from what an agent can
observe at the start: two bundles can diverge because of something that happens
later. ``InducedGuard.accuracy`` is cross-validated, and
``induce_guards`` keeps the weak ones instead of dropping them, flagged. A
bifurcation that cannot be predicted is a finding about the design -- it means
the tree structure discovered offline is not realisable as a reactive guard --
and hiding it behind a filter would be the wrong kind of tidy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: Below this cross-validated accuracy a guard is flagged as not separating.
#: 0.65 is deliberately low: the point is to surface honest weakness, not to
#: assert a quality bar the data may not support.
WEAK_GUARD_ACCURACY = 0.65


@dataclass
class InducedGuard:
    """A fitted guard, in physical units."""

    name: str
    weights: np.ndarray            # in feature units (per metre, per second, ...)
    bias: float
    feature_names: list[str]
    accuracy: float                # cross-validated
    train_accuracy: float
    balance: float                 # fraction of the minority class
    num_samples: int
    node_id: int = -1

    @property
    def is_weak(self) -> bool:
        return self.accuracy < WEAK_GUARD_ACCURACY

    def decide(self, features: np.ndarray) -> np.ndarray:
        """True = take the left branch."""
        return (features @ self.weights + self.bias) >= 0.0

    def top_terms(self, count: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.weights))[:count]
        return [(self.feature_names[i], float(self.weights[i])) for i in order]

    def describe(self, count: int = 3) -> str:
        terms = " ".join(f"{weight:+.3g}*{name}" for name, weight in self.top_terms(count))
        flag = "  [WEAK]" if self.is_weak else ""
        return f"{self.name}: {terms} {self.bias:+.3g} >= 0  (acc {self.accuracy:.3f}){flag}"


def fit_guard(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    name: str = "guard",
    C: float = 1.0,
    penalty: str = "l2",
    folds: int = 5,
    seed: int = 0,
) -> InducedGuard:
    """Soft-margin linear SVM, returned in physical units.

    ``labels`` is boolean: True for the left branch (the positive class).
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError(f"guard {name!r} needs both classes present")

    def make():
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=C, penalty=penalty, loss="squared_hinge",
                dual=penalty == "l2", class_weight="balanced",
                max_iter=20000, random_state=seed,
            ),
        )

    minority = min(np.bincount(labels))
    usable_folds = max(2, min(folds, minority))
    if minority >= 2:
        scores = cross_val_score(make(), features, labels, cv=usable_folds)
        accuracy = float(scores.mean())
    else:
        # One sample on a side: cross-validation is meaningless. Say so by
        # reporting nan rather than a flattering resubstitution score.
        accuracy = float("nan")

    model = make().fit(features, labels)
    scaler, svm = model.named_steps["standardscaler"], model.named_steps["linearsvc"]

    # Exact change of variables back to physical units -- see the module docstring.
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    weights = svm.coef_[0] / scale
    bias = float(svm.intercept_[0] - np.sum(svm.coef_[0] * scaler.mean_ / scale))

    return InducedGuard(
        name=name,
        weights=weights,
        bias=bias,
        feature_names=list(feature_names),
        accuracy=accuracy if np.isfinite(accuracy) else 0.0,
        train_accuracy=float(model.score(features, labels)),
        balance=float(minority / len(labels)),
        num_samples=int(len(labels)),
    )


def induce_guards(
    features: np.ndarray,
    bifurcations,
    feature_names: Sequence[str],
    C: float = 1.0,
    penalty: str = "l2",
    seed: int = 0,
) -> list[InducedGuard]:
    """One guard per bifurcation, in tree order. Weak guards are kept and flagged."""
    guards = []
    for index, bifurcation in enumerate(bifurcations):
        members = np.concatenate([bifurcation.left, bifurcation.right])
        labels = np.zeros(len(members), dtype=bool)
        labels[: len(bifurcation.left)] = True
        guards.append(
            fit_guard(
                features[members], labels, feature_names,
                name=f"guard_{index}_node{bifurcation.node_id}",
                C=C, penalty=penalty, seed=seed,
            )
        )
        guards[-1].node_id = bifurcation.node_id
    return guards


def guard_report(guards: Sequence[InducedGuard]) -> str:
    """Human-readable summary -- this is the auditability deliverable."""
    lines = [f"{len(guards)} induced guards:"]
    for guard in guards:
        lines.append("  " + guard.describe())
    weak = [g for g in guards if g.is_weak]
    if weak:
        lines.append("")
        lines.append(
            f"  {len(weak)} of {len(guards)} do not separate from observable state "
            f"(cross-validated accuracy < {WEAK_GUARD_ACCURACY}). Those bifurcations "
            "exist in the offline trajectory bundle but are not decidable at tick time."
        )
    return "\n".join(lines)


def to_condition(guard: InducedGuard):
    """Convert to the Behavior Tree's Condition node."""
    from src.flow2bt.bt import Condition

    return Condition(guard.name, guard.weights, guard.bias)
