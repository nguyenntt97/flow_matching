"""Subsystem 6 (assembly): dendrogram -> Guarded Fallback tree."""

import numpy as np
import pytest

from src.flow2bt.assembly import assemble, name_leaves_by_geometry, tree_report
from src.flow2bt.bt import NO_ACTION, RUNNING, action_leaves, condition_nodes
from src.flow2bt.clustering import induce, leaf_prototypes
from src.flow2bt.conditions import induce_guards

NAMES = ["ttc", "d_lat", "speed"]
DT = 0.2


def scenario(per_mode=60, steps=11, seed=0):
    """Three modes whose initial state predicts which one is taken."""
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


@pytest.fixture
def built():
    trajectories, features = scenario()
    dendrogram = induce(trajectories, DT, num_leaves=3)
    guards = induce_guards(features, dendrogram.bifurcations, NAMES)
    tree = assemble(dendrogram, guards, NAMES)
    return trajectories, features, dendrogram, tree


def test_tree_has_one_action_per_leaf_and_one_condition_per_split(built):
    _, _, dendrogram, tree = built
    assert len(action_leaves(tree.root)) == dendrogram.num_leaves == 3
    assert len(condition_nodes(tree.root)) == len(dendrogram.bifurcations) == 2
    assert sorted(leaf.leaf_index for leaf in action_leaves(tree.root)) == [0, 1, 2]


def test_every_agent_always_reaches_an_action(built):
    """Totality: the fallback structure must leave no agent without a command.

    Checked on random features, not just the training ones -- a tick during a
    rollout will see states the induction never saw.
    """
    _, features, _, tree = built
    rng = np.random.default_rng(0)
    random_states = rng.normal(loc=features.mean(0), scale=features.std(0) * 5, size=(4000, 3))
    for batch in (features, random_states):
        result = tree.tick(batch)
        assert np.all(result.status == RUNNING)
        assert np.all(result.action != NO_ACTION)


def test_tree_routes_agents_to_the_leaf_their_state_implies(built):
    """End-to-end: induction -> guards -> tree must reproduce the clustering."""
    trajectories, features, dendrogram, tree = built
    routed = tree.tick(features).action
    agreement = (routed == dendrogram.labels).mean()
    assert agreement > 0.95, f"tree routed only {agreement:.1%} of agents to their own leaf"


def test_guards_are_matched_by_node_id_not_position(built):
    """A reordering must not silently attach the wrong guard to a split."""
    trajectories, features = scenario()
    dendrogram = induce(trajectories, DT, num_leaves=3)
    guards = induce_guards(features, dendrogram.bifurcations, NAMES)
    forward = assemble(dendrogram, guards, NAMES)
    reversed_order = assemble(dendrogram, list(reversed(guards)), NAMES)
    assert np.array_equal(
        forward.tick(features).action, reversed_order.tick(features).action
    )


def test_missing_guard_is_refused(built):
    _, features, dendrogram, _ = built
    guards = induce_guards(features, dendrogram.bifurcations, NAMES)
    with pytest.raises(ValueError, match="no guard for internal node"):
        assemble(dendrogram, guards[:-1], NAMES)


def test_topology_is_nested_not_flat(built):
    """The dendrogram's structure must survive into the tree.

    A flat Fallback over three leaves would also route correctly and would
    throw away what Subsystem 3 was for, so depth is the thing to assert.
    """
    _, _, _, tree = built
    text = tree.render()
    indents = [len(line) - len(line.lstrip()) for line in text.splitlines()]
    assert max(indents) >= 6, f"tree is too shallow to be nested:\n{text}"


def test_leaves_are_named_from_measured_geometry():
    """Design 05 names eight modes as a hypothesis; naming must follow the data."""
    steps = 11
    t = np.linspace(0, 1, steps)
    prototypes = {
        0: np.stack([2.6 * t, np.zeros(steps)], axis=1),              # march
        1: np.stack([2.6 * t, 0.8 * np.sin(np.pi * t) + 0.8 * t], axis=1),   # left
        2: np.stack([2.6 * t, -0.8 * np.sin(np.pi * t) - 0.8 * t], axis=1),  # right
        3: np.zeros((steps, 2)),                                      # stop
        4: np.stack([-1.0 * t, np.zeros(steps)], axis=1),             # backstep
    }
    names = name_leaves_by_geometry(prototypes)
    assert names[0].startswith("march")
    assert names[1].startswith("swerve_left")
    assert names[2].startswith("swerve_right")
    assert names[3].startswith("stop")
    assert names[4].startswith("backstep")


def test_named_leaves_reach_the_tree(built):
    trajectories, features, dendrogram, _ = built
    prototypes = leaf_prototypes(trajectories, dendrogram.labels)
    names = name_leaves_by_geometry(prototypes)
    guards = induce_guards(features, dendrogram.bifurcations, NAMES)
    tree = assemble(dendrogram, guards, NAMES, leaf_names=names)
    rendered = tree.render()
    for name in names.values():
        assert name in rendered


def test_report_is_auditable(built):
    _, _, _, tree = built
    text = tree_report(tree)
    assert "action leaves" in text and "induced guards" in text
    # Guards must render in feature terms, which is the interpretability claim.
    assert any(feature in text for feature in NAMES)
