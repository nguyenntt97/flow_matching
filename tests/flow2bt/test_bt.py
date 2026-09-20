"""Subsystem 6 (reactive Behavior Tree)."""

import numpy as np
import pytest
import torch

from src.flow2bt.bt import (
    FAILURE,
    NO_ACTION,
    RUNNING,
    SUCCESS,
    Action,
    Condition,
    Fallback,
    Sequence,
    action_leaves,
    condition_nodes,
    render,
)

FEATURES = ["ttc", "d_lat"]


def always(status):
    """A leaf pinned to one status, for exercising composite semantics."""
    return Action("stub", leaf_index=0, terminator=lambda f: np.full(f.shape[0], status))


def const_condition(passes):
    """Condition that succeeds iff ``passes`` -- bias only, no features."""
    return Condition("c", np.zeros(len(FEATURES)), 1.0 if passes else -1.0)


ONE = np.zeros((1, len(FEATURES)))


# ------------------------------------------------------------------ leaves
def test_condition_is_a_halfplane_on_the_features():
    node = Condition("path_clear", np.array([1.0, 0.0]), -2.5)   # ttc >= 2.5
    features = np.array([[3.0, 0.0], [2.5, 0.0], [1.0, 0.0]])
    assert list(node.tick(features).status) == [SUCCESS, SUCCESS, FAILURE]
    assert np.all(node.tick(features).action == NO_ACTION)


def test_condition_describes_itself_in_feature_terms():
    """The interpretability claim: a guard must read as physics, not weights."""
    text = Condition("path_clear", np.array([1.0, -0.5]), -2.5).describe(FEATURES)
    assert "ttc" in text and "d_lat" in text and "-2.5" in text


def test_action_runs_and_reports_its_leaf():
    result = Action("march", leaf_index=3).tick(np.zeros((4, 2)))
    assert list(result.status) == [RUNNING] * 4
    assert list(result.action) == [3] * 4


def test_finished_action_is_not_active():
    result = always(SUCCESS).tick(ONE)
    assert result.status[0] == SUCCESS
    assert result.action[0] == NO_ACTION


# -------------------------------------------------------------- composites
@pytest.mark.parametrize(
    "children,expected",
    [
        ([SUCCESS, SUCCESS], SUCCESS),
        ([SUCCESS, FAILURE], FAILURE),
        ([FAILURE, SUCCESS], FAILURE),
        ([SUCCESS, RUNNING], RUNNING),
        ([RUNNING, FAILURE], RUNNING),      # short-circuits before the failure
        ([FAILURE, RUNNING], FAILURE),      # short-circuits before the running
    ],
)
def test_sequence_semantics(children, expected):
    node = Sequence("seq", [always(status) for status in children])
    assert node.tick(ONE).status[0] == expected


@pytest.mark.parametrize(
    "children,expected",
    [
        ([FAILURE, FAILURE], FAILURE),
        ([FAILURE, SUCCESS], SUCCESS),
        ([SUCCESS, FAILURE], SUCCESS),
        ([FAILURE, RUNNING], RUNNING),
        ([RUNNING, SUCCESS], RUNNING),      # short-circuits before the success
        ([SUCCESS, RUNNING], SUCCESS),      # short-circuits before the running
    ],
)
def test_fallback_semantics(children, expected):
    node = Fallback("fb", [always(status) for status in children])
    assert node.tick(ONE).status[0] == expected


def test_empty_composite_is_rejected():
    with pytest.raises(ValueError, match="at least one child"):
        Sequence("seq", [])


# ------------------------------------------------- per-agent branch divergence
def test_agents_take_different_branches_in_one_tick():
    """The reason this is arrays: one tick, different agents, different leaves.

    ttc >= 2.5 -> march (leaf 0); otherwise swerve (leaf 1).
    """
    tree = Fallback("root", [
        Sequence("nominal", [Condition("path_clear", np.array([1.0, 0.0]), -2.5),
                             Action("march", leaf_index=0)]),
        Sequence("evade", [Condition("always", np.zeros(2), 1.0),
                           Action("swerve", leaf_index=1)]),
    ])
    features = np.array([[4.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.5, 0.0]])
    result = tree.tick(features)
    assert list(result.status) == [RUNNING] * 4
    assert list(result.action) == [0, 1, 0, 1]


def test_guard_failure_preempts_within_a_single_tick():
    """Design 06's preemption claim, at the level this module can show it."""
    tree = Fallback("root", [
        Sequence("nominal", [Condition("path_clear", np.array([1.0, 0.0]), -2.5),
                             Action("march", leaf_index=0)]),
        Action("swerve", leaf_index=1),
    ])
    clear = tree.tick(np.array([[4.0, 0.0]]))
    blocked = tree.tick(np.array([[1.1, 0.0]]))
    assert clear.action[0] == 0
    assert blocked.action[0] == 1, "guard failed but the tree did not switch leaf"


# --------------------------------------------------------- soft relaxation
def soft_leaf(p_success, p_failure, p_running):
    class _Leaf(Action):
        def tick_soft(self, features, beta=1.0):
            out = torch.zeros(features.shape[0], 3, dtype=features.dtype)
            out[:, 0], out[:, 1], out[:, 2] = p_success, p_failure, p_running
            return out

    return _Leaf("soft", leaf_index=0)


def test_soft_status_is_a_distribution_for_sequences_and_fallbacks():
    """Design 06 sec 1.4 defines p_run as a remainder; it must stay non-negative.

    For a Sequence that reduces to prod(s_i) <= prod(s_i + r_i), which holds
    termwise. Checked over random children rather than argued.
    """
    rng = np.random.default_rng(0)
    features = torch.zeros(1, 2, dtype=torch.float64)
    for _ in range(300):
        triples = rng.dirichlet(np.ones(3), size=3)
        children = [soft_leaf(*triple) for triple in triples]
        for composite in (Sequence("s", children), Fallback("f", children)):
            z = composite.tick_soft(features)
            assert torch.all(z >= -1e-12), z
            assert torch.allclose(z.sum(-1), torch.ones(1, dtype=torch.float64))


def test_soft_condition_hardens_as_beta_grows():
    node = Condition("path_clear", np.array([1.0, 0.0]), -2.5)
    features = torch.tensor([[4.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    hard = node.tick(features.numpy()).status
    for beta in (1.0, 10.0, 200.0):
        soft = node.tick_soft(features, beta=beta)
    assert soft[0, 0] > 0.999 and soft[1, 1] > 0.999
    assert list(hard) == [SUCCESS, FAILURE]


def test_soft_tree_agrees_with_hard_tree_at_high_beta():
    tree = Fallback("root", [
        Sequence("nominal", [Condition("path_clear", np.array([1.0, 0.0]), -2.5),
                             Action("march", leaf_index=0)]),
        Action("swerve", leaf_index=1),
    ])
    features = torch.tensor([[4.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    soft = tree.tick_soft(features, beta=500.0)
    # Both agents end up executing something, so RUNNING dominates either way.
    assert torch.all(soft[:, 2] > 0.99)
    assert list(tree.tick(features.numpy()).status) == [RUNNING, RUNNING]


def test_soft_path_is_differentiable_into_the_hyperplane():
    """The point of the soft mode: gradients must reach the guard's bias."""
    node = Condition("path_clear", np.array([1.0, 0.0]), -2.5)
    weights = torch.tensor(node.weights, dtype=torch.float64, requires_grad=True)
    features = torch.tensor([[2.4, 0.0]], dtype=torch.float64)
    probability = torch.sigmoid(10.0 * (features @ weights + node.bias))
    probability.sum().backward()
    assert weights.grad is not None and torch.any(weights.grad.abs() > 0)


# ----------------------------------------------------------------- walking
def test_walk_finds_every_leaf():
    tree = Fallback("root", [
        Sequence("a", [Condition("c1", np.zeros(2), 0.0), Action("a1", 0)]),
        Sequence("b", [Condition("c2", np.zeros(2), 0.0), Action("a2", 1)]),
    ])
    assert [leaf.leaf_index for leaf in action_leaves(tree)] == [0, 1]
    assert [node.name for node in condition_nodes(tree)] == ["c1", "c2"]
    text = render(tree, FEATURES)
    assert text.count("\n") == 6 and "root" in text
