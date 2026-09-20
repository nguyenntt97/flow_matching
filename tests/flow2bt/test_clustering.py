"""Subsystem 3 (topological induction & bifurcations)."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.flow2bt.clustering import (
    agreement_with_labels,
    induce,
    leaf_dispersion,
    leaf_prototypes,
    merge_gaps,
    suggest_num_leaves,
    trajectory_embedding,
)

DT = 0.2


def three_modes(per_mode=40, steps=11, noise=0.02, seed=0):
    """Straight / left / right -- three genuinely distinct strategies."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    bundles, labels = [], []
    for mode, lateral in enumerate((0.0, 1.0, -1.0)):
        base = np.stack([2.6 * t, lateral * np.sin(np.pi * t)], axis=1)
        bundles.append(base[None, ...] + rng.normal(0, noise, (per_mode, steps, 2)))
        labels += [mode] * per_mode
    return np.concatenate(bundles), np.array(labels)


# ------------------------------------------------------------- the lift
def test_embedding_reproduces_the_design_metric():
    """The whole reason there is no M x M matrix: the lift must BE the metric."""
    rng = np.random.default_rng(0)
    trajectories = rng.normal(size=(6, 11, 2))
    lambda_term, lambda_vel = 2.0, 0.5

    embedded = trajectory_embedding(trajectories, DT, lambda_term, lambda_vel)
    lifted = squareform(pdist(embedded, "sqeuclidean"))

    velocity = np.gradient(trajectories, DT, axis=1)
    direct = np.zeros_like(lifted)
    for i in range(len(trajectories)):
        for j in range(len(trajectories)):
            position_term = ((trajectories[i] - trajectories[j]) ** 2).sum() * DT
            terminal = lambda_term * ((trajectories[i, -1] - trajectories[j, -1]) ** 2).sum()
            velocity_term = lambda_vel * ((velocity[i] - velocity[j]) ** 2).sum() * DT
            direct[i, j] = position_term + terminal + velocity_term

    assert np.allclose(lifted, direct)


def test_terminal_weight_separates_destinations():
    """lambda_term must dominate when paths overlap but end apart."""
    steps = 11
    shared = np.zeros((2, steps, 2))
    shared[0, -1] = [0.0, 1.0]
    shared[1, -1] = [0.0, -1.0]
    light = trajectory_embedding(shared, DT, lambda_term=0.0, lambda_vel=0.0)
    heavy = trajectory_embedding(shared, DT, lambda_term=100.0, lambda_vel=0.0)
    assert np.linalg.norm(heavy[0] - heavy[1]) > np.linalg.norm(light[0] - light[1])


def test_velocity_weight_separates_stopping_from_walking():
    """Design 03 sec 1.1: lambda_vel exists to tell a yield from a march."""
    steps = 11
    t = np.linspace(0, 1, steps)
    walking = np.stack([2.6 * t, np.zeros(steps)], axis=1)
    stopping = np.stack([2.6 * t**0.25, np.zeros(steps)], axis=1)  # same end, slower start
    pair = np.stack([walking, stopping])
    without = trajectory_embedding(pair, DT, lambda_term=1.0, lambda_vel=0.0)
    with_velocity = trajectory_embedding(pair, DT, lambda_term=1.0, lambda_vel=10.0)
    assert (
        np.linalg.norm(with_velocity[0] - with_velocity[1])
        > np.linalg.norm(without[0] - without[1])
    )


def test_embedding_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(M, T, d\)"):
        trajectory_embedding(np.zeros((4, 11)), DT)


# ------------------------------------------------------------ induction
def test_recovers_three_planted_modes():
    trajectories, truth = three_modes()
    dendrogram = induce(trajectories, DT, num_leaves=3)
    assert dendrogram.num_leaves == 3
    assert agreement_with_labels(dendrogram.labels, truth)["adjusted_rand"] > 0.99


def test_elbow_finds_the_planted_number_of_modes():
    trajectories, _ = three_modes()
    dendrogram = induce(trajectories, DT, num_leaves=3)
    assert suggest_num_leaves(dendrogram.linkage_matrix) == 3


def test_bifurcation_count_matches_the_cut():
    trajectories, _ = three_modes()
    dendrogram = induce(trajectories, DT, num_leaves=4)
    assert len(dendrogram.bifurcations) == 3          # k leaves -> k-1 splits


def test_bifurcation_branches_partition_their_node():
    trajectories, _ = three_modes()
    dendrogram = induce(trajectories, DT, num_leaves=5)
    for bifurcation in dendrogram.bifurcations:
        assert len(np.intersect1d(bifurcation.left, bifurcation.right)) == 0
        assert bifurcation.size == len(bifurcation.left) + len(bifurcation.right)
        assert 0.0 < bifurcation.balance <= 1.0


def test_root_bifurcation_is_the_deepest_split():
    """The first split must be the tallest merge -- that is what makes it the root."""
    trajectories, _ = three_modes()
    dendrogram = induce(trajectories, DT, num_leaves=4)
    root = dendrogram.bifurcations[0]
    assert root.depth == 0
    assert root.height == max(b.height for b in dendrogram.bifurcations)
    assert root.size == len(trajectories)


def test_depth_increases_down_the_tree():
    trajectories, _ = three_modes(per_mode=30)
    dendrogram = induce(trajectories, DT, num_leaves=6)
    by_depth = {}
    for bifurcation in dendrogram.bifurcations:
        by_depth.setdefault(bifurcation.depth, []).append(bifurcation.size)
    # a deeper split can only ever cover fewer trajectories than the root
    assert max(by_depth) > 0
    assert max(by_depth[max(by_depth)]) < len(trajectories)


def test_min_branch_drops_slivers():
    trajectories, _ = three_modes(per_mode=40)
    outlier = trajectories[:1] + np.array([50.0, 50.0])
    bundle = np.concatenate([trajectories, outlier])
    permissive = induce(bundle, DT, num_leaves=5, min_branch=1)
    strict = induce(bundle, DT, num_leaves=5, min_branch=5)
    assert len(strict.bifurcations) < len(permissive.bifurcations)


def test_induce_needs_enough_trajectories():
    with pytest.raises(ValueError, match="at least 8"):
        induce(np.zeros((4, 11, 2)), DT, num_leaves=8)


# ------------------------------------------------------------- leaves
def test_leaf_prototypes_are_close_to_the_planted_shapes():
    trajectories, truth = three_modes(noise=0.01)
    dendrogram = induce(trajectories, DT, num_leaves=3)
    prototypes = leaf_prototypes(trajectories, dendrogram.labels)
    assert len(prototypes) == 3
    for leaf, prototype in prototypes.items():
        members = trajectories[dendrogram.labels == leaf]
        assert np.abs(prototype - members.mean(axis=0)).max() < 1e-12


def test_dispersion_flags_a_leaf_that_should_have_been_split():
    """Design 05 assumes leaves are unimodal; this is the check on that."""
    trajectories, _ = three_modes(noise=0.01)
    tight = induce(trajectories, DT, num_leaves=3)
    coarse = induce(trajectories, DT, num_leaves=1)
    assert max(leaf_dispersion(trajectories, tight.labels).values()) < 0.1
    assert max(leaf_dispersion(trajectories, coarse.labels).values()) > 0.3


def test_merge_gaps_length():
    trajectories, _ = three_modes()
    dendrogram = induce(trajectories, DT, num_leaves=3)
    assert len(merge_gaps(dendrogram.linkage_matrix)) == len(trajectories) - 2


def test_agreement_is_one_for_identical_partitions():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert agreement_with_labels(labels, labels)["adjusted_rand"] == pytest.approx(1.0)
    relabelled = np.array([2, 2, 0, 0, 1, 1])
    assert agreement_with_labels(relabelled, labels)["adjusted_rand"] == pytest.approx(1.0)
