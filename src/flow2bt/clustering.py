"""Subsystem 3: topological induction -- bifurcations out of a trajectory bundle.

Design doc: ``survey/design/03_topological_induction_bifurcations.md``.

The design's pairwise metric (sec 1.1) is::

    D(xi_i, xi_j) = int ||xi_i - xi_j||^2 dt
                  + lambda_term ||xi_i(T) - xi_j(T)||^2
                  + lambda_vel int ||xi_i' - xi_j'||^2 dt

Do not build the M x M matrix
------------------------------
Written out, every term is a weighted squared Euclidean distance, so ``D`` is
exactly ``||phi(xi_i) - phi(xi_j)||^2`` for the explicit lift

    phi(xi) = [ xi * sqrt(dt) ,  sqrt(lambda_term) * xi(T) ,  xi' * sqrt(lambda_vel * dt) ]

Running Ward on ``phi`` directly is both cheaper (O(M) memory instead of
O(M^2)) and *more correct*: ``scipy``'s Ward implementation assumes Euclidean
inputs, and feeding it a precomputed matrix it cannot verify is a silent
modelling error. The lift makes the assumption true by construction.

What this is for
----------------
Two outputs feed the rest of the pipeline:

* **leaf clusters** -- unimodal bundles that Subsystem 5 fits one DMP each to;
* **bifurcations** -- for each internal split, the two membership sets, which
  are the positive and negative classes Subsystem 4 fits a guard hyperplane to.

The honest question this module is built to answer
--------------------------------------------------
Design 03 claims reverse-time flow bifurcation discovers structure that
CrowdES's flat KMeans over endpoints cannot. That is testable:
``agreement_with_labels`` scores the induced leaves against the B=8 behaviour
labels upstream already assigns. A high score means the hierarchy recovered the
same modes (and the contribution is the *tree*, not the partition); a low score
means they are genuinely different partitions and that difference has to be
justified on its merits. Either answer is reportable; assuming the first is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, to_tree


@dataclass
class Bifurcation:
    """One internal split of the dendrogram."""

    node_id: int
    depth: int
    height: float               # Ward merge cost -- the "how distinct" measure
    left: np.ndarray            # indices of trajectories taking the left branch
    right: np.ndarray
    left_child_id: int
    right_child_id: int

    @property
    def size(self) -> int:
        return len(self.left) + len(self.right)

    @property
    def balance(self) -> float:
        """0.5 = an even split. Very lopsided splits rarely make useful guards."""
        return min(len(self.left), len(self.right)) / max(self.size, 1)


@dataclass
class Dendrogram:
    linkage_matrix: np.ndarray
    labels: np.ndarray                          # (M,) leaf id per trajectory, 0-based
    bifurcations: list[Bifurcation]
    leaf_members: dict[int, np.ndarray] = field(default_factory=dict)
    #: scipy node id of the root, and of every internal node kept by the cut.
    #: Subsystem 6 walks these to mirror the dendrogram's topology in the tree.
    root_id: int = -1
    internal_ids: frozenset[int] = frozenset()
    #: scipy node id -> cut-cluster label, for the nodes the cut turns into leaves.
    node_to_leaf: dict[int, int] = field(default_factory=dict)

    @property
    def num_leaves(self) -> int:
        return len(self.leaf_members)


def trajectory_embedding(
    trajectories: np.ndarray,
    dt: float,
    lambda_term: float = 1.0,
    lambda_vel: float = 1.0,
) -> np.ndarray:
    """Lift ``(M, T, d)`` trajectories so plain Euclidean distance equals ``D``.

    See the module docstring for why this is a lift rather than a distance
    matrix. Returns ``(M, T*d + d + T*d)``.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    if trajectories.ndim != 3:
        raise ValueError(f"expected (M, T, d), got {trajectories.shape}")

    position = trajectories * np.sqrt(dt)
    terminal = trajectories[:, -1, :] * np.sqrt(lambda_term)
    velocity = np.gradient(trajectories, dt, axis=1) * np.sqrt(lambda_vel * dt)

    return np.concatenate(
        [position.reshape(len(trajectories), -1), terminal,
         velocity.reshape(len(trajectories), -1)],
        axis=1,
    )


def _collect(node, out: dict) -> np.ndarray:
    """Leaf indices under ``node``, memoised into ``out``."""
    if node.id in out:
        return out[node.id]
    if node.is_leaf():
        members = np.array([node.id])
    else:
        members = np.concatenate([_collect(node.left, out), _collect(node.right, out)])
    out[node.id] = members
    return members


def induce(
    trajectories: np.ndarray,
    dt: float,
    num_leaves: int = 8,
    lambda_term: float = 1.0,
    lambda_vel: float = 1.0,
    min_branch: int = 1,
) -> Dendrogram:
    """Ward agglomeration over the lifted bundle, cut to ``num_leaves``.

    ``min_branch`` drops bifurcations whose smaller side has fewer members than
    this: a split that separates two trajectories out of five hundred is noise,
    and fitting a guard to it produces a condition that fires essentially never.
    """
    embedding = trajectory_embedding(trajectories, dt, lambda_term, lambda_vel)
    if len(embedding) < num_leaves:
        raise ValueError(f"need at least {num_leaves} trajectories, got {len(embedding)}")

    linkage_matrix = linkage(embedding, method="ward")
    labels = fcluster(linkage_matrix, num_leaves, criterion="maxclust") - 1

    root, nodes = to_tree(linkage_matrix, rd=True)
    members: dict[int, np.ndarray] = {}
    _collect(root, members)

    # The cut into `num_leaves` clusters uses exactly the top num_leaves-1
    # merges, so those are the internal nodes of the tree we actually assemble.
    internal = sorted(
        (node for node in nodes if not node.is_leaf()),
        key=lambda n: n.dist,
        reverse=True,
    )[: max(num_leaves - 1, 0)]
    kept = {node.id for node in internal}

    # Depth by descending from the root through kept nodes only, so it is the
    # depth in the *assembled* tree rather than in the full agglomeration.
    depths: dict[int, int] = {}

    def descend(node, depth: int) -> None:
        if node.is_leaf() or node.id not in kept:
            return
        depths[node.id] = depth
        descend(node.left, depth + 1)
        descend(node.right, depth + 1)

    descend(root, 0)

    bifurcations = []
    for node in internal:
        left = members[node.left.id]
        right = members[node.right.id]
        if min(len(left), len(right)) < min_branch:
            continue
        bifurcations.append(
            Bifurcation(
                node_id=int(node.id),
                depth=int(depths.get(node.id, 0)),
                height=float(node.dist),
                left=np.sort(left),
                right=np.sort(right),
                left_child_id=int(node.left.id),
                right_child_id=int(node.right.id),
            )
        )
    bifurcations.sort(key=lambda b: (b.depth, -b.height))

    leaf_members = {
        int(leaf): np.where(labels == leaf)[0] for leaf in np.unique(labels)
    }

    # Any child of a kept node that is not itself kept is a leaf of the cut
    # tree. Its cut-cluster label is that of any trajectory beneath it -- they
    # all share one, because the cut is exactly these merges.
    node_to_leaf = {}
    for node in internal:
        for child in (node.left, node.right):
            if child.id not in kept:
                node_to_leaf[int(child.id)] = int(labels[members[child.id][0]])

    return Dendrogram(
        linkage_matrix=linkage_matrix,
        labels=labels,
        bifurcations=bifurcations,
        leaf_members=leaf_members,
        root_id=int(root.id),
        internal_ids=frozenset(int(i) for i in kept),
        node_to_leaf=node_to_leaf,
    )


def merge_gaps(linkage_matrix: np.ndarray) -> np.ndarray:
    """Successive differences in merge height -- the design's ``Delta Var``.

    A large gap means the next merge fused two genuinely distinct bundles, so
    the number of leaves *above* that gap is a natural cut.
    """
    heights = linkage_matrix[:, 2]
    return np.diff(heights)


def suggest_num_leaves(linkage_matrix: np.ndarray, max_leaves: int = 16) -> int:
    """Elbow: the cut just above the largest merge-height gap.

    Reported alongside the configured cut so a mismatch between what the data
    suggests and what CrowdES's B=8 imposes is visible rather than assumed away.
    """
    gaps = merge_gaps(linkage_matrix)
    if len(gaps) == 0:
        return 1
    window = gaps[-max_leaves:]
    offset = len(gaps) - len(window)
    return int(len(linkage_matrix) + 1 - (offset + int(np.argmax(window)) + 1))


def agreement_with_labels(induced: np.ndarray, reference: np.ndarray) -> dict:
    """Adjusted Rand / mutual information against an existing partition.

    Used to ask whether the induced hierarchy recovers CrowdES's B=8 KMeans
    behaviour modes. See the module docstring.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    return {
        "adjusted_rand": float(adjusted_rand_score(reference, induced)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(reference, induced)),
        "num_induced": int(len(np.unique(induced))),
        "num_reference": int(len(np.unique(reference))),
    }


def leaf_prototypes(trajectories: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean trajectory per leaf -- the demonstration each DMP is fitted to.

    A mean is only meaningful because the leaves are (claimed to be) unimodal;
    ``leaf_dispersion`` is the number that says whether that claim holds.
    """
    return {
        int(leaf): trajectories[labels == leaf].mean(axis=0)
        for leaf in np.unique(labels)
    }


def leaf_dispersion(trajectories: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """RMS deviation from the leaf mean, in metres.

    This is the check on design 05's premise that a leaf bundle is unimodal
    enough for one DMP to represent. A leaf with large dispersion should be
    split further, not compressed.
    """
    out = {}
    for leaf in np.unique(labels):
        bundle = trajectories[labels == leaf]
        out[int(leaf)] = float(
            np.sqrt(((bundle - bundle.mean(axis=0)) ** 2).sum(-1).mean())
        )
    return out
