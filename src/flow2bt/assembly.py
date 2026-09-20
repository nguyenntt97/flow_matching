"""Subsystem 6 (assembly): dendrogram + guards + primitives -> a runnable tree.

Design doc: ``survey/design/06_reactive_bt_assembly_execution.md`` sec 1.3.

The design's Guarded Fallback is::

    T = Fallback( Sequence(C_1, A_1), Sequence(C_2, A_2), ..., A_default )

A binary dendrogram maps onto that directly. Each bifurcation becomes::

    Fallback( Sequence( guard_k, <left subtree> ),
              <right subtree> )

"if the guard holds, take the left branch; otherwise fall back to the right."
The right branch needs no guard of its own -- it is the fallback -- which is
what makes the tree total: every agent reaches some action leaf on every tick,
so there is no tick on which the crowd has no command. ``test_assembly.py``
checks that on random feature matrices rather than leaving it to the argument.

Preserving the dendrogram's topology matters
--------------------------------------------
The alternative -- flattening to one Fallback over all eight modes -- would also
run, and would lose the thing Subsystem 3 was for. In the nested form a guard
near the root separates coarse strategies and only the agents that took that
branch are ever tested against the finer guards below it. That is also what
makes the weak-guard report meaningful: a weak guard deep in the tree costs
little, a weak guard at the root invalidates everything under it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import to_tree

from src.flow2bt.bt import Action, Condition, Fallback, Node, Sequence as SequenceNode
from src.flow2bt.clustering import Dendrogram
from src.flow2bt.conditions import InducedGuard


@dataclass
class AssembledTree:
    root: Node
    feature_names: list[str]
    leaf_names: dict[int, str]
    guards: list[InducedGuard]

    def tick(self, features: np.ndarray):
        return self.root.tick(features)

    def render(self) -> str:
        from src.flow2bt.bt import render

        return render(self.root, self.feature_names)


def assemble(
    dendrogram: Dendrogram,
    guards: Sequence[InducedGuard],
    feature_names: Sequence[str],
    leaf_names: Optional[dict[int, str]] = None,
) -> AssembledTree:
    """Build the Guarded Fallback tree mirroring ``dendrogram``'s topology.

    ``guards`` must be the ones ``induce_guards`` produced from the same
    dendrogram -- they are matched by ``node_id``, not by position, so a
    reordering cannot silently attach the wrong guard to a split.
    """
    by_node = {guard.node_id: guard for guard in guards}
    missing = dendrogram.internal_ids - set(by_node)
    if missing:
        raise ValueError(
            f"no guard for internal node(s) {sorted(missing)}; "
            "induce_guards must be run on this dendrogram's bifurcations"
        )

    _, nodes = to_tree(dendrogram.linkage_matrix, rd=True)
    node_by_id = {node.id: node for node in nodes}
    names = dict(leaf_names or {})

    def build(node_id: int) -> Node:
        if node_id not in dendrogram.internal_ids:
            leaf = dendrogram.node_to_leaf[node_id]
            return Action(names.get(leaf, f"leaf_{leaf}"), leaf_index=leaf)

        node = node_by_id[node_id]
        guard = by_node[node_id]
        return Fallback(
            f"split_{node_id}",
            [
                SequenceNode(
                    f"take_left_{node_id}",
                    [Condition(guard.name, guard.weights, guard.bias), build(node.left.id)],
                ),
                build(node.right.id),
            ],
        )

    return AssembledTree(
        root=build(dendrogram.root_id),
        feature_names=list(feature_names),
        leaf_names=names,
        guards=list(guards),
    )


def name_leaves_by_geometry(
    prototypes: dict[int, np.ndarray],
    stop_speed: float = 0.25,
    swerve_lateral: float = 0.25,
) -> dict[int, str]:
    """Label each leaf from what its prototype actually does.

    Design 05 sec 1.2 *names* eight modes (march, swerve left/right, yield,
    stop, overtake, group, backstep). Those names are a hypothesis about what
    clustering will find, not a result. This assigns names from measured
    geometry -- net displacement, lateral excursion, mean speed -- so the report
    says what the leaves are rather than what they were expected to be. The
    vocabulary is deliberately smaller than the design's: distinctions like
    "overtake" versus "march" are not recoverable from a single agent's own
    path.
    """
    names = {}
    for leaf, prototype in prototypes.items():
        displacement = prototype[-1] - prototype[0]
        duration = max(len(prototype) - 1, 1)
        forward = float(displacement[0])
        lateral = float(displacement[1])
        speed = float(np.linalg.norm(np.diff(prototype, axis=0), axis=1).mean() * duration)

        if speed < stop_speed:
            label = "stop"
        elif forward < 0:
            label = "backstep"
        elif abs(lateral) < swerve_lateral:
            label = "march"
        else:
            label = "swerve_left" if lateral > 0 else "swerve_right"
        names[leaf] = f"{label}_{leaf}"
    return names


def tree_report(tree: AssembledTree) -> str:
    from src.flow2bt.bt import action_leaves, condition_nodes
    from src.flow2bt.conditions import guard_report

    return "\n".join([
        tree.render(),
        "",
        f"{len(condition_nodes(tree.root))} conditions, "
        f"{len(action_leaves(tree.root))} action leaves",
        "",
        guard_report(tree.guards),
    ])
