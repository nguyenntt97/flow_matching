"""Subsystem 6: the reactive Behavior Tree, evaluated for the whole crowd at once.

Design doc: ``survey/design/06_reactive_bt_assembly_execution.md``.

Two evaluation modes over one tree definition:

* **hard** (``tick``) -- the runtime path. Returns a status and the index of the
  action leaf each agent is executing.
* **soft** (``tick_soft``) -- the differentiable relaxation of design 06 sec 1.4,
  used to calibrate condition thresholds by gradient rather than by hand.

Why this is arrays and not objects
----------------------------------
A conventional BT library walks per-instance node objects per tick. At the tick
rates this design argues for, times a crowd, that is thousands of Python object
traversals per simulated second. Here a node evaluates every agent in one
vectorised call, so tree depth costs numpy ops rather than interpreter frames.

The cost is that nothing can literally short-circuit: different agents take
different branches, so every child is evaluated and combined with a per-agent
decided-mask. Conditions are dot products, so that is cheap; action leaves are
*selected*, not executed, and only the selected primitive is stepped.

A note on the soft formulas
---------------------------
Design 06 sec 1.4 gives ``p_run = 1 - p_succ - p_fail``, which only defines a
distribution if that is non-negative. For a Sequence it expands to
``prod(s_i) <= prod(s_i + r_i)``, which holds termwise, and the Fallback case is
symmetric -- so the formulas are consistent as written. ``test_bt.py`` pins it
rather than leaving it assumed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence as SequenceType

import numpy as np

#: Status codes. Kept as small ints so a status vector is a plain int array.
FAILURE = 0
SUCCESS = 1
RUNNING = 2

STATUS_NAMES = {FAILURE: "FAILURE", SUCCESS: "SUCCESS", RUNNING: "RUNNING"}

#: Sentinel in the ``action`` vector meaning "no action leaf is active".
NO_ACTION = -1


@dataclass
class TickResult:
    status: np.ndarray          # (N,) int
    action: np.ndarray          # (N,) int, NO_ACTION where nothing is selected

    def __len__(self) -> int:
        return int(self.status.shape[0])


class Node(ABC):
    """A tree node. ``name`` is what shows up in traces and the SMV export."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def tick(self, features: np.ndarray) -> TickResult:
        """Hard evaluation. ``features`` is (N, d)."""

    @abstractmethod
    def tick_soft(self, features, beta: float = 1.0):
        """Soft evaluation. Returns a torch tensor (N, 3) = [succ, fail, run]."""

    @abstractmethod
    def describe(self, feature_names: SequenceType[str]) -> str:
        """One human-auditable line. This is the interpretability claim, so it
        has to render in physical units, not weights."""

    def walk(self):
        yield self


class Condition(Node):
    r"""``w . phi(s) + b >= 0``  ->  SUCCESS, else FAILURE. Never RUNNING.

    This is the guard the design induces by SVM at each bifurcation
    (design 04). It is a single dot product, which is what makes the tick cheap
    and what makes the node translatable to a linear constraint for the SMT
    export.
    """

    def __init__(self, name: str, weights: np.ndarray, bias: float):
        super().__init__(name)
        self.weights = np.asarray(weights, dtype=float).ravel()
        self.bias = float(bias)

    def margin(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias

    def tick(self, features: np.ndarray) -> TickResult:
        status = np.where(self.margin(features) >= 0.0, SUCCESS, FAILURE)
        return TickResult(status, np.full(features.shape[0], NO_ACTION))

    def tick_soft(self, features, beta: float = 1.0):
        import torch

        weights = torch.as_tensor(self.weights, dtype=features.dtype, device=features.device)
        probability = torch.sigmoid(beta * (features @ weights + self.bias))
        return torch.stack(
            [probability, 1.0 - probability, torch.zeros_like(probability)], dim=-1
        )

    def describe(self, feature_names: SequenceType[str]) -> str:
        terms = [
            f"{weight:+.4g}*{feature_names[index]}"
            for index, weight in enumerate(self.weights)
            if abs(weight) > 1e-9
        ]
        return f"{self.name}: {' '.join(terms) or '0'} {self.bias:+.4g} >= 0"


class Action(Node):
    """Selects a DMP leaf. Returns RUNNING unless a terminator says otherwise.

    The primitive is not stepped here -- the tree decides *which* leaf is active
    and the runtime steps the bank once, for every agent, afterwards.
    """

    def __init__(self, name: str, leaf_index: int, terminator=None):
        super().__init__(name)
        self.leaf_index = int(leaf_index)
        self.terminator = terminator

    def tick(self, features: np.ndarray) -> TickResult:
        num_agents = features.shape[0]
        if self.terminator is None:
            status = np.full(num_agents, RUNNING)
        else:
            status = np.asarray(self.terminator(features), dtype=int)
        action = np.where(status == RUNNING, self.leaf_index, NO_ACTION)
        return TickResult(status, action)

    def tick_soft(self, features, beta: float = 1.0):
        import torch

        out = torch.zeros(features.shape[0], 3, dtype=features.dtype, device=features.device)
        out[:, 2] = 1.0        # RUNNING
        return out

    def describe(self, feature_names: SequenceType[str]) -> str:
        return f"{self.name}: run primitive #{self.leaf_index}"


class _Composite(Node):
    def __init__(self, name: str, children: SequenceType[Node]):
        super().__init__(name)
        if not children:
            raise ValueError(f"composite {name!r} needs at least one child")
        self.children = list(children)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def describe(self, feature_names: SequenceType[str]) -> str:
        return f"{self.name} ({type(self).__name__.lower()}, {len(self.children)} children)"

    def _tick_children(self, features, halt_on: int) -> TickResult:
        """Left-to-right with a per-agent decided mask.

        ``halt_on`` is the status that stops the scan (FAILURE for a Sequence,
        SUCCESS for a Fallback). RUNNING always stops it. Whichever child
        settles an agent also supplies that agent's active action.
        """
        num_agents = features.shape[0]
        decided = np.zeros(num_agents, dtype=bool)
        status = np.full(num_agents, SUCCESS if halt_on == FAILURE else FAILURE)
        action = np.full(num_agents, NO_ACTION)

        for child in self.children:
            if decided.all():
                break
            result = child.tick(features)
            settles = (~decided) & ((result.status == halt_on) | (result.status == RUNNING))
            status = np.where(settles, result.status, status)
            action = np.where(settles, result.action, action)
            decided = decided | settles

        return TickResult(status, action)


class Sequence(_Composite):
    """``->``: all children must succeed. FAILURE or RUNNING short-circuits."""

    def tick(self, features: np.ndarray) -> TickResult:
        return self._tick_children(features, halt_on=FAILURE)

    def tick_soft(self, features, beta: float = 1.0):
        import torch

        children = [child.tick_soft(features, beta) for child in self.children]
        p_success = torch.stack([c[:, 0] for c in children], -1).prod(-1)
        p_failure = 1.0 - torch.stack([1.0 - c[:, 1] for c in children], -1).prod(-1)
        return torch.stack([p_success, p_failure, 1.0 - p_success - p_failure], dim=-1)


class Fallback(_Composite):
    """``?``: first child to succeed wins. This is the preemption mechanism."""

    def tick(self, features: np.ndarray) -> TickResult:
        return self._tick_children(features, halt_on=SUCCESS)

    def tick_soft(self, features, beta: float = 1.0):
        import torch

        children = [child.tick_soft(features, beta) for child in self.children]
        p_failure = torch.stack([c[:, 1] for c in children], -1).prod(-1)
        p_success = 1.0 - torch.stack([1.0 - c[:, 0] for c in children], -1).prod(-1)
        return torch.stack([p_success, p_failure, 1.0 - p_success - p_failure], dim=-1)


def render(node: Node, feature_names: SequenceType[str], indent: int = 0) -> str:
    """ASCII rendering of the whole tree -- the auditability claim, made concrete."""
    lines = ["  " * indent + node.describe(feature_names)]
    for child in getattr(node, "children", []):
        lines.append(render(child, feature_names, indent + 1))
    return "\n".join(lines)


def action_leaves(root: Node) -> list[Action]:
    return [node for node in root.walk() if isinstance(node, Action)]


def condition_nodes(root: Node) -> list[Condition]:
    return [node for node in root.walk() if isinstance(node, Condition)]
