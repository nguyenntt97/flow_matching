"""Subsystem 7b: SMV export and bounded falsification."""

import numpy as np
import pytest

from src.flow2bt.assembly import assemble
from src.flow2bt.cbf import CBFConfig
from src.flow2bt.clustering import induce
from src.flow2bt.conditions import induce_guards
from src.flow2bt.falsify import corridor_scenario, falsify, head_on_scenario, report, run_episode
from src.flow2bt.verification import to_smv
from src.runtime.controllers import WaypointController

NAMES = ["ttc", "d_lat", "speed"]


@pytest.fixture(scope="module")
def tree():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 11)
    bundles, starts = [], []
    for lateral, ttc in ((0.0, 8.0), (1.0, 1.5), (-1.0, 1.4)):
        base = np.stack([2.6 * t, lateral * np.sin(np.pi * t)], axis=1)
        bundles.append(base[None, ...] + rng.normal(0, 0.02, (60, 11, 2)))
        starts.append(np.stack([
            rng.normal(ttc, 0.2, 60), rng.normal(-lateral, 0.2, 60), rng.normal(1.3, 0.05, 60),
        ], axis=1))
    trajectories, features = np.concatenate(bundles), np.concatenate(starts)
    dendrogram = induce(trajectories, 0.2, num_leaves=3)
    return assemble(dendrogram, induce_guards(features, dendrogram.bifurcations, NAMES), NAMES)


# ------------------------------------------------------------------- SMV
def test_smv_declares_every_guard_and_leaf(tree):
    text = to_smv(tree)
    assert "MODULE main" in text
    for guard in tree.guards:
        assert guard.name.replace("-", "_") in text
    assert "leaf : -1..2;" in text


def test_smv_states_its_own_scope(tree):
    """The overclaim this file exists to prevent must be written into the file."""
    text = to_smv(tree)
    assert "DECISION LOGIC only" in text
    assert "NOT established by checking this file" in text


def test_smv_only_asserts_properties_about_the_tree(tree):
    """No closed-loop LTLSPEC: the model has no dynamics to support one."""
    text = to_smv(tree)
    specs = [line for line in text.splitlines() if line.startswith("LTLSPEC")]
    assert specs == ["LTLSPEC G (leaf >= 0)"]
    assert "distance" not in " ".join(specs)


def test_smv_records_guards_in_physical_units(tree):
    text = to_smv(tree)
    assert "*speed" in text and "*ttc" in text
    assert "cross-validated accuracy" in text


# ---------------------------------------------------------- falsification
def test_episode_runs_and_agents_arrive():
    start, goal, speed = head_on_scenario(np.random.default_rng(0), 10.0, 0.0, 1.3)
    episode = run_episode(start, goal, speed, WaypointController(), None, horizon=20.0)
    assert episode.reached.all(), "agents did not reach their goals without a filter"
    assert np.isfinite(episode.positions).all()


def test_filter_holds_separation_regardless_of_symmetry():
    """The barrier converges to exactly d_min -- measured 0.4499 at every offset."""
    from src.flow2bt.cbf import CBFFilter

    config = CBFConfig(d_min=0.45)
    rng = np.random.default_rng(0)
    for offset in (0.0, 0.01, 0.2, 0.5):
        start, goal, speed = head_on_scenario(rng, 10.0, offset, 1.3)
        episode = run_episode(
            start, goal, speed, WaypointController(), CBFFilter(config), horizon=30.0
        )
        assert episode.min_separation >= config.d_min - 1e-3, (
            f"offset {offset}: separation {episode.min_separation:.4f}"
        )
        assert "G !collision" not in episode.violations(config.d_min)


def test_perfectly_symmetric_head_on_deadlocks():
    """The counterexample design 07 sec 1.2 predicts, found by bounded search.

    With zero lateral offset the pairwise constraint's normal ``-2*dp`` lies
    along x, so the filter can brake but cannot break the tie sideways. Both
    agents stop nose to nose exactly on the barrier and neither ever arrives.
    Design 07 sec 1.2.3's remedy -- an asymmetric tie-breaking guard -- is the
    right one, and this is the test that would show it working.
    """
    from src.flow2bt.cbf import CBFFilter

    config = CBFConfig(d_min=0.45)
    rng = np.random.default_rng(0)

    start, goal, speed = head_on_scenario(rng, 10.0, 0.0, 1.3)
    symmetric = run_episode(
        start, goal, speed, WaypointController(), CBFFilter(config), horizon=30.0
    )
    assert symmetric.violations(config.d_min) == ["F at_goal", "G F moving"]
    assert not symmetric.reached.any()
    assert symmetric.stalled.all()
    final = np.linalg.norm(symmetric.positions[-1, 0] - symmetric.positions[-1, 1])
    assert final == pytest.approx(config.d_min, abs=1e-3), "they should stall on the barrier"

    # One centimetre of asymmetry is enough to resolve it.
    start, goal, speed = head_on_scenario(rng, 10.0, 0.01, 1.3)
    nudged = run_episode(
        start, goal, speed, WaypointController(), CBFFilter(config), horizon=30.0
    )
    assert nudged.reached.all()
    assert nudged.violations(config.d_min) == []


def test_violations_name_the_properties():
    from src.flow2bt.falsify import Episode

    episode = Episode(
        positions=np.zeros((2, 2, 2)), min_separation=0.1,
        reached=np.array([True, False]), stalled=np.array([False, True]), relaxed_ticks=0,
    )
    assert set(episode.violations(0.45)) == {"G !collision", "F at_goal", "G F moving"}


def test_clean_episode_reports_no_violations():
    from src.flow2bt.falsify import Episode

    episode = Episode(
        positions=np.zeros((2, 2, 2)), min_separation=1.0,
        reached=np.array([True, True]), stalled=np.array([False, False]), relaxed_ticks=0,
    )
    assert episode.violations(0.45) == []


def test_falsify_searches_and_reports():
    summary = falsify(
        lambda: WaypointController(), CBFConfig(d_min=0.2),
        episodes=6, seed=0, scenario="head_on", horizon=15.0,
    )
    assert summary["episodes"] == 6
    assert np.isfinite(summary["worst_separation"])
    text = report(summary)
    assert "bounded falsification" in text
    # The honesty caveat must survive: absence of a counterexample is not proof.
    if not summary["violations"]:
        assert "NOT a proof" in text


def test_corridor_scenario_splits_the_flow():
    start, goal, speed = corridor_scenario(np.random.default_rng(0), 6, 3.0, 1.3)
    half = len(start) // 2
    assert len(start) == len(goal) == len(speed)
    assert (start[:half, 0] < 0).all() and (start[half:, 0] > 0).all()
    assert (goal[:half, 0] > 0).all() and (goal[half:, 0] < 0).all()


def test_corridor_scenario_does_not_spawn_agents_on_top_of_each_other():
    """A falsifier must not blame the controller for its own sampling.

    Sampling y uniformly started 128 of 200 episodes inside d_min; capping the
    count by what the corridor physically holds brings that to 4 of 300, with
    the worst spawn gap 0.40 m rather than 0.003 m.
    """
    rng = np.random.default_rng(0)
    worst = np.inf
    for _ in range(300):
        start, _, _ = corridor_scenario(rng, int(rng.integers(2, 9)), rng.uniform(1.5, 4.0), 1.3)
        distance = np.linalg.norm(start[:, None] - start[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        worst = min(worst, float(distance.min()))
    assert worst > 0.35, f"worst spawn separation {worst:.4f} m"


def test_corridor_density_is_capped_by_width():
    """Four abreast at 0.6 m needs 2.4 m; asking for it in 1.5 m is infeasible."""
    narrow, _, _ = corridor_scenario(np.random.default_rng(0), 8, 1.5, 1.3)
    wide, _, _ = corridor_scenario(np.random.default_rng(0), 8, 4.0, 1.3)
    assert len(narrow) < 8
    assert len(wide) >= len(narrow)


def test_no_collision_counterexamples_at_the_benchmark_threshold():
    """200 randomised episodes at d_min = 0.2 m, the threshold metrics.py uses.

    Bounded search, so this is evidence and not a proof -- but a regression that
    breaks forward invariance should trip it.
    """
    for scenario in ("head_on", "corridor"):
        summary = falsify(
            lambda: WaypointController(), CBFConfig(d_min=0.2),
            episodes=25, seed=2, scenario=scenario, horizon=25.0,
        )
        assert "G !collision" not in summary["violations"], summary["violations"]
        assert summary["mean_arrival_rate"] > 0.95


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        falsify(lambda: WaypointController(), episodes=1, scenario="nope")
