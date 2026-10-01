"""Open-loop, agent-level evaluation of an induced Behavior Tree bundle.

    python -m src.evaluate_bt_openloop data=eth bundle=outputs/induce_flow_r4_eth/.../bundle.pkl

Scores the tree on the same held-out windows ``src/evaluate_agent.py`` scores
the teacher on (every window of the test split, 8 observed -> 10 future frames
at 5 fps), so its ADE/FDE sit next to the teacher's argmax-latent ADE/FDE and
minADE_20 on equal terms. No CBF, no neighbours reacting, no emitter: the error
is the tree's and its primitives' alone.

Every window is predicted four ways, and the differences between them are the
point -- each isolates one stage of the distillation:

* ``oracle_proto``  -- best of the 8 leaf prototypes (the mean trajectory each
  leaf's DMP is fitted to), rotated onto the window's navmesh direction.
  minADE_8 / minFDE_8 of the leaf set: what is lost to **quantising** the future
  distribution onto 8 fixed shapes, with perfect routing.
* ``routed_proto``  -- the prototype of the leaf the tree actually selects.
  The gap to ``oracle_proto`` is what **routing** costs.
* ``routed_dmp``    -- that leaf's DMP, integrated from the agent's live position
  and velocity exactly as ``BTController`` does. The gap to ``routed_proto`` is
  what **executing** the leaf as a primitive costs.
* ``oracle_dmp``    -- best of the 8 DMP rollouts; minADE_8 of the executed set.

Plus a constant-velocity extrapolation, as the floor any learned predictor has
to clear.

Replaying the runtime, not an idealised DMP
-------------------------------------------
``routed_dmp`` mirrors ``BTController.acceleration``: the goal is fixed on entry
at ``position + R(nav) @ displacement``, the phase starts at 1 from the agent's
current velocity, and the primitive is re-entered (same leaf, fresh goal) when
it arrives within ``arrival_tolerance`` or its phase drops below
``phase_floor``. It integrates at ``integration_dt``, which defaults to the
runtime's actual tick, 1 / (5 fps x 4 substeps) = 50 ms.

Two things are necessarily different from closed loop, and both are held fixed
rather than guessed at: the leaf is selected once from the observed window (no
re-tick against predicted neighbours), and the navmesh direction is the one
observed at t=0 (the A* polyline is not available per window).

Routing accuracy on held-out data
---------------------------------
The induction report's ``tree_routing_fidelity`` is in-sample: the tree is
scored on the very states whose labels its guards were fitted to. Here each test
future is labelled with its nearest leaf prototype in the navmesh frame, and
``routing_accuracy`` is how often the tree picks that leaf. It is an
approximation of the dendrogram's label (which clusters in a position+velocity
embedding, not plain L2) and should be read as such.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import src._upstream  # noqa: F401
from src.util.paths import register_resolvers

register_resolvers()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data.dotdict_bridge import to_crowdes_config  # noqa: E402
from src.flow2bt.features import FEATURE_NAMES, build_features, extract_geometry  # noqa: E402
from src.flow2bt.primitives import integrate_step, rollout_dmp  # noqa: E402
from src.induce_flow2bt import load_dataset, nav_frame_rotation  # noqa: E402
from src.util.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)

KEYS = ("traj_hist", "traj_fut", "neighbor", "control", "goal", "attr", "environment")


def rollout_leaves(bank, leaf, velocity, nav_to_world, num_frames, frame_dt, cfg):
    """Integrate each agent's leaf as ``BTController`` would. Returns ``(N, T, 2)``.

    ``leaf`` (N,), ``velocity`` (N, 2) world, ``nav_to_world`` (N, 2, 2). The agent
    starts at the origin, which is where the dataset puts its last observation.
    """
    count = len(leaf)
    displacement = np.stack([d.displacement for d in bank.dmps])[leaf]       # (N, 2) nav
    displacement = np.einsum("nij,nj->ni", nav_to_world, displacement)        # world

    position = np.zeros((count, 2))
    velocity = velocity.astype(float).copy()
    entry = position.copy()
    goal = position + displacement
    phase_s = np.ones(count)
    tau = bank.tau[leaf]

    substeps = max(1, int(round(frame_dt / float(cfg.integration_dt))))
    tick = frame_dt / substeps
    out = np.empty((count, num_frames, 2))
    for frame in range(num_frames):
        for _ in range(substeps):
            reached = np.linalg.norm(goal - position, axis=1) < float(cfg.arrival_tolerance)
            expired = phase_s < float(cfg.phase_floor)
            restart = reached | expired
            phase_s = np.where(restart, 1.0, phase_s)
            entry = np.where(restart[:, None], position, entry)
            goal = np.where(restart[:, None], position + displacement, goal)

            acceleration = bank.acceleration(
                leaf, position, velocity, goal, entry, phase_s, tau,
                rotation=nav_to_world if cfg.rotate_forcing else None,
            )
            phase_s = bank.step_phase(phase_s, tick, tau)
            position, velocity = integrate_step(position, velocity, acceleration, tick)
        out[:, frame] = position
    return out


def errors(pred, truth):
    """Per-window (ADE, FDE) for ``(N, T, 2)`` arrays."""
    distance = np.linalg.norm(pred - truth, axis=-1)
    return distance.mean(axis=-1), distance[:, -1]


@hydra.main(version_base="1.3", config_path="configs", config_name="eval_bt_openloop")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(int(cfg.seed))
    crowdes_cfg = to_crowdes_config(cfg.data)
    simulator_cfg = crowdes_cfg["crowd_simulator"]["simulator"]
    fps = float(simulator_cfg["simulator_fps"])
    frame_dt = 1.0 / fps

    with open(cfg.bundle, "rb") as handle:
        bundle = pickle.load(handle)
    if "prototypes" not in bundle:
        raise ValueError(
            f"{cfg.bundle} predates prototype export; re-run src/induce_flow2bt.py"
        )
    if list(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("bundle was induced on a different feature set")
    tree, bank = bundle["tree"], bundle["bank"]
    leaves = sorted(bundle["prototypes"])
    prototypes = np.stack([bundle["prototypes"][k] for k in leaves])           # (L, T, 2) nav
    num_leaves = len(leaves)

    # Replay error of each DMP against the prototype it was fitted to, in metres.
    # The induction report's ``dmp_fit_residual`` is the forcing-term regression
    # residual, in forcing units scaled by (g - x0) -- not a distance.
    replay = {}
    for k in range(num_leaves):
        for label, step in (("runtime_dt", float(cfg.integration_dt)), ("10ms", 0.01)):
            path = rollout_dmp(bank.dmps[k], frame_dt, integration_dt=step)
            replay.setdefault(bank.names[k], {})[label] = float(
                np.sqrt(np.mean(np.sum((path - prototypes[k]) ** 2, axis=-1)))
            )

    if cfg.split == "test":
        logger.warning("opening the HELD-OUT TEST SPLIT (evaluation only)")
    dataset = load_dataset(crowdes_cfg, str(cfg.split))
    total = len(dataset) if cfg.limit is None else min(int(cfg.limit), len(dataset))

    sums: dict[str, float] = {}
    leaf_counts = np.zeros(num_leaves, dtype=int)
    confusion = np.zeros((num_leaves, num_leaves), dtype=int)   # [routed, nearest]
    clearance_flips = 0
    count = 0

    def add(name, ade, fde):
        sums[f"{name}/ade"] = sums.get(f"{name}/ade", 0.0) + float(ade.sum())
        sums[f"{name}/fde"] = sums.get(f"{name}/fde", 0.0) + float(fde.sum())

    for start in range(0, total, int(cfg.batch_size)):
        stop = min(start + int(cfg.batch_size), total)
        items = [dataset[i] for i in range(start, stop)]
        batch = {k: np.stack([item[k].numpy() for item in items]) for k in KEYS}
        truth = batch["traj_fut"]
        num_frames = truth.shape[1]

        geometry = extract_geometry(
            batch["traj_hist"], batch["neighbor"], batch["control"],
            environment=batch["environment"], fps=fps,
            pixel_meter=float(simulator_cfg["environment_pixel_meter"]),
        )
        features = build_features(geometry, batch["goal"], batch["attr"])
        routed = tree.tick(features).action
        routed = np.where(routed < 0, 0, routed)                  # totality guard, as runtime

        # The runtime's feature_fn passes sdf=None, so it ticks with clearance=0.
        runtime_features = features.copy()
        runtime_features[:, FEATURE_NAMES.index("clearance")] = 0.0
        runtime_routed = tree.tick(runtime_features).action
        runtime_routed = np.where(runtime_routed < 0, 0, runtime_routed)
        clearance_flips += int((runtime_routed != routed).sum())

        world_to_nav = nav_frame_rotation(batch["control"])
        nav_to_world = np.transpose(world_to_nav, (0, 2, 1))

        # Prototypes, one per leaf, in each window's world frame: (N, L, T, 2)
        proto_world = np.einsum("nij,ltj->nlti", nav_to_world, prototypes)
        proto_ade = np.linalg.norm(proto_world - truth[:, None], axis=-1).mean(-1)   # (N, L)
        proto_fde = np.linalg.norm(proto_world[:, :, -1] - truth[:, None, -1], axis=-1)
        rows = np.arange(len(items))
        add("routed_proto", proto_ade[rows, routed], proto_fde[rows, routed])
        add("oracle_proto", proto_ade.min(1), proto_fde.min(1))

        # Held-out routing accuracy against the nearest prototype (nav frame, L2).
        truth_nav = np.einsum("nij,ntj->nti", world_to_nav, truth)
        nearest = np.linalg.norm(
            truth_nav[:, None] - prototypes[None], axis=-1
        ).mean(-1).argmin(1)
        np.add.at(confusion, (routed, nearest), 1)
        np.add.at(leaf_counts, routed, 1)
        # Prototype of the *correct* leaf: routing perfect, quantisation only,
        # but by the same labelling rule -- the like-for-like reference for routed_proto.
        add("nearest_proto", proto_ade[rows, nearest], proto_fde[rows, nearest])

        # DMPs: routed leaf, then all leaves for the oracle.
        velocity = geometry.velocity
        pred = rollout_leaves(bank, routed, velocity, nav_to_world, num_frames, frame_dt, cfg)
        add("routed_dmp", *errors(pred, truth))

        every = np.repeat(np.arange(num_leaves)[None], len(items), axis=0).reshape(-1)
        all_pred = rollout_leaves(
            bank, every, np.repeat(velocity, num_leaves, axis=0),
            np.repeat(nav_to_world, num_leaves, axis=0), num_frames, frame_dt, cfg,
        ).reshape(len(items), num_leaves, num_frames, 2)
        dmp_ade = np.linalg.norm(all_pred - truth[:, None], axis=-1).mean(-1)
        dmp_fde = np.linalg.norm(all_pred[:, :, -1] - truth[:, None, -1], axis=-1)
        add("oracle_dmp", dmp_ade.min(1), dmp_fde.min(1))

        steps = np.arange(1, num_frames + 1)[None, :, None] * frame_dt
        add("constant_velocity", *errors(velocity[:, None, :] * steps, truth))

        count += len(items)
        logger.info("%d / %d windows", count, total)

    metrics = {key: value / count for key, value in sums.items()}
    routing_accuracy = float(np.trace(confusion) / confusion.sum())
    report = {
        "bundle": str(cfg.bundle),
        "split": str(cfg.split),
        "num_windows": count,
        "integration_dt": float(cfg.integration_dt),
        "rotate_forcing": bool(cfg.rotate_forcing),
        "metrics": metrics,
        "routing_accuracy_vs_nearest_prototype": routing_accuracy,
        "runtime_clearance_zero_routing_changes": clearance_flips / count,
        "leaf_names": [bank.names[k] for k in range(num_leaves)],
        "dmp_replay_rms_m": replay,
        "routed_leaf_share": (leaf_counts / count).tolist(),
        "nearest_leaf_share": (confusion.sum(0) / count).tolist(),
        "confusion_routed_by_nearest": confusion.tolist(),
    }

    out_dir = Path(cfg.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    OmegaConf.save(cfg, out_dir / "eval_config.yaml")

    print()
    for name in ("constant_velocity", "oracle_proto", "nearest_proto", "routed_proto",
                 "routed_dmp", "oracle_dmp"):
        print(f"{name:18s} ADE {metrics[name + '/ade']:.4f}  FDE {metrics[name + '/fde']:.4f}")
    print(f"routing accuracy vs nearest prototype: {routing_accuracy:.3f}")
    print(f"routing changes with runtime's clearance=0: {clearance_flips / count:.3%}")
    print("DMP replay RMS vs prototype (m):", {k: round(v["runtime_dt"], 3) for k, v in replay.items()})
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
