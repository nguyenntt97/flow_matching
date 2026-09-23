"""The offline compilation pipeline: Subsystems 2 -> 6, end to end.

    python -m src.induce_flow2bt data=eth source=ground_truth
    python -m src.induce_flow2bt data=eth source=flow ckpt=outputs/flow_eth/.../last.ckpt

Produces the bundle ``src/evaluate_flow2bt.py`` loads with
``runtime.controller=bt``: an assembled Behavior Tree, its guard hyperplanes in
physical units, and one DMP per leaf.

Everything is induced in the navmesh frame
------------------------------------------
Futures are rotated so that each agent's navmesh waypoint lies along +x before
clustering and before the DMPs are fitted. This is the frame CrowdES already
clusters its own B=8 endpoints in (``simulator_model.py:200-220``), and it is
what makes a leaf mean "swerve left *relative to where I am being guided*"
rather than "move toward world -y".

Skipping it is not a cosmetic error. On a first pass without rotation, three of
the eight leaves came out labelled ``backstep`` -- they were simply the agents
walking in the -x world direction -- and the fitted primitives then dragged
agents travelling the other way backwards.

``source`` is the ablation that matters
---------------------------------------
Design 03 claims the *flow teacher* is what makes the bifurcation structure
discoverable. That is only a claim until the alternatives are run:

* ``ground_truth`` -- induce from the recorded futures in the dataset. No
  teacher at all. If this produces an equally good tree, the teacher earned
  nothing and the honest report says so.
* ``parity``       -- induce from CrowdES's own sampled futures.
* ``flow``         -- induce from the flow teacher's ensemble, which is the
  design's actual proposal and the only source that can draw *many* samples per
  conditioning state (ground truth has exactly one).

That last difference is the real argument for a teacher, and it is structural
rather than a matter of quality: a bifurcation is a statement about a
distribution over futures from one state, and a dataset contains one future per
state. ``rollouts_per_state`` is what makes ``flow`` and ``parity`` able to
answer a question ``ground_truth`` cannot.
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
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data.dotdict_bridge import to_crowdes_config  # noqa: E402
from src.data.simulator_dataset import ParitySimulatorDataset  # noqa: E402
from src.flow2bt.assembly import assemble, name_leaves_by_geometry, tree_report  # noqa: E402
from src.flow2bt.clustering import (  # noqa: E402
    agreement_with_labels,
    induce,
    leaf_dispersion,
    leaf_prototypes,
    suggest_num_leaves,
)
from src.flow2bt.conditions import guard_report, induce_guards  # noqa: E402
from src.flow2bt.features import FEATURE_NAMES, build_features, extract_geometry  # noqa: E402
from src.flow2bt.primitives import DMPBank, fit_dmp  # noqa: E402
from src.util.fingerprint import build_fingerprint  # noqa: E402
from src.util.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def load_dataset(crowdes_cfg, split: str):
    simulator_cfg = crowdes_cfg["crowd_simulator"]["simulator"]
    cache = (
        Path(crowdes_cfg["dataset"]["dataset_preprocessed_path"])
        / "cache"
        / f"{split}_simulator.{build_fingerprint(simulator_cfg)}.pickle"
    )
    if not cache.is_file():
        raise FileNotFoundError(
            f"no dataset cache at {cache}. Build it with:\n"
            f"  python -m src.train data={crowdes_cfg['dataset']['dataset_name']} "
            "build_cache_only=true"
        )
    return ParitySimulatorDataset(
        crowdes_cfg, split, allow_test_split=(split == "test"), cache_path=str(cache)
    )


def nav_frame_rotation(control: np.ndarray) -> np.ndarray:
    """``(N, 2, 2)`` rotations taking the navmesh direction onto +x.

    Degenerate (near-zero) control vectors fall back to the identity, matching
    upstream's convention in ``normalize_rotation_scale`` of treating a missing
    direction as +x.
    """
    norm = np.linalg.norm(control, axis=1, keepdims=True)
    unit = np.where(norm > 1e-4, control / np.maximum(norm, 1e-12), np.array([1.0, 0.0]))
    cos, sin = unit[:, 0], unit[:, 1]
    # Inverse rotation: [[cos, sin], [-sin, cos]]
    return np.stack([
        np.stack([cos, sin], axis=1),
        np.stack([-sin, cos], axis=1),
    ], axis=1)


def to_nav_frame(trajectories: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate ``(N, T, 2)`` futures into each agent's navmesh frame."""
    return np.einsum("nij,ntj->nti", rotation, trajectories)


def gather_batch(dataset, indices):
    keys = ("traj_hist", "traj_fut", "neighbor", "control", "goal", "attr", "environment")
    return {
        key: torch.stack([dataset[int(i)][key] for i in indices]) for key in keys
    }


@torch.no_grad()
def sample_futures(model, batch, rollouts: int, device) -> np.ndarray:
    """``(N, rollouts, T, 2)`` sampled futures from a simulator-protocol model."""
    moved = {key: value.to(device) for key, value in batch.items()}
    out = []
    for _ in range(rollouts):
        result = model(
            moved["traj_hist"], None, moved["goal"], moved["attr"],
            moved["control"].clone(), moved["neighbor"], moved["environment"],
            sampling=True,
        )
        out.append(result.preds.cpu().numpy())
    return np.stack(out, axis=1)


def load_teacher(source: str, checkpoint: str, device):
    if source == "flow":
        from src.systems.simulator_system import SimulatorLitModule

        try:
            system = SimulatorLitModule.load_from_checkpoint(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            system = SimulatorLitModule.load_from_checkpoint(checkpoint, map_location=device)
        return system.eval().to(device).model
    if source == "parity":
        from src.evaluate_agent import load_model

        return load_model(checkpoint, device)
    raise ValueError(f"source {source!r} needs no teacher")


@hydra.main(version_base="1.3", config_path="configs", config_name="induce")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(int(cfg.seed))
    crowdes_cfg = to_crowdes_config(cfg.data)

    dataset = load_dataset(crowdes_cfg, str(cfg.split))
    rng = np.random.default_rng(int(cfg.seed))
    indices = rng.choice(len(dataset), size=min(int(cfg.num_states), len(dataset)), replace=False)
    batch = gather_batch(dataset, indices)
    logger.info("induction set: %d conditioning states from %d", len(indices), len(dataset))

    simulator_cfg = crowdes_cfg["crowd_simulator"]["simulator"]
    fps = float(simulator_cfg["simulator_fps"])
    dt = 1.0 / fps

    rotation = nav_frame_rotation(batch["control"].numpy())

    # ---- Subsystem 2: the trajectory ensemble
    if cfg.source == "ground_truth":
        # One future per state -- so every state contributes a single member and
        # the dendrogram is over states, not over a distribution per state.
        bundle = batch["traj_fut"].numpy()
        rollouts = 1
        if cfg.nav_frame:
            bundle = to_nav_frame(bundle, rotation)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        teacher = load_teacher(str(cfg.source), str(cfg.ckpt), device)
        rollouts = int(cfg.rollouts_per_state)
        sampled = sample_futures(teacher, batch, rollouts, device)
        bundle = sampled.reshape(-1, sampled.shape[-2], 2)
        if cfg.nav_frame:
            bundle = to_nav_frame(bundle, np.repeat(rotation, rollouts, axis=0))
    logger.info("ensemble: %s (rollouts per state: %d)", bundle.shape, rollouts)

    # ---- Subsystem 4 (features), repeated to line up with the ensemble
    geometry = extract_geometry(
        batch["traj_hist"].numpy(), batch["neighbor"].numpy(), batch["control"].numpy(),
        environment=batch["environment"].numpy(), fps=fps,
        pixel_meter=float(simulator_cfg["environment_pixel_meter"]),
    )
    features = build_features(geometry, batch["goal"].numpy(), batch["attr"].numpy())
    features = np.repeat(features, rollouts, axis=0)

    # ---- Subsystem 3: bifurcations
    dendrogram = induce(
        bundle, dt,
        num_leaves=int(cfg.num_leaves),
        lambda_term=float(cfg.lambda_term),
        lambda_vel=float(cfg.lambda_vel),
        min_branch=int(cfg.min_branch),
    )
    elbow = suggest_num_leaves(dendrogram.linkage_matrix)
    logger.info(
        "dendrogram: %d leaves (elbow suggests %d), %d bifurcations",
        dendrogram.num_leaves, elbow, len(dendrogram.bifurcations),
    )

    # Does the induced hierarchy recover CrowdES's own B=8 behaviour modes?
    crowdes_labels = None
    if cfg.compare_to_crowdes_modes:
        from src.models.crowdes_parity import ParityCrowdESSimulator

        wrapper = ParityCrowdESSimulator(hidden_dim=64, latent_dim=int(simulator_cfg["latent_dim"]))
        endpoints = torch.tensor(bundle[:, -1, :], dtype=torch.float32)
        controls = torch.repeat_interleave(batch["control"], rollouts, dim=0).float()
        wrapper.fit_endpoint_clusters(endpoints, controls)
        normalised, *_ = wrapper.net.normalize_rotation_scale(endpoints, controls.clone())
        distance = normalised.unsqueeze(1) - wrapper.endpoint_cluster_centers
        crowdes_labels = torch.argmin(torch.linalg.norm(distance, dim=2), dim=1).numpy()

    agreement = (
        agreement_with_labels(dendrogram.labels, crowdes_labels)
        if crowdes_labels is not None else None
    )
    if agreement:
        logger.info("agreement with CrowdES B=8 KMeans modes: %s", agreement)

    # ---- Subsystem 5: one DMP per leaf
    prototypes = leaf_prototypes(bundle, dendrogram.labels)
    dispersion = leaf_dispersion(bundle, dendrogram.labels)
    names = name_leaves_by_geometry(prototypes)
    primitives = [
        fit_dmp(
            prototypes[leaf], dt,
            num_basis=int(cfg.dmp_basis), alpha_s=float(cfg.dmp_alpha_s),
            K=float(cfg.dmp_stiffness), name=names[leaf],
        )
        for leaf in sorted(prototypes)
    ]
    bank = DMPBank(primitives)

    # ---- Subsystem 4: guards, and Subsystem 6: assembly
    guards = induce_guards(
        features, dendrogram.bifurcations, FEATURE_NAMES,
        C=float(cfg.svm_C), penalty=str(cfg.svm_penalty), seed=int(cfg.seed),
    )
    tree = assemble(dendrogram, guards, FEATURE_NAMES, leaf_names=names)

    routed = tree.tick(features).action
    fidelity = float((routed == dendrogram.labels).mean())
    logger.info("tree reproduces %.1f%% of the induced leaf assignment", 100 * fidelity)

    out_dir = Path(cfg.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bundle.pkl", "wb") as handle:
        pickle.dump({"tree": tree, "bank": bank, "feature_names": FEATURE_NAMES}, handle)

    report = {
        "source": str(cfg.source),
        "nav_frame": bool(cfg.nav_frame),
        "dataset": crowdes_cfg["dataset"]["dataset_name"],
        "num_states": int(len(indices)),
        "rollouts_per_state": rollouts,
        "ensemble": list(bundle.shape),
        "num_leaves": dendrogram.num_leaves,
        "elbow_suggests": int(elbow),
        "leaf_names": {str(k): v for k, v in names.items()},
        "leaf_sizes": {str(k): int(len(v)) for k, v in dendrogram.leaf_members.items()},
        "leaf_dispersion_m": {str(k): v for k, v in dispersion.items()},
        "dmp_fit_residual": {p.name: p.fit_residual for p in primitives},
        "tree_routing_fidelity": fidelity,
        "agreement_with_crowdes_modes": agreement,
        "guards": [
            {
                "name": g.name, "accuracy": g.accuracy, "train_accuracy": g.train_accuracy,
                "balance": g.balance, "samples": g.num_samples, "weak": g.is_weak,
                "weights": {n: float(w) for n, w in zip(FEATURE_NAMES, g.weights)},
                "bias": g.bias,
            }
            for g in guards
        ],
    }
    (out_dir / "induction.json").write_text(json.dumps(report, indent=2))
    (out_dir / "tree.txt").write_text(tree_report(tree))
    OmegaConf.save(cfg, out_dir / "induce_config.yaml")

    print()
    print(tree_report(tree))
    print()
    print(f"leaf dispersion (m): { {k: round(v, 3) for k, v in dispersion.items()} }")
    if agreement:
        print(f"agreement with CrowdES B=8 modes: ARI={agreement['adjusted_rand']:.3f}")
    print(f"tree routing fidelity: {fidelity:.3f}")
    print()
    print(f"wrote {out_dir / 'bundle.pkl'}")


if __name__ == "__main__":
    main()
