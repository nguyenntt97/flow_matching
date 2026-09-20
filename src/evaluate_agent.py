"""Agent-level evaluation of a trained simulator on the held-out test split.

    python -m src.evaluate_agent ckpt=outputs/.../last.ckpt data=eth

Two metric families, and they are not the same kind of claim:

* ``ade`` / ``fde`` with ``sampling=False`` is the argmax-latent number and is
  the same arithmetic upstream's *validation* loop computes -- but upstream
  computes it on the train split, so this is a comparable statistic on a
  different split, not a reproduction of an upstream number.
* ``minADE_K`` / ``minFDE_K`` with ``sampling=True`` is the usual
  trajectory-prediction convention. Upstream never computes it. There is no
  upstream number to match; this is our own baseline.

This is the only place ``allow_test_split=True`` is set.

``ckpt`` may be either a Lightning ``.ckpt`` or a directory holding an HF export
(``config.json`` + ``pytorch_model.bin``). The directory form is what upstream's
released model zoo ships, so it is how the *published* CrowdES simulator gets
scored on exactly the same code path as ours -- no retraining needed to have a
baseline, and no separate scoring script to drift.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import src._upstream  # noqa: F401
from src.util.paths import register_resolvers

register_resolvers()

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.data.dotdict_bridge import to_crowdes_config  # noqa: E402
from src.data.simulator_dataset import ParitySimulatorDataset  # noqa: E402
from src.systems.simulator_system import SimulatorLitModule  # noqa: E402
from src.util.fingerprint import build_fingerprint  # noqa: E402
from src.util.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def load_model(checkpoint: str, device):
    """Load either a Lightning ``.ckpt`` or an HF export directory.

    The HF path exists so upstream's released model zoo is scored by the same
    code as ours. It refuses a Flow2BT export for the same reason
    ``src/evaluate_scene.py`` does: those directories carry an unused regression
    decoder that would load silently and produce a meaningless number.
    """
    path = Path(checkpoint)
    if path.is_dir():
        if (path / "FLOW2BT_HEAD.json").is_file():
            raise ValueError(
                f"{path} is a Flow2BT head export; its net.* decoder is unused and "
                "untrained. Load it with FlowMatchingSimulator.load_flow_head instead."
            )
        from src._upstream import CrowdESSimulatorModel
        from src.models.crowdes_parity import ParityCrowdESSimulator

        released = CrowdESSimulatorModel.from_pretrained(str(path))
        wrapper = ParityCrowdESSimulator(
            history_length=released.config.history_length,
            future_length=released.config.future_length,
            env_size=released.config.env_size,
            env_dim=released.config.env_dim,
            neighbor_max_num=released.config.neighbor_max_num,
            latent_dim=released.config.latent_dim,
            hidden_dim=released.config.hidden_dim,
        )
        wrapper.net.load_state_dict(released.state_dict())
        logger.info("loaded HF export %s (%s)", path, released.config.model_type)
        return wrapper.eval().to(device)

    system = SimulatorLitModule.load_from_checkpoint(checkpoint, map_location=device)
    return system.eval().to(device).model


def _batches(dataset, indices, batch_size):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate_split(model, dataset, indices, batch_size, num_samples, device):
    """Return (ade, fde, min_ade, min_fde) summed-then-averaged over the subset."""
    totals = defaultdict(float)
    count = 0

    for batch in _batches(dataset, indices, batch_size):
        batch = {k: v.to(device) for k, v in batch.items()}
        size = batch["traj_hist"].shape[0]
        count += size

        deterministic = model(
            batch["traj_hist"], None, batch["goal"], batch["attr"],
            batch["control"].clone(), batch["neighbor"], batch["environment"],
            sampling=False,
        )
        distance = torch.linalg.norm(deterministic.preds - batch["traj_fut"], dim=-1)
        totals["ade"] += distance.mean(dim=-1).sum().item()
        totals["fde"] += distance[:, -1].sum().item()

        if num_samples > 0:
            best_ade = None
            best_fde = None
            for _ in range(num_samples):
                sampled = model(
                    batch["traj_hist"], None, batch["goal"], batch["attr"],
                    batch["control"].clone(), batch["neighbor"], batch["environment"],
                    sampling=True,
                )
                d = torch.linalg.norm(sampled.preds - batch["traj_fut"], dim=-1)
                ade, fde = d.mean(dim=-1), d[:, -1]
                best_ade = ade if best_ade is None else torch.minimum(best_ade, ade)
                best_fde = fde if best_fde is None else torch.minimum(best_fde, fde)
            totals[f"min_ade_{num_samples}"] += best_ade.sum().item()
            totals[f"min_fde_{num_samples}"] += best_fde.sum().item()

    return {k: v / max(count, 1) for k, v in totals.items()}, count


@hydra.main(version_base="1.3", config_path="configs", config_name="eval_agent")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(int(cfg.seed))

    crowdes_cfg = to_crowdes_config(cfg.data)
    simulator_cfg = crowdes_cfg["crowd_simulator"]["simulator"]
    fingerprint = build_fingerprint(simulator_cfg)
    cache_path = (
        Path(crowdes_cfg["dataset"]["dataset_preprocessed_path"])
        / "cache"
        / f"{cfg.split}_simulator.{fingerprint}.pickle"
    )

    if cfg.split == "test":
        logger.warning(
            "opening the HELD-OUT TEST SPLIT for %s. This is evaluation only; "
            "no training run may do this.",
            cfg.data.dataset.dataset_name,
        )

    dataset = ParitySimulatorDataset(
        crowdes_cfg,
        cfg.split,
        allow_test_split=(cfg.split == "test"),
        cache_path=str(cache_path),
    )

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model = load_model(str(cfg.ckpt), device)

    if getattr(model, "needs_endpoint_clusters", False) and not bool(
        getattr(model, "centers_fitted", torch.tensor(False))
    ):
        # Inference only uses the centres in the training branch, so this is
        # survivable -- but it means the checkpoint predates the buffer fix.
        logger.warning("checkpoint carries no fitted endpoint cluster centres")

    # Per-scene, using idx2data so the scene attribution is upstream's own.
    per_scene_indices: dict[str, list[int]] = defaultdict(list)
    for index, (scene, _) in enumerate(dataset.idx2data):
        per_scene_indices[scene].append(index)

    results = {}
    for scene, indices in sorted(per_scene_indices.items()):
        metrics, count = evaluate_split(
            model, dataset, indices, int(cfg.batch_size), int(cfg.num_samples), device
        )
        metrics["num_samples"] = count
        results[scene] = metrics
        logger.info("%s: %s", scene, metrics)

    if not results:
        raise RuntimeError(f"no samples in split {cfg.split!r} for {cfg.data.dataset.dataset_name}")

    # Aggregate weighted by sample count, matching a single pass over the split.
    total = sum(v["num_samples"] for v in results.values())
    metric_keys = [k for k in next(iter(results.values())) if k != "num_samples"]
    overall = {
        key: sum(v[key] * v["num_samples"] for v in results.values()) / max(total, 1)
        for key in metric_keys
    }
    overall["num_samples"] = total
    logger.info("OVERALL (%s/%s): %s", cfg.data.dataset.dataset_name, cfg.split, overall)

    out_dir = Path(cfg.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": cfg.data.dataset.dataset_name,
        "split": cfg.split,
        "ckpt": str(cfg.ckpt),
        "overall": overall,
        "per_scene": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    OmegaConf.save(cfg, out_dir / "eval_config.yaml")
    logger.info("wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
