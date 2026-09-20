"""Baseline anchors for the collision-rate frontier.

    python -m src.tools.anchors data=eth

``utils/metrics.py::compute_metrics`` scores the collision rate of the
*generated* scenario only (``metrics.py:617``) and never computes it for the
ground truth. That omission matters for this project: the design's target is
"C_R = 0.0%", but the metric counts agent-frames where any other agent is within
**0.2 m centre to centre** (``metrics.py:477``), and real pedestrians do that all
the time. Without the ground-truth number there is nothing to say whether 0.0%
is an achievement or an artefact of pushing the crowd apart.

So this prints, per scene:

* the ground-truth collision rate under the benchmark's own definition;
* the same at a sweep of thresholds, which is the x-axis the CBF's ``d_min``
  will be swept along;
* the nearest-neighbour distance distribution, which is what a ``d_min`` choice
  is really trading against.

It reads only the test split's ground truth. No model, no checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import src._upstream  # noqa: F401
from src.util.paths import register_resolvers

register_resolvers()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.data.dotdict_bridge import to_crowdes_config  # noqa: E402
from src.util.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)

#: Thresholds to report the collision rate at. 0.2 is the benchmark's own;
#: the rest bracket the d_min values the CBF sweep will use.
THRESHOLDS = (0.1, 0.2, 0.3, 0.45, 0.6, 0.7, 1.0)


def collision_rate(trajectory_meter, threshold: float) -> float:
    """``measure_collision_rate`` with the threshold exposed.

    Reproduces ``utils/metrics.py:457-481`` exactly -- agents involved (not
    pairs) over total agent-frame rows -- with 0.2 m lifted out as a parameter.
    """
    collisions = 0
    for _, frame_data in trajectory_meter.groupby("frame"):
        points = frame_data[["x", "y"]].values
        distance = np.linalg.norm(points[:, None] - points, axis=-1)
        np.fill_diagonal(distance, np.inf)
        collisions += np.count_nonzero((distance < threshold).any(axis=0))
    return collisions / max(len(trajectory_meter), 1)


def nearest_neighbour_distances(trajectory_meter) -> np.ndarray:
    """Per agent-frame, distance to the closest other agent in that frame."""
    out = []
    for _, frame_data in trajectory_meter.groupby("frame"):
        points = frame_data[["x", "y"]].values
        if len(points) < 2:
            continue
        distance = np.linalg.norm(points[:, None] - points, axis=-1)
        np.fill_diagonal(distance, np.inf)
        out.append(distance.min(axis=1))
    return np.concatenate(out) if out else np.array([])


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_scene")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(int(cfg.seed))

    crowdes_cfg = to_crowdes_config(cfg.data)
    dataset_name = crowdes_cfg["dataset"]["dataset_name"]

    # Lazy, as elsewhere: these pull POT / dtaidistance / diffusers.
    from utils.dataloader.evaluation_dataloader import EvaluationDataset
    from utils.trajectory import image_to_world_trajectory

    dataset = EvaluationDataset(crowdes_cfg, "test")
    logger.info("ground-truth anchors for %s: %d test scenes", dataset_name, len(dataset.scene_list))

    per_scene, all_distances, total_rows = {}, [], 0
    for index, scene in enumerate(dataset.scene_list):
        data = dataset[index]
        meter = image_to_world_trajectory(data["trajectory_dense"], data["H"])
        rates = {f"{t:g}": collision_rate(meter, t) for t in THRESHOLDS}
        distances = nearest_neighbour_distances(meter)

        per_scene[scene] = {
            "rows": int(len(meter)),
            "agents": int(meter["agent_id"].nunique()),
            "collision_rate": rates,
            "nn_distance_percentiles": {
                p: float(np.percentile(distances, p)) for p in (1, 5, 25, 50)
            } if len(distances) else {},
        }
        all_distances.append(distances)
        total_rows += len(meter)
        logger.info("%s: %d rows, C_R@0.2m = %.4f", scene, len(meter), rates["0.2"])

    distances = np.concatenate([d for d in all_distances if len(d)])
    overall = {
        f"{t:g}": float((distances < t).mean()) for t in THRESHOLDS
    }

    print()
    print(f"GROUND TRUTH ANCHORS -- {dataset_name} test split ({total_rows} agent-frames)")
    print()
    print("  Collision rate at each threshold (the benchmark uses 0.2 m):")
    for threshold in THRESHOLDS:
        marker = "  <- utils/metrics.py" if threshold == 0.2 else ""
        print(f"    d < {threshold:4.2f} m : {overall[f'{threshold:g}']:7.4f}{marker}")
    print()
    print("  Nearest-neighbour distance, all agent-frames:")
    for percentile in (1, 5, 25, 50, 75):
        print(f"    p{percentile:<3d} : {np.percentile(distances, percentile):6.3f} m")

    out_dir = Path(cfg.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "split": "test",
        "note": (
            "Ground-truth collision rate under utils/metrics.py::measure_collision_rate, "
            "which compute_metrics computes only for generated scenarios. A generated "
            "C_R far BELOW these numbers means the crowd was pushed apart, not that it "
            "became realistic."
        ),
        "total_rows": total_rows,
        "collision_rate": overall,
        "nn_distance_percentiles": {
            str(p): float(np.percentile(distances, p)) for p in (1, 5, 10, 25, 50, 75, 90)
        },
        "per_scene": per_scene,
    }
    (out_dir / "anchors.json").write_text(json.dumps(payload, indent=2))
    print()
    print(f"wrote {out_dir / 'anchors.json'}")


if __name__ == "__main__":
    main()
