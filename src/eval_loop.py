"""Scene-level evaluation loop -- a faithful port of ``CrowdES/evaluate.py::main``.

Why a port rather than a call
-----------------------------
``CrowdES/evaluate.py`` cannot be imported on Python 3.11 at all::

    evaluate.py:52
    print(f'Generated scenario: {len(generated_scenario['agent_id'].unique())} ...')
                                                       ^ SyntaxError

Nesting the same quote character inside an f-string expression is only legal
from Python 3.12 (PEP 701). This is a fourth upstream defect alongside the three
``src/README.md`` already records, and the house rule is to work around the
submodule rather than patch it -- so the loop is reproduced here.

What is preserved exactly
-------------------------
* ``TRIALS = 20`` per scene, and ``seed=trial`` passed to ``generate``
  (upstream marks the constant "DO NOT CHANGE THIS!").
* ``reproducibility_settings(seed)`` once, before anything else.
* The metric call: ``compute_metrics(generated, trajectory_dense, size, H)``.
* Aggregation: plain mean over every (scene, trial) for each metric.

What is added
-------------
* ``framework_factory``, so the same loop drives ``CrowdESFramework`` and the
  Flow2BT runtime and the two are scored by identical code.
* Per-trial records, not just the mean. A mean over 20 trials hides whether a
  collision rate is a stable property or one bad trial, and the whole point of
  the frontier is to compare distributions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Upstream's constant, marked "DO NOT CHANGE THIS!" -- changing it makes the
#: numbers incomparable to the published table.
TRIALS = 20


def default_framework_factory(crowdes_cfg):
    from CrowdES.inference_model import CrowdESFramework

    return CrowdESFramework(crowdes_cfg)


def evaluate_scenes(
    crowdes_cfg,
    seed: int = 0,
    trials: int = TRIALS,
    framework_factory: Optional[Callable] = None,
    scene_limit: Optional[int] = None,
    label: str = "CrowdES",
    viz: bool = False,
    viz_trials: int = 1,
    viz_fps: int = 5,
    viz_max_seconds: Optional[float] = None,
    out_dir: Optional[Path] = None,
) -> dict:
    """Drive a framework over every test scene and score it. Returns a summary dict."""
    from utils.dataloader.evaluation_dataloader import EvaluationDataset
    from utils.metrics import compute_metrics
    from utils.utils import reproducibility_settings

    reproducibility_settings(seed=seed)

    dataset = EvaluationDataset(crowdes_cfg, "test")
    framework = (framework_factory or default_framework_factory)(crowdes_cfg)

    scene_list = dataset.scene_list
    if scene_limit is not None:
        scene_list = scene_list[:scene_limit]

    records = []
    for scene_index, scene in enumerate(scene_list):
        data = dataset[scene_index]
        for trial in range(trials):
            logger.info(
                "[%s] scene %d/%d %s, trial %d/%d",
                label, scene_index + 1, len(scene_list), scene, trial + 1, trials,
            )
            framework.initialize_scene(
                data["img"], data["seg"], data["walkable"], data["navmesh"], data["H"]
            )
            generated = framework.generate(data["size"]["length"], seed=trial)
            generated["scene"] = scene

            metrics = compute_metrics(
                generated, data["trajectory_dense"], data["size"], data["H"]
            )
            flat = {
                f"{group}/{name}": float(value)
                for group, entries in metrics.items()
                for name, value in entries.items()
            }
            flat["_scene"] = scene
            flat["_trial"] = trial
            flat["_agents"] = int(generated["agent_id"].nunique())
            flat["_gt_agents"] = int(data["size"]["num_agents"])
            records.append(flat)

            headline = {
                k.split("/")[-1]: round(flat[k], 5)
                for k in flat
                if k.endswith(("Collision", "Density", "Kineamtics", "DTW", "Diversity"))
            }
            logger.info("[%s]   %s agents=%d", label, headline, flat["_agents"])

            # Optional scene trajectory & crowd visualization
            if viz and out_dir is not None and trial < viz_trials:
                from src.util.video import render_scene_video

                scene_dir = out_dir / scene
                scene_dir.mkdir(parents=True, exist_ok=True)
                video_path = scene_dir / f"scenario_trial_{trial:02d}.m4v"
                logger.info("[%s] rendering scene video to %s", label, video_path)
                render_scene_video(
                    video_path=video_path,
                    scene_bg=data.get("bg", data["img"]),
                    generated_scenario=generated,
                    scenario_length=int(data["size"]["length"]),
                    scene=scene,
                    trial=trial,
                    dataset_fps=int(crowdes_cfg["dataset"]["dataset_fps"]),
                    simulator_fps=int(crowdes_cfg["crowd_simulator"]["simulator"]["simulator_fps"]),
                    max_seconds=viz_max_seconds,
                )
                logger.info("[%s] rendered %s", label, video_path)


    keys = [k for k in records[0] if not k.startswith("_")]
    summary = {
        "label": label,
        "trials": trials,
        "scenes": list(scene_list),
        "mean": {k: float(np.mean([r[k] for r in records])) for k in keys},
        "std": {k: float(np.std([r[k] for r in records])) for k in keys},
        "records": records,
    }
    return summary


def write_summary(summary: dict, out_dir: Path, name: str = "scene_metrics.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(summary, indent=2))
    return path


def print_summary(summary: dict) -> None:
    print()
    print(f"SCENE-LEVEL METRICS -- {summary['label']} "
          f"({len(summary['scenes'])} scenes x {summary['trials']} trials)")
    print()
    for key in sorted(summary["mean"]):
        if key.startswith("Deprecated"):
            continue
        print(f"  {key:52s} {summary['mean'][key]:9.5f}  +/- {summary['std'][key]:.5f}")
