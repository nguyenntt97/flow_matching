"""Scene-level CrowdES benchmark.

    python -m src.evaluate_scene data=eth

TRIALS=20 per scene, driving ``CrowdESFramework`` end to end and scoring with
``utils/metrics.py::compute_metrics``: quadrat / population / density EMDs,
travel distance-velocity-acceleration-time, DTW diversity, collision rate and
origin-goal JDE.

The loop itself lives in ``src/eval_loop.py`` because upstream's
``CrowdES/evaluate.py`` does not import on Python 3.11 -- see the docstring
there.

``CrowdESFramework.__init__`` loads emitter_pre, emitter *and* simulator
unconditionally, so all three must be present. The emitter pair comes from the
upstream release tag v1.0-model; the simulator is ours, placed by training with
``export.mirror_upstream=true``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import src._upstream  # noqa: F401
from src.util.paths import register_resolvers

register_resolvers()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.data.dotdict_bridge import to_crowdes_config  # noqa: E402
from src.util.export import resolve_mirror_dir  # noqa: E402
from src.util.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)

_RELEASE = "https://github.com/InhwanBae/Crowd-Behavior-Generation/releases (tag v1.0-model)"


def check_checkpoints(crowdes_cfg) -> None:
    dataset_name = crowdes_cfg["dataset"]["dataset_name"]
    required = {
        "emitter_pre": crowdes_cfg["crowd_emitter"]["emitter_pre"]["checkpoint_dir"],
        "emitter": crowdes_cfg["crowd_emitter"]["emitter"]["checkpoint_dir"],
        "simulator": crowdes_cfg["crowd_simulator"]["simulator"]["checkpoint_dir"],
    }
    missing = []
    for label, template in required.items():
        path = Path(resolve_mirror_dir(template, dataset_name))
        if not (path / "config.json").is_file():
            missing.append(f"  {label}: {path}")
        elif (path / "FLOW2BT_HEAD.json").is_file():
            raise ValueError(
                f"{path} is a Flow2BT head export, not a CrowdES simulator.\n"
                "Its net.* weights still carry upstream's UNUSED regression decoder, so "
                "CrowdESFramework would load and run that decoder and report a plausible "
                "but meaningless baseline. Evaluate it with src/evaluate_flow2bt.py."
            )
    if missing:
        raise FileNotFoundError(
            "CrowdESFramework needs all three checkpoints; these are absent:\n"
            + "\n".join(missing)
            + f"\n\nDownload the emitter pair from {_RELEASE}.\n"
            "Produce the simulator with: python -m src.train data="
            f"{dataset_name} export.mirror_upstream=true"
        )


@hydra.main(version_base="1.3", config_path="configs", config_name="eval_scene")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(int(cfg.seed))

    crowdes_cfg = to_crowdes_config(cfg.data)
    check_checkpoints(crowdes_cfg)

    # Lazy: this pulls POT, dtaidistance, diffusers and pathfinder.pyrvo.
    from src.eval_loop import TRIALS, evaluate_scenes, print_summary, write_summary

    logger.info("running scene-level evaluation (TRIALS=%d per scene)", TRIALS)
    summary = evaluate_scenes(
        crowdes_cfg,
        seed=int(cfg.seed),
        trials=int(cfg.get("trials", TRIALS)),
        scene_limit=cfg.get("scene_limit"),
        label=str(cfg.get("label", "CrowdES")),
    )
    print_summary(summary)
    logger.info("wrote %s", write_summary(summary, Path(cfg.run_dir)))


if __name__ == "__main__":
    main()
