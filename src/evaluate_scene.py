"""Scene-level CrowdES benchmark via upstream's own evaluation loop.

    python -m src.evaluate_scene data=eth

Runs ``CrowdES/evaluate.py::main`` unchanged (TRIALS=20 per scene), which
drives ``CrowdESFramework`` end to end and scores with
``utils/metrics.py::compute_metrics``: quadrat / population / density EMDs,
travel distance-velocity-acceleration-time, DTW diversity, collision rate and
origin-goal JDE.

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
    from CrowdES.evaluate import main as upstream_evaluate

    logger.info("running upstream CrowdES.evaluate (TRIALS=20 per scene)")
    metrics = upstream_evaluate(crowdes_cfg, seed=int(cfg.seed))
    logger.info("scene-level metrics: %s", metrics)


if __name__ == "__main__":
    main()
