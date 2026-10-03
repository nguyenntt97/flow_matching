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
from src.util.timing import TimedSimulator, maybe_timer, time_method  # noqa: E402

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

    timer = maybe_timer(bool(cfg.get("timing", False)), int(cfg.get("timing_warmup", 3)))
    simulator_ckpt = cfg.get("simulator_ckpt")

    def factory(config):
        """CrowdESFramework, optionally with a different simulator and/or timed.

        ``simulator_ckpt`` swaps in any simulator-protocol model ``load_model``
        accepts -- in practice the flow teacher's Lightning ``.ckpt``. That is
        not the refused case above: the refusal is for a Flow2BT *head export*
        loaded through the HF config, which silently runs an untrained decoder.
        Here the actual trained model runs, behind the same call site.
        """
        from CrowdES.inference_model import CrowdESFramework

        framework = CrowdESFramework(config)
        released = framework.CrowdES_simulator
        model = released
        if simulator_ckpt:
            from src.evaluate_agent import load_model

            model = load_model(str(simulator_ckpt), framework.device)
            latent = getattr(model, "latent_dim", released.config.latent_dim)
            if int(latent) != int(released.config.latent_dim):
                raise ValueError(
                    f"{simulator_ckpt} has latent_dim {latent}; the framework's "
                    f"behaviour-state buffers use {released.config.latent_dim}"
                )
            logger.info("simulator replaced by %s", simulator_ckpt)
        if simulator_ckpt or timer is not None:
            sim = config["crowd_simulator"]["simulator"]
            chunk_seconds = float(sim["future_length"]) / float(sim["simulator_fps"])
            # The adapter also supplies .config, which framework code reads.
            framework.CrowdES_simulator = TimedSimulator(
                model, timer, "scene/simulator_forward", chunk_seconds, config_source=released
            )
        if timer is not None:
            # End to end per emitter window: preprocessing (neighbours, navmesh
            # control points, environment crops) + forwards + walkable snap.
            time_method(
                framework, "process_simulator", timer, "scene/process_simulator_window",
                agents_fn=lambda f: len(f.agent_ids_in_current_scene),
                sim_seconds_fn=lambda f: f.window_frame / f.simulator_fps,
                cuda=True,
            )
        return framework

    logger.info("running scene-level evaluation (TRIALS=%d per scene)", TRIALS)
    summary = evaluate_scenes(
        crowdes_cfg,
        framework_factory=factory,
        seed=int(cfg.seed),
        trials=int(cfg.get("trials", TRIALS)),
        scene_limit=cfg.get("scene_limit"),
        label=str(cfg.get("label", "CrowdES")),
        viz=bool(cfg.get("viz", False)),
        viz_trials=int(cfg.get("viz_trials", 1)),
        viz_fps=int(cfg.get("viz_fps", 5)),
        viz_max_seconds=cfg.get("viz_max_seconds"),
        out_dir=Path(cfg.run_dir),
    )
    if timer is not None:
        summary["timing"] = timer.summary()
        logger.info("\n%s", timer.report(f"INFERENCE TIMING ({cfg.get('label', 'CrowdES')})"))
    print_summary(summary)

    logger.info("wrote %s", write_summary(summary, Path(cfg.run_dir)))


if __name__ == "__main__":
    main()
