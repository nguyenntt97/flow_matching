"""Scene-level evaluation of the Flow2BT reactive runtime, with a d_min sweep.

    python -m src.evaluate_flow2bt data=eth
    python -m src.evaluate_flow2bt data=eth d_min_sweep=[0.2,0.45] trials=5
    python -m src.evaluate_flow2bt data=eth runtime.use_cbf=false     # ablation

Scored by the same ``src/eval_loop.py`` as the baseline, so the two are
comparable by construction rather than by care.

The sweep exists because "C_R = 0.0%" is not self-evidently the goal. The
benchmark scores collisions at 0.2 m centre-to-centre and the ground truth is
already essentially collision-free there (measured 1e-4 on eth), while real
pedestrians come within the design's proposed 0.7 m on 8.7% of agent-frames. A
filter tuned to 0.7 m therefore buys a collision number by making the crowd
less like the data, and the realism metrics are where that shows up. Reporting
the frontier says so; reporting one point does not.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import src._upstream  # noqa: F401
from src.util.paths import register_resolvers

register_resolvers()

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data.dotdict_bridge import to_crowdes_config  # noqa: E402
from src.util.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def build_controller(cfg, crowdes_cfg):
    """Return ``(controller, feature_fn)`` for the configured nominal policy."""
    from src.runtime.controllers import WaypointController

    kind = str(cfg.runtime.controller)
    if kind == "waypoint":
        return WaypointController(), None

    if kind == "bt":
        import pickle

        from src.flow2bt.features import Geometry, build_features
        from src.runtime.controllers import BTController

        bundle_path = cfg.runtime.bt_bundle
        if not bundle_path:
            raise ValueError("runtime.controller=bt needs runtime.bt_bundle=<path>")
        with open(bundle_path, "rb") as handle:
            bundle = pickle.load(handle)

        def feature_fn(state):
            geometry = Geometry(
                position=state.position, velocity=state.velocity,
                neighbor_position=state.neighbor_position,
                neighbor_velocity=state.neighbor_velocity,
                neighbor_mask=state.neighbor_mask,
                heading=_heading(state), control=state.waypoint - state.position,
                sdf=None, sdf_gradient=None,
            )
            return build_features(geometry, state.goal - state.position)

        return BTController(bundle["tree"], bundle["bank"]), feature_fn

    raise ValueError(f"unknown runtime.controller {kind!r}")


def _heading(state):
    import numpy as np

    norm = np.linalg.norm(state.velocity, axis=1, keepdims=True)
    fallback = state.waypoint - state.position
    fallback_norm = np.linalg.norm(fallback, axis=1, keepdims=True)
    fallback = np.divide(fallback, np.maximum(fallback_norm, 1e-9))
    return np.where(norm > 1e-6, state.velocity / np.maximum(norm, 1e-12), fallback)


@hydra.main(version_base="1.3", config_path="configs", config_name="eval_flow2bt")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(int(cfg.seed))
    crowdes_cfg = to_crowdes_config(cfg.data)

    from src.eval_loop import evaluate_scenes, print_summary, write_summary
    from src.flow2bt.cbf import CBFConfig
    from src.runtime.flow2bt_framework import make_framework_class

    framework_class = make_framework_class()
    sweep = list(cfg.d_min_sweep) if cfg.runtime.use_cbf else [None]

    out_dir = Path(cfg.run_dir)
    frontier = []
    for d_min in sweep:
        cbf_kwargs = OmegaConf.to_container(cfg.cbf, resolve=True)
        if d_min is not None:
            cbf_kwargs["d_min"] = float(d_min)
        cbf_config = CBFConfig(**cbf_kwargs)

        controller, feature_fn = build_controller(cfg, crowdes_cfg)
        label = (
            f"Flow2BT[{cfg.runtime.controller}"
            f"{'+cbf@' + format(cbf_config.d_min, 'g') if cfg.runtime.use_cbf else ',no-cbf'}]"
        )

        holder = {}

        def factory(config, _holder=holder, _controller=controller, _cbf=cbf_config):
            framework = framework_class(
                config,
                controller=_controller,
                cbf_config=_cbf,
                use_cbf=bool(cfg.runtime.use_cbf),
                substeps=int(cfg.runtime.substeps),
                dense_output=bool(cfg.runtime.dense_output),
                agent_radius=float(cfg.runtime.agent_radius),
                feature_fn=feature_fn,
            )
            _holder["framework"] = framework
            return framework

        logger.info("running %s", label)
        summary = evaluate_scenes(
            crowdes_cfg, seed=int(cfg.seed), trials=int(cfg.trials),
            framework_factory=factory, scene_limit=cfg.scene_limit, label=label,
        )
        framework = holder["framework"]
        summary["runtime"] = {
            "controller": str(cfg.runtime.controller),
            "substeps": int(cfg.runtime.substeps),
            "tick_dt": float(framework.tick_dt),
            "use_cbf": bool(cfg.runtime.use_cbf),
            "dense_output": bool(cfg.runtime.dense_output),
            "cbf": cbf_kwargs,
            "diagnostics": {
                key: (float(value) if value != float("inf") else None)
                for key, value in framework.diagnostics.items()
            },
            "path_planning": dict(framework.follower.stats),
            "raw_collision": framework.raw_collision_rate(0.2),
            "raw_collision_at_dmin": framework.raw_collision_rate(cbf_config.d_min),
        }
        print_summary(summary)
        name = f"flow2bt_{cfg.runtime.controller}_dmin{cbf_config.d_min:g}.json"
        write_summary(summary, out_dir, name)

        frontier.append({
            "label": label,
            "d_min": None if d_min is None else float(d_min),
            "collision": summary["mean"]["Agent-Level Accuracy Metrics/Collision"],
            "density": summary["mean"]["Scene-Level Realism Metrics/Density"],
            "kinematics": summary["mean"]["Agent-Level Accuracy Metrics/Kineamtics"],
            "dtw": summary["mean"]["Agent-Level Accuracy Metrics/DTW"],
            "diversity": summary["mean"]["Agent-Level Accuracy Metrics/Diversity"],
            "min_separation": summary["runtime"]["diagnostics"]["min_separation"],
            "relaxed_agent_ticks": summary["runtime"]["diagnostics"]["relaxed_agent_ticks"],
            "agent_ticks": summary["runtime"]["diagnostics"]["agent_ticks"],
            "raw_collision": summary["runtime"]["raw_collision"]["rate"],
        })

    (out_dir / "frontier.json").write_text(json.dumps(frontier, indent=2))
    print()
    print("COLLISION / REALISM FRONTIER")
    print(f"  {'d_min':>7} {'Collision':>10} {'raw_Col':>9} {'Density':>9} {'Kinem.':>9} "
          f"{'DTW':>9} {'Divers.':>9} {'min_sep':>9}")
    for row in frontier:
        d_min = "none" if row["d_min"] is None else f"{row['d_min']:.2f}"
        separation = row["min_separation"]
        print(f"  {d_min:>7} {row['collision']:10.5f} {row['raw_collision']:9.5f} "
              f"{row['density']:9.5f} {row['kinematics']:9.5f} {row['dtw']:9.5f} "
              f"{row['diversity']:9.5f} "
              f"{(separation if separation is not None else float('nan')):9.4f}")
    print()
    print("  Collision = after upstream's snap + Kalman + 5->25fps interpolation")
    print("  raw_Col   = on the simulated state, which is all the filter controls")
    print()
    print(f"wrote {out_dir / 'frontier.json'}")


if __name__ == "__main__":
    main()
