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
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


import src._upstream  # noqa: F401
from src.util.paths import register_resolvers

register_resolvers()

import cv2  # noqa: E402
import hydra  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
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


class _VideoWriter:
    """Writes video frames using ffmpeg CLI (with libx264 + yuv420p) for VS Code playback."""

    def __init__(self, path: Path, fps: float, width: int = 800, height: int = 800):
        self.path = path
        self.proc = None
        self.cv2_writer = None

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-loglevel", "error",
                str(path),
            ]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            except Exception as e:
                logger.warning("ffmpeg spawn failed (%s), falling back to OpenCV VideoWriter", e)
                self.proc = None

        if self.proc is None:
            for tag in ("avc1", "H264", "mp4v"):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*tag)
                    candidate = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
                    if candidate.isOpened():
                        self.cv2_writer = candidate
                        break
                except Exception:
                    continue
            if self.cv2_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.cv2_writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.write(frame_bgr.tobytes())
        elif self.cv2_writer is not None:
            self.cv2_writer.write(frame_bgr)

    def release(self) -> None:
        if self.proc is not None:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait()
            self.proc = None
        if self.cv2_writer is not None:
            self.cv2_writer.release()
            self.cv2_writer = None


def render_trajectory_video(
    video_path: Path,
    traj_hist: np.ndarray,
    traj_fut: np.ndarray,
    pred_det: np.ndarray,
    sampled_preds: np.ndarray | None,
    neighbors: np.ndarray | None,
    control: np.ndarray | None,
    goal: np.ndarray | None,
    scene: str,
    sample_idx: int,
    fps: int = 5,
) -> None:
    """Render a single trajectory sample into an MP4/M4V video."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    H = traj_hist.shape[0]
    F = traj_fut.shape[0]
    total_frames = H + F

    ade = float(np.linalg.norm(pred_det - traj_fut, axis=-1).mean())
    fde = float(np.linalg.norm(pred_det[-1] - traj_fut[-1]))

    # Compute bounding box with margin
    all_pts = [traj_hist, traj_fut, pred_det]
    if sampled_preds is not None and len(sampled_preds) > 0:
        all_pts.append(sampled_preds.reshape(-1, 2))
    if neighbors is not None:
        valid_neigh = neighbors[np.any(neighbors != 0, axis=(1, 2))]
        if len(valid_neigh) > 0:
            all_pts.append(valid_neigh.reshape(-1, 2))
    if control is not None and np.linalg.norm(control) < 25.0:
        all_pts.append(control.reshape(1, 2))
    if goal is not None and np.linalg.norm(goal) < 25.0:
        all_pts.append(goal.reshape(1, 2))

    pts_concat = np.concatenate(all_pts, axis=0)
    x_min, x_max = float(pts_concat[:, 0].min() - 1.5), float(pts_concat[:, 0].max() + 1.5)
    y_min, y_max = float(pts_concat[:, 1].min() - 1.5), float(pts_concat[:, 1].max() + 1.5)

    span = max(x_max - x_min, y_max - y_min, 4.0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    x_lim = (cx - span / 2.0, cx + span / 2.0)
    y_lim = (cy - span / 2.0, cy + span / 2.0)

    w, h = 800, 800
    writer = _VideoWriter(video_path, float(fps), w, h)


    dt = 1.0 / max(float(fps), 1.0)

    # Active neighbors
    active_neighbors = []
    if neighbors is not None:
        for n_i in range(neighbors.shape[0]):
            if np.any(neighbors[n_i] != 0):
                active_neighbors.append(neighbors[n_i])

    last_bgr = None
    for step in range(total_frames):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.set_facecolor("#161b22")
        fig.patch.set_facecolor("#0d1117")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.3, color="#8b949e")
        ax.set_xlabel("X (meters relative to startpoint)", color="#c9d1d9", fontsize=10)
        ax.set_ylabel("Y (meters relative to startpoint)", color="#c9d1d9", fontsize=10)
        ax.tick_params(colors="#8b949e", labelsize=9)

        # Plot origin (0, 0)
        ax.scatter([0], [0], color="#58a6ff", s=60, marker="o", facecolors="none", edgecolors="#58a6ff", linewidths=1.5, zorder=2)

        # Draw NavMesh control waypoint if present
        if control is not None and np.linalg.norm(control) < 25.0:
            ax.scatter([control[0]], [control[1]], color="#f0883e", s=140, marker="*", zorder=6, label=r"NavMesh $c_{t,nav}$")

        # Draw Destination Goal if present and distinct from control
        if goal is not None and np.linalg.norm(goal) < 25.0:
            if control is None or np.linalg.norm(goal - control) > 0.3:
                ax.scatter([goal[0]], [goal[1]], color="#d29922", s=110, marker="D", zorder=6, label="Goal Destination")

        # Draw Neighbors in past
        for n_idx, n_traj in enumerate(active_neighbors):
            t_neigh = min(step, H - 1)
            ax.plot(n_traj[:t_neigh + 1, 0], n_traj[:t_neigh + 1, 1], color="#8b949e", linestyle=":", linewidth=1.5, alpha=0.6)
            ax.scatter([n_traj[t_neigh, 0]], [n_traj[t_neigh, 1]], color="#bc8cff", s=45, marker="^", alpha=0.8,
                       label="Neighbor" if n_idx == 0 else None, zorder=4)

        if step < H:
            # Historical observation phase
            ax.plot(traj_hist[:step + 1, 0], traj_hist[:step + 1, 1], color="#58a6ff", linewidth=2.5, label="Observed Past", zorder=5)
            ax.scatter(traj_hist[:step + 1, 0], traj_hist[:step + 1, 1], color="#58a6ff", s=25, zorder=5)

            curr_pos = traj_hist[step]
            ax.scatter([curr_pos[0]], [curr_pos[1]], color="#79c0ff", s=120, edgecolors="#ffffff", linewidths=2, zorder=8)

            t_elapsed = (step - (H - 1)) * dt
            mode_text = f"OBSERVING PAST (t = {t_elapsed:.1f}s)"
            hud_text = (
                f"Scene: {scene} | Sample #{sample_idx}\n"
                f"History Step: {step + 1}/{H}\n"
                f"Mode: Observing footstep history"
            )
        else:
            # Future rollout phase
            k = step - H  # future index 0 .. F-1

            # Full past history
            ax.plot(traj_hist[:, 0], traj_hist[:, 1], color="#58a6ff", linewidth=1.5, linestyle="--", alpha=0.7, label="Observed Past", zorder=3)
            ax.scatter(traj_hist[:, 0], traj_hist[:, 1], color="#58a6ff", s=15, alpha=0.7, zorder=3)

            # Sampled predictions (if any)
            if sampled_preds is not None and len(sampled_preds) > 0:
                for s_i in range(len(sampled_preds)):
                    ax.plot(sampled_preds[s_i, :, 0], sampled_preds[s_i, :, 1], color="#f0883e", linewidth=1.0, alpha=0.3,
                            label="Sampled Predictions" if s_i == 0 else None, zorder=3)

            # Ground truth future path
            ax.plot(traj_fut[:, 0], traj_fut[:, 1], color="#3fb950", linewidth=1.5, linestyle=":", alpha=0.6, zorder=4)
            ax.plot(traj_fut[:k + 1, 0], traj_fut[:k + 1, 1], color="#3fb950", linewidth=2.5, label="Ground Truth Future", zorder=5)
            ax.scatter(traj_fut[:k + 1, 0], traj_fut[:k + 1, 1], color="#3fb950", s=30, zorder=5)

            # Model predicted future path
            ax.plot(pred_det[:, 0], pred_det[:, 1], color="#f85149", linewidth=1.5, linestyle=":", alpha=0.6, zorder=4)
            ax.plot(pred_det[:k + 1, 0], pred_det[:k + 1, 1], color="#f85149", linewidth=2.5, label="Model Prediction", zorder=5)
            ax.scatter(pred_det[:k + 1, 0], pred_det[:k + 1, 1], color="#f85149", s=30, marker="s", zorder=5)

            # Current positions
            gt_curr = traj_fut[k]
            pred_curr = pred_det[k]
            ax.scatter([gt_curr[0]], [gt_curr[1]], color="#56d364", s=120, edgecolors="#ffffff", linewidths=2, zorder=8)
            ax.scatter([pred_curr[0]], [pred_curr[1]], color="#ff7b72", s=120, marker="s", edgecolors="#ffffff", linewidths=2, zorder=8)

            # Error line between GT and pred
            ax.plot([gt_curr[0], pred_curr[0]], [gt_curr[1], pred_curr[1]], color="#ff7b72", linestyle="--", linewidth=1.5, zorder=7)
            step_err = float(np.linalg.norm(gt_curr - pred_curr))

            t_elapsed = (k + 1) * dt
            mode_text = f"PREDICTED FUTURE (t = +{t_elapsed:.1f}s)"
            hud_text = (
                f"Scene: {scene} | Sample #{sample_idx}\n"
                f"Future Step: +{k + 1}/{F} (+{t_elapsed:.1f}s)\n"
                f"Step Error: {step_err:.3f} m\n"
                f"Trajectory ADE: {ade:.3f} m | FDE: {fde:.3f} m"
            )

        # Draw HUD box
        ax.text(
            0.03, 0.97, hud_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            color="#f0f6fc",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#30363d", alpha=0.9),
            zorder=10,
        )

        ax.set_title(f"{scene} | {mode_text}", color="#f0f6fc", fontsize=11, fontweight="bold", pad=12)
        ax.legend(loc="lower right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        bgr = cv2.resize(bgr, (w, h))
        writer.write(bgr)
        last_bgr = bgr
        plt.close(fig)

    # Freeze on the final frame for 1 second (fps frames)
    if last_bgr is not None:
        for _ in range(fps):
            writer.write(last_bgr)

    writer.release()


@torch.no_grad()
def visualize_scene_trajectories(
    model,
    dataset,
    indices: list[int],
    scene: str,
    output_dir: Path,
    limit: int | None = 50,
    fps: int = 5,
    num_samples: int = 0,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Render individual .m4v videos for trajectories in this scene."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_indices = indices[:limit] if limit is not None else indices

    count = 0
    for idx in target_indices:
        item = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}

        deterministic = model(
            batch["traj_hist"], None, batch["goal"], batch["attr"],
            batch["control"].clone(), batch["neighbor"], batch["environment"],
            sampling=False,
        )
        pred_det = deterministic.preds[0].detach().cpu().numpy()

        sampled_preds = None
        if num_samples > 0:
            samples = []
            for _ in range(min(num_samples, 5)):
                s = model(
                    batch["traj_hist"], None, batch["goal"], batch["attr"],
                    batch["control"].clone(), batch["neighbor"], batch["environment"],
                    sampling=True,
                )
                samples.append(s.preds[0].detach().cpu().numpy())
            sampled_preds = np.stack(samples, axis=0)

        traj_hist = item["traj_hist"].cpu().numpy()
        traj_fut = item["traj_fut"].cpu().numpy()
        neighbors = item["neighbor"].cpu().numpy() if "neighbor" in item else None
        control = item["control"].cpu().numpy() if "control" in item else None
        goal = item["goal"].cpu().numpy() if "goal" in item else None

        video_path = output_dir / f"traj_{idx:05d}.mp4"
        render_trajectory_video(
            video_path=video_path,
            traj_hist=traj_hist,
            traj_fut=traj_fut,
            pred_det=pred_det,
            sampled_preds=sampled_preds,
            neighbors=neighbors,
            control=control,
            goal=goal,
            scene=scene,
            sample_idx=idx,
            fps=fps,
        )
        count += 1

    return count


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

    # Optional trajectory visualization (sub directory per scene, a m4v video per trajectory)
    if bool(cfg.get("viz", False)):
        viz_limit = cfg.get("viz_limit", 50)
        fps = int(cfg.get("viz_fps", 5))
        viz_sub = str(cfg.get("viz_dir", "") or "").strip()
        num_samples = int(cfg.get("num_samples", 0))

        logger.info(
            "visualizing trajectories (viz_limit=%s per scene, fps=%d)",
            viz_limit,
            fps,
        )
        for scene, indices in sorted(per_scene_indices.items()):
            scene_out_dir = (out_dir / viz_sub / scene) if viz_sub else (out_dir / scene)
            count = visualize_scene_trajectories(
                model=model,
                dataset=dataset,
                indices=indices,
                scene=scene,
                output_dir=scene_out_dir,
                limit=viz_limit,
                fps=fps,
                num_samples=num_samples,
                device=device,
            )
            logger.info("visualized %d trajectories for %s in %s", count, scene, scene_out_dir)



if __name__ == "__main__":
    main()
