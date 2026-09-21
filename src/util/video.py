"""Video generation utilities for agent and scene evaluation.

Uses the system ffmpeg binary (with libx264 + yuv420p) to guarantee universal
playback compatibility across VS Code's built-in media player and web browsers.
Falls back to OpenCV VideoWriter if ffmpeg is unavailable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

logger = logging.getLogger(__name__)

# Standard color palette for crowd agents
AGENT_COLORS = [
    (234, 56, 41),   # Red
    (246, 133, 17),  # Orange
    (248, 217, 4),   # Yellow
    (170, 216, 22),  # Lime
    (39, 187, 54),   # Green
    (0, 143, 93),    # Teal
    (15, 181, 174),  # Cyan
    (51, 197, 232),  # Light blue
    (56, 146, 243),  # Blue
    (104, 109, 244), # Indigo
    (137, 61, 231),  # Purple
    (224, 85, 226),  # Magenta
    (222, 61, 130),  # Pink
]


class VideoWriter:
    """Robust video writer targeting H.264 / yuv420p for VS Code playback."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        # Ensure even dimensions for H.264
        self.width = (int(width) // 2) * 2
        self.height = (int(height) // 2) * 2
        self.proc: Optional[subprocess.Popen] = None
        self.cv2_writer: Optional[cv2.VideoWriter] = None

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-loglevel", "error",
                str(self.path),
            ]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            except Exception as e:
                logger.warning("ffmpeg spawn failed (%s), falling back to OpenCV", e)
                self.proc = None

        if self.proc is None:
            for tag in ("avc1", "H264", "mp4v"):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*tag)
                    candidate = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
                    if candidate.isOpened():
                        self.cv2_writer = candidate
                        break
                except Exception:
                    continue
            if self.cv2_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.cv2_writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

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

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def render_scene_video(
    video_path: Path,
    scene_bg: np.ndarray,
    generated_scenario: pd.DataFrame,
    scenario_length: int,
    scene: str,
    trial: int,
    dataset_fps: int = 25,
    simulator_fps: int = 5,
    max_seconds: Optional[float] = None,
) -> None:
    """Render the full continuous crowd simulation for a scene trial into .m4v."""
    bg = np.array(scene_bg)
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    elif bg.shape[2] == 4:
        bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2BGR)
    elif bg.shape[2] == 3:
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    h, w = bg.shape[:2]
    w = (w // 2) * 2
    h = (h // 2) * 2
    bg = cv2.resize(bg, (w, h))

    scale_factor = max(w, h) / 720.0
    line_thickness = max(1, round(scale_factor * 1.5))
    circle_size = max(5, round(scale_factor * 12))
    trail_len = int(simulator_fps * 3)  # 3 seconds of trailing path

    frame_step = max(1, dataset_fps // simulator_fps)
    max_frames = scenario_length
    if max_seconds is not None:
        max_frames = min(max_frames, int(max_seconds * dataset_fps))

    frames_to_render = list(range(0, max_frames, frame_step))
    if not frames_to_render:
        return

    writer = VideoWriter(video_path, fps=simulator_fps, width=w, height=h)

    # Pre-index scenario by frame for fast retrieval
    df = generated_scenario[generated_scenario["frame"].between(0, max_frames)]
    grouped = {f: g for f, g in df.groupby("frame")}

    # Track recent history for each agent for smooth trails
    agent_trails: dict[int, list[tuple[int, int]]] = {}
    last_frame_bgr = None

    for frame in frames_to_render:
        frame_img = bg.copy()
        current_time_sec = frame / dataset_fps
        active_agents = grouped.get(frame, pd.DataFrame())

        # Update and draw trails
        current_active_ids = set()
        for _, row in active_agents.iterrows():
            aid = int(row["agent_id"])
            current_active_ids.add(aid)
            pt = (int(round(row["x"])), int(round(row["y"])))
            if aid not in agent_trails:
                agent_trails[aid] = []
            agent_trails[aid].append(pt)
            if len(agent_trails[aid]) > trail_len:
                agent_trails[aid].pop(0)

        # Remove departed agents
        for aid in list(agent_trails.keys()):
            if aid not in current_active_ids:
                del agent_trails[aid]

        # Draw trails
        for aid, pts in agent_trails.items():
            if len(pts) > 1:
                color = AGENT_COLORS[aid % len(AGENT_COLORS)][::-1]
                for i in range(len(pts) - 1):
                    alpha = (i + 1) / len(pts)
                    thickness = max(1, int(line_thickness * alpha))
                    cv2.line(frame_img, pts[i], pts[i + 1], color, thickness)

        # Draw active agents
        agent_positions = []
        for _, row in active_agents.iterrows():
            aid = int(row["agent_id"])
            atype = int(row["agent_type"])
            ax, ay = int(round(row["x"])), int(round(row["y"]))
            agent_positions.append((ax, ay, aid))
            color = AGENT_COLORS[aid % len(AGENT_COLORS)][::-1]

            if atype == 0:  # Pedestrian
                cv2.circle(frame_img, (ax, ay), circle_size, color, -1)
                cv2.circle(frame_img, (ax, ay), circle_size, (255, 255, 255), 1)
            elif atype == 1:  # Rider / Bike
                pts = np.array([
                    [ax, ay - circle_size],
                    [ax - circle_size, ay + circle_size],
                    [ax + circle_size, ay + circle_size]
                ], np.int32)
                cv2.fillPoly(frame_img, [pts], color)
                cv2.polylines(frame_img, [pts], True, (255, 255, 255), 1)
            else:  # Vehicle
                cv2.rectangle(
                    frame_img,
                    (ax - circle_size, ay - circle_size),
                    (ax + circle_size, ay + circle_size),
                    color, -1
                )
                cv2.rectangle(
                    frame_img,
                    (ax - circle_size, ay - circle_size),
                    (ax + circle_size, ay + circle_size),
                    (255, 255, 255), 1
                )

        # Highlight close encounters / near collisions
        for i in range(len(agent_positions)):
            for j in range(i + 1, len(agent_positions)):
                p1, p2 = agent_positions[i], agent_positions[j]
                d = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if d < circle_size * 2.2:  # Proximity warning
                    cv2.circle(frame_img, (p1[0], p1[1]), int(circle_size * 1.6), (0, 0, 255), 2)
                    cv2.circle(frame_img, (p2[0], p2[1]), int(circle_size * 1.6), (0, 0, 255), 2)

        # Draw HUD Box
        hud_w, hud_h = 320, 95
        sub_img = frame_img[10:10 + hud_h, 10:10 + hud_w]
        black_rect = np.zeros_like(sub_img)
        cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0, sub_img)
        frame_img[10:10 + hud_h, 10:10 + hud_w] = sub_img
        cv2.rectangle(frame_img, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 80), 1)

        cv2.putText(
            frame_img,
            f"Scene: {scene} (Trial #{trial})",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_img,
            f"Time: {current_time_sec:.1f}s / {scenario_length / dataset_fps:.1f}s",
            (20, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_img,
            f"Active Population: {len(active_agents)} agents",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 220, 100),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame_img)
        last_frame_bgr = frame_img

    # Freeze on final frame for 1 second
    if last_frame_bgr is not None:
        for _ in range(simulator_fps):
            writer.write(last_frame_bgr)

    writer.release()

