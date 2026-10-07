from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import BODY_NAMES
from .math_utils import quat_apply_inverse
from .schemas import MotionClip


def load_motion_clips(path: Path, anchor_body_name: str = "torso_link") -> list[MotionClip]:
    motion_files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No .npz motion files found at {path}")

    anchor_idx = BODY_NAMES.index(anchor_body_name)
    clips: list[MotionClip] = []
    for motion_file in motion_files:
        data = np.load(motion_file)
        required = [
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{motion_file} is missing keys: {missing}")
        body_count = data["body_pos_w"].shape[1]
        if body_count > len(BODY_NAMES):
            raise ValueError(f"{motion_file} has {body_count} bodies, but only {len(BODY_NAMES)} names are known")

        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        length = int(data["joint_pos"].shape[0])
        first_anchor_pos = body_pos[0, anchor_idx]
        last_anchor_pos = body_pos[length - 1, anchor_idx]
        first_anchor_quat = body_quat[0, anchor_idx]
        local_delta = quat_apply_inverse(first_anchor_quat, last_anchor_pos - first_anchor_pos)[:2]

        kick_leg = None
        if "kick_leg" in data.files:
            raw = data["kick_leg"]
            try:
                kick_leg = str(raw.item()).strip().lower()
            except ValueError:
                kick_leg = str(raw).strip().lower()

        fps_arr = np.asarray(data["fps"]).reshape(-1)
        clips.append(
            MotionClip(
                path=motion_file,
                name=motion_file.stem,
                fps=float(fps_arr[0]),
                joint_pos=np.asarray(data["joint_pos"], dtype=np.float32),
                joint_vel=np.asarray(data["joint_vel"], dtype=np.float32),
                body_pos_w=body_pos,
                body_quat_w=body_quat,
                body_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=np.float32),
                body_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float32),
                local_displacement_xy=local_delta.astype(np.float32),
                length=length,
                kick_leg=kick_leg,
            )
        )
    return clips


def select_motion(clips: list[MotionClip], ball_local_xy: np.ndarray) -> tuple[MotionClip, float]:
    errors = [float(np.linalg.norm(clip.local_displacement_xy - ball_local_xy)) for clip in clips]
    best_idx = int(np.argmin(errors))
    return clips[best_idx], errors[best_idx]

