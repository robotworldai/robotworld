#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportUnusedFunction=false
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Convert collect_vla_data.py HDF5 datasets to LeRobot format.

Auto-reads HDF5 metadata (state_format, action_format, orientation_rep, fps,
cameras, dimensions) and builds LeRobot features accordingly.

Supports all combinations produced by collect_vla_data.py:
- State:  joint | eef_pose | both
- Action: abs_joint | rel_joint | abs_eef_pose | rel_eef_pose
- Orient: quat (4D) | euler (3D)
- Cameras: any number, any resolution (auto-detected)

Also accepts my_lift-style HDF5: action_format ``absolute_joint_target_8d`` (mapped to
abs_joint), and root attr ``control_freq_hz`` when ``fps`` is absent.

Examples::

    # Auto-detect everything from HDF5 metadata
    python convert_to_lerobot.py --input data.hdf5 --output my_dataset

    # Override target FPS (downsample from HDF5 fps)
    python convert_to_lerobot.py --input data.hdf5 --output my_dataset --target_fps 10

    # Only successful episodes
    python convert_to_lerobot.py --input data.hdf5 --output my_dataset --success_only
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from task_prompts import (
    LEGACY_DEFAULT_PROMPT,
    resolve_prompt_from_task,
)


logger = logging.getLogger(__name__)


def _task_slug(task_name: str) -> str:
    normalized = task_name.lower()
    for suffix in ("-datacollection-v0", "-play-v0", "-ik-abs-v0", "-ik-rel-v0", "-v0"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    tokens = [
        token for token in normalized.split("-")
        if token and token not in {"franka", "panda", "ur5", "ur10", "xarm", "kinova"}
    ]
    if not tokens:
        return normalized.replace("-", "_")
    return "_".join(tokens)


def _resolve_output_dir(task_name: str, requested_output: str, base_dir: Path) -> str:
    output_path = Path(requested_output)
    if requested_output.startswith(".") or output_path.is_absolute() or output_path.parent != Path("."):
        return str(output_path)
    return str(base_dir / _task_slug(task_name) / output_path.name)


# ---------------------------------------------------------------------------
# HDF5 introspection
# ---------------------------------------------------------------------------


def _read_meta(f: h5py.File) -> dict:
    """Extract metadata from HDF5 attrs and first demo."""
    attrs = dict(f.attrs)
    # Decode bytes -> str
    for k, v in attrs.items():
        if isinstance(v, bytes):
            attrs[k] = v.decode("utf-8")

    demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[-1]))
    if not demos:
        raise ValueError("HDF5 contains no demos")
    first = f[f"data/{demos[0]}"]

    # Detect obs datasets
    obs_keys: list[str] = []
    if "obs" in first:
        obs_keys = list(first["obs"].keys())

    # Detect cameras (keys ending with _rgb)
    cam_keys = [k for k in obs_keys if k.endswith("_rgb")]
    scalar_keys = [k for k in obs_keys if not k.endswith("_rgb")]

    # Read shapes from first demo
    shapes: dict[str, tuple[int, ...]] = {}
    for k in obs_keys:
        shapes[k] = first[f"obs/{k}"].shape[1:]  # skip T dimension

    # Action shape (actions may be a Dataset or a Group containing "action")
    if "actions" in first:
        actions_node = first["actions"]
        if isinstance(actions_node, h5py.Dataset):
            action_shape = actions_node.shape[1:]
        elif isinstance(actions_node, h5py.Group) and "action" in actions_node:
            action_shape = actions_node["action"].shape[1:]
        else:
            action_shape = (0,)
    else:
        action_shape = (0,)

    afmt = attrs.get("action_format", "abs_joint")
    if afmt == "absolute_joint_target_8d":
        # my_lift collect_vla_data.py: 7 joint targets + gripper, same as abs_joint
        afmt = "abs_joint"

    fps_val = attrs.get("fps")
    if fps_val is None:
        fps_val = attrs.get("control_freq_hz")
    if fps_val is None:
        fps_val = 30

    meta = {
        "demos": demos,
        "n_demos": len(demos),
        "obs_keys": obs_keys,
        "cam_keys": cam_keys,
        "scalar_keys": scalar_keys,
        "shapes": shapes,
        "action_shape": action_shape,
        # From collect_vla_data.py attrs
        "state_format": attrs.get("state_format", "joint"),
        "action_format": afmt,
        "orientation_rep": attrs.get("orientation_rep", "quat"),
        "control_mode": attrs.get("control_mode", "joint_pos"),
        "fps": float(fps_val),
        "prompt": attrs.get("prompt", LEGACY_DEFAULT_PROMPT),
        "task": attrs.get("task", "unknown"),
        "n_arm_joints": int(attrs.get("n_arm_joints", 7)),
    }
    return meta


# ---------------------------------------------------------------------------
# Feature / name generation
# ---------------------------------------------------------------------------


def _joint_names(n: int) -> list[str]:
    return [f"j{i + 1}" for i in range(n)]


def _eef_pos_names() -> list[str]:
    return ["x", "y", "z"]


def _orient_names(orient_rep: str) -> list[str]:
    if orient_rep == "euler":
        return ["roll", "pitch", "yaw"]
    return ["qw", "qx", "qy", "qz"]


def _build_state_names(meta: dict) -> list[str]:
    """Build ordered state dimension names from metadata."""
    names: list[str] = []
    sfmt = meta["state_format"]
    if sfmt in ("joint", "both"):
        names += _joint_names(meta["n_arm_joints"])
    if sfmt in ("eef_pose", "both"):
        names += _eef_pos_names()
        names += _orient_names(meta["orientation_rep"])
    names.append("gripper")
    return names


def _build_action_names(meta: dict) -> list[str]:
    """Build ordered action dimension names from metadata.

    For rel_eef_pose, the stored action may be 7D (dx,dy,dz + 3D rotation delta + gripper)
    as in ik_rel / SpaceMouse, or 8D if rotation delta is quaternion. We align names
    with the actual action_shape from HDF5.
    """
    afmt = meta["action_format"]
    action_len = int(meta["action_shape"][0]) if meta["action_shape"] else 0
    names: list[str] = []
    if afmt in ("abs_joint", "rel_joint"):
        prefix = "" if afmt == "abs_joint" else "d_"
        names += [f"{prefix}j{i + 1}" for i in range(meta["n_arm_joints"])]
    elif afmt in ("abs_eef_pose", "rel_eef_pose"):
        prefix = "" if afmt == "abs_eef_pose" else "d_"
        names += [f"{prefix}{c}" for c in _eef_pos_names()]
        orient_rep = meta["orientation_rep"]
        # rel_eef_pose with 7D total = 3 pos + 3 rot (euler/axis-angle) + 1 gripper
        if afmt == "rel_eef_pose" and action_len == 7:
            names += ["d_rx", "d_ry", "d_rz"]
        else:
            names += [f"{prefix}{c}" for c in _orient_names(orient_rep)]
    names.append("gripper")
    return names


def _build_state_vector(demo_obs: h5py.Group, idx: int, meta: dict) -> np.ndarray:
    """Assemble a single state vector from HDF5 obs at frame idx."""
    parts: list[np.ndarray] = []
    sfmt = meta["state_format"]
    if sfmt in ("joint", "both") and "joint_positions" in demo_obs:
        parts.append(demo_obs["joint_positions"][idx].astype(np.float32))
    if sfmt in ("eef_pose", "both"):
        if "eef_pos" in demo_obs:
            parts.append(demo_obs["eef_pos"][idx].astype(np.float32))
        if "eef_orient" in demo_obs:
            parts.append(demo_obs["eef_orient"][idx].astype(np.float32))
    if "gripper_state" in demo_obs:
        parts.append(demo_obs["gripper_state"][idx].astype(np.float32))
    if not parts:
        raise ValueError(f"No state data found for state_format={sfmt}")
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(
    hdf5_path: str,
    output_dir: str,
    target_fps: float | None,
    success_only: bool,
    robot_type: str,
) -> None:
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        meta = _read_meta(f)
        # Legacy HDF5 files (or runs that didn't pass --prompt) carry the
        # placeholder prompt.  Upgrade it to a task-specific string so that
        # every LeRobot episode gets a meaningful instruction.
        if not meta["prompt"] or meta["prompt"] == LEGACY_DEFAULT_PROMPT:
            resolved = resolve_prompt_from_task(meta["task"], default=LEGACY_DEFAULT_PROMPT)
            if resolved != meta["prompt"]:
                print(
                    f"[INFO] Upgrading legacy prompt for task '{meta['task']}': "
                    f"'{meta['prompt']}' -> '{resolved}'"
                )
            meta["prompt"] = resolved
        output_dir = _resolve_output_dir(
            meta["task"],
            output_dir,
            Path(__file__).resolve().parent / "lerobot_dataset",
        )

        source_fps = meta["fps"]
        if target_fps is None or target_fps >= source_fps:
            skip = 1
            out_fps = int(source_fps)
        else:
            skip = max(1, round(source_fps / target_fps))
            out_fps = int(round(source_fps / skip))

        state_names = _build_state_names(meta)
        action_names = _build_action_names(meta)
        state_dim = len(state_names)
        action_dim = len(action_names)

        print(f"Task           : {meta['task']}")
        print(f"State format   : {meta['state_format']}")
        print(f"Action format  : {meta['action_format']}")
        print(f"Orientation    : {meta['orientation_rep']}")
        print(f"Control mode   : {meta['control_mode']}")
        print(f"Demos          : {meta['n_demos']}")
        print(f"Source FPS     : {source_fps}")
        print(f"Target FPS     : {out_fps} (skip={skip})")
        print(f"State dim      : {state_dim}  {state_names}")
        print(f"Action dim     : {action_dim}  {action_names}")
        print(f"Cameras        : {meta['cam_keys'] or 'none'}")

        # Build LeRobot features
        features: dict = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": action_names,
            },
        }

        for cam_key in meta["cam_keys"]:
            h, w, c = meta["shapes"][cam_key]
            cam_label = cam_key.replace("_rgb", "")
            features[f"observation.images.{cam_label}"] = {
                "dtype": "video",
                "shape": (c, h, w),
                "names": ["channels", "height", "width"],
            }

        dataset = LeRobotDataset.create(
            repo_id=output_dir,
            fps=out_fps,
            robot_type=robot_type,
            features=features,
        )

        converted = 0
        skipped = 0
        for demo_key in tqdm(meta["demos"], desc="Converting"):
            demo = f[f"data/{demo_key}"]

            if success_only and not demo.attrs.get("success", True):
                skipped += 1
                continue

            obs = demo["obs"]

            # Read actions
            if "actions" in demo and isinstance(demo["actions"], h5py.Dataset):
                raw_actions = demo["actions"][:]
            elif "actions" in demo and "action" in demo["actions"]:
                raw_actions = demo["actions/action"][:]
            else:
                print(f"  [WARN] {demo_key}: no actions found, skipping")
                skipped += 1
                continue

            num_frames = raw_actions.shape[0]
            frame_indices = list(range(0, num_frames, skip))

            for i in frame_indices:
                frame_data: dict = {}

                # State
                state = _build_state_vector(obs, i, meta)
                frame_data["observation.state"] = state

                # Action
                # Keep the action aligned with the current frame:
                # input = state/image at frame i
                # target = recorded RL intent issued at frame i
                frame_data["action"] = raw_actions[i].astype(np.float32)

                # Cameras
                for cam_key in meta["cam_keys"]:
                    cam_label = cam_key.replace("_rgb", "")
                    img = obs[cam_key][i]
                    if img.ndim == 3 and img.shape[-1] in (3, 4):
                        img = img[..., :3].transpose(2, 0, 1)
                    frame_data[f"observation.images.{cam_label}"] = img

                frame_data["task"] = meta["prompt"]
                dataset.add_frame(frame_data)

            dataset.save_episode()
            converted += 1

        dataset.finalize()
        print(f"\nDone: {converted} episodes converted, {skipped} skipped")
        print(f"Output: {output_dir}")
        print(f"FPS: {out_fps}Hz")


def _build_action_from_skip(
    obs: h5py.Group,
    raw_actions: np.ndarray,
    cur_i: int,
    tgt_i: int,
    meta: dict,
) -> np.ndarray:
    """Return the action recorded at the current sampled frame.

    The collector already stores the final action sent to the environment for
    each frame. For ACT training we must preserve this current-frame alignment
    instead of rebuilding an action from future observations.
    """
    del obs, tgt_i, meta
    return raw_actions[cur_i].astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert collect_vla_data.py HDF5 to LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 path")
    parser.add_argument("--output", type=str, required=True, help="LeRobot repo_id / output dir")
    parser.add_argument(
        "--target_fps", type=float, default=None,
        help="Target FPS for LeRobot dataset; default = use HDF5 source fps",
    )
    parser.add_argument(
        "--success_only", action="store_true",
        help="Only convert episodes marked as success",
    )
    parser.add_argument("--robot_type", type=str, default="franka", help="Robot type label")
    args = parser.parse_args()

    convert(
        hdf5_path=args.input,
        output_dir=args.output,
        target_fps=args.target_fps,
        success_only=args.success_only,
        robot_type=args.robot_type,
    )
