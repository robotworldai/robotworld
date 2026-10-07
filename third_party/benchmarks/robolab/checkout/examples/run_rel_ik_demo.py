# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# isort: skip_file
import argparse
import cv2  # Must import before isaaclab. Do not remove.
import sys
import traceback
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Relative IK controller demo.")
parser.add_argument("--task", type=str, default="BananaInBowlTask")
parser.add_argument("--num_steps", type=int, default=720)
parser.add_argument("--delta", type=float, default=0.02, help="Delta per step for translation (m).")
parser.add_argument("--rot_scale", type=float, default=2.0, help="Multiplier applied to --delta for rotation DOFs.")
parser.add_argument("--phase_steps", type=int, default=60, help="Steps per +/- phase per DOF.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa
from robolab.constants import TASK_DIR  # noqa
from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs  # noqa
from robolab.core.environments.runtime import create_env, end_episode  # noqa
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg  # noqa
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa
from robolab.robots.droid import DroidCfg, DroidRelIKActionCfg, ProprioceptionObservationCfg, WristCameraCfg, contact_gripper  # noqa
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg  # noqa
from robolab.variations.camera import EgocentricMirroredCameraCfg  # noqa
from robolab.variations.lighting import SphereLightCfg  # noqa

robolab.constants.VERBOSE = False
robolab.constants.RECORD_IMAGE_DATA = False


def register_rel_ik_envs(task: str):
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({"image_obs": ImageObsCfg(), "proprio_obs": ProprioceptionObservationCfg(), "viewport_cam": ViewportCameraCfg()})
    scene_cameras = [c for c in WRIST_LEFT if c is not WristCameraCfg]
    auto_discover_and_create_cfgs(task_dir=TASK_DIR, task_subdirs=["benchmark"], tasks=task, pattern="*.py", env_prefix="", env_postfix="RelIK", observations_cfg=ObservationCfg(), actions_cfg=DroidRelIKActionCfg(), robot_cfg=DroidCfg, camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg], lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper, dt=1 / (60 * 2), render_interval=8, decimation=8, seed=1)


def main():
    register_rel_ik_envs(args_cli.task)
    task_envs = get_envs(task=args_cli.task)
    if not task_envs:
        print(f"No environments found for task '{args_cli.task}'. Check --task name.")
        simulation_app.close()
        return
    env_name = task_envs[0]
    print(f"Running environment: {env_name}")
    print(f"delta={args_cli.delta}  rot_scale={args_cli.rot_scale}  phase_steps={args_cli.phase_steps}")
    env, _ = create_env(env_name, num_envs=1, use_fabric=True)
    obs, _ = env.reset()
    dof_names = ["x", "y", "z", "euler_x", "euler_y", "euler_z"]
    dof_deltas = [args_cli.delta] * 3 + [args_cli.delta * args_cli.rot_scale] * 3
    steps_per_dof = 2 * args_cli.phase_steps
    cycle_len = len(dof_names) * steps_per_dof
    action = torch.zeros(1, 7, device=env.device)
    for step in range(args_cli.num_steps):
        cycle_step = step % cycle_len
        dof_idx = cycle_step // steps_per_dof
        within_dof = cycle_step % steps_per_dof
        direction = 1 if within_dof < args_cli.phase_steps else -1
        if within_dof == 0:
            print(f"[{step:04d}] --- DOF {dof_idx} ({dof_names[dof_idx]}) {'+ ' if direction == 1 else '- '} ---")
        action.zero_()
        action[0, dof_idx] = direction * dof_deltas[dof_idx]
        obs, _, term, trunc, _ = env.step(action)
        proprio = obs.get("proprio_obs", {})
        ee_pos = proprio.get("ee_pos")
        ee_quat = proprio.get("ee_quat")
        if ee_pos is not None and ee_quat is not None:
            p = ee_pos[0].cpu()
            q = ee_quat[0].cpu()
            print(f"[{step:04d}] pos({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})  quat({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})")
        if term or trunc:
            obs, _ = env.reset()
    end_episode(env)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
