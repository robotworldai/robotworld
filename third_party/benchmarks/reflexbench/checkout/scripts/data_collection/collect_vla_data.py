#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""VLA data collection for ReflexBench tasks.

Actions recorded in the dataset are the **raw values sent to the controller**
(after action_scale), NOT post-hoc state differences.  This guarantees that a
VLA trained on this data can feed its outputs directly to the same controller
at eval time and reproduce the recorded behaviour.

The action format is **automatically determined** by the ``--control`` mode:

=============  ====================  =========================
--control      Action format         Dimensions
=============  ====================  =========================
joint_pos      abs_joint             N_arm + 1 gripper
ik_abs         abs_eef_pose          7 (x,y,z, roll,pitch,yaw euler) + 1
ik_rel         rel_eef_pose          6 (dx,dy,dz, roll,pitch,yaw) + 1
=============  ====================  =========================

Observations use **quaternion (w, x, y, z)** for EEF orientation. Actions
match the controller: ik_abs uses euler (roll, pitch, yaw) for rotation;
ik_rel uses euler (roll, pitch, yaw) for the rotation delta.

State (observations) is always in the **same format as the action**:
- ``joint_pos`` -> state = joint_positions + gripper
- ``ik_abs`` / ``ik_rel`` -> state = eef_pos + eef_orient (quat) + gripper

Supports:
- Any registered ReflexBench task (via --task)
- Configurable robot arm control frequency / dataset FPS (via --fps).
- Optional camera image recording (auto-detected or specified)
- Policy modes: trained checkpoint (skrl), random, zero, task-specific planning
  (cuRobo), or teleoperation (SpaceMouse, IK relative control). Conveyor Belt
  Pick-and-Place requires a checkpoint and does not support planning.

HDF5 output layout::

    /attrs  -> task, prompt, control_mode, action_format, state_format, fps, ...
    /data/demo_0/
        obs/
            gripper_state   (T, 1)          # always
            joint_positions (T, N_arm)      # only if control=joint_pos
            eef_pos         (T, 3)          # only if control=ik_abs|ik_rel
            eef_orient      (T, 4)          # quaternion (w,x,y,z)
            <cam>_rgb       (T, H, W, 3)   # per camera
        actions             (T, D)          # raw controller input (see table)
        attrs: num_samples, success

Examples::

    # Trained IK-rel checkpoint, 10 FPS
    python collect_vla_data.py \\
        --task ConveyorBeltPickAndPlace-Franka-DataCollection-v0 \\
        --policy checkpoint --checkpoint /path/to/model.pt \\
        --control ik_rel --fps 10

    # Random policy, joint pos control
    python collect_vla_data.py \\
        --task ConveyorBeltPickAndPlace-Franka-v0 \\
        --policy random --num_episodes 50 --control joint_pos

    # SpaceMouse teleoperation (auto: num_envs=1, control=ik_rel)
    python collect_vla_data.py \\
        --task ConveyorBeltPickAndPlace-Franka-DataCollection-v0 \\
        --policy teleoperation --num_episodes 20 \\
        --pos_sensitivity 0.4 --rot_sensitivity 0.8
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="VLA data collection for ReflexBench tasks.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

g_task = parser.add_argument_group("Task & policy")
g_task.add_argument("--task", type=str, required=True, help="Registered task name")
g_task.add_argument(
    "--policy", type=str, default="checkpoint",
    choices=["checkpoint", "random", "zero", "planning", "teleoperation"],
    help="Policy to drive data collection (default: checkpoint). "
         "'planning' uses a task-specific cuRobo policy where supported; "
         "Conveyor Belt Pick-and-Place requires 'checkpoint'. "
         "'teleoperation' uses SpaceMouse for human teleoperation (forces num_envs=1, control=ik_rel).",
)
g_task.add_argument(
    "--planning_speed", type=float, default=1.0,
    help="Planning policy: trajectory speed scale (default: 1.0). "
         ">1 = faster (fewer interpolated points), e.g. 2.0 for ~2x speed.",
)
g_task.add_argument("--checkpoint", type=str, default=None, help="Path to skrl .pt checkpoint")
g_task.add_argument(
    "--control", type=str, default="joint_pos",
    choices=["joint_pos", "ik_abs", "ik_rel"],
    help="Policy control mode matching the environment and checkpoint "
         "(joint_pos: 7+1, ik_abs: 6+1 abs pose, ik_rel: 6+1 rel pose). Default: joint_pos",
)

g_col = parser.add_argument_group("Collection")
g_col.add_argument("--num_envs", type=int, default=20)
g_col.add_argument("--num_episodes", type=int, default=210, help="Total episodes to save")
g_col.add_argument("--output", type=str, default=None, help="Output HDF5 path (auto if omitted)")
g_col.add_argument("--seed", type=int, default=42, help="Base random seed for collector-managed resets")
g_col.add_argument("--save_failed", action="store_true", help="Also save failed episodes")
g_col.add_argument(
    "--prompt", type=str, default=None,
    help="Language instruction written to HDF5 attrs['prompt']. "
         "If omitted, auto-filled from scripts/data_collection/task_prompts.py "
         "keyed by --task; falls back to 'manipulate the object' if unknown.",
)
g_col.add_argument(
    "--balanced_bins", type=int, default=7,
    help="If > 0, balance episodes across Y-axis bins (e.g. 8 bins over conveyor range). "
         "Ensures uniform coverage of object starting positions.",
)
g_col.add_argument(
    "--balanced_y_range", type=str, default="0.0,0.4",
    help="Y-range for balanced binning as 'min,max' (default: 0.0,0.4).",
)
g_fmt = parser.add_argument_group("Dataset format")
g_fmt.add_argument(
    "--fps", type=float, default=None,
    help="Robot arm control frequency AND dataset FPS (Hz). "
    "Adjusts decimation while keeping physics dt unchanged. "
    "Default: use task config (e.g. 50 Hz).",
)

g_fmt.add_argument(
    "--action_scale", type=float, default=1.0,
    help="Scale applied to the RL policy's arm output before env.step(). "
         "The SCALED action is what gets recorded (= actual controller input). "
         "Does NOT affect gripper. Default: 1.0",
)

g_teleop = parser.add_argument_group("Teleoperation (SpaceMouse)")
g_teleop.add_argument(
    "--pos_sensitivity", type=float, default=0.4,
    help="SpaceMouse position sensitivity (default: 0.4)",
)
g_teleop.add_argument(
    "--rot_sensitivity", type=float, default=0.8,
    help="SpaceMouse rotation sensitivity (default: 0.8)",
)

g_hw = parser.add_argument_group("Robot / sensor config")
g_hw.add_argument("--arm_joints", type=str, default="panda_joint.*", help="Arm joint regex")
g_hw.add_argument("--finger_joints", type=str, default="panda_finger.*", help="Finger joint regex")
g_hw.add_argument("--ee_frame", type=str, default="ee_frame", help="EE FrameTransformer name in scene")
g_hw.add_argument("--cam_names", type=str, default="", help="Cameras (comma-sep); empty = auto-detect")
g_hw.add_argument("--disable_fabric", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
task_key = args_cli.task.lower().replace("-", "_")
if args_cli.policy == "planning" and "conveyorbeltpickandplace" in task_key.replace("_", ""):
    parser.error(
        "Conveyor Belt Pick-and-Place does not support --policy planning; "
        "use --policy checkpoint with --checkpoint instead."
    )
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-app imports
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import FrameTransformer  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reflexbench.tasks  # noqa: F401, E402

from task_prompts import resolve_prompt as _resolve_task_prompt  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def finalize_completed_episode(collector: VLADataCollector, eid: int, hf: h5py.File, terminal_phase: int):
    """
    Handle the end of an episode: success determination, balanced binning, and saving.
    """
    # 1. Determine success. WhackAMole's phase 4 only means the popup window ended;
    # success requires at least one registered hit.
    if "whackamole" in collector.task.lower() or "whack_a_mole" in collector.task.lower():
        terminal_hits = -1
        if hasattr(collector.menv, "_terminal_valid_hits"):
            terminal_hits = int(collector.menv._terminal_valid_hits[eid].item())
        success = terminal_hits > 0
    else:
        success = terminal_phase == collector.success_terminal_phase

    # 2. Balanced binning logic
    save_this = True
    if success and collector.balanced_bins > 0:
        obj_y = collector._get_episode_start_y(eid)
        # Find which bin this starting position falls into
        bin_idx = np.clip(
            np.searchsorted(collector.bin_edges[1:], obj_y), 0, collector.balanced_bins - 1
        )
        if collector.bin_counts[bin_idx] >= collector.target_per_bin:
            # This bin is already full, we don't need more samples from here
            success = False
            save_this = False
            # print(f"  [SKIP] bin {bin_idx} is full (y={obj_y:.3f})")

    # 3. Save to HDF5
    if save_this and (success or collector.save_failed):
        collector._save(eid, hf, success)
        # Increment bin count if it was a successful balanced sample
        if success and collector.balanced_bins > 0:
            obj_y = collector._get_episode_start_y(eid)
            bin_idx = np.clip(
                np.searchsorted(collector.bin_edges[1:], obj_y), 0, collector.balanced_bins - 1
            )
            collector.bin_counts[bin_idx] += 1
    else:
        if not success:
            # Optional: more descriptive failure logging
            # print(f"  [SKIP] env {eid} failed (terminal_phase={terminal_phase})")
            pass

    # 4. Reset buffers for the next episode in this environment
    collector._reset_buf(eid)


def _load_skrl_cfg(task_name: str) -> dict:
    """Resolve ``skrl_cfg_entry_point`` from the gym registry and load the YAML.

    If this task has no skrl_cfg_entry_point (e.g. DataCollection variant),
    tries a base task: e.g. "Foo-DataCollection-v0" -> "Foo-v0".
    """
    def _get_entry(name: str) -> str | None:
        try:
            spec = gym.spec(name)
            return (spec.kwargs or {}).get("skrl_cfg_entry_point")
        except Exception:
            return None

    entry = _get_entry(task_name)
    if entry is None:
        base = task_name.replace("-DataCollection-v0", "-v0").replace("-Play-v0", "-v0")
        if base != task_name:
            entry = _get_entry(base)
            if entry is not None:
                pass
    if entry is None:
        raise ValueError(
            f"Task '{task_name}' has no skrl_cfg_entry_point. Use --policy random/zero."
        )
    module_path, filename = entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    module_file = getattr(mod, "__file__", None)
    if module_file is None:
        raise ValueError(f"Unable to resolve skrl config module file for '{module_path}'")
    cfg_path = os.path.join(os.path.dirname(module_file), filename)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# Control mode -> action format mapping.
# The action format is fully determined by which controller the env uses;
# recording anything else would break eval-time consistency.
_CONTROL_TO_ACTION_FMT: dict[str, str] = {
    "joint_pos": "abs_joint",
    "ik_abs": "abs_eef_pose",
    "ik_rel": "rel_eef_pose",
}


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


def _resolve_output_path(task_name: str, requested_output: str | None, base_dir: Path, default_name: str) -> str:
    if requested_output is None:
        return str(base_dir / _task_slug(task_name) / default_name)

    output_path = Path(requested_output)
    if requested_output.startswith(".") or output_path.is_absolute() or output_path.parent != Path("."):
        return str(output_path)

    return str(base_dir / _task_slug(task_name) / output_path.name)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class VLADataCollector:
    """Generic VLA data collector for any ReflexBench task."""

    ARM_ACTION_SCALE = 0.1
    PHASE3_GRIPPER_DELAY = 3

    def __init__(self, args: argparse.Namespace):
        self.task: str = args.task
        self.policy_mode: str = args.policy
        self.checkpoint: str | None = args.checkpoint
        self.num_envs: int = args.num_envs
        self.num_episodes: int = args.num_episodes
        self.output_path: str = _resolve_output_path(
            self.task,
            args.output,
            Path(__file__).resolve().parent / "collected_data",
            f"{args.task.replace('-', '_').lower()}.hdf5",
        )
        self.seed: int = args.seed
        self.save_failed: bool = args.save_failed
        # Resolve language prompt: user-specified overrides the per-task table.
        self.prompt: str = _resolve_task_prompt(self.task, args.prompt)
        self.target_fps: float | None = args.fps
        self.arm_pattern: str = args.arm_joints
        self.finger_pattern: str = args.finger_joints
        self.ee_frame_name: str = args.ee_frame
        self.cam_arg: str = args.cam_names
        self.control_mode: str = args.control
        self.action_scale: float = args.action_scale
        self.planning_speed: float = args.planning_speed
        self.pos_sensitivity: float = args.pos_sensitivity
        self.rot_sensitivity: float = args.rot_sensitivity
        # Balanced binning
        task_name = self.task.lower()
        if "ballthrowing" in task_name or "ball_throwing" in task_name:
            self.success_terminal_phase = 2
        else:
            self.success_terminal_phase = 4
        use_balanced_binning = (
            ("pickplace" in task_name or "pick_place" in task_name)
            and "conveyor" in task_name
        )
        self.balanced_bins: int = args.balanced_bins if use_balanced_binning else 0
        if self.balanced_bins > 0:
            y_parts = args.balanced_y_range.split(",")
            self.y_min, self.y_max = float(y_parts[0]), float(y_parts[1])
        else:
            self.y_min, self.y_max = 0.0, 0.4

        # if args.balanced_bins > 0 and not use_balanced_binning:
        #     print(f"[INFO] Balanced binning disabled for task: {self.task}")

        # Teleoperation forces single-env IK relative control
        if self.policy_mode == "teleoperation":
            if self.num_envs != 1:
                # print(f"[INFO] Teleoperation: overriding num_envs {self.num_envs} -> 1")
                self.num_envs = 1
            if self.control_mode != "ik_rel":
                # print(f"[INFO] Teleoperation: overriding control '{self.control_mode}' -> 'ik_rel'")
                self.control_mode = "ik_rel"

        # Action and state format are determined by the control mode - state
        # matches action so that eval uses the same controller and observation space.
        self.action_fmt: str = _CONTROL_TO_ACTION_FMT[self.control_mode]
        self.state_fmt: str = "joint" if self.control_mode == "joint_pos" else "eef_pose"

        if self.policy_mode == "checkpoint" and not self.checkpoint:
            raise ValueError("--checkpoint is required when --policy=checkpoint")

        self.needs_eef: bool = self.control_mode in ("ik_abs", "ik_rel")

        # filled by setup()
        self.bufs: dict[int, dict] = {}
        self.max_phase: dict[int, int] = {}
        self.saved: int = 0
        self.attempts: int = 0
        self._episode_index: int = 0
        self.phase3_step_counter: torch.Tensor | None = None
        self.phase3_start_steps: dict[int, int | None] = {}
        self.ik_controller = None
        self.ee_body_idx: int | None = None
        self.ee_jacobi_idx: int | None = None
        self.box_target_pose: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        env_cfg = parse_env_cfg(
            self.task,
            device=args_cli.device,
            num_envs=self.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )

        # Override arm_action to match the trained policy's control mode
        self._apply_control_mode(env_cfg)

        # --fps directly controls robot arm control frequency.
        # Adjust decimation while keeping sim.dt (physics rate) unchanged.
        # Cameras, events and other simulation components still update at the
        # simulator cadence; we do not override render_interval here.
        if self.target_fps is not None:
            new_dec = max(1, round(1.0 / (env_cfg.sim.dt * self.target_fps)))
            old_freq = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
            env_cfg.decimation = new_dec
            if hasattr(env_cfg, "robot_control_freq"):
                env_cfg.robot_control_freq = self.target_fps
            actual_freq = 1.0 / (env_cfg.sim.dt * new_dec)
            # print(
            #     f"[INFO] --fps {self.target_fps} -> control freq: "
            #     f"{old_freq:.1f} Hz -> {actual_freq:.1f} Hz "
            #     f"(decimation={new_dec}, sim_dt={env_cfg.sim.dt})"
            # )

        env_cfg.sim.render_interval = 1

        self.sim_dt: float = env_cfg.sim.dt
        self.decimation: int = env_cfg.decimation
        self.ctrl_freq: float = 1.0 / (self.sim_dt * self.decimation)
        self.actual_fps: float = self.ctrl_freq

        self.base_env = gym.make(self.task, cfg=env_cfg)
        self.menv: ManagerBasedRLEnv = self.base_env.unwrapped

        # For conveyor_belt_pick_and_place: success = object in box (not max_phase)
        self._success_object_in_box = None
        if "pickplace" in self.task.lower():
            try:
                from reflexbench.tasks.manager_based.conveyor_belt_pick_and_place.mdp.terminations import (
                    object_in_box,
                )
                self._success_object_in_box = object_in_box
            except Exception:
                pass

        # Detect joints
        robot = self.menv.scene["robot"]
        self.arm_ids = robot.find_joints([self.arm_pattern])[0]
        self.finger_ids = robot.find_joints([self.finger_pattern])[0]
        self.n_arm: int = len(self.arm_ids)

        # Check EEF availability when using IK control
        if self.needs_eef and self.ee_frame_name not in self.menv.scene.keys():
            raise RuntimeError(
                f"EE frame '{self.ee_frame_name}' not in scene "
                f"{list(self.menv.scene.keys())}. "
                "Use --control joint_pos or specify correct --ee_frame."
            )

        # Cameras
        self.cam_names: list[str] = self._detect_cameras()

        # Policy
        if self.policy_mode == "checkpoint":
            self._setup_checkpoint()
        elif self.policy_mode == "planning":
            self._setup_planning()
        elif self.policy_mode == "teleoperation":
            self._setup_teleoperation()
        self.step_env = (
            self.wrapped_env if self.policy_mode == "checkpoint" else self.base_env
        )

        # Action dim from the actual env action space (guaranteed consistent)
        self.action_dim: int = self.base_env.action_space.shape[-1]

        self.state_dims: dict[str, int] = {"gripper_state": 1}
        if self.state_fmt == "joint":
            self.state_dims["joint_positions"] = self.n_arm
        else:
            self.state_dims["eef_pos"] = 3
            self.state_dims["eef_orient"] = 4

        for i in range(self.num_envs):
            self._reset_buf(i)

        # print(f"[INFO] Arm joints ({self.n_arm}): ids={list(self.arm_ids)}")
        # print(f"[INFO] Finger joints ({len(self.finger_ids)}): ids={list(self.finger_ids)}")
        # print(f"[INFO] Cameras: {self.cam_names or 'none'}")
        # print(
        #     f"[INFO] Ctrl freq={self.ctrl_freq:.1f}Hz = dataset FPS"
        # )
        # print(f"[INFO] Action dim={self.action_dim}  State dims={self.state_dims}")

        # Balanced binning setup
        if self.balanced_bins > 0:
            n = self.balanced_bins
            self.bin_edges = np.linspace(self.y_min, self.y_max, n + 1)
            self.bin_counts = np.zeros(n, dtype=int)
            self.target_per_bin = max(1, self.num_episodes // n)
            # print(
            #     f"[INFO] Balanced binning: {n} bins over y=[{self.y_min}, {self.y_max}], "
            #     f"target={self.target_per_bin}/bin"
            # )

    def _reset_step_env(self):
        return self.step_env.reset()

    def _init_phase3_ik_override(self) -> None:
        """Initialize the phase-3 IK override used by the original working pipeline."""
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        robot = self.menv.scene["robot"]
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        )
        self.ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=self.menv.device
        )
        ee_body_idx = int(robot.find_bodies("panda_hand")[0][0])
        self.ee_body_idx = ee_body_idx
        self.ee_jacobi_idx = ee_body_idx - 1
        self.box_target_pose = torch.tensor(
            [0.0, -0.4, 0.3, 0.0, 1.0, 0.0, 0.0],
            device=self.menv.device,
        ).unsqueeze(0)
        self.phase3_step_counter = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.menv.device
        )

    def _compute_phase3_ik_targets(self) -> torch.Tensor:
        """Compute absolute joint targets for the phase-3 box-above release pose."""
        if (
            self.ik_controller is None
            or self.box_target_pose is None
            or self.ee_jacobi_idx is None
        ):
            raise RuntimeError("Phase-3 IK override requested before IK setup.")

        robot = self.menv.scene["robot"]
        ee_frame: FrameTransformer = self.menv.scene[self.ee_frame_name]

        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pos_w,
            ee_quat_w,
        )

        jacobian = robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.arm_ids
        ]
        current_joint_pos = robot.data.joint_pos[:, self.arm_ids]

        target_pose = self.box_target_pose.expand(self.num_envs, -1)
        self.ik_controller.set_command(target_pose)
        return self.ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, current_joint_pos
        )

    def _apply_control_mode(self, env_cfg) -> None:
        """Override env_cfg.actions.arm_action to match the trained checkpoint's control mode."""
        if self.control_mode == "joint_pos":
            return  # default for most configs, no change needed

        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        env_cfg.scene.robot.spawn.activate_contact_sensors = True

        if self.control_mode == "ik_abs":
            env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.107]
                ),
            )
            # print("[INFO] Control mode: IK absolute pose (action dim = 6 + 1 gripper)")
        elif self.control_mode == "ik_rel":
            env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=True,
                    ik_method="dls",
                ),
                scale=1.0,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.107]
                ),
            )
            # print("[INFO] Control mode: IK relative pose (action dim = 6 + 1 gripper)")

    def _detect_cameras(self) -> list[str]:
        if self.cam_arg:
            requested = [n.strip() for n in self.cam_arg.split(",") if n.strip()]
            valid = [n for n in requested if n in self.menv.scene.keys()]
            if len(valid) < len(requested):
                missing = set(requested) - set(valid)
                # print(f"[WARN] Cameras not found in scene: {missing}")
                pass
            return valid
        found = []
        for key in self.menv.scene.keys():
            ent = self.menv.scene[key]
            if (
                hasattr(ent, "data")
                and hasattr(ent.data, "output")
                and isinstance(ent.data.output, dict)
                and "rgb" in ent.data.output
            ):
                found.append(key)
        return found

    def _setup_checkpoint(self) -> None:
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from skrl.utils.runner.torch import Runner

        self.wrapped_env = SkrlVecEnvWrapper(self.base_env)
        cfg = _load_skrl_cfg(self.task)
        cfg["trainer"]["close_environment_at_exit"] = False
        cfg["agent"]["experiment"]["write_interval"] = 0
        cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        self.runner = Runner(self.wrapped_env, cfg)
        # print(f"[INFO] Loading checkpoint: {self.checkpoint}")
        self.runner.agent.load(self.checkpoint)
        self.runner.agent.set_running_mode("eval")

        self.menv.enable_ik_override = False
        os.environ.pop("PHASE3_GRIPPER_OPEN", None)
        if self.control_mode == "joint_pos":
            self._init_phase3_ik_override()
            # print(
            #     "[INFO] Checkpoint collection: phase 0-2 use RL; "
            #     "phase 3 uses IK arm target + delayed gripper release"
            # )
        else:
            os.environ["PHASE3_GRIPPER_OPEN"] = "1"
            # print(
            #     "[INFO] Checkpoint collection: phase 0-2 use RL; "
            #     "phase 3 only forces delayed gripper release"
            # )

    def _setup_teleoperation(self) -> None:
        from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg

        self._teleop_discard = False

        def _on_discard():
            self._teleop_discard = True
            # print("[TELEOP] Right button pressed - episode will be discarded & env reset")

        self.teleop_interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
        )
        self.teleop_interface.add_callback("R", _on_discard)
        # Disable env's IK override so action stays 7D (ik_rel) and human has full control
        self.menv.enable_ik_override = False
        # print(f"[INFO] SpaceMouse teleoperation ready")
        # print(f"  Left  button: toggle gripper (open/close)")
        # print(f"  Right button: discard current episode & reset env")
        # print(f"  Pos sensitivity: {self.pos_sensitivity}  Rot sensitivity: {self.rot_sensitivity}")

    def _setup_planning(self) -> None:
        self.menv.enable_ik_override = False
        ctrl_dt = self.sim_dt * self.decimation
        task_name = self.task.lower()
        if "ballthrowing" in task_name or "ball_throwing" in task_name:
            from planning_policy import BallThrowingPlanningPolicy

            policy_cls = BallThrowingPlanningPolicy
            policy_name = "ball-throwing swing"
        elif "ballcatch" in task_name or "ball_catching" in task_name:
            from planning_policy import BallCatchingPlanningPolicy

            policy_cls = BallCatchingPlanningPolicy
            policy_name = "ball catching intercept"
        elif "rotatingpeginsertion" in task_name or "rotating_peg_insertion" in task_name:
            from planning_policy import RotatingPegInsertionPlanningPolicy

            policy_cls = RotatingPegInsertionPlanningPolicy
            policy_name = "rotating peg insertion hover-descend"
        elif "rollingballinterception" in task_name or "rolling_ball_interception" in task_name:
            from planning_policy import RollingBallInterceptionPlanningPolicy

            policy_cls = RollingBallInterceptionPlanningPolicy
            policy_name = "rolling-ball interception"
        elif "whackamole" in task_name or "whack_a_mole" in task_name:
            from planning_policy import WhackAMolePlanningPolicy

            policy_cls = WhackAMolePlanningPolicy
            policy_name = "whack-a-mole reactive strike"
        else:
            raise ValueError(
                f"Task '{self.task}' does not have a planning policy. "
                "Use a supported planning task or select another --policy mode."
            )

        self.planning_policy = policy_cls(
            self.menv,
            ctrl_dt=ctrl_dt,
            control_mode=self.control_mode,
            speed_scale=self.planning_speed,
        )
        # print(
        #     f"[INFO] Planning policy: {policy_name}  "
        #     f"(control={self.control_mode}, ctrl_dt={ctrl_dt:.4f}s, speed_scale={self.planning_speed})"
        # )

    # ------------------------------------------------------------------ #
    # Buffers
    # ------------------------------------------------------------------ #

    def _reset_buf(self, eid: int) -> None:
        self.bufs[eid] = {
            "joint_pos": [],
            "gripper": [],
            "eef_pos": [],
            "eef_quat": [],
            "actions": [],
        }
        for c in self.cam_names:
            self.bufs[eid][f"{c}_rgb"] = []
        self.max_phase[eid] = 0
        self.phase3_start_steps[eid] = None
        if self.phase3_step_counter is not None:
            self.phase3_step_counter[eid] = 0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def _get_episode_start_y(self, eid: int) -> float:
        """Get the object Y position (local) for balanced binning."""
        try:
            obj = self.menv.scene["object"]
            obj_y = obj.data.root_pos_w[eid, 1].item()
            origin_y = self.menv.scene.env_origins[eid, 1].item()
            return obj_y - origin_y
        except Exception:
            return 0.0

    def _gripper_state(self, eid: int) -> float:
        robot = self.menv.scene["robot"]
        mean_finger = robot.data.joint_pos[eid, self.finger_ids].mean().item()
        return 1.0 if mean_finger > 0.03 else 0.0

    def _record(self, eid: int) -> None:
        buf = self.bufs[eid]
        robot = self.menv.scene["robot"]

        buf["joint_pos"].append(
            robot.data.joint_pos[eid, self.arm_ids].cpu().numpy().copy()
        )
        buf["gripper"].append(
            np.array([self._gripper_state(eid)], dtype=np.float32)
        )

        if self.needs_eef:
            ee = self.menv.scene[self.ee_frame_name]
            r_pos = robot.data.root_pos_w[eid : eid + 1]
            r_quat = robot.data.root_quat_w[eid : eid + 1]
            e_pos = ee.data.target_pos_w[eid : eid + 1, 0, :]
            e_quat = ee.data.target_quat_w[eid : eid + 1, 0, :]
            p_b, q_b = subtract_frame_transforms(r_pos, r_quat, e_pos, e_quat)
            buf["eef_pos"].append(p_b[0].cpu().numpy().copy())
            buf["eef_quat"].append(q_b[0].cpu().numpy().copy())

        for c in self.cam_names:
            rgb = (
                self.menv.scene[c]
                .data.output["rgb"][eid][..., :3]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            buf[f"{c}_rgb"].append(rgb)


    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def _save(self, eid: int, hf: h5py.File, success: bool) -> None:
        buf = self.bufs[eid]
        T = len(buf["actions"])
        if T < 1:
            return

        grp = hf.create_group(f"data/demo_{self.saved}")
        obs = grp.create_group("obs")

        obs.create_dataset(
            "gripper_state",
            data=np.array(buf["gripper"][:T]),
            compression="gzip",
        )

        if self.state_fmt == "joint":
            obs.create_dataset(
                "joint_positions",
                data=np.array(buf["joint_pos"][:T]),
                compression="gzip",
            )
        else:
            obs.create_dataset(
                "eef_pos",
                data=np.array(buf["eef_pos"][:T]),
                compression="gzip",
            )
            obs.create_dataset(
                "eef_orient",
                data=np.array(buf["eef_quat"][:T]),
                compression="gzip",
            )

        for c in self.cam_names:
            k = f"{c}_rgb"
            if buf[k]:
                obs.create_dataset(
                    k, data=np.array(buf[k][:T]), compression="gzip"
                )

        grp.create_dataset(
            "actions", data=np.array(buf["actions"]), compression="gzip",
        )
        grp.attrs["num_samples"] = T
        grp.attrs["success"] = success

        self.saved += 1
        print(f"[COLLECT] {self.saved}/{self.num_episodes}")

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #

    def _act(self, obs: torch.Tensor) -> torch.Tensor:
        if self.policy_mode == "checkpoint":
            with torch.inference_mode():
                out = self.runner.agent.act(obs, timestep=0, timesteps=0)
                return out[-1].get("mean_actions", out[0]).detach().clone()
        if self.policy_mode == "planning":
            return self.planning_policy.compute_action()
        if self.policy_mode == "teleoperation":
            raw = self.teleop_interface.advance()
            return raw.unsqueeze(0).to(self.menv.device)
        shape = self.base_env.action_space.shape
        dev = self.menv.device
        if self.policy_mode == "random":
            return 2 * torch.rand(shape, device=dev) - 1
        return torch.zeros(shape, device=dev)

    def _apply_phase3_gripper_override(self, action: torch.Tensor) -> torch.Tensor:
        """Match the original working phase-3 collection semantics."""
        if self.policy_mode != "checkpoint":
            return action
        if not hasattr(self.menv, "task_phase"):
            return action

        phase3_mask = self.menv.task_phase == 3
        if not phase3_mask.any():
            return action

        if self.control_mode == "joint_pos" and self.phase3_step_counter is not None:
            action = action.clone()
            raw_ik_targets = self._compute_phase3_ik_targets()
            robot = self.menv.scene["robot"]
            current_joint_pos = robot.data.joint_pos[:, self.arm_ids]
            ik_rel_action = (raw_ik_targets - current_joint_pos) / self.ARM_ACTION_SCALE
            ik_rel_action = ik_rel_action.clamp(-1.0, 1.0)
            action[phase3_mask, : self.n_arm] = ik_rel_action[phase3_mask]

            self.phase3_step_counter[phase3_mask] += 1
            gripper_open_mask = phase3_mask & (
                self.phase3_step_counter > self.PHASE3_GRIPPER_DELAY
            )
            gripper_hold_mask = phase3_mask & (
                self.phase3_step_counter <= self.PHASE3_GRIPPER_DELAY
            )
            action[gripper_open_mask, -1] = 1.0
            action[gripper_hold_mask, -1] = -1.0
            return action

        if not hasattr(self.menv, "ik_step_counter"):
            return action
        action = action.clone()
        phase3_release = self.menv.ik_step_counter[phase3_mask] >= 3
        action[phase3_mask, -1] = torch.where(
            phase3_release,
            torch.ones_like(action[phase3_mask, -1]),
            -torch.ones_like(action[phase3_mask, -1]),
        )
        return action

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def collect(self) -> None:
        self.setup()
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        hf = h5py.File(self.output_path, "w")
        hf.create_group("data")
        for k, v in {
            "task": self.task,
            "prompt": self.prompt,
            "seed": self.seed,
            "control_mode": self.control_mode,
            "state_format": self.state_fmt,
            "action_format": self.action_fmt,
            "fps": self.actual_fps,
            "sim_dt": self.sim_dt,
            "decimation": self.decimation,
            "control_freq": self.ctrl_freq,
            "action_dim": self.action_dim,
            "action_scale": self.action_scale,
            "arm_joints_pattern": self.arm_pattern,
            "finger_joints_pattern": self.finger_pattern,
            "n_arm_joints": self.n_arm,
        }.items():
            hf.attrs[k] = v
        for name, dim in self.state_dims.items():
            hf.attrs[f"state_dim/{name}"] = dim

        obs, _ = self._reset_step_env()
        if self.policy_mode == "teleoperation":
            self.teleop_interface.reset()

        # print(f"\n{'=' * 60}")
        # print(f"  Task         : {self.task}")
        # print(f"  Policy       : {self.policy_mode}")
        # print(f"  Control mode : {self.control_mode}")
        # print(f"  Episodes     : {self.num_episodes}")
        # print(f"  State format : {self.state_fmt} (same as action)")
        # print(f"  Action format: {self.action_fmt}  (dim={self.action_dim})")
        # print(f"  Action scale : {self.action_scale} (arm only, gripper unchanged)")
        # print(f"  EEF orient   : quat (w,x,y,z) - matches controller")
        # print(f"  Ctrl/FPS     : {self.actual_fps:.1f} Hz")
        # print(f"  Cameras      : {self.cam_names or 'none'}")
        # print(f"  Save failed  : {self.save_failed}")
        # print(f"  Output       : {self.output_path}")
        # if self.policy_mode == "teleoperation":
        #     print(f"  SpaceMouse   : pos_sens={self.pos_sensitivity}, rot_sens={self.rot_sensitivity}")
        # print(f"{'=' * 60}\n")

        has_phase = hasattr(self.menv, "task_phase")

        try:
            while self.saved < self.num_episodes:
                # Teleoperation: check if user requested discard via right button
                if self.policy_mode == "teleoperation" and self._teleop_discard:
                    n_steps = len(self.bufs[0]["actions"])
                    # print(f"  [DISCARD] episode discarded by user ({n_steps} steps)")
                    self._reset_buf(0)
                    self._teleop_discard = False
                    self.teleop_interface.reset()
                    obs, _ = self._reset_step_env()
                    continue

                # Track max phase seen (before the step may reset it)
                if has_phase:
                    for eid in range(self.num_envs):
                        ph = self.menv.task_phase[eid].item()
                        self.max_phase[eid] = max(self.max_phase[eid], ph)
                        if ph == 3 and self.phase3_start_steps[eid] is None:
                            self.phase3_start_steps[eid] = len(self.bufs[eid]["joint_pos"])

                # Record observation (before stepping)
                for eid in range(self.num_envs):
                    self._record(eid)

                action = self._act(obs)
                if self.action_scale != 1.0:
                    action = action.clone()
                    action[..., :-1] *= self.action_scale
                action = self._apply_phase3_gripper_override(action)

                # Record the final action sent to the env.
                # joint_pos: target = current_joint + arm_scale * final env action
                # ik modes: record the final env action as-is.
                robot = self.menv.scene["robot"]
                for eid in range(self.num_envs):
                    raw = action[eid].detach().cpu().numpy().copy()
                    if self.control_mode == "joint_pos":
                        cur_jp = robot.data.joint_pos[eid, self.arm_ids].cpu().numpy()
                        arm_scale = 0.1  # RelativeJointPositionActionCfg scale
                        abs_target = cur_jp + arm_scale * raw[:self.n_arm]
                        gripper_intent = np.array(
                            [1.0 if raw[-1] > 0 else 0.0], dtype=np.float32
                        )
                        self.bufs[eid]["actions"].append(
                            np.concatenate([abs_target, gripper_intent])
                        )
                    else:
                        self.bufs[eid]["actions"].append(raw)

                obs, _rew, terminated, truncated, _info = self.step_env.step(action)
                dones = terminated | truncated

                for eid in range(self.num_envs):
                    if not dones[eid]:
                        continue
                    self.attempts += 1
                    self._episode_index = self.attempts

                    # Success phase can be task-specific (BallThrowing completes at phase 2).
                    terminal_phase = -1
                    if hasattr(self.menv, "_terminal_task_phase"):
                        terminal_phase = self.menv._terminal_task_phase[eid].item()
                    finalize_completed_episode(self, eid, hf, terminal_phase)
                    if self.policy_mode == "planning":
                        self.planning_policy.reset_env(eid)
                    if self.policy_mode == "teleoperation":
                        self.teleop_interface.reset()

                    if self.saved >= self.num_episodes:
                        break

        except KeyboardInterrupt:
            pass
        finally:
            hf.attrs["total_demos"] = self.saved
            hf.attrs["total_attempts"] = self.attempts
            hf.close()
            self.base_env.close()
            rate = self.saved / max(1, self.attempts) * 100
            # print(f"\n{'=' * 60}")
            # print(f"  Saved {self.saved} episodes -> {self.output_path}")
            # print(f"  Attempts: {self.attempts}  Success rate: {rate:.1f}%")
            # print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    VLADataCollector(args_cli).collect()
    simulation_app.close()


if __name__ == "__main__":
    main()
