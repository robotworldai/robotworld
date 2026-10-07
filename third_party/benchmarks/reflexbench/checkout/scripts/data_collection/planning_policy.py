"""Task-specific cuRobo planning policies for data collection.

This module provides shared trajectory-planning and action-formatting helpers
for the five ReflexBench tasks that support planning-based collection. Conveyor
Belt Pick-and-Place is intentionally excluded because its demonstrations must
be collected with a trained RL checkpoint.

Supports three control modes (``--control``):

- **joint_pos** : cuRobo joint trajectory sent directly to
  ``JointPositionAction``.
- **ik_abs** : FK of each trajectory step -> absolute Cartesian pose
  sent to ``DiffIKAction(use_relative_mode=False)``.
- **ik_rel** : FK of target vs. current EE pose -> delta sent to
  ``DiffIKAction(use_relative_mode=True)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch

from reflexbench.tasks.manager_based.rolling_ball_interception.constants import (
    RAMP_CENTER_POS as BALL_RAMP_CENTER_POS,
    RAMP_EXIT_ZONE_CENTER_POS as BALL_RAMP_EXIT_ZONE_CENTER_POS,
    RAMP_EXIT_ZONE_SIZE as BALL_RAMP_EXIT_ZONE_SIZE,
    RAMP_QUAT as BALL_RAMP_QUAT,
    RAMP_RAIL_LEFT_POS as BALL_RAMP_RAIL_LEFT_POS,
    RAMP_RAIL_RIGHT_POS as BALL_RAMP_RAIL_RIGHT_POS,
    RAMP_RAIL_SIZE as BALL_RAMP_RAIL_SIZE,
    RAMP_SIZE as BALL_RAMP_SIZE,
)

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import subtract_frame_transforms


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _axis_angle_between(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
    """Compute axis-angle delta from q_from to q_to (both w,x,y,z).

    Returns (3,) axis-angle vector.
    """
    w0, x0, y0, z0 = q_from[0], q_from[1], q_from[2], q_from[3]
    q_from_inv = torch.tensor([w0, -x0, -y0, -z0], device=q_from.device)

    w1, x1, y1, z1 = q_to[0], q_to[1], q_to[2], q_to[3]
    wi, xi, yi, zi = q_from_inv[0], q_from_inv[1], q_from_inv[2], q_from_inv[3]
    dw = wi * w1 - xi * x1 - yi * y1 - zi * z1
    dx = wi * x1 + xi * w1 + yi * z1 - zi * y1
    dy = wi * y1 - xi * z1 + yi * w1 + zi * x1
    dz = wi * z1 + xi * y1 - yi * x1 + zi * w1

    if dw < 0:
        dw, dx, dy, dz = -dw, -dx, -dy, -dz

    angle = 2.0 * torch.acos(torch.clamp(dw, -1.0, 1.0))
    axis = torch.stack([dx, dy, dz])
    norm = torch.norm(axis)
    if norm < 1e-8:
        return torch.zeros(3, device=q_from.device)
    return axis / norm * angle


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion (w, x, y, z)."""
    q_xyz = q[1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=0)
    return v + q[0] * t + torch.cross(q_xyz, t, dim=0)


# ------------------------------------------------------------------ #
#  Per-env trajectory state
# ------------------------------------------------------------------ #


@dataclass
class _EnvState:
    """Mutable per-environment trajectory replay state."""

    joint_traj: torch.Tensor | None = None  # (T, n_joints)
    cart_traj_pos: torch.Tensor | None = None  # (T, 3)  - FK positions
    cart_traj_quat: torch.Tensor | None = None  # (T, 4)  - FK quaternions (w,x,y,z)
    step_idx: int = 0
    last_phase: int = -1
    gripper_open: bool = True
    plan_failed: bool = False
    settle_counter: int = -1  # counts down while waiting for object to settle


class _SwingTrajectory(TypedDict):
    joints: torch.Tensor
    release_step: int


# ------------------------------------------------------------------ #
#  CuRoboPlanningPolicy
# ------------------------------------------------------------------ #


class CuRoboPlanningPolicy:
    """Shared cuRobo helpers for task-specific planning policies.

    Parameters
    ----------
    env : ManagerBasedRLEnv
        The unwrapped Isaac Lab environment.
    ctrl_dt : float
        Duration of one control step (seconds).
    control_mode : str
        One of ``"joint_pos"``, ``"ik_abs"``, ``"ik_rel"``.
    """

    JOINT_NAMES = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        ctrl_dt: float,
        control_mode: str = "joint_pos",
        speed_scale: float = 1.0,
        active_phases: set[int] | None = None,
    ):
        self.env = env
        self.ctrl_dt = ctrl_dt
        self.control_mode = control_mode
        self.speed_scale = max(0.25, min(4.0, speed_scale))  # clamp for safety
        self.active_phases = active_phases  # None = all phases; set = only these phases
        self.device = env.device
        self.num_envs = env.num_envs

        robot = env.scene["robot"]
        self.arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]
        self.n_arm = len(self.arm_joint_ids)
        self.default_joint_pos = robot.data.default_joint_pos[0, self.arm_joint_ids].clone()

        # Read the relative-joint-position action scale from the env cfg so we
        # compute env_action = (q_target - q_current) / scale correctly.
        # Different tasks may use different action scales.
        try:
            self.joint_action_scale = float(env.cfg.actions.arm_action.scale)
        except Exception:
            self.joint_action_scale = 0.1
        if self.joint_action_scale <= 0.0:
            self.joint_action_scale = 0.1

        # joint_pos: 7 arm + 1 gripper = 8
        # ik_abs:   7 (pos+quat) + 1 gripper = 8
        # ik_rel:   6 (delta pos + axis-angle) + 1 gripper = 7
        self._action_dim = 7 if control_mode == "ik_rel" else 8
        self._init_curobo()

        self.states: list[_EnvState] = [_EnvState() for _ in range(self.num_envs)]

    # ------------------------------------------------------------------ #
    #  cuRobo setup
    # ------------------------------------------------------------------ #

    def _init_curobo(self) -> None:
        tensor_args = TensorDeviceType(device=torch.device(self.device))

        world_config = {
            "cuboid": {
                "ground": {
                    "dims": [4.0, 4.0, 0.02],
                    "pose": [0.0, 0.0, -0.01, 1, 0, 0, 0],
                },
                "ramp_surface": {
                    "dims": list(BALL_RAMP_SIZE),
                    "pose": [
                        *BALL_RAMP_CENTER_POS,
                        *BALL_RAMP_QUAT,
                    ],
                },
                "ramp_exit_zone": {
                    "dims": list(BALL_RAMP_EXIT_ZONE_SIZE),
                    "pose": [
                        *BALL_RAMP_EXIT_ZONE_CENTER_POS,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                },
                "ramp_rail_left": {
                    "dims": list(BALL_RAMP_RAIL_SIZE),
                    "pose": [
                        *BALL_RAMP_RAIL_LEFT_POS,
                        *BALL_RAMP_QUAT,
                    ],
                },
                "ramp_rail_right": {
                    "dims": list(BALL_RAMP_RAIL_SIZE),
                    "pose": [
                        *BALL_RAMP_RAIL_RIGHT_POS,
                        *BALL_RAMP_QUAT,
                    ],
                },
            },
        }

        # Always interpolate at ctrl_dt for max resolution; subsample later by speed_scale
        mg_config = MotionGenConfig.load_from_robot_config(
            "franka.yml",
            world_config,
            tensor_args=tensor_args,
            interpolation_dt=self.ctrl_dt,
        )
        self.motion_gen = MotionGen(mg_config)
        self.motion_gen.warmup()

        robot_cfg_dict = load_yaml(
            join_path(get_robot_configs_path(), "franka.yml")
        )["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        # print("[PlanningPolicy] cuRobo MotionGen initialised  "
        #       f"(interpolation_dt={self.ctrl_dt:.4f}s, speed_scale={self.speed_scale:.2f})")

    # ------------------------------------------------------------------ #
    #  FK helper
    # ------------------------------------------------------------------ #

    def _fk(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward kinematics via cuRobo.  q: (n_joints,) -> (pos(3), quat(4 wxyz))."""
        q_batch = q.unsqueeze(0).to(dtype=torch.float32)
        state = self.kin_model.get_state(q_batch)
        return state.ee_position[0].clone(), state.ee_quaternion[0].clone()

    def _fk_batch(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FK for (T, n_joints) -> (T, 3), (T, 4)."""
        state = self.kin_model.get_state(q.to(dtype=torch.float32))
        return state.ee_position, state.ee_quaternion

    def _subsample_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """Subsample trajectory by speed_scale, always keeping the last point.

        With speed_scale=2, takes every 2nd point -> half the steps,
        each step's joint delta is ~2x larger -> arm moves ~2x faster.
        """
        if self.speed_scale <= 1.0:
            return traj
        stride = max(1, int(self.speed_scale))
        indices = list(range(0, len(traj), stride))
        if indices[-1] != len(traj) - 1:
            indices.append(len(traj) - 1)
        return traj[indices]

    # ------------------------------------------------------------------ #
    #  Planning helpers
    # ------------------------------------------------------------------ #

    def _current_joints(self, eid: int) -> torch.Tensor:
        robot = self.env.scene["robot"]
        return robot.data.joint_pos[eid, self.arm_joint_ids].clone()

    def _ee_pose_base(self, eid: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Current EE pose in robot base frame (pos(3), quat(4 wxyz))."""
        robot = self.env.scene["robot"]
        ee = self.env.scene["ee_frame"]
        r_pos = robot.data.root_pos_w[eid: eid + 1]
        r_quat = robot.data.root_quat_w[eid: eid + 1]
        e_pos = ee.data.target_pos_w[eid: eid + 1, 0, :]
        e_quat = ee.data.target_quat_w[eid: eid + 1, 0, :]
        p_b, q_b = subtract_frame_transforms(r_pos, r_quat, e_pos, e_quat)
        return p_b[0], q_b[0]

    def _plan_to_pose(
        self, eid: int, target_pos: torch.Tensor, target_quat: torch.Tensor,
    ) -> torch.Tensor | None:
        """Plan trajectory from current joints to a Cartesian target.

        Returns joint trajectory (T, 7) or None on failure.
        """
        q_cur = self._current_joints(eid)
        start = JointState.from_position(
            q_cur.unsqueeze(0).to(dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )
        goal = Pose(
            target_pos.unsqueeze(0).to(dtype=torch.float32),
            target_quat.unsqueeze(0).to(dtype=torch.float32),
        )
        result = self.motion_gen.plan_single(
            start, goal, MotionGenPlanConfig(max_attempts=10)
        )
        if not result.success.item():
            # print(f"    [cuRobo] plan_single FAILED to "
            #       f"pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}), "
            #       f"status={result.status}")
            return None
        traj = result.get_interpolated_plan()
        return self._subsample_traj(traj.position)  # (T', 7)

    def _plan_to_joints(
        self, eid: int, target_q: torch.Tensor,
    ) -> torch.Tensor | None:
        """Plan trajectory to a joint configuration.  Returns (T,7) or None."""
        q_cur = self._current_joints(eid)
        start = JointState.from_position(
            q_cur.unsqueeze(0).to(dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )
        goal = JointState.from_position(
            target_q.unsqueeze(0).to(dtype=torch.float32),
            joint_names=self.JOINT_NAMES,
        )
        result = self.motion_gen.plan_single_js(
            start, goal, MotionGenPlanConfig(max_attempts=10)
        )
        if not result.success.item():
            # print(f"    [cuRobo] plan_single_js FAILED, status={result.status}")
            return None
        traj = result.get_interpolated_plan()
        return self._subsample_traj(traj.position)

    def _precompute_fk(self, st: _EnvState) -> None:
        """Fill Cartesian trajectory caches from joint trajectory (ik_abs/ik_rel)."""
        if st.joint_traj is None:
            return
        if self.control_mode == "joint_pos":
            return
        pos, quat = self._fk_batch(st.joint_traj)
        st.cart_traj_pos = pos
        st.cart_traj_quat = quat

    # ------------------------------------------------------------------ #
    #  Shared phase helper
    # ------------------------------------------------------------------ #

    def _plan_return_home(self, eid: int) -> None:
        """Phase 4: return arm to default joint position."""
        st = self.states[eid]
        traj = self._plan_to_joints(eid, self.default_joint_pos)
        if traj is None:
            # print(f"  [WARN] env {eid}: return-home planning failed")
            st.plan_failed = True
            return
        st.joint_traj = traj
        st.step_idx = 0
        st.gripper_open = True
        self._precompute_fk(st)

    # ------------------------------------------------------------------ #
    #  Action formatting
    # ------------------------------------------------------------------ #

    def _format_action_joint(
        self, q_target: torch.Tensor, eid: int
    ) -> torch.Tensor:
        """joint_pos (relative): env_action = (target - current) / scale.

        ``scale`` is read from the env's ``RelativeJointPositionActionCfg`` at
        init time (see ``self.joint_action_scale``); different tasks use
        different tasks may use different scales.
        """
        current = self._current_joints(eid)
        return (q_target - current) / self.joint_action_scale

    def _format_action_ik_abs(
        self, pos: torch.Tensor, quat: torch.Tensor, _eid: int
    ) -> torch.Tensor:
        """ik_abs: (x,y,z, qw,qx,qy,qz) - 7D absolute EE pose."""
        return torch.cat([pos, quat])

    def _format_action_ik_rel(
        self, pos_target: torch.Tensor, quat_target: torch.Tensor, eid: int,
    ) -> torch.Tensor:
        """ik_rel: delta from current EE pose."""
        ee_pos, ee_quat = self._ee_pose_base(eid)
        d_pos = pos_target - ee_pos
        d_rot = _axis_angle_between(ee_quat, quat_target)
        return torch.cat([d_pos, d_rot])

    def _hold_action(self, eid: int) -> torch.Tensor:
        """Action that keeps the arm at its current position."""
        if self.control_mode == "joint_pos":
            # Relative control: zero action = stay at current position
            return torch.zeros(7, device=self.device)
        if self.control_mode == "ik_abs":
            pos, quat = self._ee_pose_base(eid)
            return self._format_action_ik_abs(pos, quat, eid)
        # ik_rel: zero delta
        return torch.zeros(6, device=self.device)

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def compute_action(self) -> torch.Tensor:
        """Compute actions for a task-specific planning policy."""
        raise NotImplementedError(
            "CuRoboPlanningPolicy only provides shared helpers; use a task-specific subclass."
        )

    def reset_env(self, eid: int) -> None:
        """Reset per-env state (called when env resets)."""
        self.states[eid] = _EnvState()


@dataclass
class _TossEnvState(_EnvState):
    swing_traj_idx: int = -1
    swing_release_step: int = -1
    swing_throw_step_stride: int = -1


class BallThrowingPlanningPolicy(CuRoboPlanningPolicy):
    """Target-conditioned swing policy for the pre-grasped BallThrowing task."""

    SWING_ACCEL_STEPS = 15
    SWING_DECEL_STEPS = 10
    SWING_SHOULDER_OFFSETS = (-0.1, 0.0, 0.1)
    SWING_RELEASE_STEP = 14

    SWING_WAYPOINT_NOISE_STD: float = 0.04
    SWING_RELEASE_JITTER: int = 0
    ACTION_NOISE_STD: float = 0.03

    SWING_REPLAY_STEP_STRIDE: int = 3
    SWING_THROW_STEP_STRIDE: int = 6
    SWING_THROW_STEP_STRIDE_NEAR: int = 1
    SWING_THROW_WINDOW_BEFORE_RELEASE: int = 6
    SWING_THROW_WINDOW_AFTER_RELEASE: int = 4

    NOMINAL_BOX_X: float = 0.8
    NOMINAL_BOX_Y: float = 0.0
    BOX_AIM_X_RANGE: float = 0.15
    BOX_AIM_YAW_GAIN: float = 2.2
    BOX_AIM_MAX_YAW: float = 0.55

    FRANKA_JOINT_LOWER = torch.tensor(
        [-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89],
        dtype=torch.float32,
    )
    FRANKA_JOINT_UPPER = torch.tensor(
        [2.89, 1.76, 2.89, -0.08, 2.89, 3.70, 2.89],
        dtype=torch.float32,
    )

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        ctrl_dt: float,
        control_mode: str = "joint_pos",
        speed_scale: float = 1.0,
        active_phases: set[int] | None = None,
    ):
        super().__init__(
            env,
            ctrl_dt,
            control_mode=control_mode,
            speed_scale=speed_scale,
            active_phases=active_phases,
        )
        self.states: list[_TossEnvState] = [_TossEnvState() for _ in range(self.num_envs)]
        self.swing_trajectories: list[_SwingTrajectory] = self._build_swing_trajectory_library()

    def _interpolate_joint_segment(
        self, start: torch.Tensor, end: torch.Tensor, steps: int,
    ) -> torch.Tensor:
        alpha = torch.linspace(0.0, 1.0, steps, device=self.device, dtype=torch.float32).unsqueeze(1)
        return start.unsqueeze(0) + alpha * (end - start).unsqueeze(0)

    def _build_swing_trajectory_library(self) -> list[_SwingTrajectory]:
        lift_pose = torch.tensor(
            [0.0, -0.9, 0.0, -2.5, 0.0, 1.5, 0.785],
            device=self.device,
            dtype=torch.float32,
        )
        swing_pose = torch.tensor(
            [0.0, 0.45, 0.0, -0.55, 0.0, 2.75, 0.785],
            device=self.device,
            dtype=torch.float32,
        )
        follow_pose = torch.tensor(
            [0.0, 0.75, 0.0, -0.35, 0.0, 3.15, 0.785],
            device=self.device,
            dtype=torch.float32,
        )

        trajectories: list[_SwingTrajectory] = []
        for shoulder_offset in self.SWING_SHOULDER_OFFSETS:
            wp1 = lift_pose.clone()
            wp2 = swing_pose.clone()
            wp3 = follow_pose.clone()
            wp1[1] += shoulder_offset
            wp2[1] += shoulder_offset
            wp3[1] += shoulder_offset

            accel = self._interpolate_joint_segment(wp1, wp2, self.SWING_ACCEL_STEPS)
            decel = self._interpolate_joint_segment(wp2, wp3, self.SWING_DECEL_STEPS)[1:]
            joints = torch.cat([accel, decel], dim=0)
            release_step = min(self.SWING_RELEASE_STEP, len(accel) - 1)
            trajectories.append({"joints": joints, "release_step": release_step})

        return trajectories

    def _box_target_position(self, eid: int) -> torch.Tensor:
        if hasattr(self.env, "box_target_position"):
            return self.env.box_target_position[eid]
        return torch.tensor(
            [self.NOMINAL_BOX_X, self.NOMINAL_BOX_Y, 0.075],
            device=self.device,
            dtype=torch.float32,
        )

    def _box_range_unit(self, eid: int) -> float:
        target = self._box_target_position(eid)
        dx = float(target[0].item() - self.NOMINAL_BOX_X)
        return max(-1.0, min(1.0, dx / self.BOX_AIM_X_RANGE))

    def _box_range_profile(self, eid: int) -> tuple[int, int, float, float, float]:
        """Piecewise throw profile: release offset, stride, joint2/4/6 offsets."""
        range_unit = self._box_range_unit(eid)
        if range_unit <= -0.875:
            return -6, 1, -0.47, -0.19, 0.26
        if range_unit <= -0.625:
            return -4, 2, -0.32, -0.13, 0.19
        if range_unit <= -0.375:
            return -2, 2, -0.12, -0.05, 0.09
        if range_unit <= -0.125:
            return 0, 3, 0.13, 0.05, -0.02
        if range_unit < 0.125:
            return 2, 4, 0.33, 0.11, -0.14
        if range_unit < 0.375:
            return 3, 4, 0.50, 0.17, -0.22
        if range_unit < 0.625:
            return 5, 5, 0.70, 0.25, -0.32
        if range_unit < 0.875:
            return 6, 5, 0.86, 0.31, -0.40
        return 7, 6, 0.98, 0.37, -0.48

    def _condition_swing_trajectory(
        self, eid: int, swing_joints: torch.Tensor, base_release_step: int
    ) -> tuple[torch.Tensor, int, int]:
        release_offset, throw_stride, joint2_offset, joint4_offset, joint6_offset = (
            self._box_range_profile(eid)
        )
        target = self._box_target_position(eid)
        dx = float(target[0].item() - self.NOMINAL_BOX_X)
        dy = float(target[1].item() - self.NOMINAL_BOX_Y)
        target_x = max(0.4, self.NOMINAL_BOX_X + dx)
        yaw_offset = math.atan2(dy, target_x) * self.BOX_AIM_YAW_GAIN
        yaw_offset = max(-self.BOX_AIM_MAX_YAW, min(self.BOX_AIM_MAX_YAW, yaw_offset))

        joint_offset = torch.zeros(7, device=self.device, dtype=swing_joints.dtype)
        joint_offset[0] = yaw_offset
        joint_offset[1] = joint2_offset
        joint_offset[3] = joint4_offset
        joint_offset[5] = joint6_offset

        ramp = torch.linspace(
            0.0,
            1.0,
            swing_joints.shape[0],
            device=self.device,
            dtype=swing_joints.dtype,
        ).pow(0.5).unsqueeze(1)
        conditioned = swing_joints + ramp * joint_offset.unsqueeze(0)
        conditioned[0] = swing_joints[0]

        if self.SWING_WAYPOINT_NOISE_STD > 0.0:
            noise = torch.randn_like(conditioned) * self.SWING_WAYPOINT_NOISE_STD
            noise[0].zero_()
            conditioned = conditioned + noise

        lower = self.FRANKA_JOINT_LOWER.to(device=self.device, dtype=conditioned.dtype)
        upper = self.FRANKA_JOINT_UPPER.to(device=self.device, dtype=conditioned.dtype)
        conditioned = torch.clamp(conditioned, min=lower, max=upper)

        release_step = base_release_step + release_offset
        if self.SWING_RELEASE_JITTER > 0:
            jitter = int(
                torch.randint(
                    -self.SWING_RELEASE_JITTER,
                    self.SWING_RELEASE_JITTER + 1,
                    (1,),
                    device=self.device,
                ).item()
            )
            release_step += jitter
        release_step = max(0, min(conditioned.shape[0] - 1, release_step))
        return conditioned, release_step, max(1, int(throw_stride))

    def _plan_swing(self, eid: int) -> None:
        st = self.states[eid]
        current_q = self._current_joints(eid)
        best_idx = min(
            range(len(self.swing_trajectories)),
            key=lambda idx: torch.norm(
                current_q - self.swing_trajectories[idx]["joints"][0]
            ).item(),
        )
        swing = self.swing_trajectories[best_idx]
        swing_joints, release_step, throw_stride = self._condition_swing_trajectory(
            eid, swing["joints"].clone(), int(swing["release_step"])
        )
        swing_start = swing_joints[0]

        transition_traj = self._plan_to_joints(eid, swing_start)
        if transition_traj is not None and len(transition_traj) > 1:
            full_traj = torch.cat([transition_traj[:-1], swing_joints], dim=0)
            release_offset = len(transition_traj) - 1
        else:
            full_traj = swing_joints
            release_offset = 0

        st.joint_traj = full_traj
        st.step_idx = 0
        st.gripper_open = False
        st.swing_traj_idx = best_idx
        st.swing_release_step = release_step + release_offset
        st.swing_throw_step_stride = throw_stride
        self._precompute_fk(st)

    def _swing_step_stride(self, st: _TossEnvState, idx: int) -> int:
        if st.swing_release_step < 0:
            return self.SWING_REPLAY_STEP_STRIDE
        throw_start = st.swing_release_step - self.SWING_THROW_WINDOW_BEFORE_RELEASE
        throw_end = st.swing_release_step + self.SWING_THROW_WINDOW_AFTER_RELEASE
        if throw_start <= idx <= throw_end:
            throw_stride = (
                st.swing_throw_step_stride
                if st.swing_throw_step_stride > 0
                else self.SWING_THROW_STEP_STRIDE
            )
            return max(1, throw_stride)
        return self.SWING_REPLAY_STEP_STRIDE

    def compute_action(self) -> torch.Tensor:
        arm_dim = self._action_dim - 1
        actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        for eid in range(self.num_envs):
            phase = int(self.env.task_phase[eid].item())
            st = self.states[eid]
            phase_active = self.active_phases is None or phase in self.active_phases

            if phase != st.last_phase:
                keep_follow_through = st.last_phase == 0 and phase == 1
                if not keep_follow_through:
                    st.joint_traj = None
                    st.cart_traj_pos = None
                    st.cart_traj_quat = None
                    st.step_idx = 0
                    st.plan_failed = False
                    st.settle_counter = -1
                    st.swing_traj_idx = -1
                    st.swing_release_step = -1
                    st.swing_throw_step_stride = -1
                st.last_phase = phase

                if not phase_active:
                    continue

                if phase == 0:
                    st.gripper_open = False
                    self._plan_swing(eid)
                elif phase == 1:
                    st.gripper_open = True
                elif phase == 2:
                    st.gripper_open = True

            if not phase_active:
                continue

            if phase == 2:
                arm_action = self._hold_action(eid)
                gripper = 1.0
            elif st.plan_failed or st.joint_traj is None:
                arm_action = self._hold_action(eid)
                gripper = 1.0 if phase >= 1 or st.gripper_open else -1.0
            elif st.step_idx < len(st.joint_traj):
                idx = st.step_idx
                if self.control_mode == "joint_pos":
                    arm_action = self._format_action_joint(st.joint_traj[idx], eid)
                else:
                    assert st.cart_traj_pos is not None
                    assert st.cart_traj_quat is not None
                    if self.control_mode == "ik_abs":
                        arm_action = self._format_action_ik_abs(
                            st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
                        )
                    else:
                        arm_action = self._format_action_ik_rel(
                            st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
                        )

                if self.ACTION_NOISE_STD > 0.0:
                    arm_action = arm_action + torch.randn_like(arm_action) * self.ACTION_NOISE_STD

                stride = self._swing_step_stride(st, idx)
                if (
                    phase == 0
                    and st.swing_release_step >= 0
                    and idx < st.swing_release_step <= idx + stride
                ):
                    stride = st.swing_release_step - idx

                if phase >= 1:
                    st.gripper_open = True
                    gripper = 1.0
                elif st.swing_release_step >= 0 and idx >= st.swing_release_step:
                    st.gripper_open = True
                    gripper = 1.0
                else:
                    st.gripper_open = False
                    gripper = -1.0
                st.step_idx += stride
            else:
                arm_action = self._hold_action(eid)
                st.gripper_open = True
                gripper = 1.0

            actions[eid, :arm_dim] = arm_action
            actions[eid, arm_dim] = gripper

        return actions

    def reset_env(self, eid: int) -> None:
        self.states[eid] = _TossEnvState()


# ------------------------------------------------------------------ #
#  BallCatchingPlanningPolicy
# ------------------------------------------------------------------ #


@dataclass
class _BallCatchEnvState(_EnvState):
    hold_counter: int = 0


class BallCatchingPlanningPolicy(CuRoboPlanningPolicy):
    """Scripted intercept policy for ball catching.

    Phases (managed by env events, not by this policy):
      0 - wait for launch
      1 - launch detected, waiting for intercept prediction
      2 - intercept predicted, move EE to intercept point
      3 - hold at intercept, absorb impact
      4 - done / return home

    The ball trajectory prediction lives in ``env.predicted_intercept_pos``
    (robot-base frame) computed by ``mdp.update_ball_tracking``. This policy
    plans the robot so the bucket catch point, not the raw EE frame, reaches
    that predicted intercept.
    """

    CATCH_QUAT = [0.0, 1.0, 0.0, 0.0]  # hand-down, bucket opening up (w,x,y,z)
    BUCKET_CATCH_POINT_Z_OFFSET = 0.05
    HOLD_STEPS_AFTER_TRAJ = 20

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        ctrl_dt: float,
        control_mode: str = "ik_abs",
        speed_scale: float = 4.0,
        active_phases: set[int] | None = None,
    ):
        super().__init__(
            env,
            ctrl_dt,
            control_mode=control_mode,
            speed_scale=speed_scale,
            active_phases=active_phases,
        )
        self.states: list[_BallCatchEnvState] = [
            _BallCatchEnvState() for _ in range(self.num_envs)
        ]

    def _bucket_pose_base(self, eid: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Bucket root pose in robot base frame (pos(3), quat(4 wxyz))."""
        robot = self.env.scene["robot"]
        bucket = self.env.scene["bucket"]
        bucket_pos_w = bucket.data.root_pos_w[eid : eid + 1, :3]
        bucket_quat_w = bucket.data.root_quat_w[eid : eid + 1]
        bucket_pos_b, bucket_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w[eid : eid + 1],
            robot.data.root_quat_w[eid : eid + 1],
            bucket_pos_w,
            bucket_quat_w,
        )
        return bucket_pos_b[0], bucket_quat_b[0]

    def _planner_ee_pose_base(self, eid: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Current cuRobo planning EE pose in robot base frame (panda_hand)."""
        return self._fk(self._current_joints(eid))

    def _plan_intercept(self, eid: int) -> None:
        st = self.states[eid]
        intercept_pos = self.env.predicted_intercept_pos[eid].clone()

        if intercept_pos.norm() < 1e-4:
            st.plan_failed = True
            return

        target_pos = intercept_pos.clone()

        ee_pos_b, _ = self._planner_ee_pose_base(eid)
        bucket_pos_b, bucket_quat_b = self._bucket_pose_base(eid)
        bucket_catch_point_b = bucket_pos_b + _quat_apply(
            bucket_quat_b,
            torch.tensor([0.0, 0.0, self.BUCKET_CATCH_POINT_Z_OFFSET], device=self.device),
        )
        ee_to_catch_point_b = bucket_catch_point_b - ee_pos_b
        target_pos -= ee_to_catch_point_b

        target_quat = torch.tensor(self.CATCH_QUAT, device=self.device, dtype=torch.float32)

        traj = self._plan_to_pose(eid, target_pos, target_quat)
        if traj is None:
            st.plan_failed = True
            return

        st.joint_traj = traj
        st.step_idx = 0
        self._precompute_fk(st)
        # print(
        #     f"  [BallCatch] env {eid} phase 2: traj={len(traj)} steps, "
        #     f"intercept=({intercept_pos[0]:.3f}, {intercept_pos[1]:.3f}, {intercept_pos[2]:.3f})"
        # )

    def compute_action(self) -> torch.Tensor:
        arm_dim = self._action_dim - 1
        actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        for eid in range(self.num_envs):
            phase = int(self.env.task_phase[eid].item())
            st = self.states[eid]
            phase_active = self.active_phases is None or phase in self.active_phases

            if phase != st.last_phase:
                st.joint_traj = None
                st.cart_traj_pos = None
                st.cart_traj_quat = None
                st.step_idx = 0
                st.plan_failed = False
                st.hold_counter = 0
                st.last_phase = phase

                if not phase_active:
                    continue

                if phase == 2:
                    self._plan_intercept(eid)
                elif phase == 4:
                    self._plan_return_home(eid)

            if not phase_active:
                continue

            if phase in (0, 1, 3):
                arm_action = self._hold_action(eid)
            elif st.plan_failed or st.joint_traj is None:
                arm_action = self._hold_action(eid)
            elif st.step_idx < len(st.joint_traj):
                idx = st.step_idx
                if self.control_mode == "joint_pos":
                    arm_action = self._format_action_joint(st.joint_traj[idx], eid)
                elif self.control_mode == "ik_abs":
                    arm_action = self._format_action_ik_abs(
                        st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
                    )
                else:
                    arm_action = self._format_action_ik_rel(
                        st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
                    )
                st.step_idx += 1
            else:
                arm_action = self._hold_action(eid)

            actions[eid, :arm_dim] = arm_action
            actions[eid, arm_dim] = 1.0  # gripper open - bucket catch, no grasp

        return actions

    def reset_env(self, eid: int) -> None:
        self.states[eid] = _BallCatchEnvState()


# ------------------------------------------------------------------ #
#  WhackAMolePlanningPolicy
# ------------------------------------------------------------------ #


@dataclass
class _WhackAMoleEnvState(_EnvState):
    wam_state: int = 0  # 0=idle, 1=striking, 2=pressing, 3=returning, 4=game_over
    current_target_mole: int = -1
    press_counter: int = 0


class WhackAMolePlanningPolicy(CuRoboPlanningPolicy):
    """Reactive strike policy for whack-a-mole.

    The env manages popup windows and press detection geometrically.
    This policy monitors ``env.active_mole_id`` each step, decides
    whether to attempt a strike based on remaining window time, plans
    a cuRobo trajectory to the mole, keeps the gripper closed while
    pressing the active mole down, then returns to a ready hover position.

    Internal state machine per env (``wam_state``):
      0 - idle / hovering at ready
      1 - executing strike trajectory toward active mole
      2 - dwelling at press position
      3 - returning to ready hover
      4 - game over, returning home
    """

    STRIKE_QUAT = [0.0, 1.0, 0.0, 0.0]  # hand-down (w,x,y,z)
    HAND_TO_TCP_Z = 0.107
    TABLE_HEIGHT = 0.20
    BOARD_SURFACE_Z = TABLE_HEIGHT + 0.02
    TCP_STRIKE_Z = BOARD_SURFACE_Z + 0.008
    TCP_READY_Z = BOARD_SURFACE_Z + 0.11
    READY_XY = (0.55, 0.0)
    # Dwell long enough for the PD controller to converge to the planned strike
    # joint config and the env-side press detector (needs 2 consecutive frames)
    # to register a hit.
    PRESS_DWELL_STEPS = 5
    MIN_WINDOW_MARGIN_STEPS = 3

    MOLE_XY = [
        (0.55, 0.00),
        (0.55, 0.10),
        (0.55, -0.10),
        (0.45, 0.00),
        (0.65, 0.00),
    ]

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        ctrl_dt: float,
        control_mode: str = "ik_rel",
        speed_scale: float = 1.0,
        active_phases: set[int] | None = None,
    ):
        super().__init__(
            env, ctrl_dt,
            control_mode=control_mode,
            speed_scale=speed_scale,
            active_phases=active_phases,
        )
        self.states: list[_WhackAMoleEnvState] = [
            _WhackAMoleEnvState() for _ in range(self.num_envs)
        ]

    def _active_mole_world_pos(self, eid: int, mole_id: int) -> torch.Tensor:
        env_origin = self.env.scene.env_origins[eid]
        x, y = self.MOLE_XY[mole_id]
        z = self.env.mole_heights[eid, mole_id]
        return torch.stack(
            [
                env_origin[0] + torch.tensor(x, device=self.device, dtype=torch.float32),
                env_origin[1] + torch.tensor(y, device=self.device, dtype=torch.float32),
                env_origin[2] + z,
            ]
        )

    def _remaining_window_steps(self, eid: int) -> int:
        if not self.env.window_active[eid]:
            return 0
        win_idx = int(self.env.current_window_idx[eid].item())
        if win_idx >= self.env.popup_start_times.shape[1]:
            return 0
        start_t = self.env.popup_start_times[eid, win_idx].item()
        dur = self.env.popup_durations[eid, win_idx].item()
        remaining_s = max(0.0, (start_t + dur) - self.env.episode_timer[eid].item())
        return int(remaining_s / self.ctrl_dt)

    def _plan_strike(self, eid: int, mole_id: int) -> bool:
        st = self.states[eid]
        x, y = self.MOLE_XY[mole_id]
        ee_frame = self.env.scene["ee_frame"]
        tip_w = ee_frame.data.target_pos_w[eid, 0, :].detach().clone()
        mole_w = self._active_mole_world_pos(eid, mole_id).detach().clone()
        target_pos = torch.tensor(
            [x, y, self.TCP_STRIKE_Z + self.HAND_TO_TCP_Z],
            device=self.device, dtype=torch.float32,
        )
        target_quat = torch.tensor(self.STRIKE_QUAT, device=self.device, dtype=torch.float32)

        # print(
        #     "  [WAM-DBG] "
        #     f"env {eid}: tip_w={tip_w.tolist()} "
        #     f"mole_w={mole_w.tolist()} "
        #     f"dz={(tip_w[2] - mole_w[2]).item():.4f} "
        #     f"planned_tip_z={self.TCP_STRIKE_Z:.4f} "
        #     f"planned_hand_z={target_pos[2].item():.4f}"
        # )

        traj = self._plan_to_pose(eid, target_pos, target_quat)
        if traj is None:
            return False

        remaining = self._remaining_window_steps(eid)
        needed = len(traj) + self.PRESS_DWELL_STEPS + self.MIN_WINDOW_MARGIN_STEPS
        if needed > remaining:
            # print(f"  [WAM] env {eid}: skip mole {mole_id} (need {needed}, have {remaining} steps)")
            return False

        st.joint_traj = traj
        st.step_idx = 0
        st.current_target_mole = mole_id
        self._precompute_fk(st)
        # print(f"  [WAM] env {eid}: strike mole {mole_id} ({len(traj)} steps, {remaining} remain)")
        return True

    def _plan_ready_pose(self, eid: int) -> bool:
        """Return to the initial (reset-time) joint configuration so the wrist
        camera can see all 5 moles, instead of hovering above the centre mole."""
        st = self.states[eid]
        traj = self._plan_to_joints(eid, self.default_joint_pos)
        if traj is None:
            return False
        st.joint_traj = traj
        st.step_idx = 0
        self._precompute_fk(st)
        return True

    def _replay_step(self, eid: int, st: _WhackAMoleEnvState) -> torch.Tensor | None:
        if st.joint_traj is None or st.step_idx >= len(st.joint_traj):
            return None
        idx = st.step_idx
        if self.control_mode == "joint_pos":
            action = self._format_action_joint(st.joint_traj[idx], eid)
        elif self.control_mode == "ik_abs":
            action = self._format_action_ik_abs(st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid)
        else:
            action = self._format_action_ik_rel(st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid)
        st.step_idx += 1
        return action

    def _strike_hold_action(
        self, eid: int, st: _WhackAMoleEnvState,
    ) -> torch.Tensor:
        """Keep commanding the planned strike endpoint during press dwell.

        Using :meth:`_hold_action` (zero relative action) would freeze the arm
        at its *current* (possibly under-shot) joint config; this helper keeps
        driving the PD controller toward ``joint_traj[-1]`` so TCP actually
        reaches the strike z inside the press-detection window.
        """
        if st.joint_traj is None or len(st.joint_traj) == 0:
            return self._hold_action(eid)
        last_idx = len(st.joint_traj) - 1
        if self.control_mode == "joint_pos":
            return self._format_action_joint(st.joint_traj[last_idx], eid)
        elif self.control_mode == "ik_abs":
            assert st.cart_traj_pos is not None and st.cart_traj_quat is not None
            return self._format_action_ik_abs(
                st.cart_traj_pos[last_idx], st.cart_traj_quat[last_idx], eid,
            )
        else:
            assert st.cart_traj_pos is not None and st.cart_traj_quat is not None
            return self._format_action_ik_rel(
                st.cart_traj_pos[last_idx], st.cart_traj_quat[last_idx], eid,
            )

    def compute_action(self) -> torch.Tensor:
        arm_dim = self._action_dim - 1
        actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        for eid in range(self.num_envs):
            task_phase = int(self.env.task_phase[eid].item())
            st = self.states[eid]

            if task_phase == 4 and st.wam_state != 4:
                self._plan_return_home(eid)
                st.wam_state = 4

            if st.wam_state == 0:
                should_strike = (
                    self.env.window_active[eid]
                    and int(self.env.active_mole_id[eid].item()) >= 0
                    and not self.env.window_hit_registered[eid]
                )
                if should_strike:
                    mid = int(self.env.active_mole_id[eid].item())
                    if self._plan_strike(eid, mid):
                        st.wam_state = 1
                        arm_action = self._replay_step(eid, st)
                        if arm_action is None:
                            arm_action = self._hold_action(eid)
                    else:
                        arm_action = self._hold_action(eid)
                else:
                    arm_action = self._hold_action(eid)

            elif st.wam_state == 1:
                if not self.env.window_active[eid]:
                    if self._plan_ready_pose(eid):
                        st.wam_state = 3
                    else:
                        st.wam_state = 0
                    arm_action = self._hold_action(eid)
                else:
                    arm_action = self._replay_step(eid, st)
                    if arm_action is None:
                        st.wam_state = 2
                        st.press_counter = self.PRESS_DWELL_STEPS
                        # Stay locked onto the planned strike joint config so
                        # the PD controller keeps driving TCP onto the mole
                        # instead of freezing at an under-shot pose.
                        arm_action = self._strike_hold_action(eid, st)

            elif st.wam_state == 2:
                arm_action = self._strike_hold_action(eid, st)
                st.press_counter -= 1
                if st.press_counter <= 0:
                    if self._plan_ready_pose(eid):
                        st.wam_state = 3
                    else:
                        st.wam_state = 0

            elif st.wam_state == 3:
                arm_action = self._replay_step(eid, st)
                if arm_action is None:
                    st.wam_state = 0
                    arm_action = self._hold_action(eid)

            elif st.wam_state == 4:
                arm_action = self._replay_step(eid, st)
                if arm_action is None:
                    arm_action = self._hold_action(eid)

            else:
                arm_action = self._hold_action(eid)

            actions[eid, :arm_dim] = arm_action
            actions[eid, arm_dim] = -1.0  # gripper stays closed while pressing moles

        return actions

    def reset_env(self, eid: int) -> None:
        self.states[eid] = _WhackAMoleEnvState()


# ------------------------------------------------------------------ #
#  RotatingPegInsertionPlanningPolicy
# ------------------------------------------------------------------ #


@dataclass
class _DiscInsertEnvState(_EnvState):
    hold_counter: int = 0
    hover_target_xy: tuple[float, float] | None = None


class RotatingPegInsertionPlanningPolicy(CuRoboPlanningPolicy):
    """Closed-loop hover/insert policy for rotating disc insertion."""

    HOVER_QUAT = [0.0, 1.0, 0.0, 0.0]
    SETTLE_STEPS = 3
    HOLD_STEPS_AFTER_INSERT = 15
    HOVER_REPLAY_STEP_STRIDE = 3
    DESCENT_REPLAY_STEP_STRIDE = 2

    PEG_TIP_TO_EE = 0.095
    DISC_TOP_Z = 0.405
    HOVER_CLEARANCE = 0.10
    PRE_INSERT_CLEARANCE = 0.05
    INSERT_DEPTH = 0.040

    HOVER_Z = DISC_TOP_Z + PEG_TIP_TO_EE + HOVER_CLEARANCE
    PRE_INSERT_Z = DISC_TOP_Z + PEG_TIP_TO_EE + PRE_INSERT_CLEARANCE
    INSERT_Z = DISC_TOP_Z + PEG_TIP_TO_EE - INSERT_DEPTH

    DISC_CENTER_X = 0.5
    DISC_CENTER_Y = 0.0
    HOLE_OFFSET_R = 0.07
    HOVER_X = DISC_CENTER_X + HOLE_OFFSET_R
    HOVER_Y = DISC_CENTER_Y

    HOVER_REPLAN_DISTANCE = 0.035
    HOVER_TARGET_LEAD_TIME = 0.80
    HOVER_ARRIVAL_EXTRA_LEAD_TIME = 0.12
    HOVER_MIN_LEAD_TIME = 0.45
    HOVER_MAX_LEAD_TIME = 1.80
    HOVER_REPLAN_LEAD_DIFF = 0.10

    DESCENT_TARGET_LEAD_TIME = 0.25
    DESCENT_ARRIVAL_EXTRA_LEAD_TIME = 0.04
    DESCENT_MIN_LEAD_TIME = 0.18
    DESCENT_MAX_LEAD_TIME = 0.90
    DESCENT_REPLAN_LEAD_DIFF = 0.06

    EE_FRAME_OFFSET_TO_PANDA_HAND = [0.0, 0.0, 0.1034]

    TRACK_XY_SPEED = 0.60
    TRACK_EXTRA_LEAD_TIME = 0.08
    TRACK_MIN_LEAD_TIME = 0.18
    TRACK_MAX_LEAD_TIME = 0.95

    INSERT_Z_SPEED = 0.30
    INSERT_EXTRA_LEAD_TIME = 0.05
    INSERT_MIN_LEAD_TIME = 0.08
    INSERT_MAX_LEAD_TIME = 0.32

    SERVO_MAX_JOINT_STEP = 0.24
    DESCENT_MAX_JOINT_STEP = 0.12
    DESCENT_ALIGN_RADIUS = 0.060
    INSERT_ALIGN_RADIUS = 0.040

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        ctrl_dt: float,
        control_mode: str = "ik_abs",
        speed_scale: float = 1.0,
        active_phases: set[int] | None = None,
    ):
        super().__init__(
            env,
            ctrl_dt,
            control_mode=control_mode,
            speed_scale=speed_scale,
            active_phases=active_phases,
        )
        self.states: list[_DiscInsertEnvState] = [
            _DiscInsertEnvState() for _ in range(self.num_envs)
        ]

    @staticmethod
    def _clamp_lead_time(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def _hole_pos_base(self, eid: int) -> torch.Tensor:
        """Current hole center in robot base frame."""
        return self._predicted_hole_pos_base(eid, 0.0)

    def _predicted_hole_pos_base(self, eid: int, lead_time: float) -> torch.Tensor:
        """Predicted hole center in robot base frame after ``lead_time`` seconds."""
        robot = self.env.scene["robot"]
        theta = (
            self.env.disc_angle[eid]
            + self.env.hole_angle_offset[eid]
            + self.env.disc_angular_velocity[eid] * lead_time
        )
        hole_local = self.env.disc_center_local[eid].clone()
        hole_local[0] += self.HOLE_OFFSET_R * torch.cos(theta)
        hole_local[1] += self.HOLE_OFFSET_R * torch.sin(theta)
        hole_local[2] = self.DISC_TOP_Z
        hole_world = hole_local + self.env.scene.env_origins[eid]
        p_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[eid: eid + 1],
            robot.data.root_quat_w[eid: eid + 1],
            hole_world.unsqueeze(0),
        )
        return p_b[0]

    def _ee_frame_target_to_panda_hand_target(
        self, ee_frame_pos: torch.Tensor, ee_frame_quat: torch.Tensor
    ) -> torch.Tensor:
        ee_frame_offset = torch.tensor(
            self.EE_FRAME_OFFSET_TO_PANDA_HAND,
            device=self.device,
            dtype=torch.float32,
        )
        return ee_frame_pos - _quat_apply(ee_frame_quat, ee_frame_offset)

    def _solve_ik_single(
        self, target_pos: torch.Tensor, target_quat: torch.Tensor, q_seed: torch.Tensor,
    ) -> torch.Tensor | None:
        goal = Pose(
            target_pos.unsqueeze(0).to(dtype=torch.float32),
            target_quat.unsqueeze(0).to(dtype=torch.float32),
        )
        ik_result = self.motion_gen.solve_ik(
            goal,
            retract_config=q_seed.unsqueeze(0).to(dtype=torch.float32),
        )
        if not ik_result.success.any():
            return None
        return ik_result.solution[ik_result.success][0].to(device=self.device)

    def _tracking_lead_time(self, eid: int) -> float:
        ee_pos, _ = self._ee_pose_base(eid)
        hole_pos = self._hole_pos_base(eid)
        xy_dist = torch.norm(ee_pos[:2] - hole_pos[:2]).item()
        return self._clamp_lead_time(
            xy_dist / self.TRACK_XY_SPEED + self.TRACK_EXTRA_LEAD_TIME,
            self.TRACK_MIN_LEAD_TIME,
            self.TRACK_MAX_LEAD_TIME,
        )

    def _descent_lead_time(self, eid: int) -> float:
        ee_pos, _ = self._ee_pose_base(eid)
        z_dist = max(0.0, float(ee_pos[2].item()) - self.INSERT_Z)
        return self._clamp_lead_time(
            z_dist / self.INSERT_Z_SPEED + self.INSERT_EXTRA_LEAD_TIME,
            self.INSERT_MIN_LEAD_TIME,
            self.INSERT_MAX_LEAD_TIME,
        )

    def _servo_to_ee_target(
        self,
        eid: int,
        target_ee_pos: torch.Tensor,
        target_quat: torch.Tensor,
        max_joint_step: float,
    ) -> torch.Tensor:
        if self.control_mode == "ik_abs":
            return self._format_action_ik_abs(target_ee_pos, target_quat, eid)
        if self.control_mode == "ik_rel":
            return self._format_action_ik_rel(target_ee_pos, target_quat, eid)

        q_cur = self._current_joints(eid)
        target_hand_pos = self._ee_frame_target_to_panda_hand_target(
            target_ee_pos, target_quat
        )
        q_target = self._solve_ik_single(target_hand_pos, target_quat, q_cur)
        if q_target is None:
            return self._hold_action(eid)
        q_step = q_cur + torch.clamp(
            q_target - q_cur,
            min=-max_joint_step,
            max=max_joint_step,
        )
        return self._format_action_joint(q_step, eid)

    def _servo_to_hole(
        self,
        eid: int,
        target_z: float,
        lead_time: float,
        max_joint_step: float,
    ) -> torch.Tensor:
        hole_pos = self._predicted_hole_pos_base(eid, lead_time)
        target_ee_pos = torch.tensor(
            [hole_pos[0].item(), hole_pos[1].item(), target_z],
            device=self.device,
            dtype=torch.float32,
        )
        target_quat = torch.tensor(self.HOVER_QUAT, device=self.device, dtype=torch.float32)
        return self._servo_to_ee_target(eid, target_ee_pos, target_quat, max_joint_step)

    def _servo_insert(self, eid: int) -> torch.Tensor:
        lead_time = self._descent_lead_time(eid)
        hole_pos = self._predicted_hole_pos_base(eid, lead_time)
        ee_pos, _ = self._ee_pose_base(eid)
        xy_error = torch.norm(ee_pos[:2] - hole_pos[:2]).item()

        if xy_error > self.DESCENT_ALIGN_RADIUS:
            target_z = self.PRE_INSERT_Z
        elif xy_error > self.INSERT_ALIGN_RADIUS:
            alpha = (
                (self.DESCENT_ALIGN_RADIUS - xy_error)
                / (self.DESCENT_ALIGN_RADIUS - self.INSERT_ALIGN_RADIUS)
            )
            target_z = self.PRE_INSERT_Z + alpha * (self.INSERT_Z - self.PRE_INSERT_Z)
        else:
            target_z = self.INSERT_Z

        target_ee_pos = torch.tensor(
            [hole_pos[0].item(), hole_pos[1].item(), target_z],
            device=self.device,
            dtype=torch.float32,
        )
        target_quat = torch.tensor(self.HOVER_QUAT, device=self.device, dtype=torch.float32)
        return self._servo_to_ee_target(
            eid, target_ee_pos, target_quat, self.DESCENT_MAX_JOINT_STEP
        )

    def _format_step(self, st: _DiscInsertEnvState, eid: int) -> torch.Tensor:
        idx = st.step_idx
        if self.control_mode == "joint_pos":
            return self._format_action_joint(st.joint_traj[idx], eid)
        elif self.control_mode == "ik_abs":
            return self._format_action_ik_abs(
                st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
            )
        else:
            return self._format_action_ik_rel(
                st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
            )

    def compute_action(self) -> torch.Tensor:
        arm_dim = self._action_dim - 1
        actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        for eid in range(self.num_envs):
            phase = int(self.env.task_phase[eid].item())
            st = self.states[eid]
            phase_active = self.active_phases is None or phase in self.active_phases

            if phase != st.last_phase:
                st.joint_traj = None
                st.cart_traj_pos = None
                st.cart_traj_quat = None
                st.step_idx = 0
                st.plan_failed = False
                st.hold_counter = 0
                st.settle_counter = -1
                st.last_phase = phase

                if not phase_active:
                    continue

                if phase == 0:
                    st.settle_counter = self.SETTLE_STEPS
                elif phase == 4:
                    self._plan_return_home(eid)

            if not phase_active:
                continue

            gripper = -1.0  # closed - holding the peg

            if phase == 0:
                if st.settle_counter > 0:
                    st.settle_counter -= 1
                    arm_action = self._hold_action(eid)
                elif st.settle_counter == 0:
                    st.settle_counter = -1
                    arm_action = self._servo_to_hole(
                        eid,
                        self.HOVER_Z,
                        self._tracking_lead_time(eid),
                        self.SERVO_MAX_JOINT_STEP,
                    )
                else:
                    arm_action = self._servo_to_hole(
                        eid,
                        self.HOVER_Z,
                        self._tracking_lead_time(eid),
                        self.SERVO_MAX_JOINT_STEP,
                    )

            elif phase in (1, 2):
                arm_action = self._servo_to_hole(
                    eid,
                    self.HOVER_Z,
                    self._tracking_lead_time(eid),
                    self.SERVO_MAX_JOINT_STEP,
                )

            elif phase == 3:
                arm_action = self._servo_insert(eid)
                st.hold_counter += 1

            elif phase == 4:
                gripper = 1.0
                if st.plan_failed or st.joint_traj is None:
                    arm_action = self._hold_action(eid)
                elif st.step_idx < len(st.joint_traj):
                    arm_action = self._format_step(st, eid)
                    st.step_idx += 1
                else:
                    arm_action = self._hold_action(eid)

            else:
                arm_action = self._hold_action(eid)

            actions[eid, :arm_dim] = arm_action
            actions[eid, arm_dim] = gripper

        return actions

    def reset_env(self, eid: int) -> None:
        self.states[eid] = _DiscInsertEnvState()


@dataclass
class _BallInterceptEnvState(_EnvState):
    hold_counter: int = 0
    debug_step_counter: int = 0
    planned_target_y: float = 0.0


class RollingBallInterceptionPlanningPolicy(CuRoboPlanningPolicy):
    CATCH_QUAT = [0.0, 1.0, 0.0, 0.0]
    HOLD_STEPS_AFTER_TRAJ = 20
    DEBUG_LOG_INTERVAL_STEPS = 25
    INTERCEPT_EXEC_STEP_STRIDE = 1
    EE_FRAME_OFFSET_TO_PANDA_HAND = [0.0, 0.0, 0.1034]

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        ctrl_dt: float,
        control_mode: str = "ik_abs",
        speed_scale: float = 1.0,
        active_phases: set[int] | None = None,
    ):
        super().__init__(
            env,
            ctrl_dt,
            control_mode=control_mode,
            speed_scale=speed_scale,
            active_phases=active_phases,
        )
        self.states: list[_BallInterceptEnvState] = [
            _BallInterceptEnvState() for _ in range(self.num_envs)
        ]

    def _ball_debug_snapshot(self, eid: int) -> str:
        ball = self.env.scene["ball"]
        robot = self.env.scene["robot"]
        ee_frame = self.env.scene["ee_frame"]

        ball_pos_w = ball.data.root_pos_w[eid, :3]
        ball_vel_w = ball.data.root_lin_vel_w[eid, :3]
        ee_pos_w = ee_frame.data.target_pos_w[eid : eid + 1, 0, :]
        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[eid : eid + 1],
            robot.data.root_quat_w[eid : eid + 1],
            ee_pos_w,
        )

        predicted_intercept = self.env.predicted_intercept_pos[eid]
        ramp_exit = (
            self.env.ramp_exit_pos[eid]
            if hasattr(self.env, "ramp_exit_pos")
            else torch.zeros(3, dtype=torch.float32, device=self.device)
        )
        predicted_time = (
            float(self.env.predicted_intercept_time[eid].item())
            if hasattr(self.env, "predicted_intercept_time")
            else float("nan")
        )
        catch_counter = (
            int(self.env.ball_in_catcher_counter[eid].item())
            if hasattr(self.env, "ball_in_catcher_counter")
            else -1
        )

        dist_to_intercept = float("nan")
        if predicted_intercept.norm().item() > 1e-6:
            dist_to_intercept = torch.norm(ee_pos_b[0] - predicted_intercept).item()
        dist_text = f"{dist_to_intercept:.3f}" if math.isfinite(dist_to_intercept) else "n/a"

        return (
            f"ball_w=({ball_pos_w[0]:.3f}, {ball_pos_w[1]:.3f}, {ball_pos_w[2]:.3f}), "
            f"ball_v_w=({ball_vel_w[0]:.3f}, {ball_vel_w[1]:.3f}, {ball_vel_w[2]:.3f})\n"
            f"    ee_b=({ee_pos_b[0, 0]:.3f}, {ee_pos_b[0, 1]:.3f}, {ee_pos_b[0, 2]:.3f}), "
            f"predicted_b=({predicted_intercept[0]:.3f}, {predicted_intercept[1]:.3f}, {predicted_intercept[2]:.3f}), "
            f"dist_to_intercept={dist_text}\n"
            f"    predicted_time={predicted_time:.3f}s, "
            f"ramp_exit=({ramp_exit[0]:.3f}, {ramp_exit[1]:.3f}, {ramp_exit[2]:.3f}), "
            f"catch_counter={catch_counter}"
        )

    def _log_ball_debug(self, eid: int, reason: str) -> None:
        # print(
        #     f"  [BallInterceptDebug] env {eid}: {reason}\n"
        #     f"    {self._ball_debug_snapshot(eid)}"
        # )
        pass

    def _ee_frame_target_to_panda_hand_target(
        self, ee_frame_pos: torch.Tensor, ee_frame_quat: torch.Tensor
    ) -> torch.Tensor:
        """Convert an ee_frame target pose into the panda_hand pose used by cuRobo."""
        ee_frame_offset = torch.tensor(
            self.EE_FRAME_OFFSET_TO_PANDA_HAND,
            device=self.device,
            dtype=torch.float32,
        )
        return ee_frame_pos - _quat_apply(ee_frame_quat, ee_frame_offset)

    def _panda_hand_pose_to_ee_frame_pose(
        self, panda_hand_pos: torch.Tensor, panda_hand_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a panda_hand pose into the task's ee_frame pose."""
        ee_frame_offset = torch.tensor(
            self.EE_FRAME_OFFSET_TO_PANDA_HAND,
            device=self.device,
            dtype=torch.float32,
        )
        ee_frame_pos = panda_hand_pos + _quat_apply(panda_hand_quat, ee_frame_offset)
        return ee_frame_pos, panda_hand_quat

    def _mouth_pose_base(self, eid: int) -> torch.Tensor:
        """Current cup mouth center in robot base frame."""
        robot = self.env.scene["robot"]
        catcher = self.env.scene["catcher"]
        catcher_pos_w = catcher.data.root_pos_w[eid : eid + 1, :3]
        catcher_quat_w = catcher.data.root_quat_w[eid : eid + 1]
        mouth_rel = torch.tensor(
            [0.0, 0.0, 0.05],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        mouth_pos_w = catcher_pos_w + _quat_apply(catcher_quat_w[0], mouth_rel[0]).unsqueeze(0)
        mouth_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w[eid : eid + 1],
            robot.data.root_quat_w[eid : eid + 1],
            mouth_pos_w,
        )
        return mouth_pos_b[0]

    def _predicted_mouth_target(self, eid: int) -> torch.Tensor:
        """Desired cup mouth target in robot base frame."""
        if hasattr(self.env, "predicted_intercept_mouth_pos"):
            return self.env.predicted_intercept_mouth_pos[eid].clone()
        return self.env.predicted_intercept_pos[eid].clone()

    def _ee_target_from_mouth_target(
        self,
        eid: int,
        mouth_target_b: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the EE target from the current physical mouth-to-EE transform."""
        ee_pos_b, ee_quat_b = self._ee_pose_base(eid)
        mouth_pos_b = self._mouth_pose_base(eid)
        mouth_rel_to_ee, _ = subtract_frame_transforms(
            ee_pos_b.unsqueeze(0),
            ee_quat_b.unsqueeze(0),
            mouth_pos_b.unsqueeze(0),
        )
        return mouth_target_b - _quat_apply(target_quat, mouth_rel_to_ee[0])

    # Maximum joint-space step per control tick (rad) for linear interpolation.
    _IK_INTERP_MAX_STEP = 0.05

    def _solve_ik_single(
        self, target_pos: torch.Tensor, target_quat: torch.Tensor, q_seed: torch.Tensor,
    ) -> torch.Tensor | None:
        """Solve IK for a single target. Returns (7,) joint config or None."""
        goal = Pose(
            target_pos.unsqueeze(0).to(dtype=torch.float32),
            target_quat.unsqueeze(0).to(dtype=torch.float32),
        )
        ik_result = self.motion_gen.solve_ik(
            goal,
            retract_config=q_seed.unsqueeze(0).to(dtype=torch.float32),
        )
        if not ik_result.success.any():
            # print(f"    [IK] FAILED  pos_err={ik_result.position_error}  rot_err={ik_result.rotation_error}")
            return None
        sol = ik_result.solution[ik_result.success][0].to(device=self.device)
        joint_delta = (sol - q_seed).norm().item()
        fk_pos, fk_quat = self._fk(sol)
        # print(
        #     f"    [IK] OK  pos_err={ik_result.position_error.item():.4f}  "
        #     f"rot_err={ik_result.rotation_error.item():.4f}  "
        #     f"joint_delta={joint_delta:.4f}\n"
        #     f"    [IK] target_hand=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  "
        #     f"FK_hand=({fk_pos[0]:.3f},{fk_pos[1]:.3f},{fk_pos[2]:.3f})  "
        #     f"FK_quat=({fk_quat[0]:.3f},{fk_quat[1]:.3f},{fk_quat[2]:.3f},{fk_quat[3]:.3f})\n"
        #     f"    [IK] target_quat=({target_quat[0]:.3f},{target_quat[1]:.3f},{target_quat[2]:.3f},{target_quat[3]:.3f})  "
        #     f"sol[:4]=[{sol[0]:.3f},{sol[1]:.3f},{sol[2]:.3f},{sol[3]:.3f}]  "
        #     f"seed[:4]=[{q_seed[0]:.3f},{q_seed[1]:.3f},{q_seed[2]:.3f},{q_seed[3]:.3f}]"
        # )
        return sol

    def _linear_joint_traj(
        self, q_start: torch.Tensor, q_end: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation in joint space, respecting action-space limits.

        With RelativeJointPositionActionCfg(scale=0.1) and action clip [-1,1],
        the max per-joint movement per step is 0.1 rad.  We compute the number
        of steps so that no single joint exceeds this limit.

        Returns (N, 7) trajectory *excluding* start, *including* end.
        """
        max_per_joint = (q_end - q_start).abs().max().item()
        action_limit_per_step = 0.1  # scale * clip_max
        n_steps = max(2, int(max_per_joint / action_limit_per_step) + 1)
        alphas = torch.linspace(0, 1, n_steps + 1, device=self.device)[1:]  # (N,)
        return q_start.unsqueeze(0) + alphas.unsqueeze(1) * (q_end - q_start).unsqueeze(0)

    def _plan_intercept(self, eid: int) -> None:
        st = self.states[eid]
        intercept_mouth_pos = self._predicted_mouth_target(eid)

        if intercept_mouth_pos.norm() < 1e-4:
            st.plan_failed = True
            self._log_ball_debug(eid, "phase 2 planning skipped because predicted intercept mouth target is zero")
            return

        target_quat = torch.tensor(
            self.CATCH_QUAT, device=self.device, dtype=torch.float32,
        )
        intercept_pos = self._ee_target_from_mouth_target(eid, intercept_mouth_pos, target_quat)
        target_pos = self._ee_frame_target_to_panda_hand_target(intercept_pos.clone(), target_quat)

        q_cur = self._current_joints(eid)
        traj = None
        method = "ik_interp"

        # Primary: IK + linear joint interpolation (fast and reliable)
        target_q = self._solve_ik_single(target_pos, target_quat, q_cur)
        if target_q is not None:
            traj = self._linear_joint_traj(q_cur, target_q)
        else:
            # Fallback: cuRobo motion planner
            method = "motion_gen"
            traj = self._plan_to_pose(eid, target_pos, target_quat)

        if traj is None:
            st.plan_failed = True
            self._log_ball_debug(eid, "phase 2 planning failed (IK + motion_gen both failed)")
            return

        st.joint_traj = traj
        st.step_idx = 0
        st.plan_failed = False
        st.planned_target_y = float(intercept_mouth_pos[1].item())
        self._precompute_fk(st)

        # DEBUG: verify trajectory endpoint
        fk_end_hand_pos, fk_end_hand_quat = self._fk(traj[-1])
        fk_cur_hand_pos, fk_cur_hand_quat = self._fk(q_cur)
        fk_end_pos, _ = self._panda_hand_pose_to_ee_frame_pose(fk_end_hand_pos, fk_end_hand_quat)
        fk_cur_pos, _ = self._panda_hand_pose_to_ee_frame_pose(fk_cur_hand_pos, fk_cur_hand_quat)
        # print(
        #     f"  [BallIntercept] env {eid} phase 2 ({method}): traj={len(traj)} steps, "
        #     f"intercept_mouth=({intercept_mouth_pos[0]:.3f}, {intercept_mouth_pos[1]:.3f}, {intercept_mouth_pos[2]:.3f})\n"
        #     f"    FK current EE=({fk_cur_pos[0]:.3f}, {fk_cur_pos[1]:.3f}, {fk_cur_pos[2]:.3f})\n"
        #     f"    FK traj[-1]  =({fk_end_pos[0]:.3f}, {fk_end_pos[1]:.3f}, {fk_end_pos[2]:.3f})\n"
        #     f"    TARGET_EE    =({intercept_pos[0]:.3f}, {intercept_pos[1]:.3f}, {intercept_pos[2]:.3f})"
        # )

    # Minimum Y drift (metres) to trigger a replan after trajectory is done.
    _REPLAN_Y_THRESHOLD = 0.015
    # Joint convergence tolerance (rad) before allowing a replan.
    _JOINT_CONVERGE_TOL = 0.02

    def _should_replan_intercept(self, eid: int) -> bool:
        st = self.states[eid]
        if st.plan_failed or st.joint_traj is None:
            return True
        # After trajectory exhausted, only replan once PD has converged AND Y drifted.
        if st.step_idx >= len(st.joint_traj):
            # Check if PD controller has converged to the target joints
            q_cur = self._current_joints(eid)
            joint_err = (q_cur - st.joint_traj[-1]).abs().max().item()
            if joint_err > self._JOINT_CONVERGE_TOL:
                return False  # Still converging - keep commanding traj[-1]
            mouth_target = self._predicted_mouth_target(eid)
            if mouth_target.norm() < 1e-4:
                return False
            y_delta = abs(float(mouth_target[1].item()) - st.planned_target_y)
            return y_delta > self._REPLAN_Y_THRESHOLD
        return False

    def compute_action(self) -> torch.Tensor:
        arm_dim = self._action_dim - 1
        actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        for eid in range(self.num_envs):
            phase = int(self.env.task_phase[eid].item())
            st = self.states[eid]
            phase_active = self.active_phases is None or phase in self.active_phases

            if phase != st.last_phase:
                prev_phase = st.last_phase
                st.joint_traj = None
                st.cart_traj_pos = None
                st.cart_traj_quat = None
                st.step_idx = 0
                st.plan_failed = False
                st.hold_counter = 0
                st.debug_step_counter = 0
                st.last_phase = phase

                if not phase_active:
                    continue

                self._log_ball_debug(eid, f"phase transition {prev_phase} -> {phase}")

                if phase == 2:
                    self._plan_intercept(eid)
                elif phase == 4:
                    self._plan_return_home(eid)

            if not phase_active:
                continue

            if phase == 2 and self._should_replan_intercept(eid):
                self._plan_intercept(eid)

            if phase in (0, 1, 3):
                arm_action = self._hold_action(eid)
                if phase == 3:
                    st.hold_counter += 1
            elif st.plan_failed or st.joint_traj is None:
                arm_action = self._hold_action(eid)
            elif st.step_idx < len(st.joint_traj):
                idx = st.step_idx
                if self.control_mode == "joint_pos":
                    arm_action = self._format_action_joint(st.joint_traj[idx], eid)
                elif self.control_mode == "ik_abs":
                    arm_action = self._format_action_ik_abs(
                        st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
                    )
                else:
                    arm_action = self._format_action_ik_rel(
                        st.cart_traj_pos[idx], st.cart_traj_quat[idx], eid,
                    )
                st.step_idx += self.INTERCEPT_EXEC_STEP_STRIDE if phase == 2 else 1
            elif phase == 2:
                # Trajectory exhausted but PD may not have converged yet.
                # Keep commanding the final target so X/Z continue converging.
                if self.control_mode == "joint_pos":
                    arm_action = self._format_action_joint(st.joint_traj[-1], eid)
                elif self.control_mode == "ik_abs":
                    arm_action = self._format_action_ik_abs(
                        st.cart_traj_pos[-1], st.cart_traj_quat[-1], eid,
                    )
                else:
                    arm_action = self._format_action_ik_rel(
                        st.cart_traj_pos[-1], st.cart_traj_quat[-1], eid,
                    )
            else:
                arm_action = self._hold_action(eid)

            actions[eid, :arm_dim] = arm_action
            actions[eid, arm_dim] = 1.0

            st.debug_step_counter += 1
            if st.debug_step_counter % self.DEBUG_LOG_INTERVAL_STEPS == 0:
                if phase == 1:
                    self._log_ball_debug(eid, "phase 1 waiting for intercept prediction")
                elif phase == 2:
                    traj_len = len(st.joint_traj) if st.joint_traj is not None else 0
                    self._log_ball_debug(
                        eid,
                        f"phase 2 progress plan_failed={st.plan_failed}, step_idx={st.step_idx}/{traj_len}",
                    )
                elif phase == 3:
                    self._log_ball_debug(
                        eid,
                        f"phase 3 holding after intercept, hold_counter={st.hold_counter}",
                    )

        return actions

    def reset_env(self, eid: int) -> None:
        self.states[eid] = _BallInterceptEnvState()
