from __future__ import annotations

import numpy as np

from .constants import MUJOCO_TO_ISAACLAB_REINDEX
from .math_utils import gravity_orientation, quat_apply_inverse
from .scene import MujocoSoccer
from .schemas import MotionClip


def _fit_obs_dim(obs: np.ndarray, obs_dim: int | None) -> np.ndarray:
    if obs_dim is None or obs.shape[0] == obs_dim:
        return obs.astype(np.float32)
    if obs.shape[0] > obs_dim:
        return obs[:obs_dim].astype(np.float32)
    padded = np.zeros(obs_dim, dtype=np.float32)
    padded[: obs.shape[0]] = obs
    return padded


def build_observation(
    env: MujocoSoccer,
    motion: MotionClip,
    time_step: int,
    last_action: np.ndarray,
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    ref_ang_vel: np.ndarray,
    ball_world: np.ndarray,
    goal_world: np.ndarray,
    default_joint_pos: np.ndarray,
    obs_dim: int | None = None,
) -> np.ndarray:
    del motion, time_step
    pelvis_quat = env.pelvis_quat
    projected_gravity = gravity_orientation(pelvis_quat)
    joint_pos_rel = (env.joint_pos - default_joint_pos)[MUJOCO_TO_ISAACLAB_REINDEX]
    joint_vel = env.joint_vel[MUJOCO_TO_ISAACLAB_REINDEX]
    target_point_pos = quat_apply_inverse(pelvis_quat, ball_world - env.pelvis_pos)
    target_destination = quat_apply_inverse(pelvis_quat, goal_world - env.pelvis_pos)
    obs = np.concatenate(
        [
            ref_joint_pos.astype(np.float32),
            ref_joint_vel.astype(np.float32),
            projected_gravity.astype(np.float32),
            ref_ang_vel.astype(np.float32),
            env.base_ang_vel.astype(np.float32),
            joint_pos_rel.astype(np.float32),
            joint_vel.astype(np.float32),
            last_action.astype(np.float32),
            target_point_pos.astype(np.float32),
            target_destination.astype(np.float32),
        ],
        axis=0,
    )
    return _fit_obs_dim(obs, obs_dim)
