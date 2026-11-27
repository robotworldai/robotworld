from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def motion_anchor_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.anchor_ang_vel_w.view(env.num_envs, -1)


def _get_motion_command(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    command: MotionCommand | None = env.command_manager.get_term(command_name)
    if command is None:
        raise RuntimeError(f"motion command '{command_name}' not found in env.command_manager")
    if not hasattr(command, "target_point_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_point_pos attribute")
    return command


def get_target_point_world(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_local = command.target_point_pos
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        return target_local + env_origins
    return target_local


def get_target_point_base(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_world = get_target_point_world(env, command_name)
    # delta = target_world - command.robot_anchor_pos_w
    delta = target_world - command.robot_pelvis_pos_w
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def _positional_encoding(vec: torch.Tensor, num_freqs: int = 6) -> torch.Tensor:
    """Apply sinusoidal positional encoding to a target tensor of shape (E, 3).

    The encoding follows Transformer-style frequencies: for each coordinate x,
    compute sin(2^k*pi*x) and cos(2^k*pi*x) for k=0..num_freqs-1, then
    concatenate with the original coordinates.
    """
    if num_freqs <= 0:
        return vec.view(vec.shape[0], -1)

    device = vec.device
    dtype = vec.dtype
    # freqs: [num_freqs]
    freqs = (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype)) * math.pi
    # vec: [E, 3] -> vec_exp: [E, 3, num_freqs]
    vec_exp = vec.unsqueeze(-1) * freqs
    sin = torch.sin(vec_exp)
    cos = torch.cos(vec_exp)
    # sin_cos: [E, 3, 2*num_freqs] -> flatten per-sample
    sin_cos = torch.cat([sin, cos], dim=-1).view(vec.shape[0], -1)
    # Concatenate original coordinates in front.
    return torch.cat([vec.view(vec.shape[0], -1), sin_cos], dim=-1)


def target_point_pos_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_point_cache"
    target_local = get_target_point_base(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        raise AttributeError("ManagerBasedEnv missing episode_length_buf required for target point caching")

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # Return cached target vector.
    return getattr(env, cache_name)
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)


def constant_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    # Constant observation path keeps the same representation as policy inputs.
    base = get_target_point_base(env, command_name)
    return base
    return _positional_encoding(base, num_freqs=6)


def blind_zone_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return target point in robot base frame with blind-zone simulation.
    
    If robot-ball (x, y) distance is outside [blind_distance_min, blind_distance_max],
    return the last visible position to emulate limited visibility.
    Thresholds are resampled from MotionCommandCfg ranges at each resample.
    """
    command = _get_motion_command(env, command_name)
    
    # Current target in robot base frame.
    target_base = get_target_point_base(env, command_name)
    
    # Compute robot-target (x, y) distance in world coordinates.
    target_world = get_target_point_world(env, command_name)
    robot_pos = command.robot_pelvis_pos_w
    # Horizontal distance only.
    distance_xy = torch.norm(target_world[:, :2] - robot_pos[:, :2], dim=-1)
    
    # Visible only when distance is within [min, max].
    in_visible_range = (distance_xy >= command.blind_distance_min) & (distance_xy <= command.blind_distance_max)
    
    # Update last visible target for visible environments.
    if torch.any(in_visible_range):
        command.last_visible_target_point_base[in_visible_range] = target_base[in_visible_range]
        command.is_in_blind_zone[in_visible_range] = False
    
    # Mark blind-zone environments.
    command.is_in_blind_zone[~in_visible_range] = True
    
    # Return last visible position in blind zone, otherwise current target.
    result = torch.where(
        command.is_in_blind_zone.unsqueeze(-1),
        command.last_visible_target_point_base,
        target_base
    )
    # print("blind zone target point:", command.blind_distance_min, command.blind_distance_max, result)
    return result


def target_destination_pos_local(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not hasattr(command, "target_destination_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_destination_pos attribute")
    # target_destination_pos is local to env origin; convert to world before subtracting robot pose.
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        target_world = command.target_destination_pos + env_origins
    else:
        target_world = command.target_destination_pos

    delta = target_world - command.robot_pelvis_pos_w
    # print("position:", quat_apply(quat_inv(command.robot_pelvis_quat_w), delta))
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def target_destination_pos_local_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_destination_local_cache"
    target_local = target_destination_pos_local(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        raise AttributeError("ManagerBasedEnv missing episode_length_buf required for target destination caching")

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # print("cache:", getattr(env, cache_name))
    return getattr(env, cache_name)
    # Positional encoding path is intentionally disabled here.
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)
    


def foot_target_point_distance(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg, command_name: str = "motion",) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    robot = env.scene[robot_cfg.name]
    foot_pos = robot.data.body_pos_w[:, robot_cfg.body_ids]
    target_world = get_target_point_world(env, command_name)
    diff = foot_pos - target_world.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)
    return dist.view(env.num_envs, -1)