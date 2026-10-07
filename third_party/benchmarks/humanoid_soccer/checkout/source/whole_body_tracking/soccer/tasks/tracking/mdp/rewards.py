from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_apply, quat_inv, quat_apply_inverse

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.observations import get_target_point_world
from soccer.tasks.tracking.mdp.kick_detection import KickContactTracker


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _map_names_to_indices(source_names: list[str], target_names: list[str]) -> list[int]:
    target_list = list(target_names)
    name_to_index = {name: idx for idx, name in enumerate(target_list)}
    indices: list[int] = []
    # Iterate all source names to map.
    for name in source_names:
        # Prefer exact matching for deterministic mapping.
        if name in name_to_index:
            indices.append(name_to_index[name])
            continue
        # If exact matching fails, attempt unique suffix matching.
        suffix_matches = [idx for idx, candidate in enumerate(target_list) if candidate.endswith(name)]
        # Accept only unique suffix matches to avoid ambiguity.
        if len(suffix_matches) == 1:
            indices.append(suffix_matches[0])
    return indices


def action_rate_l2_clip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward.clamp(max=100.0)


def waist_action_rate_l2_clip(env: ManagerBasedRLEnv, waist_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    if waist_cfg is None:
        raise ValueError("waist_cfg cannot be None")
    robot = env.scene[waist_cfg.name]
    idx = torch.as_tensor(robot.find_joints(waist_cfg.joint_names, preserve_order=True)[0], device=env.device)
    return torch.sum(torch.square(env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]), dim=1).clamp(max=100.0)


def _get_kick_tracker(command: MotionCommand) -> KickContactTracker:
    tracker = getattr(command, "kick_contact_tracker", None)
    if tracker is None:
        raise RuntimeError("MotionCommand is missing kick_contact_tracker; ensure command setup is up to date.")
    return tracker


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_foot_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, foot_body_names: list[str] | None = None
) -> torch.Tensor:
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, foot_body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def foot_distance(env: ManagerBasedRLEnv, threshold: float, std: float, foot_cfg: SceneEntityCfg | None = None,) -> torch.Tensor:
    """Encourage a minimum separation between both feet to avoid crossing/overlap."""
    if foot_cfg is None:
        raise ValueError("foot_distance requires foot_cfg to identify feet.")
    robot = env.scene[foot_cfg.name]
    left_foot_idx = foot_cfg.body_ids[0]
    right_foot_idx = foot_cfg.body_ids[1]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]  # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]  # [num_envs, 3]
    distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)  # [num_envs]
    reward = torch.where(
        distance >= threshold,
        torch.tensor(1., device=distance.device),
        1.0 * torch.exp(-((distance / threshold - 1)**2) / (std ** 2))
    )
    return reward


def feet_slip_penalty(env: ManagerBasedRLEnv, foot_cfg: SceneEntityCfg, slip_force_threshold: float,) -> torch.Tensor:
    """Penalize foot linear velocity when the foot is in contact.

    A contact is detected when the contact force sensor reports an upward (positive Z)
    force larger than ``slip_force_threshold`` on the foot bodies provided by
    ``foot_cfg``. The penalty mirrors the Isaac Gym style reward, summing the squared
    linear velocity of feet that are currently in contact.
    """

    if foot_cfg is None:
        raise ValueError("foot_cfg cannot be None for _reward_feet_slip_penalty")
    contact_sensor = None
    sensors = getattr(env.scene, "sensors", None)
    if sensors is not None:
        try:
            contact_sensor = sensors["contact_forces"] if isinstance(sensors, dict) else getattr(sensors, "contact_forces", None)
        except (KeyError, AttributeError, TypeError):
            contact_sensor = None
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    device = env.device
    num_envs = env.num_envs
    forces = None
    forces_data = contact_sensor.data
    if hasattr(forces_data, "net_forces_w_history"):
        forces_hist = forces_data.net_forces_w_history
        if forces_hist.numel() > 0:
            forces = forces_hist.to(device)
            if forces.ndim >= 4:
                forces = forces.max(dim=1).values
    if forces is None:
        if hasattr(forces_data, "net_forces_w"):
            forces = forces_data.net_forces_w
            if forces is not None and forces.numel() > 0:
                forces = forces.to(device)
            else:
                return torch.zeros(num_envs, device=device, dtype=torch.float32)
        else:
            return torch.zeros(num_envs, device=device, dtype=torch.float32)
    if forces.ndim < 3:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)

    robot = env.scene[foot_cfg.name]

    foot_indices_key = tuple(foot_cfg.body_names)
    if not hasattr(contact_sensor, '_foot_indices_cache'):
        contact_sensor._foot_indices_cache = {}
    if foot_indices_key not in contact_sensor._foot_indices_cache:
        foot_sensor_indices = contact_sensor.find_bodies(foot_cfg.body_names, preserve_order=True)[0]
        contact_sensor._foot_indices_cache[foot_indices_key] = torch.as_tensor(
            foot_sensor_indices, device=device, dtype=torch.long
        )
    foot_indices = contact_sensor._foot_indices_cache[foot_indices_key]

    max_foot_idx = int(foot_indices.max()) if len(foot_indices) > 0 else -1
    if forces.shape[1] <= max_foot_idx:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)
    vertical_forces = forces[:, foot_indices, 2]
    contact_mask = vertical_forces > slip_force_threshold
    foot_vel_w = robot.data.body_lin_vel_w[:, foot_indices]
    penalize = torch.where(
        contact_mask.unsqueeze(-1), 
        torch.square(foot_vel_w), 
        torch.zeros_like(foot_vel_w)
    )
    if penalize.numel() > 10000:  # Heuristic threshold; tune if needed.
        return penalize.reshape(num_envs, -1).sum(dim=1)
    else:
        return torch.sum(penalize, dim=(1, 2))
    

def target_point_proximity(env: ManagerBasedRLEnv, std: float, command_name: str = "motion",) -> torch.Tensor:
    """Reward proximity to the target point (ball) and freeze at first kick contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    
    # Compute current proximity reward.
    base_xy = command.robot_anchor_pos_w[..., :2]
    target = get_target_point_world(env, command_name).to(device=base_xy.device, dtype=base_xy.dtype)
    diff_xy = base_xy - target[..., :2]
    error = torch.sum(diff_xy * diff_xy, dim=-1)
    proximity_reward = torch.exp(-error / std**2)
    
    # Query kick-contact status.
    contact_awarded = tracker.get_contact_awarded()
    frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Freeze reward for environments that just kicked this step.
    new_kick_mask = contact_awarded & (frozen_reward == 0.0)
    if torch.any(new_kick_mask):
        new_kick_ids = torch.nonzero(new_kick_mask, as_tuple=False).squeeze(-1)
        tracker.freeze_proximity_reward(new_kick_ids, proximity_reward[new_kick_ids])
        frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Return frozen reward after contact; otherwise return current reward.
    reward = torch.where(contact_awarded, frozen_reward, proximity_reward)
    return reward


def target_point_contact(env: ManagerBasedRLEnv, 
        horizontal_force_threshold: float = 0.0,
        command_name: str = "motion",
        ball_sensor_name: str = "soccer_ball_contact",
        foot_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
    """One-shot reward for contacting the ball at first valid touch."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward
    # print(event.new_contact.to(reward.dtype))
    reward_scale = torch.zeros_like(reward)
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            reward_scale[foot_info.env_ids] = correct.to(reward_scale.dtype)
            correct_mask[foot_info.env_ids] = correct

    tracker.record_expected_success(event.new_contact, correct_mask)
    # print("contact", event.new_contact.to(reward.dtype) * reward_scale)
    return event.new_contact.to(reward.dtype) * reward_scale

def sideways_kick(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Single-shot reward encouraging foot swing along the expected lateral axis.
    Left kick expects foot velocity along local -Y; right kick expects local +Y.
    """
    if foot_cfg is None:
        raise ValueError("sideways_kick_reward requires foot_cfg to identify kicking feet.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() == 0:
        return reward

    robot = command.robot
    foot_vel_w = robot.data.body_lin_vel_w[foot_info.env_ids, foot_info.body_indices]
    foot_quat_w = robot.data.body_quat_w[foot_info.env_ids, foot_info.body_indices]

    vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)
    vel_norm = torch.norm(vel_local, dim=-1)

    expected_leg = foot_info.expected.to(device=env.device, dtype=torch.int8)
    desired_sign = torch.zeros(expected_leg.shape, device=env.device, dtype=torch.float32)
    desired_sign = torch.where(expected_leg == 0, torch.full_like(desired_sign, -1.0), desired_sign)
    desired_sign = torch.where(expected_leg == 1, torch.full_like(desired_sign, 1.0), desired_sign)

    directional_component = vel_local[:, 1] * desired_sign
    axis_component = torch.clamp(directional_component, min=0.0)

    alignment = torch.where(vel_norm > 1e-6, axis_component / vel_norm, torch.zeros_like(vel_norm))
    reward[foot_info.env_ids] = alignment.to(reward.dtype)

    # Reward only when expected leg is valid and contact leg matches expectation.
    valid_expectation = expected_leg >= 0
    correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
    wrong_mask = ~correct_foot
    if torch.any(wrong_mask):
        reward[foot_info.env_ids[wrong_mask]] = 0.0
    # print("sideways_kick reward:", reward)
    return reward



def ball_velocity_direction_alignment(
    env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward alignment between ball velocity direction and pre-kick target-to-destination direction.

    Active only for a short window after contact with the expected foot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    vel_xy = vel[:, :2]  # x-y plane projection
    vel_xy_norm = torch.norm(vel_xy, dim=-1, keepdim=True)
    vel_norm = torch.norm(vel, dim=-1, keepdim=True)
    
    # Direction vector from pre-kick target point (ball) to destination.
    direction = command.target_destination_pos - command.initial_target_point_pos  # [num_envs, 3]
    direction_xy = direction[:, :2]
    dir_norm = torch.norm(direction_xy, dim=-1, keepdim=True)

    valid_mask = (vel_norm.squeeze(-1) > velocity_threshold) & (vel_xy_norm.squeeze(-1) > 1e-6) & (
        dir_norm.squeeze(-1) > 1e-6
    )

    # Track average angle based on initial direction vectors.
    avg_angle = torch.tensor(0.0, device=env.device, dtype=torch.float32)
    if torch.any(valid_mask):
        dir_unit_valid = direction_xy[valid_mask] / dir_norm[valid_mask]
        vel_unit_valid = vel_xy[valid_mask] / vel_xy_norm[valid_mask]
        cos_theta_valid = torch.sum(vel_unit_valid * dir_unit_valid, dim=-1).clamp(-1.0, 1.0)
        theta_valid = torch.acos(cos_theta_valid)
        avg_angle = theta_valid.mean()
    if hasattr(command, "metrics"):
        command.metrics["ball_velocity_dir_alignment_angle"] = torch.full(
            (env.num_envs,), avg_angle.item(), device=env.device, dtype=torch.float32
        )
    
    # Reward window.
    timer_name = f"_{command_name}_dir_align_timer"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    # Trigger reward window on expected-foot contact.
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            # Open the window only for correct-foot contacts.
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    # Validate speeds in active_mask to avoid division by zero.
    speed_valid = (vel_xy_norm.squeeze(-1) > 1e-6) & (dir_norm.squeeze(-1) > 1e-6)
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        dir_unit = direction_xy[active_mask] / dir_norm[active_mask]
        vel_unit = vel_xy[active_mask] / vel_xy_norm[active_mask]
        cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
        error = torch.acos(cos_theta) ** 2
        reward[active_mask] = torch.exp(-error / (std ** 2))

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    # print("ball_velocity_direction_alignment reward:", timer,reward)
    return reward


def ball_speed_reward(env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
    """Reward ball speed within a short window after expected-foot contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    speed_xy = torch.norm(vel[:, :2], dim=-1)  # x-y plane speed

    timer_name = f"_{command_name}_speed_timer"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    # Trigger reward window on expected-foot contact.
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            # Open the window only for correct-foot contacts.
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    # Validate speed in active_mask to avoid division by zero.
    speed_valid = speed_xy > 1e-6
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        reward_active = 1.0 - torch.exp(-(speed_xy[active_mask] ** 2) / (std ** 2))
        reward[active_mask] = reward_active

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    # print("ball_speed_reward:", reward)
    return reward

def ball_z_speed_penalty_reward(env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    ) -> torch.Tensor:
    """Penalize excessive vertical ball speed in a short post-activation window."""
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    z_speed = vel[:, 2]  # vertical speed
    speed = torch.norm(vel, dim=-1)

    valid_mask = speed > velocity_threshold

    timer_name = f"_{command_name}_z_speed_timer"
    prev_name = f"_{command_name}_z_speed_prev"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    prev_valid = getattr(env, prev_name, None)
    if prev_valid is None or prev_valid.shape[0] != env.num_envs:
        prev_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        prev_valid = prev_valid.to(device=env.device, dtype=torch.bool)

    rising_mask = valid_mask & (~prev_valid)
    timer[rising_mask] = 5
    active_mask = timer > 0

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        scale = std if std > 0 else 1.0
        reward[active_mask] = torch.tanh(torch.abs(z_speed[active_mask]) / (scale + 1e-8))

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    setattr(env, prev_name, valid_mask.to(dtype=torch.bool))
    # print("ball_z_speed_penalty_reward:", reward)
    return reward


def pelvis_orientation(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Penalize pelvis pitch/roll tilt to keep the robot upright."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    gravity_vec_w = robot.data.GRAVITY_VEC_W
    
    # Project gravity vector to pelvis local frame.
    pelvis_proj_gravity = quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    # print("pelvis_proj_gravity:", gravity_vec_w, pelvis_proj_gravity)
    return torch.sum(torch.square(pelvis_proj_gravity[:, :2]), dim=1)