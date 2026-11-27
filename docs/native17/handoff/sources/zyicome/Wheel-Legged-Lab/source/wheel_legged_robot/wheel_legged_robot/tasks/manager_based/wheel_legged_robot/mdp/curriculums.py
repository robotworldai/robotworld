# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)

# Wheel-legged robot specific curriculum functions can be added here, for example, to adjust the command ranges based on the robot's performance or other metrics.
def wheel_legged_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    command_name: str = "wheel_legged_commands",
    angular_reward_term_name: str | None = None,
    range_multiplier: tuple[float, float] = (0.1, 1.0),
    threshold: float = 0.9,
    step_size: float = 0.05,
    initial_ang_vel_limit: float = 0.5,
    final_ang_vel_limit: float = 2.0,
    angular_threshold: float = 0.85,
    angular_step_size: float = 0.25,
    min_episodes: int = 256,
) -> torch.Tensor:
    """
    轮腿机器人命令课程：基于一批已结束 episode 的跟踪表现扩大线速度范围。

    参数：
        reward_term_name: 用于判断能力的奖励项名称（例如 "track_lin_vel"）
        command_name: 命令项名称
        range_multiplier: (初始缩放, 最终缩放)
        threshold: 奖励阈值（达到最大奖励的多少比例才扩大）
        step_size: 每次扩大的步长
        angular_reward_term_name: 角速度课程使用的奖励项；为 None 时禁用角向课程
        initial_ang_vel_limit: 初始 yaw 角速度限幅
        final_ang_vel_limit: 最终 yaw 角速度限幅
        angular_threshold: 角速度课程升级阈值
        angular_step_size: yaw 角速度限幅每次增加量
        min_episodes: 每次判断课程升级前至少收集的已结束 episode 数

    Isaac Lab 会在各环境 reset 时调用 curriculum。训练器通常会随机化初始
    episode 进度，因此环境是异步 reset 的，不能依赖
    ``common_step_counter % max_episode_length == 0``。这里累计每次 reset 的
    episode 得分，达到一个稳定窗口后再判断升级。
    """
    cmd_term = env.command_manager.get_term(command_name)
    ranges = cmd_term.cfg.ranges

    # 首次调用时保存配置中的完整范围，并切到课程初始范围。
    if not hasattr(env, "_orig_lin_vel_x"):
        env._orig_lin_vel_x = torch.tensor(ranges.lin_vel_x, device=env.device)
        env._curr_lin_vel_x = env._orig_lin_vel_x * range_multiplier[0]
        ranges.lin_vel_x = tuple(env._curr_lin_vel_x.tolist())
        env._orig_ang_vel_yaw = torch.tensor(ranges.ang_vel_yaw, device=env.device)
        configured_ang_limit = torch.min(torch.abs(env._orig_ang_vel_yaw))
        env._final_ang_vel_limit = min(float(configured_ang_limit), final_ang_vel_limit)
        env._curr_ang_vel_limit = min(initial_ang_vel_limit, env._final_ang_vel_limit)
        if angular_reward_term_name is not None:
            ranges.ang_vel_yaw = (-env._curr_ang_vel_limit, env._curr_ang_vel_limit)
        env._command_curriculum_score_sum = torch.zeros((), device=env.device)
        env._command_curriculum_angular_score_sum = torch.zeros((), device=env.device)
        env._command_curriculum_episode_count = 0
        env._command_curriculum_last_score = torch.zeros((), device=env.device)
        env._command_curriculum_last_angular_score = torch.zeros((), device=env.device)
        # common_step_counter == 0 的 reset 是环境初始化，不是已完成 episode。
        return env._curr_lin_vel_x[1].clone()

    if len(env_ids) == 0:
        return env._curr_lin_vel_x[1].clone()

    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    # episode_sums 已含 weight 和 dt。使用最大 episode 时长归一化会自然惩罚
    # 提前摔倒的轨迹，而完整存活且精确跟踪时 score 接近 1。
    reward_scale = max(float(reward_cfg.weight), 1.0e-6)
    episode_scores = episode_sums / (env.max_episode_length_s * reward_scale)
    env._command_curriculum_score_sum += episode_scores.sum()
    if angular_reward_term_name is not None:
        angular_sums = env.reward_manager._episode_sums[angular_reward_term_name][env_ids]
        angular_reward_cfg = env.reward_manager.get_term_cfg(angular_reward_term_name)
        angular_reward_scale = max(float(angular_reward_cfg.weight), 1.0e-6)
        angular_scores = angular_sums / (env.max_episode_length_s * angular_reward_scale)
        env._command_curriculum_angular_score_sum += angular_scores.sum()
    env._command_curriculum_episode_count += int(episode_scores.numel())

    if env._command_curriculum_episode_count >= min_episodes:
        mean_score = (
            env._command_curriculum_score_sum / env._command_curriculum_episode_count
        )
        env._command_curriculum_last_score = mean_score.detach().clone()
        if angular_reward_term_name is not None:
            mean_angular_score = (
                env._command_curriculum_angular_score_sum
                / env._command_curriculum_episode_count
            )
            env._command_curriculum_last_angular_score = mean_angular_score.detach().clone()

        if mean_score.item() >= threshold:
            final_range = env._orig_lin_vel_x * range_multiplier[1]
            delta = torch.tensor([-step_size, step_size], device=env.device)
            new_range = env._curr_lin_vel_x + delta
            env._curr_lin_vel_x = torch.stack(
                (
                    torch.maximum(new_range[0], final_range[0]),
                    torch.minimum(new_range[1], final_range[1]),
                )
            )
            ranges.lin_vel_x = tuple(env._curr_lin_vel_x.tolist())

        if (
            angular_reward_term_name is not None
            and mean_angular_score.item() >= angular_threshold
        ):
            env._curr_ang_vel_limit = min(
                env._curr_ang_vel_limit + angular_step_size,
                env._final_ang_vel_limit,
            )
            ranges.ang_vel_yaw = (
                -env._curr_ang_vel_limit,
                env._curr_ang_vel_limit,
            )

        # 固定大小的独立窗口，避免早期样本永久影响之后的课程判断。
        env._command_curriculum_score_sum.zero_()
        env._command_curriculum_angular_score_sum.zero_()
        env._command_curriculum_episode_count = 0

    # 返回当前最大速度（用于日志）
    return env._curr_lin_vel_x[1].clone()


def moving_jump_speed_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel",
    heading_reward_term_name: str = "track_heading",
    command_name: str = "wheel_legged_commands",
    speed_levels: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    initial_level: int = 0,
    tracking_threshold: float = 0.78,
    jump_success_threshold: float = 0.75,
    soft_landing_threshold: float = 0.75,
    heading_threshold: float = 0.80,
    min_episodes: int = 512,
    min_jump_attempts: int = 1024,
    consecutive_passes: int = 2,
) -> torch.Tensor:
    """Discrete moving-jump curriculum gated by locomotion and jump quality.

    A locomotion-only curriculum can advance after learning to drive while the
    jump collapses. This curriculum therefore requires four simultaneous
    conditions over completed episodes: forward-speed tracking, successful
    jumps, soft landings, and heading alignment.
    """
    cmd_term = env.command_manager.get_term(command_name)
    ranges = cmd_term.cfg.ranges

    if not hasattr(env, "_moving_jump_curriculum_level"):
        if len(speed_levels) == 0:
            raise ValueError("moving_jump_speed_curriculum requires speed levels.")
        if any(b <= a for a, b in zip(speed_levels, speed_levels[1:])):
            raise ValueError(f"speed_levels must be strictly increasing: {speed_levels}")
        env._moving_jump_speed_levels = tuple(float(level) for level in speed_levels)
        if not 0 <= initial_level < len(env._moving_jump_speed_levels):
            raise ValueError(
                f"initial_level must be in [0, {len(env._moving_jump_speed_levels) - 1}], "
                f"got {initial_level}."
            )
        env._moving_jump_curriculum_level = int(initial_level)
        env._moving_jump_curriculum_passes = 0
        env._moving_jump_episode_count = 0
        env._moving_jump_attempt_count = torch.zeros((), device=env.device)
        env._moving_jump_success_count = torch.zeros((), device=env.device)
        env._moving_jump_soft_count = torch.zeros((), device=env.device)
        env._moving_jump_tracking_score_sum = torch.zeros((), device=env.device)
        env._moving_jump_heading_score_sum = torch.zeros((), device=env.device)
        env._command_curriculum_last_score = torch.zeros((), device=env.device)
        env._moving_jump_curriculum_last_success = torch.zeros((), device=env.device)
        env._moving_jump_curriculum_last_soft = torch.zeros((), device=env.device)
        env._moving_jump_curriculum_last_heading = torch.zeros((), device=env.device)
        limit = env._moving_jump_speed_levels[env._moving_jump_curriculum_level]
        ranges.lin_vel_x = (-limit, limit)
        return torch.tensor(limit, device=env.device)

    current_limit = env._moving_jump_speed_levels[
        env._moving_jump_curriculum_level
    ]
    if len(env_ids) == 0 or not hasattr(env, "_jump_attempts"):
        return torch.tensor(current_limit, device=env.device)

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    tracking_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    tracking_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    tracking_scale = max(float(tracking_cfg.weight), 1.0e-6)
    tracking_scores = torch.clamp(
        tracking_sums / (env.max_episode_length_s * tracking_scale), 0.0, 1.0
    )

    heading_sums = env.reward_manager._episode_sums[heading_reward_term_name][env_ids]
    heading_cfg = env.reward_manager.get_term_cfg(heading_reward_term_name)
    heading_scale = max(float(heading_cfg.weight), 1.0e-6)
    heading_scores = torch.clamp(
        heading_sums / (env.max_episode_length_s * heading_scale), 0.0, 1.0
    )

    env._moving_jump_tracking_score_sum += tracking_scores.sum()
    env._moving_jump_heading_score_sum += heading_scores.sum()
    env._moving_jump_episode_count += int(env_ids.numel())
    env._moving_jump_attempt_count += env._jump_attempts[env_ids].sum()
    env._moving_jump_success_count += env._jump_successes[env_ids].sum()
    env._moving_jump_soft_count += env._jump_soft_landings[env_ids].sum()

    enough_samples = (
        env._moving_jump_episode_count >= min_episodes
        and env._moving_jump_attempt_count.item() >= min_jump_attempts
    )
    if enough_samples:
        episode_count = max(env._moving_jump_episode_count, 1)
        attempt_count = env._moving_jump_attempt_count.clamp_min(1.0)
        tracking_score = (
            env._moving_jump_tracking_score_sum / episode_count
        )
        heading_score = env._moving_jump_heading_score_sum / episode_count
        jump_success = env._moving_jump_success_count / attempt_count
        soft_landing = env._moving_jump_soft_count / attempt_count

        env._command_curriculum_last_score = tracking_score.detach().clone()
        env._moving_jump_curriculum_last_success = jump_success.detach().clone()
        env._moving_jump_curriculum_last_soft = soft_landing.detach().clone()
        env._moving_jump_curriculum_last_heading = heading_score.detach().clone()

        passed = (
            tracking_score.item() >= tracking_threshold
            and jump_success.item() >= jump_success_threshold
            and soft_landing.item() >= soft_landing_threshold
            and heading_score.item() >= heading_threshold
        )
        if passed:
            env._moving_jump_curriculum_passes += 1
        else:
            # Require consecutive good windows; one regression cancels the
            # streak instead of allowing stale historical performance.
            env._moving_jump_curriculum_passes = 0

        if (
            env._moving_jump_curriculum_passes >= consecutive_passes
            and env._moving_jump_curriculum_level
            < len(env._moving_jump_speed_levels) - 1
        ):
            env._moving_jump_curriculum_level += 1
            env._moving_jump_curriculum_passes = 0
            current_limit = env._moving_jump_speed_levels[
                env._moving_jump_curriculum_level
            ]
            ranges.lin_vel_x = (-current_limit, current_limit)

        env._moving_jump_tracking_score_sum.zero_()
        env._moving_jump_heading_score_sum.zero_()
        env._moving_jump_attempt_count.zero_()
        env._moving_jump_success_count.zero_()
        env._moving_jump_soft_count.zero_()
        env._moving_jump_episode_count = 0

    return torch.tensor(
        env._moving_jump_speed_levels[env._moving_jump_curriculum_level],
        device=env.device,
    )


def obstacle_geometry_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    height_levels: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08),
    width_levels: tuple[float, ...] = (0.035, 0.050, 0.065, 0.080),
    speed_min_levels: tuple[float, ...] = (0.45, 0.50, 0.60, 0.70),
    speed_max_levels: tuple[float, ...] = (0.60, 0.65, 0.75, 0.75),
    initial_level: int = 0,
    success_threshold: float = 0.65,
    clear_threshold: float = 0.80,
    collision_threshold: float = 0.05,
    min_trials: int = 1024,
    consecutive_passes: int = 2,
) -> torch.Tensor:
    """Advance compatible obstacle geometry and approach-speed ranges."""
    level_count = len(height_levels)
    if (
        not level_count
        or len(width_levels) != level_count
        or len(speed_min_levels) != level_count
        or len(speed_max_levels) != level_count
    ):
        raise ValueError(
            "height, width, speed-min and speed-max levels must have equal "
            "non-zero length."
        )
    if any(
        low <= 0.0 or high < low
        for low, high in zip(speed_min_levels, speed_max_levels)
    ):
        raise ValueError("Each obstacle speed range must satisfy 0 < min <= max.")

    def apply_speed_range(level: int) -> None:
        # Fixed Play geometry is an evaluation override; do not silently
        # replace a command range supplied on the CLI in that mode.
        oracle_cfg = getattr(env.cfg, "obstacle_oracle", None)
        if oracle_cfg is not None and (
            oracle_cfg.fixed_height is not None or oracle_cfg.fixed_width is not None
        ):
            return
        ranges = env.command_manager.get_term("wheel_legged_commands").cfg.ranges
        ranges.lin_vel_x = (
            float(speed_min_levels[level]),
            float(speed_max_levels[level]),
        )

    if not hasattr(env, "_obstacle_curriculum_level"):
        if not 0 <= initial_level < len(height_levels):
            raise ValueError(f"initial_level must be in [0, {len(height_levels) - 1}].")
        env._obstacle_height_levels = tuple(float(v) for v in height_levels)
        env._obstacle_width_levels = tuple(float(v) for v in width_levels)
        env._obstacle_speed_min_levels = tuple(float(v) for v in speed_min_levels)
        env._obstacle_speed_max_levels = tuple(float(v) for v in speed_max_levels)
        env._obstacle_curriculum_level = int(initial_level)
        env._obstacle_curriculum_passes = 0
        env._obstacle_curriculum_trials = torch.zeros((), device=env.device)
        env._obstacle_curriculum_clears = torch.zeros((), device=env.device)
        env._obstacle_curriculum_collisions = torch.zeros((), device=env.device)
        env._obstacle_curriculum_successes = torch.zeros((), device=env.device)
        env._obstacle_curriculum_last_clear = torch.zeros((), device=env.device)
        env._obstacle_curriculum_last_collision = torch.zeros((), device=env.device)
        env._obstacle_curriculum_last_success = torch.zeros((), device=env.device)
        apply_speed_range(env._obstacle_curriculum_level)
        return torch.tensor(float(initial_level), device=env.device)

    if len(env_ids) == 0 or not hasattr(env, "_obstacle_trials"):
        return torch.tensor(float(env._obstacle_curriculum_level), device=env.device)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    env._obstacle_curriculum_trials += env._obstacle_trials[env_ids].sum()
    env._obstacle_curriculum_clears += env._obstacle_clears[env_ids].sum()
    env._obstacle_curriculum_collisions += env._obstacle_collisions[env_ids].sum()
    env._obstacle_curriculum_successes += env._obstacle_successes[env_ids].sum()

    if env._obstacle_curriculum_trials.item() >= min_trials:
        trials = env._obstacle_curriculum_trials.clamp_min(1.0)
        clear_rate = env._obstacle_curriculum_clears / trials
        collision_rate = env._obstacle_curriculum_collisions / trials
        success_rate = env._obstacle_curriculum_successes / trials
        env._obstacle_curriculum_last_clear = clear_rate.detach().clone()
        env._obstacle_curriculum_last_collision = collision_rate.detach().clone()
        env._obstacle_curriculum_last_success = success_rate.detach().clone()
        passed = (
            success_rate.item() >= success_threshold
            and clear_rate.item() >= clear_threshold
            and collision_rate.item() <= collision_threshold
        )
        env._obstacle_curriculum_passes = (
            env._obstacle_curriculum_passes + 1 if passed else 0
        )
        if (
            env._obstacle_curriculum_passes >= consecutive_passes
            and env._obstacle_curriculum_level < len(env._obstacle_height_levels) - 1
        ):
            env._obstacle_curriculum_level += 1
            env._obstacle_curriculum_passes = 0
            apply_speed_range(env._obstacle_curriculum_level)
        env._obstacle_curriculum_trials.zero_()
        env._obstacle_curriculum_clears.zero_()
        env._obstacle_curriculum_collisions.zero_()
        env._obstacle_curriculum_successes.zero_()
    return torch.tensor(float(env._obstacle_curriculum_level), device=env.device)


def wheel_legged_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg=None,
    command_name: str = "wheel_legged_commands",
    minimum_command_speed: float = 0.15,
    promotion_tracking_ratio: float = 0.70,
    demotion_tracking_ratio: float = 0.35,
    minimum_command_distance: float = 0.75,
) -> torch.Tensor:
    """Adjust terrain rows from command-aligned progress over an episode.

    Net world displacement is unsuitable because the command can reverse during
    an episode.  The robust environment therefore accumulates commanded travel
    and travel in the requested body-x direction.  Turning in place and nearly
    stationary commands neither promote nor demote a terrain.
    """
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    terrain = env.scene.terrain
    if terrain.cfg.terrain_type != "generator":
        return torch.zeros((), device=env.device)
    if len(env_ids) == 0:
        return torch.mean(terrain.terrain_levels.float())

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if not hasattr(env, "_terrain_commanded_distance"):
        raise RuntimeError(
            "wheel_legged_terrain_levels requires WheelLeggedRobustEnv progress buffers."
        )
    commanded_distance = env._terrain_commanded_distance[env_ids]
    tracked_distance = env._terrain_tracking_distance[env_ids]
    tracking_ratio = tracked_distance / commanded_distance.clamp_min(1.0e-6)
    # Keep the speed parameter meaningful if episode duration is changed.
    enough_command = commanded_distance >= max(
        minimum_command_distance,
        minimum_command_speed * env.max_episode_length_s * 0.25,
    )
    move_up = enough_command & (tracking_ratio >= promotion_tracking_ratio)
    move_down = enough_command & (tracking_ratio < demotion_tracking_ratio)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
