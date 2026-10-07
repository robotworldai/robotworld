# Copyright (c) 2024-2026
# SPDX-License-Identifier: BSD-3-Clause

"""Shared play and debug helpers for the wheel-legged training scripts."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

import torch


# Environment ``extras["log"]`` keys and their compact terminal labels.
WHEEL_LEGGED_DEBUG_METRICS: tuple[tuple[str, str], ...] = (
    ("cmd_vx_abs", "|vx|"),
    ("cmd_vx_limit", "vx_lim"),
    ("curriculum_score", "cur"),
    ("moving_jump_curriculum_level", "MJ档"),
    ("moving_jump_curriculum_success", "MJ成功"),
    ("moving_jump_curriculum_soft", "MJ柔落"),
    ("moving_jump_curriculum_heading", "MJ航向"),
    ("moving_jump_curriculum_passes", "MJ连过"),
    ("cmd_wz_limit", "wz_lim"),
    ("curriculum_angular_score", "cur_wz"),
    ("vel_x_abs", "|vel|"),
    ("vx_err_inst", "vx_err"),
    ("vx_tracking_gain", "gain"),
    ("vx_sign_match", "sign"),
    ("cmd_wz_abs", "|wz|"),
    ("omega_z_abs", "|ωz|"),
    ("wz_err_inst", "wz_err"),
    ("wz_tracking_gain", "gain_wz"),
    ("heading_err_abs", "hdg_err"),
    ("cmd_height", "cmd_h"),
    ("base_height", "height"),
    ("h_err_inst", "h_err"),
    ("theta0_L", "θ0_L"),
    ("theta0_R", "θ0_R"),
    ("L0_L", "L0_L"),
    ("L0_R", "L0_R"),
    ("rf0_pos", "rf0"),
    ("lf0_pos", "lf0"),
    ("joint_margin_min", "joint_margin"),
    ("torque_saturation", "tau_clip"),
    ("motor_enable_scale", "motor"),
    ("tilt_angle", "tilt"),
    ("wheel_action_abs", "|a_w|"),
    ("wheel_action_clip_fraction", "clip_w"),
    ("wheel_vel_abs", "|ω_w|"),
    ("recovery_success_rate", "R成功"),
    ("recovery_failure_rate", "R失败"),
    ("recovery_mean_time", "R用时"),
    ("recovery_pending_rate", "R进行"),
    ("terrain_level", "T档"),
    ("terrain_tracking_ratio", "T跟踪"),
    ("jump_phase_crouch", "J蹲"),
    ("jump_phase_thrust", "J蹬"),
    ("jump_phase_flight", "J空"),
    ("jump_phase_landing", "J落"),
    ("jump_phase_recovery", "J稳"),
    ("jump_target_height", "J目标"),
    ("jump_target_distance", "J目标距"),
    ("jump_target_vx", "J目标vx"),
    ("jump_takeoff_vz", "J起速"),
    ("jump_takeoff_vx_error", "J起vx误差"),
    ("jump_landing_vz", "J落速"),
    ("jump_landing_vx_error", "J落vx误差"),
    ("jump_landing_leg_length", "J落L"),
    ("jump_apex_rise", "J机身升"),
    ("jump_wheel_clearance", "J轮高"),
    ("jump_air_time", "J空时"),
    ("jump_success_rate", "J成功"),
    ("jump_soft_landing_rate", "J柔落"),
    ("jump_landing_position_error", "J落点误差"),
    ("jump_target_landing_rate", "J落点成功"),
    ("jump_fail_crouch_rate", "F蹲"),
    ("jump_fail_thrust_rate", "F蹬"),
    ("jump_fail_unload_rate", "F卸"),
    ("jump_fail_performance_rate", "F性能"),
    ("jump_fail_recovery_rate", "F恢复"),
    ("jump_crouch_l0", "J蹲L"),
    ("jump_thrust_l0", "J蹬L"),
    ("jump_thrust_l0_dot", "J蹬dL"),
    ("obstacle_height", "O高"),
    ("obstacle_width", "O宽"),
    ("obstacle_curriculum_level", "O档"),
    ("obstacle_curriculum_success", "O课成功"),
    ("obstacle_curriculum_clear", "O课跨越"),
    ("obstacle_curriculum_collision", "O课碰撞"),
    ("obstacle_trigger_error", "O触发误差"),
    ("obstacle_min_clearance", "O净空"),
    ("obstacle_clear_rate", "O跨越"),
    ("obstacle_collision_rate", "O碰撞"),
    ("obstacle_collision_force_mean", "O撞力均"),
    ("obstacle_collision_force_peak", "O撞力峰"),
    ("obstacle_collision_base_fraction", "O撞基"),
    ("obstacle_collision_upper_leg_fraction", "O撞大腿"),
    ("obstacle_collision_lower_leg_fraction", "O撞小腿"),
    ("obstacle_collision_wheel_fraction", "O撞轮"),
    ("obstacle_success_rate", "O成功"),
    ("obstacle_perception_valid_rate", "P有效"),
    ("obstacle_perception_trigger_distance", "P触发距"),
    ("obstacle_perception_distance_error", "P距误差"),
    ("obstacle_perception_height_error", "P高误差"),
    ("obstacle_perception_width_error", "P宽误差"),
    ("obstacle_perception_distance", "P距离"),
    ("obstacle_perception_height", "P高度"),
    ("obstacle_perception_width", "P宽度"),
)


def _as_scalar(value) -> float | None:
    """Convert scalar/tensor metric values to a finite Python float."""
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.float().mean().item()
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value


def aggregate_debug_metrics(extras: Sequence[Mapping]) -> dict[str, float]:
    """Average selected debug metrics across one RSL-RL rollout."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in extras:
        for key, _ in WHEEL_LEGGED_DEBUG_METRICS:
            if key not in item:
                continue
            value = _as_scalar(item[key])
            if value is None:
                continue
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: total / counts[key] for key, total in totals.items()}


def format_debug_metrics(info: Mapping[str, object]) -> str:
    """Format wheel-legged metrics as one compact terminal line."""
    parts: list[str] = []
    for key, label in WHEEL_LEGGED_DEBUG_METRICS:
        if key not in info:
            continue
        value = _as_scalar(info[key])
        if value is not None:
            parts.append(f"{label}={value:.4f}")
    return " │ " + "  ".join(parts) + " │" if parts else ""


class TrainingEarlyStop(RuntimeError):
    """Raised after an iteration has been logged when a stage gate is satisfied."""


def install_rsl_rl_debug_logger(runner, metric_callback=None) -> None:
    """Add a compact debug line while preserving RSL-RL/TensorBoard logging.

    RSL-RL stores environment ``extras["log"]`` entries in ``logger.ep_extras``.
    The official logger normally prints every entry on a separate line. This
    wrapper keeps all scalar writes, requests the compact official summary when
    supported, and prints the selected robot metrics in one line.
    """

    logger = runner.logger
    original_log = logger.log
    supports_minimal = "print_minimal" in inspect.signature(original_log).parameters

    def log_with_wheel_legged_debug(*args, **kwargs):
        debug_info = aggregate_debug_metrics(getattr(logger, "ep_extras", ()))
        if supports_minimal:
            kwargs["print_minimal"] = True
        result = original_log(*args, **kwargs)
        debug_line = format_debug_metrics(debug_info)
        if debug_line and not getattr(logger, "disable_logs", False):
            print(debug_line)
        if metric_callback is not None:
            metric_callback(debug_info, kwargs.get("it", args[0] if args else None))
        return result

    logger.log = log_with_wheel_legged_debug


def install_rsl_rl_action_std_bounds(
    runner, *, min_std: float = 0.05, max_std: float = 0.50
) -> None:
    """Clamp learned Gaussian exploration noise after every PPO update."""
    actor = getattr(runner.alg, "actor", None)
    distribution = getattr(actor, "distribution", None)
    if distribution is None:
        raise RuntimeError("RSL-RL actor has no supported action distribution.")

    def clamp_std():
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                distribution.std_param.clamp_(min=min_std, max=max_std)
            elif hasattr(distribution, "log_std_param"):
                distribution.log_std_param.clamp_(
                    min=torch.log(torch.tensor(min_std)).item(),
                    max=torch.log(torch.tensor(max_std)).item(),
                )
            else:
                raise RuntimeError("Unsupported RSL-RL Gaussian std parameterization.")

    clamp_std()
    original_update = runner.alg.update

    def update_with_bounded_std(*args, **kwargs):
        result = original_update(*args, **kwargs)
        clamp_std()
        return result

    runner.alg.update = update_with_bounded_std
    print(f"[INFO] Action exploration std bounded to [{min_std:.3f}, {max_std:.3f}].")


class WheelLeggedKeyboardController:
    """Inject keyboard locomotion, jump and power-cycle requests during playback."""

    def __init__(
        self,
        env,
        *,
        vx_limit: float,
        wz_limit: float,
        default_height: float = 0.21,
        height_step: float = 0.01,
        jump_height: float | None = None,
        jump_distance: float | None = None,
    ):
        self.env = env.unwrapped
        self.command_term = self.env.command_manager.get_term("wheel_legged_commands")
        self.height_min = float(self.command_term.cfg.ranges.height[0])
        self.height_max = float(self.command_term.cfg.ranges.height[1])
        self.default_height = min(max(default_height, self.height_min), self.height_max)
        self.height = self.default_height
        self.height_step = height_step
        self.jump_command_term = self.env.command_manager._terms.get("jump_command")
        self._jump_pulse_steps_remaining = 0
        self._jump_pulse_steps = 1
        self._power_cycle_requested = False
        self.jump_height = 0.0
        self.jump_distance = 0.0
        if self.jump_command_term is not None:
            jump_range = self.jump_command_term.cfg.target_height_range
            default_jump_height = 0.5 * (float(jump_range[0]) + float(jump_range[1]))
            self.jump_height = (
                default_jump_height if jump_height is None else float(jump_height)
            )
            self.jump_height = min(
                max(self.jump_height, float(jump_range[0])),
                float(jump_range[1]),
            )
            distance_range = self.jump_command_term.cfg.target_distance_range
            default_jump_distance = 0.5 * (
                float(distance_range[0]) + float(distance_range[1])
            )
            self.jump_distance = (
                default_jump_distance
                if jump_distance is None
                else float(jump_distance)
            )
            self.jump_distance = min(
                max(self.jump_distance, float(distance_range[0])),
                float(distance_range[1]),
            )
            self._jump_pulse_steps = max(
                1,
                round(
                    float(self.jump_command_term.cfg.trigger_pulse_time)
                    / float(self.env.step_dt)
                ),
            )

        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

        self.keyboard = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=vx_limit,
                v_y_sensitivity=0.0,
                omega_z_sensitivity=wz_limit,
                sim_device=str(self.env.device),
            )
        )
        self.keyboard.add_callback("R", self._raise_height)
        self.keyboard.add_callback("F", self._lower_height)
        self.keyboard.add_callback("L", self._reset_height)
        self.keyboard.add_callback("K", self._request_power_cycle)
        if self.jump_command_term is not None:
            self.keyboard.add_callback("J", self._request_jump)

        # Keep CommandManager, policy observations and debug metrics on one
        # command source. Reset/resampling must not replace keyboard commands.
        self.command_term.cfg.heading_command = False
        self.command_term.cfg.resampling_time_range = (1.0e9, 1.0e9)
        self.command_term._resample_command = lambda env_ids: None
        self.command_term.set_debug_vis(False)
        if self.jump_command_term is not None:
            # Disable the autonomous JumpCommand sampler. J emits the same
            # finite trigger pulse duration used during policy training.
            self.jump_command_term.cfg.jump_probability = 0.0
            self.jump_command_term.cfg.resampling_time_range = (1.0e9, 1.0e9)
            self.jump_command_term._enabled.zero_()
            self.jump_command_term._command[:, 0] = 0.0
            self.jump_command_term._command[:, 1] = self.jump_height
            self.jump_command_term._command[:, 2] = self.jump_distance
        self.update()

        print(self.keyboard)
        print(
            "\tRaise body height: R\n"
            "\tLower body height: F\n"
            "\tPower off / restart: K\n"
            f"\tHeight range: [{self.height_min:.2f}, {self.height_max:.2f}] m\n"
            "\tLateral keys are disabled for this non-holonomic robot."
        )
        if self.jump_command_term is not None:
            print(
                "\tJump: J (one jump per press)\n"
                f"\tJump target height: {self.jump_height:.3f} m"
                f", distance: {self.jump_distance:.3f} m"
            )

    def _raise_height(self):
        self.height = min(self.height + self.height_step, self.height_max)
        print(f"[KEYBOARD] height={self.height:.3f} m")

    def _lower_height(self):
        self.height = max(self.height - self.height_step, self.height_min)
        print(f"[KEYBOARD] height={self.height:.3f} m")

    def _reset_height(self):
        self.height = self.default_height
        print(f"[KEYBOARD] commands reset, height={self.height:.3f} m")

    def _request_jump(self):
        # A pulse during an active maneuver cannot start another jump and
        # should not be queued to fire unexpectedly after recovery.
        if hasattr(self.env, "jump_phase") and int(self.env.jump_phase[0].item()) != 0:
            print("[KEYBOARD] jump ignored: previous jump is still active.")
            return
        self._jump_pulse_steps_remaining = self._jump_pulse_steps
        print(
            "[KEYBOARD] jump requested, "
            f"height={self.jump_height:.3f} m, distance={self.jump_distance:.3f} m"
        )

    def _request_power_cycle(self):
        self._power_cycle_requested = True

    def consume_power_cycle_request(self) -> bool:
        """Return and clear the edge-triggered K-key request."""
        requested = self._power_cycle_requested
        self._power_cycle_requested = False
        return requested

    def stop_commands(self, *, cancel_jump: bool = True) -> None:
        """Hold all user commands at zero while actuator power is unavailable."""
        command = self.command_term.command
        command[:, 0] = 0.0
        command[:, 1] = 0.0
        command[:, 2] = self.height
        command[:, 3] = 0.0
        if cancel_jump and self.jump_command_term is not None:
            self._jump_pulse_steps_remaining = 0
            self.jump_command_term.command[:, 0] = 0.0

    def cancel_active_jump(self) -> None:
        """Cancel Play-only jump state before simulating actuator power loss."""
        self.stop_commands(cancel_jump=True)
        if hasattr(self.env, "jump_phase"):
            self.env.jump_phase.zero_()
            self.env.jump_phase_time.zero_()

    def update(self):
        """Copy keyboard state and advance the finite jump trigger pulse."""
        keyboard_command = self.keyboard.advance().to(self.command_term.command.device)
        command = self.command_term.command
        command[:, 0] = keyboard_command[0]
        command[:, 1] = keyboard_command[2]
        command[:, 2] = self.height
        command[:, 3] = 0.0
        if self.jump_command_term is not None:
            jump_command = self.jump_command_term.command
            jump_command[:, 0] = float(self._jump_pulse_steps_remaining > 0)
            jump_command[:, 1] = self.jump_height
            jump_command[:, 2] = self.jump_distance
            self._jump_pulse_steps_remaining = max(
                0, self._jump_pulse_steps_remaining - 1
            )


def camera_follow(env):
    """Smoothly follow environment zero with the viewer camera."""
    import isaaclab.utils.math as math_utils

    if env.unwrapped.viewport_camera_controller is None:
        return
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = torch.tensor([-3.0, 0.0, 0.5], dtype=torch.float32, device=robot_pos.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )
