# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

"""Jump commands, observations and phase-aware rewards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .actions import VMCAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


JUMP_PHASE_IDLE = 0
JUMP_PHASE_CROUCH = 1
JUMP_PHASE_THRUST = 2
JUMP_PHASE_FLIGHT = 3
JUMP_PHASE_LANDING = 4
JUMP_PHASE_RECOVERY = 5
NUM_JUMP_PHASES = 6


class JumpVMCAction(VMCAction):
    """VMC action with a phase reference and learned leg-length residual.

    The open-loop reference is only applied to both leg-length channels. Leg
    angle and wheel-speed actions remain fully controlled by the policy.
    """

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if not hasattr(self._env, "jump_phase"):
            return
        state_cfg = self._env.cfg.jump_state_machine
        if not state_cfg.use_leg_reference_assist:
            return

        phase = self._env.jump_phase
        target_length = jump_leg_reference(self._env)
        landing = (phase == JUMP_PHASE_FLIGHT) | (phase == JUMP_PHASE_LANDING)
        active = (
            (phase == JUMP_PHASE_CROUCH)
            | (phase == JUMP_PHASE_THRUST)
            | landing
        )
        target_action = torch.clamp(
            (target_length - self.cfg.l0_offset) / self.cfg.action_scale_l0,
            -1.0,
            1.0,
        )
        residual_scale = torch.full(
            (self.num_envs,), state_cfg.leg_action_residual_scale, device=self.device
        )
        if state_cfg.use_landing_assist:
            prelanding_threshold = (
                state_cfg.prelanding_start_vz
                if state_cfg.use_split_flight_control
                else 0.0
            )
            descending = (phase == JUMP_PHASE_FLIGHT) & (
                self._env.scene["robot"].data.root_lin_vel_w[:, 2]
                < prelanding_threshold
            )
            residual_scale = torch.where(
                descending,
                torch.full_like(residual_scale, state_cfg.prelanding_action_residual_scale),
                residual_scale,
            )
            residual_scale = torch.where(
                phase == JUMP_PHASE_LANDING,
                torch.full_like(residual_scale, state_cfg.landing_action_residual_scale),
                residual_scale,
            )
        for action_index in (1, 4):
            assisted = target_action + residual_scale * self._processed_actions[:, action_index]
            self._processed_actions[:, action_index] = torch.where(
                active,
                torch.clamp(assisted, -self.cfg.action_clip, self.cfg.action_clip),
                self._processed_actions[:, action_index],
            )

    def _get_l0_control_parameters(self):
        """Increase thrust authority and landing damping only in the new stage."""
        l0_kp, l0_kd, feedforward = super()._get_l0_control_parameters()
        state_cfg = self._env.cfg.jump_state_machine
        if not state_cfg.use_phase_dependent_gains:
            return l0_kp, l0_kd, feedforward

        phase = self._env.jump_phase
        thrust = (phase == JUMP_PHASE_THRUST).unsqueeze(-1)
        flight = (phase == JUMP_PHASE_FLIGHT).unsqueeze(-1)
        root_vz = self._env.scene["robot"].data.root_lin_vel_w[:, 2]
        descending_flight = (
            (phase == JUMP_PHASE_FLIGHT) & (root_vz < 0.0)
        ).unsqueeze(-1)
        prelanding_flight = (
            (phase == JUMP_PHASE_FLIGHT)
            & (root_vz < state_cfg.prelanding_start_vz)
        ).unsqueeze(-1)
        retract_flight = flight & ~prelanding_flight
        landing = (phase == JUMP_PHASE_LANDING).unsqueeze(-1)
        touchdown_contact = (
            (phase == JUMP_PHASE_FLIGHT)
            & self._env.jump_wheel_contact.any(dim=1)
            & (self._env.scene["robot"].data.root_lin_vel_w[:, 2] < 0.0)
        ).unsqueeze(-1)
        absorbing = landing | descending_flight

        l0_kp = torch.where(thrust, l0_kp * state_cfg.thrust_kp_scale, l0_kp)
        if state_cfg.use_split_flight_control:
            l0_kp = torch.where(
                retract_flight,
                l0_kp * state_cfg.flight_retract_kp_scale,
                l0_kp,
            )
            l0_kd = torch.where(
                retract_flight,
                l0_kd * state_cfg.flight_retract_kd_scale,
                l0_kd,
            )
        if state_cfg.use_split_flight_control:
            l0_kd = torch.where(
                prelanding_flight,
                l0_kd * state_cfg.prelanding_kd_scale,
                l0_kd,
            )
            l0_kd = torch.where(
                landing, l0_kd * state_cfg.landing_kd_scale, l0_kd
            )
        else:
            l0_kd = torch.where(
                absorbing, l0_kd * state_cfg.landing_kd_scale, l0_kd
            )
        feedforward = torch.where(
            thrust,
            feedforward * state_cfg.thrust_feedforward_scale,
            feedforward,
        )
        # The nominal feed-forward balances body weight on the ground.  In
        # flight it has no load to balance and otherwise launches the wheels
        # past their requested pre-landing length.
        if state_cfg.use_split_flight_control:
            # During ascent/apex the wheels must be pulled toward the body.
            # Weight-support feed-forward opposes that motion in free flight,
            # so use an independent near-zero scale. Re-enable support only
            # once the pre-landing extension starts.
            feedforward = torch.where(
                retract_flight,
                feedforward * state_cfg.flight_retract_feedforward_scale,
                feedforward,
            )
            feedforward = torch.where(
                prelanding_flight,
                feedforward * state_cfg.prelanding_feedforward_scale,
                feedforward,
            )
        else:
            feedforward = torch.where(
                flight,
                feedforward * state_cfg.flight_feedforward_scale,
                feedforward,
            )
        feedforward = torch.where(
            landing,
            feedforward * state_cfg.landing_feedforward_scale,
            feedforward,
        )
        # Contact confirmation deliberately takes multiple frames, but impact
        # support must begin after the very first measured contact rather than
        # waiting for the phase transition.
        feedforward = torch.where(
            touchdown_contact,
            torch.full_like(feedforward, self.cfg.feedforward_force)
            * state_cfg.landing_feedforward_scale,
            feedforward,
        )
        return l0_kp, l0_kd, feedforward


def _height_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalize commanded jump height for height-conditioned references."""
    state_cfg = env.cfg.jump_state_machine
    command = env.command_manager.get_command("jump_command")
    denominator = max(
        state_cfg.reference_height_max - state_cfg.reference_height_min, 1.0e-6
    )
    return torch.clamp(
        (command[:, 1] - state_cfg.reference_height_min) / denominator, 0.0, 1.0
    )


def jump_leg_reference(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the continuous phase-aware physical leg-length reference.

    The baseline stage retains its original fixed references.  When landing
    assistance is enabled, the legs retract during ascent, extend smoothly
    during descent, and then compress smoothly after contact to absorb energy.
    """
    state_cfg = env.cfg.jump_state_machine
    phase = env.jump_phase
    target = torch.full(
        (env.num_envs,), env.action_manager.get_term("vmc").cfg.l0_offset, device=env.device
    )

    crouch_target = torch.full_like(target, state_cfg.crouch_length)
    if state_cfg.use_height_conditioned_crouch:
        fraction = _height_fraction(env)
        crouch_target = state_cfg.crouch_length + fraction * (
            state_cfg.high_jump_crouch_length - state_cfg.crouch_length
        )
    target = torch.where(phase == JUMP_PHASE_CROUCH, crouch_target, target)
    target = torch.where(
        phase == JUMP_PHASE_THRUST,
        torch.full_like(target, state_cfg.thrust_length),
        target,
    )

    if state_cfg.use_landing_assist:
        root_vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
        # 0 at the apex threshold and 1 at/below the full-extension speed.
        descent_denominator = max(
            state_cfg.prelanding_start_vz - state_cfg.prelanding_full_vz, 1.0e-6
        )
        descent_fraction = torch.clamp(
            (state_cfg.prelanding_start_vz - root_vz) / descent_denominator,
            0.0,
            1.0,
        )
        flight_target = state_cfg.flight_retract_length + descent_fraction * (
            state_cfg.prelanding_length - state_cfg.flight_retract_length
        )
        target = torch.where(phase == JUMP_PHASE_FLIGHT, flight_target, target)

        compression_fraction = torch.clamp(
            env.jump_phase_time / max(state_cfg.landing_compression_time, 1.0e-6),
            0.0,
            1.0,
        )
        landing_target = state_cfg.prelanding_length + compression_fraction * (
            state_cfg.landing_absorption_length - state_cfg.prelanding_length
        )
        target = torch.where(phase == JUMP_PHASE_LANDING, landing_target, target)
    else:
        landing = (phase == JUMP_PHASE_FLIGHT) | (phase == JUMP_PHASE_LANDING)
        target = torch.where(
            landing,
            torch.full_like(target, state_cfg.landing_length),
            target,
        )
    return target


class JumpCommand(CommandTerm):
    """Generate ``[trigger, target_height, target_distance]`` jump commands.

    The trigger is a short pulse inside each command cycle. Some cycles contain
    no jump so the final policy continues practicing ordinary locomotion.
    """

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._trigger_delay = torch.zeros(self.num_envs, device=self.device)
        self._enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pulse_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        count = len(env_ids)
        if count == 0:
            return
        self._elapsed[env_ids] = 0.0
        self._enabled[env_ids] = torch.rand(count, device=self.device) < self.cfg.jump_probability
        self._trigger_delay[env_ids] = torch.empty(count, device=self.device).uniform_(
            *self.cfg.trigger_delay_range
        )
        target_height = torch.empty(count, device=self.device).uniform_(*self.cfg.target_height_range)
        target_distance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.target_distance_range
        )
        if self.cfg.couple_distance_to_velocity:
            target_distance = self._distance_from_velocity(env_ids)
        self._command[env_ids, 0] = 0.0
        self._command[env_ids, 1] = torch.where(
            self._enabled[env_ids], target_height, torch.zeros_like(target_height)
        )
        self._command[env_ids, 2] = torch.where(
            self._enabled[env_ids], target_distance, torch.zeros_like(target_distance)
        )

    def _update_command(self):
        self._elapsed += self._env.step_dt
        pulse = (
            self._enabled
            & (self._elapsed >= self._trigger_delay)
            & (self._elapsed < self._trigger_delay + self.cfg.trigger_pulse_time)
        )
        # Recompute and latch the target on the rising edge. This uses the
        # locomotion command that will actually be held by the jump state
        # machine, instead of a possibly stale value from command resampling.
        pulse_start = pulse & ~self._pulse_active
        if self.cfg.couple_distance_to_velocity and pulse_start.any():
            env_ids = pulse_start.nonzero(as_tuple=False).squeeze(-1)
            self._command[env_ids, 2] = self._distance_from_velocity(env_ids)
        self._command[:, 0] = pulse.float()
        self._pulse_active[:] = pulse

    def _distance_from_velocity(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Predict signed landing displacement from commanded takeoff speed."""
        count = len(env_ids)
        if count == 0:
            return torch.empty(0, device=self.device)
        locomotion_command = self._env.command_manager.get_command(
            self.cfg.velocity_command_name
        )
        vx = locomotion_command[env_ids, 0]
        noise = torch.empty(count, device=self.device).uniform_(
            *self.cfg.distance_noise_range
        )
        distance = vx * self.cfg.expected_air_time * noise
        return torch.clamp(
            distance,
            min=self.cfg.target_distance_range[0],
            max=self.cfg.target_distance_range[1],
        )

    def _update_metrics(self):
        pass


@configclass
class JumpCommandCfg(CommandTermCfg):
    """Configuration for externally triggered small jumps."""

    class_type: type[CommandTerm] = JumpCommand
    resampling_time_range: tuple[float, float] = (4.0, 6.0)
    jump_probability: float = 0.7
    trigger_delay_range: tuple[float, float] = (0.8, 1.5)
    trigger_pulse_time: float = 0.10
    target_height_range: tuple[float, float] = (0.03, 0.07)
    target_distance_range: tuple[float, float] = (0.0, 0.0)
    couple_distance_to_velocity: bool = False
    velocity_command_name: str = "wheel_legged_commands"
    expected_air_time: float = 0.16
    distance_noise_range: tuple[float, float] = (1.0, 1.0)


def jump_phase_one_hot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Six-state phase encoding ordered as idle/crouch/thrust/flight/landing/recovery."""
    if not hasattr(env, "jump_phase"):
        return torch.zeros(env.num_envs, NUM_JUMP_PHASES, device=env.device)
    return torch.nn.functional.one_hot(env.jump_phase, num_classes=NUM_JUMP_PHASES).float()


def jump_wheel_contacts(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary left/right wheel contact state."""
    if not hasattr(env, "jump_wheel_contact"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env.jump_wheel_contact.float()


def jump_command(
    env: ManagerBasedRLEnv,
    command_name: str = "jump_command",
    scale: tuple[float, float, float] = (1.0, 10.0, 5.0),
) -> torch.Tensor:
    """Scaled trigger, desired rise and desired forward distance."""
    try:
        command = env.command_manager.get_command(command_name)
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return command * torch.tensor(scale, device=env.device)


def jump_phase_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time in the current phase, clipped and normalized to one second."""
    if not hasattr(env, "jump_phase_time"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return torch.clamp(env.jump_phase_time, 0.0, 1.0).unsqueeze(-1)


def _phase_mask(env: ManagerBasedRLEnv, *phases: int) -> torch.Tensor:
    if not hasattr(env, "jump_phase"):
        return torch.zeros(env.num_envs, device=env.device)
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for phase in phases:
        mask |= env.jump_phase == phase
    return mask.float()


def jump_gated_lin_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize vertical velocity only outside crouch/thrust/flight."""
    vz = env.scene["robot"].data.root_lin_vel_b[:, 2]
    gate = _phase_mask(env, JUMP_PHASE_IDLE, JUMP_PHASE_LANDING, JUMP_PHASE_RECOVERY)
    return vz.square() * gate


def jump_gated_base_height_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.05,
) -> torch.Tensor:
    """Track normal commanded height only when it does not oppose takeoff."""
    command = env.command_manager.get_command(command_name)
    error = env.base_height - command[:, 2]
    upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    gate = _phase_mask(env, JUMP_PHASE_IDLE, JUMP_PHASE_RECOVERY)
    return torch.exp(-error.square() / std**2) * upright * gate


def jump_crouch_tracking(env: ManagerBasedRLEnv, std: float = 0.015) -> torch.Tensor:
    """Track the state-machine crouch length during preparation."""
    if not hasattr(env, "L0"):
        return torch.zeros(env.num_envs, device=env.device)
    target = jump_leg_reference(env)
    error = env.L0.mean(dim=1) - target
    return torch.exp(-error.square() / std**2) * _phase_mask(env, JUMP_PHASE_CROUCH)


def jump_leg_action_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Imitate the proven open-loop leg-length references in early training.

    This is deliberately a curriculum shaping term. It gives useful credit
    before the delayed physical state has reached the requested leg length.
    """
    if not hasattr(env, "jump_phase"):
        return torch.zeros(env.num_envs, device=env.device)
    term = env.action_manager.get_term("vmc")
    cfg = term.cfg
    actions = term.processed_actions
    target_length = jump_leg_reference(env)
    target_action = torch.clamp(
        (target_length - cfg.l0_offset) / cfg.action_scale_l0, -1.0, 1.0
    )
    error = 0.5 * (
        (actions[:, 1] - target_action).square()
        + (actions[:, 4] - target_action).square()
    )
    active = _phase_mask(
        env, JUMP_PHASE_CROUCH, JUMP_PHASE_THRUST, JUMP_PHASE_FLIGHT, JUMP_PHASE_LANDING
    )
    return torch.clamp(1.0 - 0.75 * error, min=0.0) * active


def jump_thrust_length_tracking(env: ManagerBasedRLEnv, std: float = 0.020) -> torch.Tensor:
    """Guide the initial policy search toward rapid leg extension during thrust."""
    if not hasattr(env, "L0"):
        return torch.zeros(env.num_envs, device=env.device)
    target = env.cfg.jump_state_machine.thrust_length
    error = env.L0.mean(dim=1) - target
    return torch.exp(-error.square() / std**2) * _phase_mask(env, JUMP_PHASE_THRUST)


def jump_thrust_extension_speed(
    env: ManagerBasedRLEnv, target_speed: float = 0.8
) -> torch.Tensor:
    """Dense credit for rapidly extending both legs during thrust."""
    if not hasattr(env, "L0_dot"):
        return torch.zeros(env.num_envs, device=env.device)
    extension_speed = env.L0_dot.mean(dim=1)
    progress = torch.clamp(extension_speed / target_speed, 0.0, 1.0)
    return progress * _phase_mask(env, JUMP_PHASE_THRUST)


def jump_takeoff_velocity(env: ManagerBasedRLEnv, std: float = 0.35) -> torch.Tensor:
    """Reward the ballistic takeoff velocity implied by target jump height."""
    command = env.command_manager.get_command("jump_command")
    target_vz = torch.sqrt(torch.clamp(2.0 * 9.81 * command[:, 1], min=1.0e-4))
    vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
    progress = torch.clamp(vz / target_vz, 0.0, 1.0)
    tracking = torch.exp(-(vz - target_vz).square() / std**2)
    return (0.7 * progress + 0.3 * tracking) * _phase_mask(env, JUMP_PHASE_THRUST)


def jump_takeoff_event_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Immediate credit assignment when a physically valid takeoff is detected."""
    if not hasattr(env, "jump_takeoff_event"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.jump_takeoff_event.float()


def jump_height_tracking(env: ManagerBasedRLEnv, std: float = 0.025) -> torch.Tensor:
    """Track base rise during confirmed flight.

    The success state separately requires confirmed takeoff and minimum air
    time, so grounded leg extension can no longer satisfy this objective.
    Wheel-center rise remains a diagnostic until terrain-relative clearance is
    introduced with the obstacle curriculum.
    """
    if not hasattr(env, "jump_start_height"):
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command("jump_command")
    rise = env.scene["robot"].data.root_pos_w[:, 2] - env.jump_start_height
    target = torch.clamp(command[:, 1], min=1.0e-4)
    progress = torch.clamp(rise / target, 0.0, 1.0)
    tracking = torch.exp(-(rise - target).square() / std**2)
    return (0.6 * progress + 0.4 * tracking) * _phase_mask(env, JUMP_PHASE_FLIGHT)


def jump_wheel_clearance_tracking(
    env: ManagerBasedRLEnv, std: float = 0.025
) -> torch.Tensor:
    """Track actual wheel-center rise while genuinely airborne.

    Base rise is not obstacle clearance: extending the legs can raise the base
    while the wheels remain near the floor. This term directly rewards the
    minimum of the two wheel-center heights relative to jump start, so both
    legs must retract during flight.
    """
    if not hasattr(env, "jump_start_wheel_height"):
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command("jump_command")
    wheel_height = env.scene["robot"].data.body_pos_w[
        :, env._jump_robot_wheel_body_ids, 2
    ].amin(dim=1)
    clearance = wheel_height - env.jump_start_wheel_height
    target = torch.clamp(command[:, 1], min=1.0e-4)
    progress = torch.clamp(clearance / target, 0.0, 1.0)
    tracking = torch.exp(-(clearance - target).square() / std**2)
    airborne = (
        (~env.jump_wheel_contact.any(dim=1))
        & env._jump_airborne_ever
        & (env.jump_phase == JUMP_PHASE_FLIGHT)
    )
    return (0.7 * progress + 0.3 * tracking) * airborne.float()


def jump_moving_velocity_tracking(
    env: ManagerBasedRLEnv, std: float = 0.20
) -> torch.Tensor:
    """Track the latched pre-jump forward speed outside free flight."""
    if not hasattr(env, "jump_target_vx"):
        return torch.zeros(env.num_envs, device=env.device)
    command = env.command_manager.get_command("wheel_legged_commands")
    active = env.jump_phase != JUMP_PHASE_IDLE
    target_vx = torch.where(active, env.jump_target_vx, command[:, 0])
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    quality = torch.exp(-(vx - target_vx).square() / std**2)
    gate = _phase_mask(
        env,
        JUMP_PHASE_IDLE,
        JUMP_PHASE_CROUCH,
        JUMP_PHASE_THRUST,
        JUMP_PHASE_LANDING,
        JUMP_PHASE_RECOVERY,
    )
    return quality * gate


def jump_flight_velocity_preservation(
    env: ManagerBasedRLEnv, std: float = 0.25
) -> torch.Tensor:
    """Preserve commanded horizontal velocity while both wheels are airborne."""
    if not hasattr(env, "jump_target_vx"):
        return torch.zeros(env.num_envs, device=env.device)
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    quality = torch.exp(-(vx - env.jump_target_vx).square() / std**2)
    airborne = (~env.jump_wheel_contact.any(dim=1)) & env._jump_airborne_ever
    return quality * airborne.float() * _phase_mask(env, JUMP_PHASE_FLIGHT)


def jump_landing_velocity_tracking(
    env: ManagerBasedRLEnv, std: float = 0.20
) -> torch.Tensor:
    """Event reward for retaining the requested speed at first touchdown."""
    if not hasattr(env, "jump_landing_vx"):
        return torch.zeros(env.num_envs, device=env.device)
    quality = torch.exp(
        -(env.jump_landing_vx - env.jump_target_vx).square() / std**2
    )
    return quality * env.jump_landing_event.float()


def jump_target_landing_progress(
    env: ManagerBasedRLEnv, std: float = 0.10
) -> torch.Tensor:
    """Dense flight reward for approaching the commanded planar landing point."""
    if not hasattr(env, "jump_target_position_w"):
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.linalg.vector_norm(
        env.scene["robot"].data.root_pos_w[:, :2] - env.jump_target_position_w,
        dim=1,
    )
    quality = torch.exp(-error.square() / std**2)
    airborne = (~env.jump_wheel_contact.any(dim=1)) & env._jump_airborne_ever
    return quality * airborne.float() * _phase_mask(env, JUMP_PHASE_FLIGHT)


def jump_target_landing_tracking(
    env: ManagerBasedRLEnv, std: float = 0.05
) -> torch.Tensor:
    """One-step reward for first-contact proximity to the commanded target."""
    if not hasattr(env, "jump_landing_position_error"):
        return torch.zeros(env.num_envs, device=env.device)
    quality = torch.exp(-env.jump_landing_position_error.square() / std**2)
    return quality * env.jump_landing_event.float()


def obstacle_clearance_tracking(
    env: ManagerBasedRLEnv, target_clearance: float = 0.025, std: float = 0.02
) -> torch.Tensor:
    """Reward wheel-bottom clearance while both wheels pass over the barrier."""
    if not hasattr(env, "obstacle_min_clearance"):
        return torch.zeros(env.num_envs, device=env.device)
    clearance = env.obstacle_min_clearance
    finite = torch.isfinite(clearance)
    progress = torch.clamp(clearance / target_clearance, 0.0, 1.0)
    tracking = torch.exp(-(clearance - target_clearance).square() / std**2)
    quality = torch.where(finite, 0.7 * progress + 0.3 * tracking, 0.0)
    return quality * _phase_mask(env, JUMP_PHASE_FLIGHT)


def obstacle_crossing_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step credit for crossing the far obstacle edge without contact."""
    if not hasattr(env, "obstacle_cross_event"):
        return torch.zeros(env.num_envs, device=env.device)
    clean = env.obstacle_cross_event & ~env.obstacle_trial_collision
    return clean.float()


def obstacle_collision_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step event for the first robot-obstacle contact in each trial."""
    if not hasattr(env, "obstacle_collision_event"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.obstacle_collision_event.float()


def jump_airborne_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small survival reward while both wheels are genuinely airborne."""
    if not hasattr(env, "jump_wheel_contact"):
        return torch.zeros(env.num_envs, device=env.device)
    airborne = ~env.jump_wheel_contact.any(dim=1)
    return airborne.float() * _phase_mask(env, JUMP_PHASE_FLIGHT)


def jump_leg_symmetry_reward(env: ManagerBasedRLEnv, std_l0: float = 0.015, std_theta: float = 0.10):
    """Keep left/right virtual legs coordinated throughout the jump."""
    if not hasattr(env, "L0"):
        return torch.zeros(env.num_envs, device=env.device)
    error = (env.L0[:, 0] - env.L0[:, 1]).square() / std_l0**2
    error += (env.theta0[:, 0] - env.theta0[:, 1]).square() / std_theta**2
    active = _phase_mask(
        env, JUMP_PHASE_CROUCH, JUMP_PHASE_THRUST, JUMP_PHASE_FLIGHT, JUMP_PHASE_LANDING
    )
    return torch.exp(-error) * active


def jump_landing_length_tracking(env: ManagerBasedRLEnv, std: float = 0.025) -> torch.Tensor:
    """Guide the legs toward a compliant, non-collapsed landing posture."""
    if not hasattr(env, "L0"):
        return torch.zeros(env.num_envs, device=env.device)
    target = jump_leg_reference(env)
    error = env.L0.mean(dim=1) - target
    return torch.exp(-error.square() / std**2) * _phase_mask(
        env, JUMP_PHASE_FLIGHT, JUMP_PHASE_LANDING
    )


def jump_landing_soft(env: ManagerBasedRLEnv, std_vz: float = 0.50, std_tilt: float = 0.20):
    """Event reward for a low-speed upright first landing contact."""
    if not hasattr(env, "jump_landing_event"):
        return torch.zeros(env.num_envs, device=env.device)
    tilt = torch.acos(
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], -1.0, 1.0)
    )
    quality = torch.exp(-(env.jump_landing_vz / std_vz).square() - (tilt / std_tilt).square())
    return quality * env.jump_landing_event.float()


def jump_landing_impact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse squared landing-speed penalty."""
    if not hasattr(env, "jump_landing_event"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.jump_landing_vz.square() * env.jump_landing_event.float()


def jump_recovery_stability(env: ManagerBasedRLEnv, std_vz: float = 0.30, std_tilt: float = 0.20):
    """Reward regaining upright, low-vertical-speed wheel contact after landing."""
    if not hasattr(env, "jump_wheel_contact"):
        return torch.zeros(env.num_envs, device=env.device)
    vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
    tilt = torch.acos(
        torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], -1.0, 1.0)
    )
    both_contact = env.jump_wheel_contact.all(dim=1).float()
    return (
        torch.exp(-(vz / std_vz).square() - (tilt / std_tilt).square())
        * both_contact
        * _phase_mask(env, JUMP_PHASE_RECOVERY)
        * env._jump_airborne_ever.float()
    )


def jump_success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step success event; use a large config weight because rewards are dt-scaled."""
    if not hasattr(env, "jump_success_event"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.jump_success_event.float()


def jump_failure_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step event for a missed takeoff or failed recovery."""
    if not hasattr(env, "jump_failure_event"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.jump_failure_event.float()
