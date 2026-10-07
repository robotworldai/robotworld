# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic power-on stand controller for the wheel-legged VMC."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PowerOnStandCfg:
    """Timing and success-neutral references for power-on standing."""

    torque_ramp_time: float = 0.60
    extend_time: float = 0.80
    stabilize_time: float = 0.60
    handoff_time: float = 1.00
    target_leg_length: float = 0.21
    balance_pitch_kp: float = 3.0
    balance_pitch_kd: float = 0.45
    balance_wheel_action_limit: float = 0.65
    balance_leg_angle_gain: float = -1.0
    balance_leg_action_limit: float = 0.80


class PowerOnStandController:
    """Vectorized PASSIVE→RAMP→EXTEND→STABILIZE→HANDOFF controller.

    The controller outputs ordinary six-dimensional VMC actions plus a separate
    motor-enable scale.  It contains no learned parameters and can blend into an
    optional locomotion policy after the chassis is upright.
    """

    PHASE_NAMES = ("passive", "ramp", "extend", "stabilize", "handoff", "complete")
    PASSIVE = 0
    RAMP = 1
    EXTEND = 2
    STABILIZE = 3
    HANDOFF = 4
    COMPLETE = 5

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        vmc_cfg,
        passive_durations: float | torch.Tensor,
        cfg: PowerOnStandCfg | None = None,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.vmc_cfg = vmc_cfg
        self.cfg = cfg or PowerOnStandCfg()
        if not vmc_cfg.l0_min <= self.cfg.target_leg_length <= vmc_cfg.l0_max:
            raise ValueError(
                "Power-on target leg length must be within the VMC range: "
                f"target={self.cfg.target_leg_length}, "
                f"range=({vmc_cfg.l0_min}, {vmc_cfg.l0_max})."
            )
        for name in ("torque_ramp_time", "extend_time", "stabilize_time", "handoff_time"):
            if getattr(self.cfg, name) <= 0.0:
                raise ValueError(f"Power-on duration {name} must be positive.")
        durations = torch.as_tensor(
            passive_durations, dtype=torch.float32, device=self.device
        )
        if durations.ndim == 0:
            durations = durations.repeat(self.num_envs)
        if durations.shape != (self.num_envs,) or torch.any(durations < 0.0):
            raise ValueError(
                f"passive_durations must be non-negative with shape ({self.num_envs},)."
            )
        self.passive_durations = durations
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.start_lengths = torch.full(
            (self.num_envs, 2),
            float(vmc_cfg.l0_min),
            device=self.device,
        )
        self.previous_phase = torch.full(
            (self.num_envs,), self.PASSIVE, dtype=torch.long, device=self.device
        )

    @staticmethod
    def _smoothstep(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.0, 1.0)
        return value.square() * (3.0 - 2.0 * value)

    def _phase(self) -> torch.Tensor:
        relative = self.elapsed - self.passive_durations
        ramp_end = self.cfg.torque_ramp_time
        extend_end = ramp_end + self.cfg.extend_time
        stabilize_end = extend_end + self.cfg.stabilize_time
        handoff_end = stabilize_end + self.cfg.handoff_time
        return torch.where(
            relative < 0.0,
            torch.full_like(relative, self.PASSIVE, dtype=torch.long),
            torch.where(
                relative < ramp_end,
                torch.full_like(relative, self.RAMP, dtype=torch.long),
                torch.where(
                    relative < extend_end,
                    torch.full_like(relative, self.EXTEND, dtype=torch.long),
                    torch.where(
                        relative < stabilize_end,
                        torch.full_like(relative, self.STABILIZE, dtype=torch.long),
                        torch.where(
                            relative < handoff_end,
                            torch.full_like(relative, self.HANDOFF, dtype=torch.long),
                            torch.full_like(relative, self.COMPLETE, dtype=torch.long),
                        ),
                    ),
                ),
            ),
        )

    def _length_actions(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (lengths - self.vmc_cfg.l0_offset) / self.vmc_cfg.action_scale_l0,
            -self.vmc_cfg.action_clip,
            self.vmc_cfg.action_clip,
        )

    def compute(
        self,
        current_lengths: torch.Tensor,
        base_pitch: torch.Tensor | None = None,
        base_pitch_rate: torch.Tensor | None = None,
        policy_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VMC action, motor scale, phase and new-handoff mask."""
        if current_lengths.shape != (self.num_envs, 2):
            raise ValueError(
                f"current_lengths must have shape ({self.num_envs}, 2), "
                f"got {tuple(current_lengths.shape)}."
            )
        if policy_actions is not None and policy_actions.shape != (self.num_envs, 6):
            raise ValueError(
                f"policy_actions must have shape ({self.num_envs}, 6), "
                f"got {tuple(policy_actions.shape)}."
            )

        phase = self._phase()
        passive = phase == self.PASSIVE
        # Continuously track the collapsed geometry so each environment begins
        # the ramp from its final passive-settle leg lengths.
        self.start_lengths = torch.where(
            passive.unsqueeze(-1), current_lengths, self.start_lengths
        )

        relative = self.elapsed - self.passive_durations
        ramp_alpha = self._smoothstep(relative / self.cfg.torque_ramp_time)
        extend_alpha = self._smoothstep(
            (relative - self.cfg.torque_ramp_time) / self.cfg.extend_time
        )
        handoff_alpha = self._smoothstep(
            (
                relative
                - self.cfg.torque_ramp_time
                - self.cfg.extend_time
                - self.cfg.stabilize_time
            )
            / self.cfg.handoff_time
        )

        motor_scale = torch.where(
            passive,
            torch.zeros_like(ramp_alpha),
            torch.where(phase == self.RAMP, ramp_alpha, torch.ones_like(ramp_alpha)),
        )
        target = torch.full_like(self.start_lengths, self.cfg.target_leg_length)
        desired_lengths = torch.where(
            (phase <= self.RAMP).unsqueeze(-1),
            self.start_lengths,
            self.start_lengths + extend_alpha.unsqueeze(-1) * (target - self.start_lengths),
        )
        desired_lengths = desired_lengths.clamp(
            self.vmc_cfg.l0_min, self.vmc_cfg.l0_max
        )
        stand_actions = torch.zeros(
            self.num_envs, 6, dtype=current_lengths.dtype, device=self.device
        )
        length_actions = self._length_actions(desired_lengths)
        stand_actions[:, 1] = length_actions[:, 0]
        stand_actions[:, 4] = length_actions[:, 1]
        if (base_pitch is None) != (base_pitch_rate is None):
            raise ValueError("base_pitch and base_pitch_rate must be provided together.")
        if base_pitch is not None:
            if base_pitch.shape != (self.num_envs,) or base_pitch_rate.shape != (
                self.num_envs,
            ):
                raise ValueError(
                    "base_pitch and base_pitch_rate must both have shape "
                    f"({self.num_envs},)."
                )
            # Drive the wheel contact point towards the leaning chassis.  The
            # outer actuator gate still guarantees zero torque in PASSIVE.
            wheel_action = (
                self.cfg.balance_pitch_kp * base_pitch
                + self.cfg.balance_pitch_kd * base_pitch_rate
            ).clamp(
                -self.cfg.balance_wheel_action_limit,
                self.cfg.balance_wheel_action_limit,
            )
            # The URDF wheel joint axes are mirrored: equal chassis motion uses
            # opposite left/right joint-velocity signs.
            stand_actions[:, 2] = wheel_action
            stand_actions[:, 5] = -wheel_action
            # Keep both virtual legs approximately vertical in the world while
            # the chassis is still lifting away from ground contact.
            leg_angle_action = (
                self.cfg.balance_leg_angle_gain
                * base_pitch
                / self.vmc_cfg.action_scale_theta
            ).clamp(
                -self.cfg.balance_leg_action_limit,
                self.cfg.balance_leg_action_limit,
            )
            stand_actions[:, 0] = leg_angle_action
            stand_actions[:, 3] = leg_angle_action

        if policy_actions is None:
            actions = stand_actions
        else:
            blend = torch.where(
                phase == self.COMPLETE,
                torch.ones_like(handoff_alpha),
                torch.where(
                    phase == self.HANDOFF,
                    handoff_alpha,
                    torch.zeros_like(handoff_alpha),
                ),
            ).unsqueeze(-1)
            actions = (1.0 - blend) * stand_actions + blend * policy_actions
        actions = actions.clamp(-self.vmc_cfg.action_clip, self.vmc_cfg.action_clip)
        handoff_started = (phase == self.HANDOFF) & (
            self.previous_phase < self.HANDOFF
        )
        self.previous_phase = phase.clone()
        return actions, motor_scale, phase, handoff_started

    def advance(self, dt: float) -> None:
        """Advance the controller clock after one environment control step."""
        self.elapsed += float(dt)

    @property
    def total_duration(self) -> torch.Tensor:
        return self.passive_durations + (
            self.cfg.torque_ramp_time
            + self.cfg.extend_time
            + self.cfg.stabilize_time
            + self.cfg.handoff_time
        )
