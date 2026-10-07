# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery and rough-terrain locomotion stages for the wheel-legged robot.

All tasks remain command-conditioned: the operator or an upstream planner
supplies ``vx``, ``wz`` and body height. The policy only performs low-level
balance, disturbance rejection and terrain adaptation; it is not an autonomous
navigation policy.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.utils import configclass

from . import mdp
from .wheel_legged_flat_env_cfg import (
    WheelLeggedCurriculumCfg,
    WheelLeggedFlatEnvCfg,
    WheelLeggedRewardsCfg,
    WheelLeggedRobotSceneCfg,
    WheelLeggedVMCEnv,
    WheelleggedObservationsCfg,
)


def progressive_random_uniform_terrain(difficulty, cfg):
    """Scale random roughness with the generator row difficulty.

    Isaac Lab's stock random-uniform height field intentionally ignores its
    ``difficulty`` argument.  Without this wrapper every curriculum row would
    expose the full roughness range, including row zero.
    """
    scaled_cfg = cfg.copy()
    maximum_amplitude = max(abs(value) for value in cfg.noise_range)
    amplitude = float(difficulty) * maximum_amplitude
    scaled_cfg.noise_range = (-amplitude, amplitude)
    return hf_terrains.random_uniform_terrain(difficulty, scaled_cfg)


ROUGH_TERRAIN_CFG = terrain_gen.HfRandomUniformTerrainCfg(
    proportion=0.25,
    noise_range=(0.0, 0.035),
    noise_step=0.005,
    border_width=0.25,
)
ROUGH_TERRAIN_CFG.function = progressive_random_uniform_terrain


TERRAIN_LOCOMOTION_CFG = TerrainGeneratorCfg(
    seed=17,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=2.0,
    border_height=1.0,
    num_rows=6,
    num_cols=8,
    color_scheme="height",
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.40,
            slope_range=(0.03, 0.20),
            platform_width=2.5,
            border_width=0.25,
        ),
        "stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.015, 0.045),
            step_width=0.35,
            platform_width=2.5,
            border_width=0.25,
        ),
        "rough": ROUGH_TERRAIN_CFG,
    },
)

TERRAIN_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=0.8,
    dynamic_friction=0.7,
    restitution=0.0,
)


@configclass
class WheelLeggedTerrainSceneCfg(WheelLeggedRobotSceneCfg):
    """Mixed slope/stair/rough scene with a safe central spawn platform."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_LOCOMOTION_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=TERRAIN_PHYSICS_MATERIAL,
        debug_vis=False,
    )


class WheelLeggedRobustEnv(WheelLeggedVMCEnv):
    """Base environment with compact robustness diagnostics."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # The wide forward scanner is useful to the critic/actor, but averaging
        # it to estimate body clearance makes an approaching stair change
        # ``base_height`` before the robot reaches it.  Clearance must use the
        # small scanner directly below the chassis.
        self._height_scanner = self.scene.sensors.get(
            "height_scanner_base", self._height_scanner
        )

        self._recovery_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._recovery_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._recovery_stable_time = torch.zeros(self.num_envs, device=self.device)
        self._recovery_reward_mask = torch.zeros(self.num_envs, device=self.device)
        self._recovery_window_attempts = 0
        self._recovery_window_successes = 0
        self._recovery_window_time = 0.0
        self._recovery_success_rate = 0.0
        self._recovery_failure_rate = 0.0
        self._recovery_mean_time = 0.0

        self._terrain_commanded_distance = torch.zeros(
            self.num_envs, device=self.device
        )
        self._terrain_tracking_distance = torch.zeros(
            self.num_envs, device=self.device
        )

        # Treat the randomized reset pose as the first recoverable trial.  Full
        # side-lying self-righting is intentionally outside this task.
        self.start_recovery_attempt(
            torch.arange(self.num_envs, device=self.device)
        )

    def start_recovery_attempt(self, env_ids):
        """Start or restart a timed recovery trial after a reset or push."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self._recovery_pending[env_ids] = True
        self._recovery_elapsed[env_ids] = 0.0
        self._recovery_stable_time[env_ids] = 0.0

    def _record_recovery_outcomes(self, success: torch.Tensor, elapsed: torch.Tensor):
        count = int(success.numel())
        if count == 0:
            return
        self._recovery_window_attempts += count
        self._recovery_window_successes += int(success.sum().item())
        self._recovery_window_time += float(elapsed[success].sum().item())
        # One result window contains enough independent environments to avoid a
        # noisy single-push success spike controlling early stopping.
        minimum_window = max(64, self.num_envs // 4)
        if self._recovery_window_attempts >= minimum_window:
            attempts = self._recovery_window_attempts
            successes = self._recovery_window_successes
            self._recovery_success_rate = successes / attempts
            self._recovery_failure_rate = 1.0 - self._recovery_success_rate
            self._recovery_mean_time = (
                self._recovery_window_time / successes if successes > 0 else 2.5
            )
            self._recovery_window_attempts = 0
            self._recovery_window_successes = 0
            self._recovery_window_time = 0.0

    def _pre_reward_update(self):
        super()._pre_reward_update()

        # Command-aligned progress is robust to command reversals.  A robot that
        # correctly drives forward and then backward should not be demoted just
        # because its final world displacement is close to zero.
        command = self.command_manager.get_command("wheel_legged_commands")[:, 0]
        actual = self._robot.data.root_lin_vel_b[:, 0]
        command_abs = command.abs()
        aligned_speed = torch.sign(command) * actual
        useful_speed = torch.minimum(
            aligned_speed.clamp_min(0.0), 1.25 * command_abs
        )
        moving = command_abs >= 0.05
        self._terrain_commanded_distance += torch.where(
            moving, command_abs, torch.zeros_like(command_abs)
        ) * self.step_dt
        self._terrain_tracking_distance += torch.where(
            moving, useful_speed, torch.zeros_like(useful_speed)
        ) * self.step_dt

        active = self._recovery_pending
        self._recovery_reward_mask = active.float()
        if not active.any():
            return
        self._recovery_elapsed[active] += self.step_dt
        gravity_b = self._robot.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
        stable = (tilt < 0.25) & (self.base_height > 0.14)
        evaluate = self._recovery_elapsed >= 0.25
        accumulating = active & evaluate & stable
        self._recovery_stable_time = torch.where(
            accumulating,
            self._recovery_stable_time + self.step_dt,
            torch.zeros_like(self._recovery_stable_time),
        )
        success = active & (self._recovery_stable_time >= 0.30)
        failure = active & (
            (tilt > 1.0)
            | (self.base_height < 0.085)
            | (self._recovery_elapsed >= 2.5)
        ) & (~success)
        resolved = success | failure
        if resolved.any():
            self._record_recovery_outcomes(
                success[resolved], self._recovery_elapsed[resolved]
            )
            self._recovery_pending[resolved] = False
            self._recovery_reward_mask[resolved] = 1.0

    def _log_debug_metrics(self):
        super()._log_debug_metrics()
        gravity_b = self._robot.data.projected_gravity_b
        tilt = (-gravity_b[:, 2]).clamp(-1.0, 1.0).acos()
        upright = (tilt < 0.25) & (self.base_height > 0.14)
        fallen = (tilt > 0.80) | (self.base_height < 0.10)
        self.extras["log"] = {
            **self.extras.get("log", {}),
            "recovery_upright_rate": upright.float().mean().item(),
            "recovery_fallen_rate": fallen.float().mean().item(),
            "recovery_pending_rate": self._recovery_pending.float().mean().item(),
            "recovery_success_rate": self._recovery_success_rate,
            "recovery_failure_rate": self._recovery_failure_rate,
            "recovery_mean_time": self._recovery_mean_time,
            "terrain_tracking_ratio": (
                self._terrain_tracking_distance.sum()
                / self._terrain_commanded_distance.sum().clamp_min(1.0e-6)
            ).item(),
            "terrain_level": (
                self.scene.terrain.terrain_levels.float().mean().item()
                if self.scene.terrain.cfg.terrain_type == "generator"
                else 0.0
            ),
        }

    def _reset_idx(self, env_ids):
        # Curriculum terms consume the previous episode's progress inside the
        # parent reset, so clear these buffers only afterwards.
        if hasattr(self, "_recovery_pending"):
            env_ids_tensor = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
            unresolved = self._recovery_pending[env_ids_tensor]
            if unresolved.any():
                unresolved_ids = env_ids_tensor[unresolved]
                self._record_recovery_outcomes(
                    torch.zeros(
                        unresolved_ids.numel(), dtype=torch.bool, device=self.device
                    ),
                    self._recovery_elapsed[unresolved_ids],
                )
        super()._reset_idx(env_ids)
        if hasattr(self, "_terrain_commanded_distance"):
            self._terrain_commanded_distance[env_ids] = 0.0
            self._terrain_tracking_distance[env_ids] = 0.0
            self.start_recovery_attempt(env_ids)


@configclass
class WheelLeggedRecoveryRewardsCfg(WheelLeggedRewardsCfg):
    """Locomotion rewards plus a dense near-fall recovery objective."""

    recovery_upright = RewardTermCfg(
        func=mdp.recovery_upright_reward,
        weight=6.0,
        params={
            "command_name": "wheel_legged_commands",
            "tilt_std": 0.30,
            "height_std": 0.07,
        },
    )


@configclass
class WheelLeggedRecoveryFlatEnvCfg(WheelLeggedFlatEnvCfg):
    """Near-fall recovery stage, not full side-lying self-righting."""

    rewards: WheelLeggedRecoveryRewardsCfg = WheelLeggedRecoveryRewardsCfg()
    env_class = WheelLeggedRobustEnv

    def __post_init__(self):
        super().__post_init__()
        # The flat parent reconstructs manager configs in __post_init__.
        self.rewards = WheelLeggedRecoveryRewardsCfg()
        # Start from mechanically recoverable disturbances. These ranges can be
        # widened only after the robot reliably survives this stage.
        pose = self.events.reset_base.params["pose_range"]
        pose.update({"roll": (-0.22, 0.22), "pitch": (-0.28, 0.28)})
        velocity = self.events.reset_base.params["velocity_range"]
        velocity.update(
            {
                "x": (-0.35, 0.35),
                "y": (-0.15, 0.15),
                "roll": (-0.8, 0.8),
                "pitch": (-0.8, 0.8),
            }
        )
        self.events.push_robot.interval_range_s = (3.0, 6.0)
        self.events.push_robot.params["max_push_vel_xy"] = 0.9
        self.terminations.bad_orientation.params["limit_angle"] = 1.15
        self.terminations.minimum_height.params["minimum_height"] = 0.085
        self.rewards.orientation.weight = -6.0
        self.rewards.base_height.weight = 4.0
        self.rewards.collision.weight = -1.0
        command = self.commands.wheel_legged_commands
        command.ranges.lin_vel_x = (-0.6, 0.6)
        command.ranges.ang_vel_yaw = (-1.2, 1.2)


@configclass
class WheelLeggedTerrainObservationsCfg(WheelleggedObservationsCfg):
    """Deployable terrain actor observation with a compact 15-ray scan."""

    @configclass
    class PolicyCfg(WheelleggedObservationsCfg.PolicyCfg):
        terrain_scan = ObsTerm(
            func=mdp.compact_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "reference_sensor_cfg": SceneEntityCfg("height_scanner_base"),
                "longitudinal_samples": 5,
                "lateral_samples": 3,
            },
            clip=(-0.20, 0.20),
            scale=5.0,
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class WheelLeggedTerrainCurriculumCfg(WheelLeggedCurriculumCfg):
    """Keep command curriculum and progressively raise terrain difficulty."""

    terrain_levels = CurriculumTermCfg(
        func=mdp.wheel_legged_terrain_levels,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "wheel_legged_commands",
            "minimum_command_speed": 0.15,
            "promotion_tracking_ratio": 0.70,
            "demotion_tracking_ratio": 0.35,
            "minimum_command_distance": 0.75,
        },
    )


@configclass
class WheelLeggedTerrainReactiveEnvCfg(WheelLeggedFlatEnvCfg):
    """Mixed-terrain curriculum with a proprioception-only actor.

    The actor observation remains compatible with the flat/recovery checkpoint.
    During training only, the critic retains the existing privileged height scan.
    """

    scene: WheelLeggedTerrainSceneCfg = WheelLeggedTerrainSceneCfg(
        num_envs=4096, env_spacing=8.0
    )
    rewards: WheelLeggedRecoveryRewardsCfg = WheelLeggedRecoveryRewardsCfg()
    curriculum: WheelLeggedTerrainCurriculumCfg = WheelLeggedTerrainCurriculumCfg()
    env_class = WheelLeggedRobustEnv

    def __post_init__(self):
        super().__post_init__()
        # The flat parent reconstructs manager configs in __post_init__.
        self.rewards = WheelLeggedRecoveryRewardsCfg()
        self.curriculum = WheelLeggedTerrainCurriculumCfg()
        # WheelLeggedFlatEnvCfg intentionally forces a plane. Restore this
        # subclass's generator without replacing the full scene object, so all
        # robot/sensor fields initialized by the parent remain intact.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TERRAIN_LOCOMOTION_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 1
        self.scene.terrain.physics_material = TERRAIN_PHYSICS_MATERIAL.copy()
        self.scene.env_spacing = 8.0

        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.25, 0.0, 0.5))
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.1, size=[1.0, 0.6]
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material

        self.events.reset_base.params["pose_range"].update(
            {"roll": (-0.08, 0.08), "pitch": (-0.10, 0.10)}
        )
        self.events.push_robot.interval_range_s = (5.0, 9.0)
        self.events.push_robot.params["max_push_vel_xy"] = 0.55
        self.terminations.bad_orientation.params["limit_angle"] = 1.0
        self.terminations.minimum_height.params["minimum_height"] = 0.09
        self.rewards.orientation.weight = -5.0
        self.rewards.base_height.weight = 3.5
        self.rewards.recovery_upright.weight = 3.0

        command = self.commands.wheel_legged_commands
        command.ranges.lin_vel_x = (-0.6, 0.6)
        command.ranges.ang_vel_yaw = (-1.2, 1.2)


@configclass
class WheelLeggedTerrainPerceptiveEnvCfg(WheelLeggedTerrainReactiveEnvCfg):
    """Optional sensor-aware actor; keyboard still supplies vx/wz/height."""

    observations: WheelLeggedTerrainObservationsCfg = WheelLeggedTerrainObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The flat parent reconstructs its observation config in __post_init__.
        # Reinstall the actor-side scan only for this optional branch.
        self.observations = WheelLeggedTerrainObservationsCfg()
        # Match the normalization/scaling used by the checkpoint-compatible
        # reactive actor and its privileged critic.
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.theta0_dot.scale = 0.05
        self.observations.policy.L0.scale = 5.0
        self.observations.policy.L0_dot.scale = 0.25
        self.observations.policy.joint_wheel_vel.scale = 0.05
        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.theta0_dot.scale = 0.05
        self.observations.critic.L0.scale = 5.0
        self.observations.critic.L0_dot.scale = 0.25
        self.observations.critic.joint_wheel_vel.scale = 0.05
        self.observations.critic.joint_acc.scale = 0.0025
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.height_scan.scale = 5.0
        self.observations.critic.torques.scale = 0.05
