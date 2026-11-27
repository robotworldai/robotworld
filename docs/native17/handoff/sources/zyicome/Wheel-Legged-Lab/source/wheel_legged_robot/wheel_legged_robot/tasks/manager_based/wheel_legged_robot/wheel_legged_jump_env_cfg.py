# Copyright (c) 2026 zyicome
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground locomotion plus externally triggered small-jump task."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.views import XformPrimView
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import (
    CurriculumTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster import patterns as ray_patterns

from . import mdp
from .wheel_legged_flat_env_cfg import (
    VmcActionsCfg,
    WheelLeggedEventCfg,
    WheelLeggedFlatEnvCfg,
    WheelLeggedRewardsCfg,
    WheelLeggedRobotSceneCfg,
    WheelLeggedTerminationsCfg,
    WheelLeggedVMCEnv,
    WheelleggedObservationsCfg,
)


@configclass
class JumpStateMachineCfg:
    """Contact-driven jump phase timings and acceptance thresholds."""

    # Stage B2: the former 0.22 m reference changed phase at 0.255 m and only
    # produced a marginal hop.  The open-loop sweep showed that holding a
    # 0.18--0.20 m crouch for roughly 0.25 s is required for a visible jump.
    crouch_length: float = 0.19
    # Transition at 0.22 m is deep enough to store useful energy while still
    # being reachable under the ±10% actuator/domain randomization.
    crouch_ready_length: float = 0.22
    thrust_length: float = 0.30
    # Retract fully in flight.  This converts base motion into useful wheel
    # clearance and leaves the entire leg stroke available for impact absorption.
    landing_length: float = 0.19
    crouch_min_time: float = 0.22
    max_crouch_time: float = 0.42
    # 保持完整蹬伸约 0.20 s。过早切换到飞行收腿会让轮端在卸载前
    # 再次压回地面，表现为“机体上升但 contact 始终存在”。
    min_thrust_time: float = 0.18
    max_thrust_time: float = 0.32
    min_release_vz: float = 0.40
    # 使用测得的实际腿长而非动作目标。0.30 m 目标在高速蹬伸时
    # 会因惯性自然越过名义值；延迟到 0.31 m 再收腿可完成轮端卸载。
    min_release_length: float = 0.31
    max_airborne_wait: float = 0.15
    min_takeoff_vz: float = 0.25
    min_flight_time: float = 0.04
    landing_time: float = 0.20
    recovery_stable_time: float = 0.50
    max_recovery_time: float = 1.50
    contact_force_threshold: float = 2.0
    contact_confirm_steps: int = 2
    start_max_tilt: float = 0.30
    start_max_speed: float = 0.40
    recovery_max_tilt: float = 0.20
    recovery_max_vz: float = 0.30
    success_height_ratio: float = 0.80
    # Zero preserves the original base-height-only acceptance rule. Clearance
    # stages set this above zero so a long-legged grounded extension cannot be
    # reported as a successful obstacle-clearing jump.
    success_wheel_clearance_ratio: float = 0.0
    success_min_air_time: float = 0.12
    success_max_recovery_vx_error: float = float("inf")
    # Optional target-landing acceptance. ``inf`` preserves every existing
    # jump task; the target-landing stage installs a finite planar tolerance.
    success_max_landing_position_error: float = float("inf")
    use_leg_reference_assist: bool = True
    # The Stage-B1 actor learned residuals around the former shallow reference.
    # Limit its authority during the B2 transfer so it cannot cancel the proven
    # deep-crouch trajectory; increase this again only after B2 converges.
    leg_action_residual_scale: float = 0.05

    # Optional Stage-C high-jump/landing assistance.  Defaults preserve the
    # already-trained Stage-B2 task and its checkpoint behavior.
    use_height_conditioned_crouch: bool = False
    reference_height_min: float = 0.05
    reference_height_max: float = 0.07
    high_jump_crouch_length: float = 0.19
    use_landing_assist: bool = False
    flight_retract_length: float = 0.19
    prelanding_length: float = 0.25
    prelanding_start_vz: float = 0.05
    prelanding_full_vz: float = -0.65
    landing_absorption_length: float = 0.21
    landing_compression_time: float = 0.18
    prelanding_action_residual_scale: float = 0.10
    landing_action_residual_scale: float = 0.15
    use_phase_dependent_gains: bool = False
    thrust_kp_scale: float = 1.0
    thrust_feedforward_scale: float = 1.0
    landing_kd_scale: float = 1.0
    flight_feedforward_scale: float = 1.0
    landing_feedforward_scale: float = 1.0
    # Optional split flight controller: retract through ascent/apex, then
    # independently tune the descending pre-extension. Defaults are inactive
    # and therefore leave Stage B2/C1 checkpoints unchanged.
    use_split_flight_control: bool = False
    flight_retract_kp_scale: float = 1.0
    flight_retract_kd_scale: float = 1.0
    flight_retract_feedforward_scale: float = 0.0
    prelanding_feedforward_scale: float = 1.0
    prelanding_kd_scale: float = 1.0
    # ``inf`` keeps the Stage-B2 success definition unchanged.  The high-jump
    # stage uses a finite intermediate limit while separately reporting the
    # stricter 0.8 m/s soft-landing rate.
    success_max_landing_speed: float = float("inf")
    soft_landing_speed: float = 0.80


@configclass
class ObstacleOracleCfg:
    """Geometry and analytic trigger parameters for the first obstacle stage."""

    spawn_distance: float = 0.90
    obstacle_width: float = 0.035
    obstacle_width_levels: tuple[float, ...] = (
        0.035,
        0.035,
        0.050,
        0.050,
        0.065,
        0.065,
        0.080,
    )
    obstacle_lateral_size: float = 1.00
    obstacle_max_height: float = 0.08
    obstacle_height_range: tuple[float, float] = (0.02, 0.04)
    # Optional exact geometry used by Play/evaluation. ``None`` keeps the
    # training curriculum active; explicit values bypass its random range.
    fixed_height: float | None = None
    fixed_width: float | None = None
    # The horizontal landing command must remain compatible with the distance
    # that can be travelled during the measured flight time.  Computing it
    # from the remaining obstacle distance made the requested landing point
    # grow when a policy slowed down, producing unreachable 0.25--0.30 m hops.
    expected_air_time: float = 0.18
    target_distance_min: float = 0.08
    target_distance_max: float = 0.16
    preparation_time: float = 0.44
    # Desired root-to-near-edge distance at takeoff.  It is derived from the
    # speed-feasible landing distance and clamped to this interval.
    takeoff_margin: float = 0.02
    max_takeoff_margin: float = 0.07
    landing_margin: float = 0.02
    minimum_clearance: float = 0.015
    wheel_radius: float = 0.0675
    contact_force_threshold: float = 2.0


@configclass
class ObstaclePerceptionCfg:
    """Calibrated depth-camera preprocessing used by the perceptive stage."""

    sensor_name: str = "front_depth_camera"
    max_distance: float = 1.20
    minimum_point_height: float = 0.012
    maximum_point_height: float = 0.14
    lateral_half_width: float = 0.35
    minimum_valid_points: int = 4
    width_prior: float = 0.05
    width_min: float = 0.02
    width_max: float = 0.10
    scan_distances: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    )
    scan_half_bin: float = 0.055


def _obstacle_body_contact_sensor(body_name: str) -> ContactSensorCfg:
    """Create a valid one-body-to-one-obstacle filtered contact sensor."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle"],
        history_length=1,
        track_air_time=False,
    )


@configclass
class WheelLeggedObstacleSceneCfg(WheelLeggedRobotSceneCfg):
    """Flat scene with one kinematic barrier cloned into every environment."""

    obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.90, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.035, 1.00, 0.08),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.20, 0.08),
                metallic=0.0,
                roughness=0.7,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
        ),
    )
    # Contact filtering is one-to-many only in Isaac Lab. Use one sensor per
    # robot body instead of a Robot/.* expression.
    obstacle_contact_base = _obstacle_body_contact_sensor("base_link")
    obstacle_contact_lf0 = _obstacle_body_contact_sensor("lf0_Link")
    obstacle_contact_lf1 = _obstacle_body_contact_sensor("lf1_Link")
    obstacle_contact_rf0 = _obstacle_body_contact_sensor("rf0_Link")
    obstacle_contact_rf1 = _obstacle_body_contact_sensor("rf1_Link")
    obstacle_contact_lwheel = _obstacle_body_contact_sensor("l_wheel_Link")
    obstacle_contact_rwheel = _obstacle_body_contact_sensor("r_wheel_Link")


@configclass
class WheelLeggedPerceptiveObstacleSceneCfg(WheelLeggedObstacleSceneCfg):
    """Obstacle scene with a scalable ray-cast forward depth camera."""

    front_depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=[
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Obstacle",
                is_shared=True,
                track_mesh_transforms=True,
            ),
        ],
        update_period=0.04,
        max_distance=1.20,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        pattern_cfg=ray_patterns.PinholeCameraPatternCfg(
            focal_length=12.0,
            horizontal_aperture=20.955,
            height=24,
            width=32,
        ),
        # In world convention +X is forward and +Z is up. A positive pitch
        # rotates the optical axis slightly toward the ground.
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.13, 0.0, 0.0),
            rot=(0.997551, 0.0, 0.069943, 0.0),
            convention="world",
        ),
        debug_vis=False,
    )


class WheelLeggedJumpEnv(WheelLeggedVMCEnv):
    """Wheel-legged environment with a six-state, contact-driven jump machine."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.jump_phase = torch.full(
            (self.num_envs,), mdp.JUMP_PHASE_IDLE, dtype=torch.long, device=self.device
        )
        self.jump_phase_time = torch.zeros(self.num_envs, device=self.device)
        self.jump_wheel_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.jump_start_height = self._robot.data.root_pos_w[:, 2].clone()
        self.jump_apex_height = self.jump_start_height.clone()
        wheel_body_ids, wheel_body_names = self._robot.find_bodies(".*wheel.*")
        if len(wheel_body_ids) != 2:
            raise RuntimeError(
                f"Jump task expected two articulation wheel bodies, got {wheel_body_names}."
            )
        self._jump_robot_wheel_body_ids = wheel_body_ids
        wheel_height = self._robot.data.body_pos_w[:, wheel_body_ids, 2].amin(dim=1)
        self.jump_start_wheel_height = wheel_height.clone()
        self.jump_wheel_apex_height = wheel_height.clone()
        self.jump_takeoff_vz = torch.zeros(self.num_envs, device=self.device)
        self.jump_target_vx = torch.zeros(self.num_envs, device=self.device)
        self.jump_takeoff_vx = torch.zeros(self.num_envs, device=self.device)
        self.jump_landing_vz = torch.zeros(self.num_envs, device=self.device)
        self.jump_landing_vx = torch.zeros(self.num_envs, device=self.device)
        self.jump_landing_leg_length = torch.zeros(self.num_envs, device=self.device)
        self.jump_takeoff_position_w = self._robot.data.root_pos_w[:, :2].clone()
        self.jump_target_position_w = self.jump_takeoff_position_w.clone()
        self.jump_landing_position_w = self.jump_takeoff_position_w.clone()
        self.jump_landing_position_error = torch.zeros(self.num_envs, device=self.device)
        self._jump_landing_position_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.jump_takeoff_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.jump_landing_event = torch.zeros_like(self.jump_takeoff_event)
        self.jump_success_event = torch.zeros_like(self.jump_takeoff_event)
        self.jump_failure_event = torch.zeros_like(self.jump_takeoff_event)
        self._jump_air_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._jump_ground_steps = torch.zeros_like(self._jump_air_steps)
        self._jump_stable_steps = torch.zeros_like(self._jump_air_steps)
        self._jump_airborne_steps = torch.zeros_like(self._jump_air_steps)
        self._jump_airborne_ever = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._jump_attempts = torch.zeros(self.num_envs, device=self.device)
        self._jump_successes = torch.zeros(self.num_envs, device=self.device)
        self._jump_soft_landings = torch.zeros(self.num_envs, device=self.device)
        self._jump_target_landings = torch.zeros(self.num_envs, device=self.device)
        self._jump_landing_position_error_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._jump_landing_position_count = torch.zeros(
            self.num_envs, device=self.device
        )
        self._jump_fail_crouch = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_thrust = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_unload = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_performance = torch.zeros(self.num_envs, device=self.device)
        self._jump_fail_recovery = torch.zeros(self.num_envs, device=self.device)

        self._jump_contact_sensor = self.scene.sensors["contact_forces"]
        wheel_ids, wheel_names = self._jump_contact_sensor.find_bodies(".*wheel.*")
        if len(wheel_ids) != 2:
            raise RuntimeError(
                f"Jump task expected two wheel bodies, got {wheel_names} "
                f"from {self._jump_contact_sensor.body_names}."
            )
        self._jump_wheel_body_ids = wheel_ids

    def _enter_jump_phase(self, mask: torch.Tensor, phase: int):
        if mask.any():
            self.jump_phase[mask] = phase
            self.jump_phase_time[mask] = 0.0
            # Contact counters are phase-local. Reusing counts accumulated in
            # CROUCH was the cause of downward unloading being called takeoff.
            if phase == mdp.JUMP_PHASE_CROUCH:
                self._jump_air_steps[mask] = 0
                self._jump_ground_steps[mask] = 0
                self._jump_stable_steps[mask] = 0
            elif phase == mdp.JUMP_PHASE_THRUST:
                self._jump_air_steps[mask] = 0
            elif phase == mdp.JUMP_PHASE_FLIGHT:
                self._jump_air_steps[mask] = 0
                self._jump_ground_steps[mask] = 0
            elif phase in (mdp.JUMP_PHASE_LANDING, mdp.JUMP_PHASE_RECOVERY):
                self._jump_stable_steps[mask] = 0

    def _pre_reward_update(self):
        cfg = self.cfg.jump_state_machine
        self.jump_takeoff_event.zero_()
        self.jump_landing_event.zero_()
        self.jump_success_event.zero_()
        self.jump_failure_event.zero_()
        self.jump_phase_time += self.step_dt

        wheel_forces = self._jump_contact_sensor.data.net_forces_w[:, self._jump_wheel_body_ids]
        self.jump_wheel_contact[:] = (
            torch.linalg.vector_norm(wheel_forces, dim=-1) > cfg.contact_force_threshold
        )
        airborne = ~self.jump_wheel_contact.any(dim=1)
        any_contact = self.jump_wheel_contact.any(dim=1)
        both_contact = self.jump_wheel_contact.all(dim=1)
        self._jump_air_steps = torch.where(
            airborne, self._jump_air_steps + 1, torch.zeros_like(self._jump_air_steps)
        )
        self._jump_ground_steps = torch.where(
            any_contact, self._jump_ground_steps + 1, torch.zeros_like(self._jump_ground_steps)
        )

        root_z = self._robot.data.root_pos_w[:, 2]
        wheel_height = self._robot.data.body_pos_w[
            :, self._jump_robot_wheel_body_ids, 2
        ].amin(dim=1)
        root_vz = self._robot.data.root_lin_vel_w[:, 2]
        root_vx = self._robot.data.root_lin_vel_b[:, 0]
        speed_x = self._robot.data.root_lin_vel_b[:, 0].abs()
        tilt = torch.acos(torch.clamp(-self._robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        active = self.jump_phase != mdp.JUMP_PHASE_IDLE
        self.jump_apex_height[active] = torch.maximum(self.jump_apex_height[active], root_z[active])
        self.jump_wheel_apex_height[active] = torch.maximum(
            self.jump_wheel_apex_height[active], wheel_height[active]
        )
        confirmed_airborne = active & self._jump_airborne_ever & airborne
        self._jump_airborne_steps[confirmed_airborne] += 1

        jump_command = self.command_manager.get_command("jump_command")
        idle = self.jump_phase == mdp.JUMP_PHASE_IDLE
        start = (
            idle
            & (jump_command[:, 0] > 0.5)
            & both_contact
            & (tilt < cfg.start_max_tilt)
            & (speed_x < cfg.start_max_speed)
        )
        if start.any():
            self.jump_start_height[start] = root_z[start]
            self.jump_apex_height[start] = root_z[start]
            self.jump_start_wheel_height[start] = wheel_height[start]
            self.jump_wheel_apex_height[start] = wheel_height[start]
            self.jump_takeoff_vz[start] = 0.0
            locomotion_command = self.command_manager.get_command(
                "wheel_legged_commands"
            )
            self.jump_target_vx[start] = locomotion_command[start, 0]
            self.jump_takeoff_vx[start] = 0.0
            self.jump_landing_vz[start] = 0.0
            self.jump_landing_vx[start] = 0.0
            self.jump_landing_position_error[start] = 0.0
            self._jump_landing_position_valid[start] = False
            self._jump_airborne_steps[start] = 0
            self._jump_airborne_ever[start] = False
            self._jump_attempts[start] += 1.0
            self._enter_jump_phase(start, mdp.JUMP_PHASE_CROUCH)

        crouch = self.jump_phase == mdp.JUMP_PHASE_CROUCH
        mean_leg_length = self.L0.mean(dim=1)
        crouch_reached = mean_leg_length <= cfg.crouch_ready_length
        begin_thrust = (
            crouch & (self.jump_phase_time >= cfg.crouch_min_time) & crouch_reached
        )
        self._enter_jump_phase(begin_thrust, mdp.JUMP_PHASE_THRUST)

        crouch = self.jump_phase == mdp.JUMP_PHASE_CROUCH
        missed_crouch = crouch & (self.jump_phase_time >= cfg.max_crouch_time)
        if missed_crouch.any():
            self.jump_failure_event[missed_crouch] = True
            self._jump_fail_crouch[missed_crouch] += 1.0
            self._enter_jump_phase(missed_crouch, mdp.JUMP_PHASE_RECOVERY)

        thrust = self.jump_phase == mdp.JUMP_PHASE_THRUST
        release = (
            thrust
            & (self.jump_phase_time >= cfg.min_thrust_time)
            & (root_vz >= cfg.min_release_vz)
            & (mean_leg_length >= cfg.min_release_length)
        )
        self._enter_jump_phase(release, mdp.JUMP_PHASE_FLIGHT)

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        takeoff = (
            flight
            & ~self._jump_airborne_ever
            & (self._jump_air_steps >= cfg.contact_confirm_steps)
            & (root_vz >= cfg.min_takeoff_vz)
        )
        if takeoff.any():
            self.jump_takeoff_vz[takeoff] = root_vz[takeoff]
            self.jump_takeoff_vx[takeoff] = root_vx[takeoff]
            self.jump_takeoff_position_w[takeoff] = self._robot.data.root_pos_w[
                takeoff, :2
            ]
            forward_b = torch.zeros(self.num_envs, 3, device=self.device)
            forward_b[:, 0] = 1.0
            forward_w = quat_apply(self._robot.data.root_quat_w, forward_b)
            forward_xy = torch.nn.functional.normalize(
                forward_w[:, :2], dim=1, eps=1.0e-6
            )
            self.jump_target_position_w[takeoff] = (
                self.jump_takeoff_position_w[takeoff]
                + forward_xy[takeoff] * jump_command[takeoff, 2].unsqueeze(-1)
            )
            self.jump_takeoff_event[takeoff] = True
            self._jump_airborne_ever[takeoff] = True

        thrust = self.jump_phase == mdp.JUMP_PHASE_THRUST
        missed_takeoff = thrust & (self.jump_phase_time >= cfg.max_thrust_time)
        if missed_takeoff.any():
            self.jump_failure_event[missed_takeoff] = True
            self._jump_fail_thrust[missed_takeoff] += 1.0
            self._enter_jump_phase(missed_takeoff, mdp.JUMP_PHASE_RECOVERY)

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        failed_to_unload = (
            flight
            & ~self._jump_airborne_ever
            & (self.jump_phase_time >= cfg.max_airborne_wait)
        )
        if failed_to_unload.any():
            self.jump_failure_event[failed_to_unload] = True
            self._jump_fail_unload[failed_to_unload] += 1.0
            self._enter_jump_phase(failed_to_unload, mdp.JUMP_PHASE_RECOVERY)

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        first_touchdown = (
            flight
            & cfg.use_landing_assist
            & self._jump_airborne_ever
            & (self._jump_ground_steps == 1)
            & (root_vz <= 0.0)
        )
        if first_touchdown.any():
            # Latch the physical first-contact state.  Waiting for the
            # multi-frame confirmation records post-impact compression speed
            # instead of the actual touchdown speed.
            self.jump_landing_vz[first_touchdown] = root_vz[first_touchdown]
            self.jump_landing_vx[first_touchdown] = root_vx[first_touchdown]
            self.jump_landing_leg_length[first_touchdown] = mean_leg_length[first_touchdown]
            self._record_jump_landing_position(first_touchdown)

        flight = self.jump_phase == mdp.JUMP_PHASE_FLIGHT
        landing = (
            flight
            & self._jump_airborne_ever
            & (self.jump_phase_time >= cfg.min_flight_time)
            & (self._jump_ground_steps >= cfg.contact_confirm_steps)
            & (root_vz <= 0.0)
        )
        if landing.any():
            if not cfg.use_landing_assist:
                # Preserve Stage-B2 reward/metric semantics exactly.
                self.jump_landing_vz[landing] = root_vz[landing]
                self.jump_landing_vx[landing] = root_vx[landing]
                self.jump_landing_leg_length[landing] = mean_leg_length[landing]
            missing_position = landing & ~self._jump_landing_position_valid
            self._record_jump_landing_position(missing_position)
            self.jump_landing_event[landing] = True
            landing_speed_for_metric = (
                self.jump_landing_vz[landing]
                if cfg.use_landing_assist
                else root_vz[landing]
            )
            self._jump_soft_landings[landing] += (
                landing_speed_for_metric.abs() <= cfg.soft_landing_speed
            ).float()
            self._jump_target_landings[landing] += (
                self.jump_landing_position_error[landing]
                <= cfg.success_max_landing_position_error
            ).float()
            self._enter_jump_phase(landing, mdp.JUMP_PHASE_LANDING)

        landing_phase = self.jump_phase == mdp.JUMP_PHASE_LANDING
        begin_recovery = landing_phase & (self.jump_phase_time >= cfg.landing_time)
        self._enter_jump_phase(begin_recovery, mdp.JUMP_PHASE_RECOVERY)

        recovery = self.jump_phase == mdp.JUMP_PHASE_RECOVERY
        stable = (
            recovery
            & both_contact
            & (tilt < cfg.recovery_max_tilt)
            & (root_vz.abs() < cfg.recovery_max_vz)
        )
        self._jump_stable_steps = torch.where(
            stable, self._jump_stable_steps + 1, torch.zeros_like(self._jump_stable_steps)
        )
        stable_required = max(1, round(cfg.recovery_stable_time / self.step_dt))
        recovered = recovery & (self._jump_stable_steps >= stable_required)
        # Base rise alone used to accept a grounded leg extension as a jump.
        # It becomes a valid height criterion when combined with confirmed
        # takeoff and a minimum continuous airborne duration.
        apex_rise = self.jump_apex_height - self.jump_start_height
        height_reached = apex_rise >= cfg.success_height_ratio * jump_command[:, 1]
        wheel_clearance = self.jump_wheel_apex_height - self.jump_start_wheel_height
        clearance_reached = (
            wheel_clearance
            >= cfg.success_wheel_clearance_ratio * jump_command[:, 1]
        )
        air_time_reached = self._jump_airborne_steps * self.step_dt >= cfg.success_min_air_time
        obstacle_success_ready = getattr(
            self,
            "jump_obstacle_success_ready",
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
        )
        successful = (
            recovered
            & self._jump_airborne_ever
            & height_reached
            & clearance_reached
            & air_time_reached
            & (self.jump_landing_vz.abs() <= cfg.success_max_landing_speed)
            & (
                self.jump_landing_position_error
                <= cfg.success_max_landing_position_error
            )
            & obstacle_success_ready
            & (
                (root_vx - self.jump_target_vx).abs()
                <= cfg.success_max_recovery_vx_error
            )
        )
        # A failure before takeoff was already counted at its actual cause.
        # Do not count it again as a performance failure after recovery.
        recovered_but_failed = recovered & self._jump_airborne_ever & ~successful
        if successful.any():
            self.jump_success_event[successful] = True
            self._jump_successes[successful] += 1.0
        if recovered_but_failed.any():
            self.jump_failure_event[recovered_but_failed] = True
            self._jump_fail_performance[recovered_but_failed] += 1.0
        self._enter_jump_phase(recovered, mdp.JUMP_PHASE_IDLE)

        recovery = self.jump_phase == mdp.JUMP_PHASE_RECOVERY
        recovery_failed = recovery & (self.jump_phase_time >= cfg.max_recovery_time)
        if recovery_failed.any():
            self.jump_failure_event[recovery_failed] = True
            self._jump_fail_recovery[recovery_failed] += 1.0
            self._enter_jump_phase(recovery_failed, mdp.JUMP_PHASE_IDLE)

    def _record_jump_landing_position(self, mask: torch.Tensor):
        """Latch first-touchdown position and its planar target error."""
        if not mask.any():
            return
        position = self._robot.data.root_pos_w[mask, :2]
        error = torch.linalg.vector_norm(
            position - self.jump_target_position_w[mask], dim=1
        )
        self.jump_landing_position_w[mask] = position
        self.jump_landing_position_error[mask] = error
        self._jump_landing_position_valid[mask] = True
        self._jump_landing_position_error_sum[mask] += error
        self._jump_landing_position_count[mask] += 1.0

    def _log_debug_metrics(self):
        super()._log_debug_metrics()
        if not hasattr(self, "jump_phase"):
            return
        log = dict(self.extras.get("log", {}))
        for phase, name in (
            (mdp.JUMP_PHASE_IDLE, "idle"),
            (mdp.JUMP_PHASE_CROUCH, "crouch"),
            (mdp.JUMP_PHASE_THRUST, "thrust"),
            (mdp.JUMP_PHASE_FLIGHT, "flight"),
            (mdp.JUMP_PHASE_LANDING, "landing"),
            (mdp.JUMP_PHASE_RECOVERY, "recovery"),
        ):
            log[f"jump_phase_{name}"] = (self.jump_phase == phase).float().mean().item()
        command = self.command_manager.get_command("jump_command")
        attempts = self._jump_attempts.sum().clamp_min(1.0)
        enabled = command[:, 1] > 0.0
        log["jump_target_height"] = (
            command[enabled, 1].mean().item() if enabled.any() else 0.0
        )
        log["jump_target_distance"] = (
            command[enabled, 2].mean().item() if enabled.any() else 0.0
        )
        valid_takeoff = self._jump_airborne_ever
        valid_landing = self.jump_landing_vz < 0.0
        log["jump_takeoff_vz"] = (
            self.jump_takeoff_vz[valid_takeoff].mean().item()
            if valid_takeoff.any()
            else 0.0
        )
        log["jump_target_vx"] = (
            self.jump_target_vx[valid_takeoff].abs().mean().item()
            if valid_takeoff.any()
            else 0.0
        )
        log["jump_takeoff_vx_error"] = (
            (self.jump_takeoff_vx[valid_takeoff] - self.jump_target_vx[valid_takeoff])
            .abs()
            .mean()
            .item()
            if valid_takeoff.any()
            else 0.0
        )
        log["jump_landing_vz"] = (
            self.jump_landing_vz[valid_landing].mean().item()
            if valid_landing.any()
            else 0.0
        )
        log["jump_landing_vx_error"] = (
            (self.jump_landing_vx[valid_landing] - self.jump_target_vx[valid_landing])
            .abs()
            .mean()
            .item()
            if valid_landing.any()
            else 0.0
        )
        log["jump_landing_leg_length"] = (
            self.jump_landing_leg_length[valid_landing].mean().item()
            if valid_landing.any()
            else 0.0
        )
        apex_rise = self.jump_apex_height - self.jump_start_height
        log["jump_apex_rise"] = (
            apex_rise[valid_takeoff].mean().item() if valid_takeoff.any() else 0.0
        )
        wheel_clearance = self.jump_wheel_apex_height - self.jump_start_wheel_height
        air_time = self._jump_airborne_steps.float() * self.step_dt
        log["jump_wheel_clearance"] = (
            wheel_clearance[valid_takeoff].mean().item() if valid_takeoff.any() else 0.0
        )
        log["jump_air_time"] = (
            air_time[valid_takeoff].mean().item() if valid_takeoff.any() else 0.0
        )
        log["jump_airborne_fraction"] = (~self.jump_wheel_contact.any(dim=1)).float().mean().item()
        log["jump_success_rate"] = (self._jump_successes.sum() / attempts).item()
        log["jump_soft_landing_rate"] = (
            self._jump_soft_landings.sum() / attempts
        ).item()
        landing_count = self._jump_landing_position_count.sum().clamp_min(1.0)
        log["jump_landing_position_error"] = (
            self._jump_landing_position_error_sum.sum() / landing_count
        ).item()
        log["jump_target_landing_rate"] = (
            self._jump_target_landings.sum() / attempts
        ).item()
        log["jump_fail_crouch_rate"] = (self._jump_fail_crouch.sum() / attempts).item()
        log["jump_fail_thrust_rate"] = (self._jump_fail_thrust.sum() / attempts).item()
        log["jump_fail_unload_rate"] = (self._jump_fail_unload.sum() / attempts).item()
        log["jump_fail_performance_rate"] = (
            self._jump_fail_performance.sum() / attempts
        ).item()
        log["jump_fail_recovery_rate"] = (self._jump_fail_recovery.sum() / attempts).item()
        if hasattr(self, "_moving_jump_curriculum_level"):
            log["moving_jump_curriculum_level"] = float(
                self._moving_jump_curriculum_level
            )
            log["moving_jump_curriculum_success"] = (
                self._moving_jump_curriculum_last_success.item()
            )
            log["moving_jump_curriculum_soft"] = (
                self._moving_jump_curriculum_last_soft.item()
            )
            log["moving_jump_curriculum_heading"] = (
                self._moving_jump_curriculum_last_heading.item()
            )
            log["moving_jump_curriculum_passes"] = float(
                self._moving_jump_curriculum_passes
            )
        crouch = self.jump_phase == mdp.JUMP_PHASE_CROUCH
        thrust = self.jump_phase == mdp.JUMP_PHASE_THRUST
        log["jump_crouch_l0"] = self.L0[crouch].mean().item() if crouch.any() else 0.0
        log["jump_thrust_l0"] = self.L0[thrust].mean().item() if thrust.any() else 0.0
        log["jump_thrust_l0_dot"] = (
            self.L0_dot[thrust].mean().item() if thrust.any() else 0.0
        )
        self.extras["log"] = log

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "jump_phase"):
            return
        self.jump_phase[env_ids] = mdp.JUMP_PHASE_IDLE
        self.jump_phase_time[env_ids] = 0.0
        self.jump_wheel_contact[env_ids] = False
        root_z = self._robot.data.root_pos_w[env_ids, 2]
        self.jump_start_height[env_ids] = root_z
        self.jump_apex_height[env_ids] = root_z
        wheel_height = self._robot.data.body_pos_w[env_ids][
            :, self._jump_robot_wheel_body_ids, 2
        ].amin(dim=1)
        self.jump_start_wheel_height[env_ids] = wheel_height
        self.jump_wheel_apex_height[env_ids] = wheel_height
        self.jump_takeoff_vz[env_ids] = 0.0
        self.jump_target_vx[env_ids] = 0.0
        self.jump_takeoff_vx[env_ids] = 0.0
        self.jump_landing_vz[env_ids] = 0.0
        self.jump_landing_vx[env_ids] = 0.0
        self.jump_landing_leg_length[env_ids] = 0.0
        root_xy = self._robot.data.root_pos_w[env_ids, :2]
        self.jump_takeoff_position_w[env_ids] = root_xy
        self.jump_target_position_w[env_ids] = root_xy
        self.jump_landing_position_w[env_ids] = root_xy
        self.jump_landing_position_error[env_ids] = 0.0
        self._jump_landing_position_valid[env_ids] = False
        self.jump_takeoff_event[env_ids] = False
        self.jump_landing_event[env_ids] = False
        self.jump_success_event[env_ids] = False
        self.jump_failure_event[env_ids] = False
        self._jump_air_steps[env_ids] = 0
        self._jump_ground_steps[env_ids] = 0
        self._jump_stable_steps[env_ids] = 0
        self._jump_airborne_steps[env_ids] = 0
        self._jump_airborne_ever[env_ids] = False
        self._jump_attempts[env_ids] = 0.0
        self._jump_successes[env_ids] = 0.0
        self._jump_soft_landings[env_ids] = 0.0
        self._jump_target_landings[env_ids] = 0.0
        self._jump_landing_position_error_sum[env_ids] = 0.0
        self._jump_landing_position_count[env_ids] = 0.0
        self._jump_fail_crouch[env_ids] = 0.0
        self._jump_fail_thrust[env_ids] = 0.0
        self._jump_fail_unload[env_ids] = 0.0
        self._jump_fail_performance[env_ids] = 0.0
        self._jump_fail_recovery[env_ids] = 0.0


class WheelLeggedObstacleOracleEnv(WheelLeggedJumpEnv):
    """Jump environment with an exact-geometry obstacle trigger."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._obstacle = self.scene["obstacle"]
        self._obstacle_xform = XformPrimView(
            "/World/envs/env_.*/Obstacle",
            device=self.device,
            sync_usd_on_fabric_write=True,
        )
        self._obstacle_contact_sensors = tuple(
            self.scene.sensors[name]
            for name in (
                "obstacle_contact_base",
                "obstacle_contact_lf0",
                "obstacle_contact_lf1",
                "obstacle_contact_rf0",
                "obstacle_contact_rf1",
                "obstacle_contact_lwheel",
                "obstacle_contact_rwheel",
            )
        )
        self.obstacle_height = torch.zeros(self.num_envs, device=self.device)
        self.obstacle_width = torch.full(
            (self.num_envs,), self.cfg.obstacle_oracle.obstacle_width, device=self.device
        )
        self.obstacle_position_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.obstacle_forward_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.obstacle_forward_w[:, 0] = 1.0
        self.obstacle_min_clearance = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        self.obstacle_trigger_error = torch.zeros(self.num_envs, device=self.device)
        self.obstacle_trigger_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.obstacle_cross_event = torch.zeros_like(self.obstacle_trigger_event)
        self.obstacle_collision_event = torch.zeros_like(self.obstacle_trigger_event)
        self.obstacle_trial_triggered = torch.zeros_like(self.obstacle_trigger_event)
        self.obstacle_trial_crossed = torch.zeros_like(self.obstacle_trigger_event)
        self.obstacle_trial_collision = torch.zeros_like(self.obstacle_trigger_event)
        self.obstacle_respawn_pending = torch.zeros_like(self.obstacle_trigger_event)
        self.jump_obstacle_success_ready = torch.zeros_like(
            self.obstacle_trigger_event
        )
        self._obstacle_trials = torch.zeros(self.num_envs, device=self.device)
        self._obstacle_clears = torch.zeros(self.num_envs, device=self.device)
        self._obstacle_collisions = torch.zeros(self.num_envs, device=self.device)
        self._obstacle_successes = torch.zeros(self.num_envs, device=self.device)
        self._obstacle_collision_force_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._obstacle_collision_force_peak = torch.zeros(
            self.num_envs, device=self.device
        )
        # base, lf0, lf1, rf0, rf1, left wheel, right wheel
        self._obstacle_collision_body_counts = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        self._place_obstacle(torch.arange(self.num_envs, device=self.device))

    def _reset_obstacle_perception(self, env_ids: torch.Tensor):
        """Reset optional perception state when an obstacle is respawned."""

    def _obstacle_control_estimate(
        self, near_distance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return distance, height and width used by the jump controller.

        The Oracle stage deliberately returns simulator truth. Perceptive
        subclasses override this hook while the physical success metrics below
        continue to use exact geometry only as training-time supervision.
        """
        return near_distance, self.obstacle_height, self.obstacle_width

    def _place_obstacle(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        cfg = self.cfg.obstacle_oracle
        root_pos = self._robot.data.root_pos_w[env_ids]
        forward_b = torch.zeros(len(env_ids), 3, device=self.device)
        forward_b[:, 0] = 1.0
        forward_w = quat_apply(self._robot.data.root_quat_w[env_ids], forward_b)
        yaw = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        forward_xy = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
        geometry_term = getattr(self.cfg.curriculum, "geometry_levels", None)
        geometry_params = getattr(geometry_term, "params", {})
        initial_level = int(geometry_params.get("initial_level", 0))
        level = int(
            getattr(self, "_obstacle_curriculum_level", initial_level)
        )
        configured_height_levels = geometry_params.get(
            "height_levels", (cfg.obstacle_height_range[1],)
        )
        height_levels = getattr(
            self, "_obstacle_height_levels", configured_height_levels
        )
        level = min(level, len(height_levels) - 1)
        if cfg.fixed_height is None:
            height_top = float(height_levels[level])
            height_low = max(0.02, height_top - 0.01)
            height = torch.empty(len(env_ids), device=self.device).uniform_(
                height_low, height_top
            )
        else:
            height = torch.full(
                (len(env_ids),), float(cfg.fixed_height), device=self.device
            )
        if cfg.fixed_width is None:
            width = float(
                getattr(
                    self,
                    "_obstacle_width_levels",
                    geometry_params.get(
                        "width_levels", cfg.obstacle_width_levels
                    ),
                )[level]
            )
        else:
            width = float(cfg.fixed_width)
        scales = torch.ones((len(env_ids), 3), device=self.device)
        scales[:, 0] = width / cfg.obstacle_width
        self._obstacle_xform.set_scales(scales, indices=env_ids)
        position_xy = root_pos[:, :2] + cfg.spawn_distance * forward_xy
        # The cuboid keeps a fixed collision shape. Lowering most of it below
        # the plane provides per-environment exposed-height randomization.
        position_z = height - 0.5 * cfg.obstacle_max_height
        pose = torch.cat(
            (
                position_xy,
                position_z.unsqueeze(-1),
                quat_from_euler_xyz(
                    torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
                ),
            ),
            dim=1,
        )
        self._obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.obstacle_height[env_ids] = height
        self.obstacle_width[env_ids] = width
        self.obstacle_position_w[env_ids] = position_xy
        self.obstacle_forward_w[env_ids] = forward_xy
        self.obstacle_min_clearance[env_ids] = float("inf")
        self.obstacle_trigger_error[env_ids] = 0.0
        self.obstacle_trial_triggered[env_ids] = False
        self.obstacle_trial_crossed[env_ids] = False
        self.obstacle_trial_collision[env_ids] = False
        self.obstacle_respawn_pending[env_ids] = False
        self.jump_obstacle_success_ready[env_ids] = False
        self._reset_obstacle_perception(env_ids)

    def _pre_reward_update(self):
        cfg = self.cfg.obstacle_oracle
        self.obstacle_trigger_event.zero_()
        self.obstacle_cross_event.zero_()
        self.obstacle_collision_event.zero_()

        respawn_ids = self.obstacle_respawn_pending.nonzero(
            as_tuple=False
        ).squeeze(-1)
        self._place_obstacle(respawn_ids)

        root_xy = self._robot.data.root_pos_w[:, :2]
        root_delta = root_xy - self.obstacle_position_w
        root_longitudinal = torch.sum(root_delta * self.obstacle_forward_w, dim=1)
        near_distance = -0.5 * self.obstacle_width - root_longitudinal
        velocity_command = self.command_manager.get_command(
            "wheel_legged_commands"
        )
        control_distance, control_height, control_width = (
            self._obstacle_control_estimate(near_distance)
        )

        body_contact_forces = []
        for sensor in self._obstacle_contact_sensors:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                raise RuntimeError(
                    "Obstacle contact sensor filter was not initialized."
                )
            body_force = torch.linalg.vector_norm(force_matrix, dim=-1)
            while body_force.ndim > 1:
                body_force = body_force.amax(dim=-1)
            body_contact_forces.append(body_force)
        body_contact_forces = torch.stack(body_contact_forces, dim=1)
        contact_force, collision_body = body_contact_forces.max(dim=1)
        collision = contact_force > cfg.contact_force_threshold
        new_collision = (
            self.obstacle_trial_triggered
            & collision
            & ~self.obstacle_trial_collision
        )
        self.obstacle_collision_event[new_collision] = True
        self.obstacle_trial_collision |= collision & self.obstacle_trial_triggered
        self._obstacle_collisions[new_collision] += 1.0
        if new_collision.any():
            self._obstacle_collision_force_sum[new_collision] += contact_force[
                new_collision
            ]
            self._obstacle_collision_force_peak[new_collision] = torch.maximum(
                self._obstacle_collision_force_peak[new_collision],
                contact_force[new_collision],
            )
            event_env_ids = new_collision.nonzero(as_tuple=False).squeeze(-1)
            self._obstacle_collision_body_counts[
                event_env_ids, collision_body[event_env_ids]
            ] += 1.0

        wheel_xy = self._robot.data.body_pos_w[
            :, self._jump_robot_wheel_body_ids, :2
        ]
        wheel_delta = wheel_xy - self.obstacle_position_w.unsqueeze(1)
        wheel_longitudinal = torch.sum(
            wheel_delta * self.obstacle_forward_w.unsqueeze(1), dim=2
        )
        # Add two control-step travel margins so a narrow barrier cannot fall
        # entirely between sampled wheel positions at higher approach speeds.
        sampling_margin = (
            velocity_command[:, 0].abs() * (2.0 * self.step_dt)
        ).unsqueeze(1)
        wheel_over = (
            wheel_longitudinal.abs()
            <= 0.5 * self.obstacle_width.unsqueeze(1) + sampling_margin
        )
        both_over = wheel_over.all(dim=1)
        wheel_far_passed = (
            wheel_longitudinal.amin(dim=1) >= 0.5 * self.obstacle_width
        )
        wheel_bottom = (
            self._robot.data.body_pos_w[
                :, self._jump_robot_wheel_body_ids, 2
            ].amin(dim=1)
            - cfg.wheel_radius
        )
        clearance = wheel_bottom - self.obstacle_height
        self.obstacle_min_clearance = torch.where(
            both_over,
            torch.minimum(self.obstacle_min_clearance, clearance),
            self.obstacle_min_clearance,
        )

        crossed = (
            self.obstacle_trial_triggered
            & wheel_far_passed
            & ~self.obstacle_trial_crossed
        )
        self.obstacle_cross_event[crossed] = True
        self.obstacle_trial_crossed |= crossed
        clean_cross = crossed & ~self.obstacle_trial_collision
        self._obstacle_clears[clean_cross] += 1.0

        jump_command = self.command_manager.get_command("jump_command")
        target_height = torch.maximum(
            control_height + cfg.minimum_clearance + 0.025,
            torch.full_like(self.obstacle_height, 0.08),
        )
        # This value is latched by the jump state machine at confirmed
        # takeoff.  Keep it ballistically compatible with the command speed;
        # tying it to ``near_distance`` creates an unreachable target whenever
        # the policy brakes during crouch/thrust.
        target_distance = torch.clamp(
            velocity_command[:, 0].abs() * cfg.expected_air_time,
            min=cfg.target_distance_min,
            max=cfg.target_distance_max,
        )
        jump_command[:, 1] = target_height
        jump_command[:, 2] = target_distance

        idle = self.jump_phase == mdp.JUMP_PHASE_IDLE
        # Place takeoff so that the feasible landing point lies just beyond
        # the far edge. Wider levels therefore trigger closer to the barrier,
        # while faster commands retain more stand-off distance.
        takeoff_standoff = torch.clamp(
            target_distance - control_width - cfg.landing_margin,
            min=cfg.takeoff_margin,
            max=cfg.max_takeoff_margin,
        )
        trigger_distance = (
            velocity_command[:, 0].abs() * cfg.preparation_time
            + takeoff_standoff
        )
        trigger = (
            idle
            & ~self.obstacle_trial_triggered
            & (control_distance > 0.0)
            & (control_distance <= trigger_distance)
        )
        jump_command[:, 0] = trigger.float()
        if trigger.any():
            self.obstacle_trigger_event[trigger] = True
            self.obstacle_trial_triggered[trigger] = True
            self.obstacle_trigger_error[trigger] = (
                near_distance[trigger] - trigger_distance[trigger]
            )
            self._obstacle_trials[trigger] += 1.0

        finite_clearance = torch.isfinite(self.obstacle_min_clearance)
        self.jump_obstacle_success_ready = (
            self.obstacle_trial_crossed
            & ~self.obstacle_trial_collision
            & finite_clearance
            & (self.obstacle_min_clearance >= cfg.minimum_clearance)
        )

        super()._pre_reward_update()

        completed = (
            self.obstacle_trial_triggered
            & (self.jump_phase == mdp.JUMP_PHASE_IDLE)
            & (self.jump_phase_time == 0.0)
        )
        obstacle_success = completed & self.jump_success_event
        self._obstacle_successes[obstacle_success] += 1.0
        self.obstacle_respawn_pending |= completed

    def _log_debug_metrics(self):
        super()._log_debug_metrics()
        if not hasattr(self, "_obstacle_trials"):
            return
        log = dict(self.extras.get("log", {}))
        trials = self._obstacle_trials.sum().clamp_min(1.0)
        finite = torch.isfinite(self.obstacle_min_clearance)
        log["obstacle_height"] = self.obstacle_height.mean().item()
        log["obstacle_width"] = self.obstacle_width.mean().item()
        fixed_geometry = (
            self.cfg.obstacle_oracle.fixed_height is not None
            or self.cfg.obstacle_oracle.fixed_width is not None
        )
        log["obstacle_curriculum_level"] = (
            -1.0
            if fixed_geometry
            else float(getattr(self, "_obstacle_curriculum_level", 0))
        )
        if hasattr(self, "_obstacle_curriculum_last_success"):
            log["obstacle_curriculum_success"] = (
                self._obstacle_curriculum_last_success.item()
            )
            log["obstacle_curriculum_clear"] = (
                self._obstacle_curriculum_last_clear.item()
            )
            log["obstacle_curriculum_collision"] = (
                self._obstacle_curriculum_last_collision.item()
            )
        log["obstacle_trigger_error"] = (
            self.obstacle_trigger_error[self.obstacle_trial_triggered]
            .abs()
            .mean()
            .item()
            if self.obstacle_trial_triggered.any()
            else 0.0
        )
        log["obstacle_min_clearance"] = (
            self.obstacle_min_clearance[finite].mean().item()
            if finite.any()
            else 0.0
        )
        log["obstacle_clear_rate"] = (self._obstacle_clears.sum() / trials).item()
        log["obstacle_collision_rate"] = (
            self._obstacle_collisions.sum() / trials
        ).item()
        collision_count = self._obstacle_collisions.sum().clamp_min(1.0)
        log["obstacle_collision_force_mean"] = (
            self._obstacle_collision_force_sum.sum() / collision_count
        ).item()
        log["obstacle_collision_force_peak"] = (
            self._obstacle_collision_force_peak.max().item()
        )
        body_counts = self._obstacle_collision_body_counts.sum(dim=0)
        log["obstacle_collision_base_fraction"] = (
            body_counts[0] / collision_count
        ).item()
        log["obstacle_collision_upper_leg_fraction"] = (
            (body_counts[1] + body_counts[3]) / collision_count
        ).item()
        log["obstacle_collision_lower_leg_fraction"] = (
            (body_counts[2] + body_counts[4]) / collision_count
        ).item()
        log["obstacle_collision_wheel_fraction"] = (
            (body_counts[5] + body_counts[6]) / collision_count
        ).item()
        log["obstacle_success_rate"] = (
            self._obstacle_successes.sum() / trials
        ).item()
        self.extras["log"] = log

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_obstacle_trials"):
            return
        self._obstacle_trials[env_ids] = 0.0
        self._obstacle_clears[env_ids] = 0.0
        self._obstacle_collisions[env_ids] = 0.0
        self._obstacle_successes[env_ids] = 0.0
        self._obstacle_collision_force_sum[env_ids] = 0.0
        self._obstacle_collision_force_peak[env_ids] = 0.0
        self._obstacle_collision_body_counts[env_ids] = 0.0
        self._place_obstacle(env_ids)


class WheelLeggedObstaclePerceptiveEnv(WheelLeggedObstacleOracleEnv):
    """Obstacle task whose controller and actor observe calibrated depth only."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        perception = self.cfg.obstacle_perception
        sample_count = len(perception.scan_distances)
        self.obstacle_depth_scan = torch.zeros(
            self.num_envs, sample_count, device=self.device
        )
        self.obstacle_perception_distance = torch.full(
            (self.num_envs,), perception.max_distance, device=self.device
        )
        self.obstacle_perception_height = torch.zeros(
            self.num_envs, device=self.device
        )
        self.obstacle_perception_width = torch.full(
            (self.num_envs,), perception.width_prior, device=self.device
        )
        self.obstacle_perception_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # A low obstacle inevitably leaves the camera's vertical field of view
        # shortly before take-off.  Keep its last depth estimate alive with
        # base odometry instead of treating the near-field blind spot as a new
        # "no obstacle" observation.
        self._obstacle_perception_last_root_xy = (
            self._robot.data.root_pos_w[:, :2].clone()
        )
        self._depth_pixel_cache = None
        self._reset_obstacle_perception(
            torch.arange(self.num_envs, device=self.device)
        )

    def _reset_obstacle_perception(self, env_ids: torch.Tensor):
        if not hasattr(self, "obstacle_depth_scan") or len(env_ids) == 0:
            return
        perception = self.cfg.obstacle_perception
        self.obstacle_depth_scan[env_ids] = 0.0
        self.obstacle_perception_distance[env_ids] = perception.max_distance
        self.obstacle_perception_height[env_ids] = 0.0
        self.obstacle_perception_width[env_ids] = perception.width_prior
        self.obstacle_perception_valid[env_ids] = False
        self._obstacle_perception_last_root_xy[env_ids] = (
            self._robot.data.root_pos_w[env_ids, :2]
        )

    def _depth_pixels(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._depth_pixel_cache
        if cache is None or cache[0] != height or cache[1] != width:
            rows, columns = torch.meshgrid(
                torch.arange(height, device=self.device, dtype=torch.float32),
                torch.arange(width, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            cache = (height, width, columns.reshape(1, -1), rows.reshape(1, -1))
            self._depth_pixel_cache = cache
        return cache[2], cache[3]

    def _update_depth_obstacle_perception(self):
        perception = self.cfg.obstacle_perception
        camera = self.scene.sensors[perception.sensor_name]
        root_position = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        forward_body = torch.zeros(self.num_envs, 3, device=self.device)
        forward_body[:, 0] = 1.0
        forward = quat_apply(root_quat, forward_body)
        forward_xy = torch.nn.functional.normalize(forward[:, :2], dim=1)

        # Propagate an already detected obstacle through the camera's close
        # blind zone. This uses only deployable base odometry, not simulator
        # obstacle truth. A fresh depth return below overwrites the prediction.
        root_motion = root_position[:, :2] - self._obstacle_perception_last_root_xy
        forward_motion = torch.sum(root_motion * forward_xy, dim=1)
        tracking = self.obstacle_perception_valid & ~self.obstacle_trial_triggered
        self.obstacle_perception_distance[tracking] -= forward_motion[tracking]
        self.obstacle_perception_distance.clamp_(
            min=-perception.width_max, max=perception.max_distance
        )
        self._obstacle_perception_last_root_xy[:] = root_position[:, :2]

        depth = camera.data.output.get("distance_to_image_plane")
        if depth is None:
            self._reconstruct_tracked_depth_scan()
            return
        if depth.ndim == 4:
            depth = depth[..., 0]
        height, width = depth.shape[1:]
        column, row = self._depth_pixels(height, width)
        depth = depth.reshape(self.num_envs, -1)
        intrinsics = camera.data.intrinsic_matrices
        fx = intrinsics[:, 0, 0].unsqueeze(1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)

        valid_depth = (
            torch.isfinite(depth)
            & (depth > 0.05)
            & (depth < perception.max_distance)
        )
        safe_depth = torch.where(valid_depth, depth, torch.zeros_like(depth))
        if hasattr(camera, "ray_hits_w"):
            # The training sensor already exposes its calibrated ray hits. A
            # physical depth image follows the deprojection branch below and
            # produces the same world-space point representation.
            points_world = camera.ray_hits_w.reshape(self.num_envs, -1, 3)
        else:
            points_camera = torch.stack(
                (
                    (column - cx) * safe_depth / fx,
                    (row - cy) * safe_depth / fy,
                    safe_depth,
                ),
                dim=-1,
            )
            point_count = points_camera.shape[1]
            camera_quat = camera.data.quat_w_ros[:, None, :].expand(
                -1, point_count, -1
            )
            points_world = quat_apply(camera_quat, points_camera)
            points_world += camera.data.pos_w[:, None, :]

        lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=1)
        delta_xy = points_world[..., :2] - root_position[:, None, :2]
        longitudinal = torch.sum(delta_xy * forward_xy[:, None, :], dim=-1)
        lateral = torch.sum(delta_xy * lateral_xy[:, None, :], dim=-1)
        ground_height = self.scene.env_origins[:, 2].unsqueeze(1)
        point_height = points_world[..., 2] - ground_height
        # The training ray caster represents the output of a conventional
        # depth-camera ground-plane removal stage: rays are evaluated against
        # elevated scene geometry after the support plane is rejected. On the
        # real robot the same mask is obtained from calibrated depth + IMU.
        elevated_depth_return = point_height > perception.minimum_point_height
        obstacle_point = (
            valid_depth
            & elevated_depth_return
            & (longitudinal > 0.05)
            & (longitudinal < perception.max_distance)
            & (lateral.abs() < perception.lateral_half_width)
            & (point_height > perception.minimum_point_height)
            & (point_height < perception.maximum_point_height)
        )
        valid_count = obstacle_point.sum(dim=1)
        detected = valid_count >= perception.minimum_valid_points

        positive_inf = torch.full_like(longitudinal, float("inf"))
        negative_inf = torch.full_like(longitudinal, -float("inf"))
        near = torch.where(obstacle_point, longitudinal, positive_inf).amin(dim=1)
        far = torch.where(obstacle_point, longitudinal, negative_inf).amax(dim=1)
        estimated_height = torch.where(
            obstacle_point, point_height, negative_inf
        ).amax(dim=1)
        observed_width = far - near
        estimated_width = torch.where(
            observed_width >= perception.width_min,
            observed_width,
            torch.full_like(observed_width, perception.width_prior),
        ).clamp(perception.width_min, perception.width_max)

        # Before triggering, refresh the estimate whenever the obstacle is
        # visible. Once triggered, retain it through crouch/thrust even if the
        # camera can no longer see the very close barrier.
        refresh = detected & ~self.obstacle_trial_triggered
        self.obstacle_perception_distance[refresh] = near[refresh]
        self.obstacle_perception_height[refresh] = estimated_height[refresh]
        self.obstacle_perception_width[refresh] = estimated_width[refresh]
        self.obstacle_perception_valid |= detected | self.obstacle_trial_triggered
        self._reconstruct_tracked_depth_scan()

    def _reconstruct_tracked_depth_scan(self):
        """Build the Oracle-compatible 10-bin scan from the tracked geometry.

        Depth supplies the geometry when the target is visible; odometry moves
        that geometry through the final near-field blind zone.  This is the
        same compact interface intended for the real depth-camera pipeline.
        """
        perception = self.cfg.obstacle_perception
        samples = torch.tensor(
            perception.scan_distances, device=self.device
        ).unsqueeze(0)
        center_distance = (
            self.obstacle_perception_distance
            + 0.5 * self.obstacle_perception_width
        ).unsqueeze(1)
        half_width = 0.5 * self.obstacle_perception_width.unsqueeze(1)
        hit = (
            self.obstacle_perception_valid.unsqueeze(1)
            & ((samples - center_distance).abs() <= half_width)
        )
        self.obstacle_depth_scan[:] = torch.where(
            hit, self.obstacle_perception_height.unsqueeze(1), 0.0
        )

    def _obstacle_control_estimate(
        self, near_distance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._update_depth_obstacle_perception()
        # Invalid frames cannot trigger a jump. ``max_distance`` is safely
        # outside every trigger range and is replaced as soon as depth detects
        # an elevated cluster.
        distance = torch.where(
            self.obstacle_perception_valid,
            self.obstacle_perception_distance,
            torch.full_like(near_distance, self.cfg.obstacle_perception.max_distance),
        )
        return (
            distance,
            self.obstacle_perception_height,
            self.obstacle_perception_width,
        )

    def _append_perception_debug_metrics(self):
        if not hasattr(self, "obstacle_perception_valid"):
            return
        log = dict(self.extras.get("log", {}))
        # Keep reporting a latched detection after take-off.  Otherwise every
        # successful trigger would immediately turn into an invalid sample and
        # make the staged-training gate measure jump phase occupancy instead of
        # perception reliability.  Errors are still evaluated only before the
        # trigger, while exact geometry is available for diagnostics.
        tracked = self.obstacle_perception_valid
        valid = tracked & ~self.obstacle_trial_triggered
        log["obstacle_perception_valid_rate"] = tracked.float().mean().item()
        triggered = tracked & self.obstacle_trial_triggered
        log["obstacle_perception_trigger_distance"] = (
            self.obstacle_perception_distance[triggered].mean().item()
            if triggered.any()
            else 0.0
        )
        if valid.any():
            root_delta = self._robot.data.root_pos_w[:, :2] - self.obstacle_position_w
            exact_distance = -0.5 * self.obstacle_width - torch.sum(
                root_delta * self.obstacle_forward_w, dim=1
            )
            log["obstacle_perception_distance_error"] = (
                self.obstacle_perception_distance[valid] - exact_distance[valid]
            ).abs().mean().item()
            log["obstacle_perception_height_error"] = (
                self.obstacle_perception_height[valid] - self.obstacle_height[valid]
            ).abs().mean().item()
            log["obstacle_perception_width_error"] = (
                self.obstacle_perception_width[valid] - self.obstacle_width[valid]
            ).abs().mean().item()
            log["obstacle_perception_distance"] = (
                self.obstacle_perception_distance[valid].mean().item()
            )
            log["obstacle_perception_height"] = (
                self.obstacle_perception_height[valid].mean().item()
            )
            log["obstacle_perception_width"] = (
                self.obstacle_perception_width[valid].mean().item()
            )
        else:
            log["obstacle_perception_distance_error"] = 0.0
            log["obstacle_perception_height_error"] = 0.0
            log["obstacle_perception_width_error"] = 0.0
            log["obstacle_perception_distance"] = 0.0
            log["obstacle_perception_height"] = 0.0
            log["obstacle_perception_width"] = 0.0
        self.extras["log"] = log

    def _log_debug_metrics(self):
        super()._log_debug_metrics()
        if not hasattr(self, "_obstacle_trials"):
            return
        log = dict(self.extras.get("log", {}))
        trials = self._obstacle_trials.sum().clamp_min(1.0)
        finite = torch.isfinite(self.obstacle_min_clearance)
        log["obstacle_height"] = self.obstacle_height.mean().item()
        log["obstacle_width"] = self.obstacle_width.mean().item()
        fixed_geometry = (
            self.cfg.obstacle_oracle.fixed_height is not None
            or self.cfg.obstacle_oracle.fixed_width is not None
        )
        # ``-1`` makes manual Play geometry distinguishable from curriculum
        # level 0 in the compact O档 diagnostic.
        log["obstacle_curriculum_level"] = (
            -1.0
            if fixed_geometry
            else float(getattr(self, "_obstacle_curriculum_level", 0))
        )
        if hasattr(self, "_obstacle_curriculum_last_success"):
            log["obstacle_curriculum_success"] = (
                self._obstacle_curriculum_last_success.item()
            )
            log["obstacle_curriculum_clear"] = (
                self._obstacle_curriculum_last_clear.item()
            )
            log["obstacle_curriculum_collision"] = (
                self._obstacle_curriculum_last_collision.item()
            )
        log["obstacle_trigger_error"] = (
            self.obstacle_trigger_error[self.obstacle_trial_triggered]
            .abs()
            .mean()
            .item()
            if self.obstacle_trial_triggered.any()
            else 0.0
        )
        log["obstacle_min_clearance"] = (
            self.obstacle_min_clearance[finite].mean().item()
            if finite.any()
            else 0.0
        )
        log["obstacle_clear_rate"] = (self._obstacle_clears.sum() / trials).item()
        log["obstacle_collision_rate"] = (
            self._obstacle_collisions.sum() / trials
        ).item()
        collision_count = self._obstacle_collisions.sum().clamp_min(1.0)
        log["obstacle_collision_force_mean"] = (
            self._obstacle_collision_force_sum.sum() / collision_count
        ).item()
        log["obstacle_collision_force_peak"] = (
            self._obstacle_collision_force_peak.max().item()
        )
        body_counts = self._obstacle_collision_body_counts.sum(dim=0)
        # Group symmetric links to keep the terminal line readable.
        log["obstacle_collision_base_fraction"] = (
            body_counts[0] / collision_count
        ).item()
        log["obstacle_collision_upper_leg_fraction"] = (
            (body_counts[1] + body_counts[3]) / collision_count
        ).item()
        log["obstacle_collision_lower_leg_fraction"] = (
            (body_counts[2] + body_counts[4]) / collision_count
        ).item()
        log["obstacle_collision_wheel_fraction"] = (
            (body_counts[5] + body_counts[6]) / collision_count
        ).item()
        log["obstacle_success_rate"] = (
            self._obstacle_successes.sum() / trials
        ).item()
        self.extras["log"] = log
        self._append_perception_debug_metrics()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)


@configclass
class WheelLeggedJumpObservationsCfg:
    """Flat-task observations plus explicit jump state."""

    @configclass
    class PolicyCfg(WheelleggedObservationsCfg.PolicyCfg):
        jump_phase = ObsTerm(func=mdp.jump_phase_one_hot, clip=(0.0, 1.0))
        wheel_contacts = ObsTerm(func=mdp.jump_wheel_contacts, clip=(0.0, 1.0))
        jump_command = ObsTerm(
            func=mdp.jump_command,
            params={"command_name": "jump_command", "scale": (1.0, 10.0, 5.0)},
            clip=(-5.0, 5.0),
        )
        jump_phase_time = ObsTerm(func=mdp.jump_phase_time, clip=(0.0, 1.0))

    @configclass
    class CriticCfg(WheelleggedObservationsCfg.CriticCfg):
        jump_phase = ObsTerm(func=mdp.jump_phase_one_hot, clip=(0.0, 1.0))
        wheel_contacts = ObsTerm(func=mdp.jump_wheel_contacts, clip=(0.0, 1.0))
        jump_command = ObsTerm(
            func=mdp.jump_command,
            params={"command_name": "jump_command", "scale": (1.0, 10.0, 5.0)},
            clip=(-5.0, 5.0),
        )
        jump_phase_time = ObsTerm(func=mdp.jump_phase_time, clip=(0.0, 1.0))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class WheelLeggedObstacleObservationsCfg(WheelLeggedJumpObservationsCfg):
    """Jump observations augmented with a compact robot-aligned forward scan."""

    @configclass
    class PolicyCfg(WheelLeggedJumpObservationsCfg.PolicyCfg):
        obstacle_forward_scan = ObsTerm(
            func=mdp.obstacle_forward_scan,
            params={
                "distances": (
                    0.10, 0.20, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 1.00,
                )
            },
            clip=(0.0, 0.10),
            scale=10.0,
        )

    @configclass
    class CriticCfg(WheelLeggedJumpObservationsCfg.CriticCfg):
        obstacle_forward_scan = ObsTerm(
            func=mdp.obstacle_forward_scan,
            params={
                "distances": (
                    0.10, 0.20, 0.30, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 1.00,
                )
            },
            clip=(0.0, 0.10),
            scale=10.0,
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class WheelLeggedPerceptiveObstacleObservationsCfg(
    WheelLeggedObstacleObservationsCfg
):
    """Depth-derived actor scan with an asymmetric Oracle critic scan."""

    @configclass
    class PolicyCfg(WheelLeggedJumpObservationsCfg.PolicyCfg):
        obstacle_forward_scan = ObsTerm(
            func=mdp.depth_camera_obstacle_scan,
            params={"sample_count": 10},
            clip=(0.0, 0.10),
            scale=10.0,
        )

    # The critic may use simulator geometry during training. It is discarded
    # on export, while the deployed actor depends only on calibrated depth.
    @configclass
    class CriticCfg(WheelLeggedObstacleObservationsCfg.CriticCfg):
        pass

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class WheelLeggedJumpCommandsCfg:
    """Low-speed locomotion command plus an independent jump trigger."""

    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        heading_command=False,
        rel_heading_envs=0.0,
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            lin_vel_x=(-0.05, 0.05),
            ang_vel_yaw=(-0.10, 0.10),
            height=(0.20, 0.22),
            heading=(0.0, 0.0),
        ),
    )
    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.0, 4.0),
        jump_probability=0.9,
        trigger_delay_range=(0.6, 1.0),
        trigger_pulse_time=0.10,
        # Height is accepted only together with confirmed takeoff and at least
        # 0.12 s of air time, so grounded leg extension cannot earn success.
        target_height_range=(0.05, 0.07),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedJumpRewardsCfg(WheelLeggedRewardsCfg):
    """Phase-aware locomotion and small-jump objectives."""

    lin_vel_z = RewardTermCfg(func=mdp.jump_gated_lin_vel_z_l2, weight=-0.5)
    base_height = RewardTermCfg(
        func=mdp.jump_gated_base_height_reward,
        weight=2.5,
        params={"command_name": "wheel_legged_commands", "std": 0.06},
    )
    dof_acc = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-5.0e-9,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    torques = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001)
    action_smooth = RewardTermCfg(func=mdp.action_smooth_l2, weight=-0.001)

    jump_crouch = RewardTermCfg(func=mdp.jump_crouch_tracking, weight=8.0, params={"std": 0.030})
    # The reference action is applied by JumpVMCAction already.  Keep only a
    # weak imitation term so it cannot dominate the physical jump objectives.
    jump_phase_action = RewardTermCfg(func=mdp.jump_leg_action_tracking, weight=1.0)
    jump_thrust_pose = RewardTermCfg(
        func=mdp.jump_thrust_length_tracking,
        weight=5.0,
        params={"std": 0.035},
    )
    jump_thrust_speed = RewardTermCfg(
        func=mdp.jump_thrust_extension_speed,
        weight=5.0,
        params={"target_speed": 1.0},
    )
    jump_takeoff = RewardTermCfg(func=mdp.jump_takeoff_velocity, weight=10.0, params={"std": 0.50})
    jump_takeoff_event = RewardTermCfg(func=mdp.jump_takeoff_event_bonus, weight=100.0)
    jump_height = RewardTermCfg(func=mdp.jump_height_tracking, weight=6.0, params={"std": 0.030})
    jump_airborne = RewardTermCfg(func=mdp.jump_airborne_reward, weight=4.0)
    jump_symmetry = RewardTermCfg(func=mdp.jump_leg_symmetry_reward, weight=0.5)
    jump_landing_pose = RewardTermCfg(
        func=mdp.jump_landing_length_tracking,
        weight=1.0,
        params={"std": 0.025},
    )
    jump_landing_soft = RewardTermCfg(
        func=mdp.jump_landing_soft,
        weight=50.0,
        params={"std_vz": 0.50, "std_tilt": 0.20},
    )
    jump_landing_impact = RewardTermCfg(func=mdp.jump_landing_impact, weight=-25.0)
    jump_recovery = RewardTermCfg(
        func=mdp.jump_recovery_stability,
        weight=1.5,
        params={"std_vz": 0.30, "std_tilt": 0.20},
    )
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=200.0)
    jump_failure = RewardTermCfg(func=mdp.jump_failure_penalty, weight=-100.0)


@configclass
class WheelLeggedJumpTerminationsCfg(WheelLeggedTerminationsCfg):
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.20},
    )
    minimum_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.05},
    )


@configclass
class WheelLeggedJumpFlatEnvCfg(WheelLeggedFlatEnvCfg):
    """Initial curriculum task: low-speed locomotion and externally triggered jumps."""

    scene: WheelLeggedRobotSceneCfg = WheelLeggedRobotSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: WheelLeggedJumpObservationsCfg = WheelLeggedJumpObservationsCfg()
    actions: VmcActionsCfg = VmcActionsCfg()
    commands: WheelLeggedJumpCommandsCfg = WheelLeggedJumpCommandsCfg()
    rewards: WheelLeggedJumpRewardsCfg = WheelLeggedJumpRewardsCfg()
    terminations: WheelLeggedJumpTerminationsCfg = WheelLeggedJumpTerminationsCfg()
    events: WheelLeggedEventCfg = WheelLeggedEventCfg()
    curriculum = None
    jump_state_machine: JumpStateMachineCfg = JumpStateMachineCfg()
    env_class = WheelLeggedJumpEnv

    def __post_init__(self):
        super().__post_init__()

        # The flat parent currently rebuilds manager configs in __post_init__,
        # so install the jump variants after the common scene/simulation setup.
        self.observations = WheelLeggedJumpObservationsCfg()
        self.actions = VmcActionsCfg()
        self.actions.vmc.class_type = mdp.JumpVMCAction
        self.commands = WheelLeggedJumpCommandsCfg()
        self.rewards = WheelLeggedJumpRewardsCfg()
        self.terminations = WheelLeggedJumpTerminationsCfg()
        self.events = WheelLeggedEventCfg()
        self.curriculum = None
        self.jump_state_machine = JumpStateMachineCfg()
        self.episode_length_s = 20.0
        # These flat-task regularizers oppose the large, intentional leg
        # excursion used for jumping. Keep only a weak branch preference.
        self.rewards.nominal_state.weight = -0.1
        self.rewards.leg_posture.weight = -0.1

        # Retain moderate domain randomization, but remove pushes during the
        # first jump curriculum so contact transitions remain learnable.
        self.events.push_robot = None

        # Preserve the proven flat-task normalization for inherited terms.
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


@configclass
class WheelLeggedHighJumpLandingCommandsCfg(WheelLeggedJumpCommandsCfg):
    """Stationary 9--12 cm jumps used before the moving-jump curriculum."""

    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.5, 4.5),
        jump_probability=0.90,
        trigger_delay_range=(0.7, 1.1),
        trigger_pulse_time=0.10,
        target_height_range=(0.09, 0.12),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedHighJumpLandingRewardsCfg(WheelLeggedJumpRewardsCfg):
    """High-jump reward balance with explicit touchdown and absorption priority."""

    jump_crouch = RewardTermCfg(
        func=mdp.jump_crouch_tracking, weight=7.0, params={"std": 0.025}
    )
    jump_phase_action = RewardTermCfg(func=mdp.jump_leg_action_tracking, weight=1.5)
    jump_thrust_pose = RewardTermCfg(
        func=mdp.jump_thrust_length_tracking,
        weight=5.0,
        params={"std": 0.030},
    )
    jump_thrust_speed = RewardTermCfg(
        func=mdp.jump_thrust_extension_speed,
        weight=6.0,
        params={"target_speed": 1.2},
    )
    jump_takeoff = RewardTermCfg(
        func=mdp.jump_takeoff_velocity, weight=12.0, params={"std": 0.55}
    )
    jump_takeoff_event = RewardTermCfg(func=mdp.jump_takeoff_event_bonus, weight=80.0)
    jump_height = RewardTermCfg(
        func=mdp.jump_height_tracking, weight=10.0, params={"std": 0.035}
    )
    jump_airborne = RewardTermCfg(func=mdp.jump_airborne_reward, weight=5.0)
    jump_landing_pose = RewardTermCfg(
        func=mdp.jump_landing_length_tracking,
        weight=4.0,
        params={"std": 0.025},
    )
    jump_landing_soft = RewardTermCfg(
        func=mdp.jump_landing_soft,
        weight=100.0,
        params={"std_vz": 0.55, "std_tilt": 0.18},
    )
    jump_landing_impact = RewardTermCfg(func=mdp.jump_landing_impact, weight=-40.0)
    jump_recovery = RewardTermCfg(
        func=mdp.jump_recovery_stability,
        weight=3.0,
        params={"std_vz": 0.25, "std_tilt": 0.18},
    )
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=250.0)
    jump_failure = RewardTermCfg(func=mdp.jump_failure_penalty, weight=-120.0)


@configclass
class WheelLeggedHighJumpLandingFlatEnvCfg(WheelLeggedJumpFlatEnvCfg):
    """Stage C1: 9--12 cm high jump with pre-extension and impact absorption."""

    commands: WheelLeggedHighJumpLandingCommandsCfg = WheelLeggedHighJumpLandingCommandsCfg()
    rewards: WheelLeggedHighJumpLandingRewardsCfg = WheelLeggedHighJumpLandingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedHighJumpLandingCommandsCfg()
        self.rewards = WheelLeggedHighJumpLandingRewardsCfg()

        state = self.jump_state_machine
        state.use_height_conditioned_crouch = True
        state.reference_height_min = 0.09
        state.reference_height_max = 0.12
        # Deeper commands receive a deeper crouch, while the low end starts
        # close to the proven Stage-B2 reference.
        state.crouch_length = 0.19
        state.high_jump_crouch_length = 0.18
        state.crouch_ready_length = 0.21
        state.crouch_min_time = 0.24
        state.max_crouch_time = 0.48
        state.thrust_length = 0.30
        state.min_thrust_time = 0.20
        state.max_thrust_time = 0.36
        state.min_release_vz = 0.50
        state.min_release_length = 0.31

        state.use_landing_assist = True
        state.flight_retract_length = 0.19
        # A useful pre-extension must exceed the normal ~0.27 m standing leg
        # length.  The 0.30 m reference plus a small airborne feed-forward
        # targets ~0.31 m actual length without the former 0.328 m overshoot.
        state.prelanding_length = 0.30
        # Start before the apex so a zero-feed-forward airborne leg has enough
        # time to reach the requested length without a last-moment impulse.
        state.prelanding_start_vz = 0.30
        state.prelanding_full_vz = -0.30
        state.landing_absorption_length = 0.21
        state.landing_compression_time = 0.20
        state.landing_time = 0.30
        state.prelanding_action_residual_scale = 0.10
        state.landing_action_residual_scale = 0.15

        state.use_phase_dependent_gains = True
        state.thrust_kp_scale = 1.25
        state.thrust_feedforward_scale = 1.15
        state.landing_kd_scale = 2.5
        state.flight_feedforward_scale = 0.55
        state.landing_feedforward_scale = 1.20
        state.recovery_stable_time = 0.50
        state.max_recovery_time = 2.0
        state.success_height_ratio = 0.80
        state.success_min_air_time = 0.12
        # Do not call a hard 1.1 m/s touchdown a success.  The separately
        # logged soft-landing rate retains the final <=0.8 m/s acceptance goal.
        state.success_max_landing_speed = 1.05
        state.soft_landing_speed = 0.80

        # Keep the current stage near-stationary.  Forward-distance learning is
        # intentionally deferred to the next task/checkpoint.
        self.commands.wheel_legged_commands.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.wheel_legged_commands.ranges.ang_vel_yaw = (-0.10, 0.10)


@configclass
class WheelLeggedJumpClearanceCommandsCfg(WheelLeggedJumpCommandsCfg):
    """Stationary 8--12 cm wheel-clearance commands."""

    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.5, 4.5),
        jump_probability=0.90,
        trigger_delay_range=(0.7, 1.1),
        trigger_pulse_time=0.10,
        target_height_range=(0.08, 0.12),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedJumpClearanceRewardsCfg(WheelLeggedHighJumpLandingRewardsCfg):
    """Prioritize physical wheel clearance without discarding safe landing."""

    # Base rise remains useful for takeoff, but it is no longer the dominant
    # height objective. The new term measures the lower of both wheel centers.
    jump_height = RewardTermCfg(
        func=mdp.jump_height_tracking, weight=5.0, params={"std": 0.040}
    )
    jump_wheel_clearance = RewardTermCfg(
        func=mdp.jump_wheel_clearance_tracking,
        weight=16.0,
        params={"std": 0.025},
    )
    jump_takeoff = RewardTermCfg(
        func=mdp.jump_takeoff_velocity, weight=14.0, params={"std": 0.60}
    )
    jump_airborne = RewardTermCfg(func=mdp.jump_airborne_reward, weight=6.0)
    # Retraction and pre-extension references are deterministic guard rails;
    # this term makes it costly for the residual actor to cancel them.
    jump_landing_pose = RewardTermCfg(
        func=mdp.jump_landing_length_tracking,
        weight=6.0,
        params={"std": 0.022},
    )
    jump_landing_soft = RewardTermCfg(
        func=mdp.jump_landing_soft,
        weight=100.0,
        params={"std_vz": 0.60, "std_tilt": 0.18},
    )
    jump_landing_impact = RewardTermCfg(func=mdp.jump_landing_impact, weight=-35.0)
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=300.0)
    jump_failure = RewardTermCfg(func=mdp.jump_failure_penalty, weight=-120.0)


@configclass
class WheelLeggedJumpClearanceFlatEnvCfg(WheelLeggedHighJumpLandingFlatEnvCfg):
    """Stage C2: retract in flight to raise both wheel centers by up to 12 cm."""

    commands: WheelLeggedJumpClearanceCommandsCfg = WheelLeggedJumpClearanceCommandsCfg()
    rewards: WheelLeggedJumpClearanceRewardsCfg = WheelLeggedJumpClearanceRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedJumpClearanceCommandsCfg()
        self.rewards = WheelLeggedJumpClearanceRewardsCfg()

        state = self.jump_state_machine
        state.reference_height_min = 0.08
        state.reference_height_max = 0.12
        state.crouch_length = 0.19
        state.high_jump_crouch_length = 0.18
        state.min_release_vz = 0.50
        # Rapidly retracting the wheels redistributes momentum inside the
        # articulation and can reduce base-link vz during the two contact
        # confirmation frames. Release itself already required 0.50 m/s, so a
        # lower post-release threshold avoids misclassifying real takeoffs as
        # unload failures.
        state.min_takeoff_vz = 0.10

        # Hold the wheels close to the body through the apex. Stage C1 began
        # pre-extension at +0.30 m/s and retained 55% weight feed-forward,
        # which effectively cancelled most of the requested retraction.
        state.flight_retract_length = 0.18
        state.prelanding_length = 0.30
        state.prelanding_start_vz = -0.08
        state.prelanding_full_vz = -0.65
        state.prelanding_action_residual_scale = 0.08
        state.landing_action_residual_scale = 0.15

        state.use_split_flight_control = True
        state.flight_retract_kp_scale = 1.8
        state.flight_retract_kd_scale = 0.50
        state.flight_retract_feedforward_scale = 0.0
        state.prelanding_feedforward_scale = 0.70
        state.prelanding_kd_scale = 1.10
        state.landing_kd_scale = 2.5
        state.landing_feedforward_scale = 1.20

        # Success now requires both useful base motion and actual wheel
        # clearance. A 0.90 ratio makes the 0.12 m command require >=0.108 m
        # during early training; the dense reward still peaks at the full
        # commanded 0.12 m.
        state.success_height_ratio = 0.75
        state.success_wheel_clearance_ratio = 0.90
        state.success_min_air_time = 0.12
        state.success_max_landing_speed = 1.10
        state.soft_landing_speed = 0.80

        self.commands.wheel_legged_commands.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.wheel_legged_commands.ranges.ang_vel_yaw = (-0.10, 0.10)


@configclass
class WheelLeggedMovingJumpCommandsCfg(WheelLeggedJumpClearanceCommandsCfg):
    """First moving-jump level: gentle forward/backward motion and yaw."""

    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 7.0),
        heading_command=False,
        rel_heading_envs=0.0,
        hold_command_during_jump=True,
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            lin_vel_x=(-0.20, 0.20),
            ang_vel_yaw=(-0.20, 0.20),
            height=(0.20, 0.22),
            heading=(0.0, 0.0),
        ),
    )
    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.5, 4.5),
        # Keep enough no-jump windows to protect ordinary locomotion.
        jump_probability=0.80,
        trigger_delay_range=(0.7, 1.1),
        trigger_pulse_time=0.10,
        target_height_range=(0.08, 0.12),
        target_distance_range=(0.0, 0.0),
    )


@configclass
class WheelLeggedMovingJumpRewardsCfg(WheelLeggedJumpClearanceRewardsCfg):
    """Preserve speed through takeoff, flight, touchdown and recovery."""

    track_lin_vel = RewardTermCfg(
        func=mdp.jump_moving_velocity_tracking,
        weight=4.0,
        params={"std": 0.18},
    )
    jump_flight_velocity = RewardTermCfg(
        func=mdp.jump_flight_velocity_preservation,
        weight=6.0,
        params={"std": 0.22},
    )
    jump_landing_velocity = RewardTermCfg(
        func=mdp.jump_landing_velocity_tracking,
        weight=80.0,
        params={"std": 0.20},
    )
    # Slightly reduce the sparse bonus so speed preservation cannot be traded
    # away merely by satisfying the vertical-jump acceptance condition.
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=260.0)


@configclass
class WheelLeggedMovingJumpFlatEnvCfg(WheelLeggedJumpClearanceFlatEnvCfg):
    """Stage C3-L1: retain 8--12 cm wheel clearance while moving at ±0.20 m/s."""

    commands: WheelLeggedMovingJumpCommandsCfg = WheelLeggedMovingJumpCommandsCfg()
    rewards: WheelLeggedMovingJumpRewardsCfg = WheelLeggedMovingJumpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedMovingJumpCommandsCfg()
        self.rewards = WheelLeggedMovingJumpRewardsCfg()

        state = self.jump_state_machine
        state.start_max_speed = 0.35
        # At the completion of RECOVERY, forward speed must again agree with
        # the velocity command latched when the jump started.
        state.success_max_recovery_vx_error = 0.15

        # Keep the proven vertical trajectory unchanged in the first moving
        # level. Higher speed and explicit target distance belong to later
        # levels, after this task passes its regression gates.
        self.commands.wheel_legged_commands.ranges.lin_vel_x = (-0.20, 0.20)
        self.commands.wheel_legged_commands.ranges.ang_vel_yaw = (-0.20, 0.20)


@configclass
class WheelLeggedMovingJumpCurriculumCommandsCfg(WheelLeggedMovingJumpCommandsCfg):
    """World-heading-aligned commands with a final ±1.0 m/s envelope."""

    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 7.0),
        heading_command=True,
        heading_control_stiffness=0.8,
        rel_heading_envs=1.0,
        hold_command_during_jump=True,
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            # The curriculum immediately installs the first ±0.20 level.
            lin_vel_x=(-1.0, 1.0),
            # This is the heading controller's correction clamp, not a random
            # yaw-rate command range.
            ang_vel_yaw=(-1.2, 1.2),
            height=(0.20, 0.22),
            # Keep every robot aligned with the world x-axis.
            heading=(0.0, 0.0),
        ),
    )


@configclass
class WheelLeggedMovingJumpCurriculumRewardsCfg(WheelLeggedMovingJumpRewardsCfg):
    """Make heading position and its generated yaw-rate command compatible."""

    track_ang_vel = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp_wl,
        weight=1.5,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    track_heading = RewardTermCfg(
        func=mdp.track_heading_exp,
        weight=1.5,
        params={
            "command_name": "wheel_legged_commands",
            "std": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class WheelLeggedMovingJumpSpeedCurriculumCfg:
    """Discrete speed levels guarded by locomotion, jump, landing and yaw."""

    speed_levels = CurriculumTermCfg(
        func=mdp.moving_jump_speed_curriculum,
        params={
            "reward_term_name": "track_lin_vel",
            "heading_reward_term_name": "track_heading",
            "command_name": "wheel_legged_commands",
            "speed_levels": (0.2, 0.4, 0.6, 0.8, 1.0),
            "initial_level": 0,
            "tracking_threshold": 0.78,
            "jump_success_threshold": 0.75,
            "soft_landing_threshold": 0.75,
            "heading_threshold": 0.80,
            "min_episodes": 512,
            "min_jump_attempts": 1024,
            "consecutive_passes": 2,
        },
    )


@configclass
class WheelLeggedMovingJumpCurriculumFlatEnvCfg(WheelLeggedMovingJumpFlatEnvCfg):
    """Stage C3: one run from ±0.20 to ±1.00 m/s with heading alignment."""

    commands: WheelLeggedMovingJumpCurriculumCommandsCfg = (
        WheelLeggedMovingJumpCurriculumCommandsCfg()
    )
    rewards: WheelLeggedMovingJumpCurriculumRewardsCfg = (
        WheelLeggedMovingJumpCurriculumRewardsCfg()
    )
    curriculum: WheelLeggedMovingJumpSpeedCurriculumCfg = (
        WheelLeggedMovingJumpSpeedCurriculumCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedMovingJumpCurriculumCommandsCfg()
        self.rewards = WheelLeggedMovingJumpCurriculumRewardsCfg()
        self.curriculum = WheelLeggedMovingJumpSpeedCurriculumCfg()
        # The L1 task intentionally rejected starts above 0.35 m/s. The
        # curriculum must permit jump triggers at every configured level.
        self.jump_state_machine.start_max_speed = 1.20


@configclass
class WheelLeggedTargetLandingCommandsCfg(WheelLeggedMovingJumpCurriculumCommandsCfg):
    """Full-range moving jumps with velocity-feasible landing displacement."""

    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(3.5, 4.5),
        jump_probability=0.85,
        trigger_delay_range=(0.7, 1.1),
        trigger_pulse_time=0.10,
        target_height_range=(0.08, 0.12),
        # The target is recomputed from the signed locomotion command when the
        # pulse starts: d = vx_cmd * expected_air_time * noise.
        target_distance_range=(-0.16, 0.16),
        couple_distance_to_velocity=True,
        expected_air_time=0.16,
        distance_noise_range=(0.90, 1.10),
    )


@configclass
class WheelLeggedTargetLandingRewardsCfg(WheelLeggedMovingJumpCurriculumRewardsCfg):
    """Add dense approach and first-touchdown target-position objectives."""

    jump_target_progress = RewardTermCfg(
        func=mdp.jump_target_landing_progress,
        weight=8.0,
        params={"std": 0.10},
    )
    jump_target_landing = RewardTermCfg(
        func=mdp.jump_target_landing_tracking,
        weight=180.0,
        params={"std": 0.05},
    )
    # Success now includes target accuracy, so its sparse credit may remain
    # comparable to the proven moving-jump stage.
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=280.0)


@configclass
class WheelLeggedTargetLandingFlatEnvCfg(
    WheelLeggedMovingJumpCurriculumFlatEnvCfg
):
    """Stage D1: land near a commanded planar point on unobstructed flat ground."""

    commands: WheelLeggedTargetLandingCommandsCfg = (
        WheelLeggedTargetLandingCommandsCfg()
    )
    rewards: WheelLeggedTargetLandingRewardsCfg = (
        WheelLeggedTargetLandingRewardsCfg()
    )
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.commands = WheelLeggedTargetLandingCommandsCfg()
        self.rewards = WheelLeggedTargetLandingRewardsCfg()
        self.curriculum = None

        # Cover the final locomotion envelope. Signed distance commands make
        # backward jumps target a point behind the takeoff pose.
        command = self.commands.wheel_legged_commands
        command.ranges.lin_vel_x = (-1.0, 1.0)
        command.ranges.ang_vel_yaw = (-1.20, 1.20)

        state = self.jump_state_machine
        state.start_max_speed = 1.0
        state.success_max_recovery_vx_error = 0.18
        state.success_max_landing_position_error = 0.05


@configclass
class WheelLeggedObstacleOracleCommandsCfg(WheelLeggedTargetLandingCommandsCfg):
    """Forward approach commands; the environment supplies every jump pulse."""

    wheel_legged_commands = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 7.0),
        heading_command=True,
        heading_control_stiffness=0.8,
        rel_heading_envs=1.0,
        hold_command_during_jump=True,
        ranges=mdp.WheelLeggedCommandCfg.Ranges(
            # Level 0 starts close to the distribution already mastered by
            # target-landing. The geometry curriculum expands this range.
            lin_vel_x=(0.45, 0.60),
            ang_vel_yaw=(-0.40, 0.40),
            height=(0.20, 0.22),
            heading=(0.0, 0.0),
        ),
    )
    jump_command = mdp.JumpCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        jump_probability=0.0,
        trigger_delay_range=(1.0e9, 1.0e9),
        trigger_pulse_time=0.10,
        target_height_range=(0.08, 0.10),
        # Overwritten every step by the oracle with vx * expected_air_time;
        # this range documents and bounds the actual training distribution.
        target_distance_range=(0.08, 0.16),
        couple_distance_to_velocity=False,
    )


@configclass
class WheelLeggedObstacleOracleRewardsCfg(WheelLeggedTargetLandingRewardsCfg):
    """Prioritize clean barrier crossing while retaining landing quality."""

    # Level 6 already crosses often, but most remaining contacts are wheel-edge
    # contacts. Strengthen the dense over-obstacle signal instead of weakening
    # the physical clean-crossing definition.
    obstacle_clearance = RewardTermCfg(
        func=mdp.obstacle_clearance_tracking,
        weight=28.0,
        params={"target_clearance": 0.025, "std": 0.020},
    )
    obstacle_crossing = RewardTermCfg(
        func=mdp.obstacle_crossing_bonus,
        weight=240.0,
    )
    obstacle_collision = RewardTermCfg(
        func=mdp.obstacle_collision_penalty,
        weight=-360.0,
    )
    # The latest level-6 logs show a ~0.20 m/s takeoff speed error and only
    # ~0.72 target-landing rate. These denser terms prevent the larger sparse
    # crossing bonus from sacrificing horizontal-speed and landing quality.
    jump_flight_velocity = RewardTermCfg(
        func=mdp.jump_flight_velocity_preservation,
        weight=10.0,
        params={"std": 0.20},
    )
    jump_landing_velocity = RewardTermCfg(
        func=mdp.jump_landing_velocity_tracking,
        weight=100.0,
        params={"std": 0.20},
    )
    jump_target_progress = RewardTermCfg(
        func=mdp.jump_target_landing_progress,
        weight=12.0,
        params={"std": 0.10},
    )
    jump_target_landing = RewardTermCfg(
        func=mdp.jump_target_landing_tracking,
        weight=220.0,
        params={"std": 0.06},
    )
    jump_success = RewardTermCfg(func=mdp.jump_success_bonus, weight=320.0)


@configclass
class WheelLeggedObstacleGeometryCurriculumCfg:
    """Fine-grained height/width curriculum for reliable 8 cm barriers."""

    geometry_levels = CurriculumTermCfg(
        func=mdp.obstacle_geometry_curriculum,
        params={
            # Height and width are not increased aggressively at the same
            # transition. This provides intermediate gradients before the final
            # 8 cm high × 8 cm wide barrier.
            "height_levels": (0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08),
            "width_levels": (0.035, 0.035, 0.050, 0.050, 0.065, 0.065, 0.080),
            # Wider obstacles require greater horizontal travel during the
            # same ~0.18 s flight. Couple the command range to geometry so no
            # level samples physically incompatible speed/width pairs.
            "speed_min_levels": (0.45, 0.45, 0.50, 0.50, 0.60, 0.60, 0.70),
            "speed_max_levels": (0.60, 0.65, 0.65, 0.70, 0.75, 0.75, 0.75),
            "initial_level": 0,
            # These are progression gates rather than final acceptance rules.
            # Once a policy has a repeatable basic skill, expose the next
            # geometry instead of waiting for near-mastery of the current one.
            "success_threshold": 0.40,
            "clear_threshold": 0.68,
            # Wheel-edge contact remains useful learning signal at the next
            # level. Base/leg collision and strict O成功 definitions are not
            # relaxed by this curriculum-only threshold.
            "collision_threshold": 0.38,
            "min_trials": 768,
            "consecutive_passes": 2,
        },
    )


@configclass
class WheelLeggedObstacleOracleFlatEnvCfg(WheelLeggedTargetLandingFlatEnvCfg):
    """Stage D2: analytic triggering over one exact-geometry low barrier."""

    scene: WheelLeggedObstacleSceneCfg = WheelLeggedObstacleSceneCfg(
        num_envs=4096, env_spacing=4.0
    )
    observations: WheelLeggedObstacleObservationsCfg = (
        WheelLeggedObstacleObservationsCfg()
    )
    commands: WheelLeggedObstacleOracleCommandsCfg = (
        WheelLeggedObstacleOracleCommandsCfg()
    )
    rewards: WheelLeggedObstacleOracleRewardsCfg = (
        WheelLeggedObstacleOracleRewardsCfg()
    )
    obstacle_oracle: ObstacleOracleCfg = ObstacleOracleCfg()
    curriculum: WheelLeggedObstacleGeometryCurriculumCfg = (
        WheelLeggedObstacleGeometryCurriculumCfg()
    )
    env_class = WheelLeggedObstacleOracleEnv

    def __post_init__(self):
        super().__post_init__()
        self.observations = WheelLeggedObstacleObservationsCfg()
        self.commands = WheelLeggedObstacleOracleCommandsCfg()
        self.rewards = WheelLeggedObstacleOracleRewardsCfg()
        self.curriculum = WheelLeggedObstacleGeometryCurriculumCfg()
        self.obstacle_oracle = ObstacleOracleCfg()
        for name in (
            "obstacle_contact_base",
            "obstacle_contact_lf0",
            "obstacle_contact_lf1",
            "obstacle_contact_rf0",
            "obstacle_contact_rf1",
            "obstacle_contact_lwheel",
            "obstacle_contact_rwheel",
        ):
            getattr(self.scene, name).update_period = self.sim.dt
        # Keep the first oracle lane approximately aligned with commanded
        # world heading. Wider yaw randomization and pushes return with the
        # perception/curriculum stage.
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.15, 0.15)
        self.events.push_robot = None

        state = self.jump_state_machine
        state.start_max_speed = 0.90
        # Obstacle success is defined primarily by actual wheel clearance,
        # clean crossing, air time and landing. Retraction can produce useful
        # wheel clearance without raising the base by 50% of the command.
        state.success_height_ratio = 0.40
        state.success_max_recovery_vx_error = 0.20
        # Six centimetres remains smaller than the final 8 cm obstacle width.
        # It removes borderline target misses from O成功 while clean crossing,
        # minimum wheel clearance, air time and landing speed remain mandatory.
        state.success_max_landing_position_error = 0.06


@configclass
class WheelLeggedObstaclePerceptiveFlatEnvCfg(
    WheelLeggedObstacleOracleFlatEnvCfg
):
    """Stage D3: depth-perceived obstacle geometry and jump triggering."""

    scene: WheelLeggedPerceptiveObstacleSceneCfg = (
        WheelLeggedPerceptiveObstacleSceneCfg(num_envs=512, env_spacing=4.0)
    )
    observations: WheelLeggedPerceptiveObstacleObservationsCfg = (
        WheelLeggedPerceptiveObstacleObservationsCfg()
    )
    obstacle_perception: ObstaclePerceptionCfg = ObstaclePerceptionCfg()
    env_class = WheelLeggedObstaclePerceptiveEnv

    def __post_init__(self):
        super().__post_init__()
        self.observations = WheelLeggedPerceptiveObstacleObservationsCfg()
        self.obstacle_perception = ObstaclePerceptionCfg()
