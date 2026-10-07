# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from legged_lab.utils.env_utils.scene import SceneCfg

import legged_lab.mdp as mdp


@configclass
class RewardCfg:
    pass

@configclass
class CurriculumCfg:
    pass

@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 10.0 
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    table: RigidObjectCfg = MISSING
    ball: RigidObjectCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []
    num_actions: int = 21
    num_joints: int = 21
    effort_limit_scale: float = 1.0

@configclass
class BallCfg:
    ball_speed_x_range: tuple = (-5.5, -4.5)
    ball_speed_y_range: tuple = (-0.8,0.8)
    ball_speed_z_range: tuple = (1.6, 1.7)
    ball_pos_y_range: tuple = (-0.2, 0.2)
    contact_threshold: float = 0.06
    ball_max_eposide_length: float = 1.5
    ball_reset_repeat: int = 5
    num_new_serves = 2
    max_serve_per_episode: int = 5


@configclass
class TableCfg:
    table_opponent_contact_x: tuple = (0.0, 1.37)
    table_opponent_contact_y: tuple = (-0.7625, 0.7625)
    table_opponent_contact_z: tuple = (0.70, 0.85) # table height - ball radius + margin
    table_own_contact_x: tuple = (-1.37, 0.0)
    table_own_contact_y: tuple = (-0.7625, 0.7625)
    table_own_contact_z: tuple = (0.70, 0.85)

@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0
    robot_pos: float = 1.0
    ball_pos: float = 1.0
    ball_linvel: float = 1.0
    perception: float = 1.0 #assuming robot_pos, ball_pos, ball_linvel can be treated uniformly
    ball_state: float = 1.0

@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0) # TODO: set zero for student policy
    lin_vel_y: tuple = (-0.5, 0.5) # TODO: set zero for student policy
    ang_vel_z: tuple = (-1.0, 1.0) # TODO: set zero for student policy
    heading: tuple = (-math.pi, math.pi) # TODO: set smaller range for student policy if needed, else set zero


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0) # TODO: Check and change to match episode length if required
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1
    ball_pos: float = 0.0 #TODO: tune this based on perception noise
    ball_linvel: float = 0.0 #TODO: tune this based on perception noise
    robot_pos: float = 0.0  # TODO: tune this based on perception noise
    perception: float = 0.0 #TODO: assuming uniform noise across perception, tune this based on perception noise
    ball_state: float = 0.0


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_locomotion_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Waist",".*_Hip_.*", ".*_Knee_.*",".*_Ankle_.*","Left_Elbow_.*","Left_Shoulder_.*","AAHead_yaw","Head_pitch"])
        },
    )
    reset_manipulation_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Right_Elbow_.*","Right_Shoulder_.*"])
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}

@configclass
class PerceptionDelayCfg:
    enable: bool = True
    params: dict = {"max_delay": 4, "min_delay": 3}

@configclass
class DomainRandCfg:
    events: EventCfg = EventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()
    perception_delay: PerceptionDelayCfg = PerceptionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()

@configclass
class ActionsCfg:
    joint_names: list = []
    """List of joint names or regex expressions that the action will be mapped to."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

@configclass
class ObservationsCfg:
    joint_names: list = []
    """List of joint names or regex expressions that the action will be mapped to."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

# @configclass
# class TTSceneCfg(SceneCfg):
#     def __init__(self, config: BaseSceneCfg, physics_dt, step_dt):
#         super().__init__(config, physics_dt, step_dt)
#         self.table: RigidObjectCfg = config.table
#         self.table.prim_path = "{ENV_REGEX_NS}/Table"
