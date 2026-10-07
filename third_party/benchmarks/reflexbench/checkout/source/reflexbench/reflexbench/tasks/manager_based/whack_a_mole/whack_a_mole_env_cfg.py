# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp

TABLE_SIZE = (0.36, 0.32, 0.20)
TABLE_CENTER_POS = (0.55, 0.0, TABLE_SIZE[2] * 0.5)
BOARD_SIZE = (0.30, 0.28, 0.02)
BOARD_CENTER_POS = (0.55, 0.0, TABLE_SIZE[2] + BOARD_SIZE[2] * 0.5)


@configclass
class WhackAMoleSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    mole_0: RigidObjectCfg = MISSING
    mole_1: RigidObjectCfg = MISSING
    mole_2: RigidObjectCfg = MISSING
    mole_3: RigidObjectCfg = MISSING
    mole_4: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.34, 0.20)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER_POS),
    )

    board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MoleBoard",
        spawn=sim_utils.CuboidCfg(
            size=BOARD_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.15)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=BOARD_CENTER_POS),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        gripper_width = ObsTerm(func=mdp.gripper_width_obs)
        task_phase = ObsTerm(func=mdp.task_phase_obs)
        mole_positions = ObsTerm(func=mdp.mole_positions_in_robot_root_frame)
        mole_heights = ObsTerm(func=mdp.mole_heights_obs)
        active_mole_index = ObsTerm(func=mdp.active_mole_index_obs)
        active_mole_position = ObsTerm(func=mdp.active_mole_position_in_robot_root_frame)
        window_remaining_time = ObsTerm(func=mdp.window_remaining_time_obs)
        valid_hits = ObsTerm(func=mdp.valid_hits_obs)
        current_window_index = ObsTerm(func=mdp.current_window_index_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_moles = EventTerm(
        func=mdp.reset_moles,
        mode="reset",
    )

    reset_task_phase = EventTerm(
        func=mdp.reset_task_phase,
        mode="reset",
    )

    check_popup_schedule = EventTerm(
        func=mdp.check_popup_schedule,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )

    update_mole_positions = EventTerm(
        func=mdp.update_mole_positions,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_completed = DoneTerm(func=mdp.task_completed)


@configclass
class CurriculumCfg:
    pass


@configclass
class WhackAMoleEnvCfg(ManagerBasedRLEnvCfg):
    sim_freq: float = 100.0
    robot_control_freq: float = 25.0

    scene: WhackAMoleSceneCfg = WhackAMoleSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    episode_length_s = 3.0

    def __post_init__(self) -> None:
        self.sim.dt = 1.0 / self.sim_freq
        self.decimation = max(1, round(self.sim_freq / self.robot_control_freq))
        self.episode_length_s = 3.0

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**19
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**19
        self.sim.physx.gpu_max_rigid_contact_count = 2**19
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        self.sim.physx.gpu_heap_capacity = 2**25
        self.sim.physx.gpu_temp_buffer_capacity = 2**24
        self.sim.physx.gpu_max_num_partitions = 8
        self.sim.physx.gpu_collision_stack_size = 2**25