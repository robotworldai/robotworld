# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda joint position control for pick-place from conveyor."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils.prims import clone
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

from ... import mdp
from ...conveyor_belt_pick_and_place_env_cfg import ConveyorBeltPickAndPlaceEnvCfg


@clone
def spawn_mac_n_cheese(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn distractor USD with rigid body and collision."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    collision_cfg = (
        cfg.collision_props if cfg.collision_props is not None else CollisionPropertiesCfg()
    )
    sim_utils.define_collision_properties(prim_path, collision_cfg)
    sim_utils.activate_contact_sensors(prim_path)
    return prim


@configclass
class FrankaConveyorBeltPickAndPlaceEnvCfg(ConveyorBeltPickAndPlaceEnvCfg):
    """Franka pick-place from conveyor with joint position control."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # Relative joint position control: target = current_pos + scale * action
        # scale=0.1: max increment 0.1 rad/step, at 25Hz = 2.5 rad/s
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.1,
            use_zero_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.2], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                activate_contact_sensors=True,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.scene.distractor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.2], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Food/mac_n_cheese.usd",
                scale=(0.6, 0.7, 1.0),
                func=spawn_mac_n_cheese,
                activate_contact_sensors=False,
                collision_props=CollisionPropertiesCfg(),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        self.scene.robot_arm_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link.*",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object.*"],
            update_period=0.0,
            history_length=2,
            debug_vis=False,
        )


@configclass
class FrankaConveyorBeltPickAndPlaceEnvCfg_PLAY(FrankaConveyorBeltPickAndPlaceEnvCfg):
    """Play/eval variant: fewer envs, no obs corruption."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
