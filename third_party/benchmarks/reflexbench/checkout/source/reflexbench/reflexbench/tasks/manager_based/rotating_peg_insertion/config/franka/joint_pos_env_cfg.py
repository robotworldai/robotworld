# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

from ... import mdp
from ...rotating_peg_insertion_env_cfg import RotatingPegInsertionEnvCfg


# Keep the arm clear of the rotating disc at reset.  The default Franka home
# pose reaches forward enough that the forearm/hand can touch the disc before
# the planning policy has a chance to move.
_DISC_INSERT_INIT_JOINT_POS: dict[str, float] = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.95,
    "panda_joint3": 0.0,
    "panda_joint4": -2.35,
    "panda_joint5": 0.0,
    "panda_joint6": 1.45,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": 0.0,
}


@configclass
class FrankaRotatingPegInsertionEnvCfg(RotatingPegInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=_DISC_INSERT_INIT_JOINT_POS),
        )
        self.scene.robot.spawn.activate_contact_sensors = True

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

        self.scene.disc = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Disc",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.40], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CylinderCfg(
                radius=0.14,
                height=0.01,
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.30)),
            ),
        )

        self.scene.peg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Peg",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.48], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CylinderCfg(
                radius=0.005,
                height=0.10,
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.1)),
            ),
        )

        self.scene.hole_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/HoleMarker",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.57, 0.0, 0.406], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CylinderCfg(
                radius=0.022,
                height=0.002,
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
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


@configclass
class FrankaRotatingPegInsertionEnvCfg_PLAY(FrankaRotatingPegInsertionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
