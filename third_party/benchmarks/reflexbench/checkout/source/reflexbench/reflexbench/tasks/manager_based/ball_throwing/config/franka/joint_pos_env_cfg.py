# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedClassDecorator=false

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

from ... import mdp
from ...ball_throwing_env_cfg import BallThrowingEnvCfg


# Finger joint initial half-aperture in meters.  The ball is 0.025 m in
# radius; we keep the fingers slightly tighter so a contact constraint is
# active from t=0.  Gravity is also disabled on the ball during phase 0
# (see :func:`mdp.apply_post_release_gravity`) so a small grip force is
# enough -- no risk of the ball free-falling before the gripper command
# ramps up.
_FINGER_INIT_HALF_APERTURE: float = 0.024


def _build_robot_cfg():
    """Franka cfg with fingers initialised closed around the ball.

    Strategy: keep all arm-joint default values from
    :data:`FRANKA_PANDA_HIGH_PD_CFG` untouched, but override the two finger
    joints so the gripper is already clamped on the ball at reset time.
    Existing ``panda_finger_.*`` regex entries are stripped first so they do
    not collide with the explicit per-joint overrides.
    """
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.activate_contact_sensors = True

    joint_pos = {
        k: v
        for k, v in dict(robot_cfg.init_state.joint_pos).items()
        if "finger" not in k
    }
    joint_pos["panda_finger_joint1"] = _FINGER_INIT_HALF_APERTURE
    joint_pos["panda_finger_joint2"] = _FINGER_INIT_HALF_APERTURE
    robot_cfg.init_state = robot_cfg.init_state.replace(joint_pos=joint_pos)
    return robot_cfg


@configclass
class FrankaBallThrowingEnvCfg(BallThrowingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = _build_robot_cfg()

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

        # Ball: blue sphere, high-friction physics material so the finger
        # contact reliably supports the ball before the gripper closes hard.
        # ``init_state.pos`` is a bootstrap only -- ``reset_ball_in_gripper``
        # re-writes the ball pose to the gripper TCP minus a 2.2 cm z-offset
        # (between the two fingers) at every reset.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.4, 0.0, 0.5], rot=[1, 0, 0, 0]
            ),
            spawn=sim_utils.SphereCfg(
                radius=0.025,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=6.0,
                    max_depenetration_velocity=2.0,
                    # Gravity is disabled at spawn so the ball cannot drop out
                    # of the gripper before the policy starts.  An interval
                    # event re-applies gravity as an external force once the
                    # task transitions to phase 1 (released).
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
                collision_props=CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5,
                    dynamic_friction=1.5,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.25, 0.85)
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # EE target sits between the two fingers (panda_hand +Z by 0.1034 m).
        # ``reset_ball_in_gripper`` places the ball at this point with a
        # configurable additional offset (default zero).
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
class FrankaBallThrowingEnvCfg_PLAY(FrankaBallThrowingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
