# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedClassDecorator=false

from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg
from .joint_pos_env_cfg import _build_robot_cfg


@configclass
class FrankaBallThrowingEnvCfg_IK_Abs(joint_pos_env_cfg.FrankaBallThrowingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # Re-apply finger-closed init pose (super already set the robot, but
        # we keep the explicit call here for clarity if subclasses override).
        self.scene.robot = _build_robot_cfg()
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]
            ),
        )
