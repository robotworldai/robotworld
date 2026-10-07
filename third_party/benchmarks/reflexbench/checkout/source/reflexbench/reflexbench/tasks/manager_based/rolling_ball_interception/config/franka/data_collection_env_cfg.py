# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from reflexbench.fabric_fixes import mark_wrist_cam_usd_dirty

from .joint_pos_env_cfg import FrankaRollingBallInterceptionEnvCfg

_roll = math.radians(235)
_pitch = math.radians(-40)
_yaw = math.radians(180)
_FIXED_CAM_QUAT_X = quat_from_euler_xyz(
    torch.tensor([_roll]),
    torch.tensor([_pitch]),
    torch.tensor([_yaw]),
).squeeze().tolist()
FIXED_CAM_QUAT = (
    _FIXED_CAM_QUAT_X[3],
    _FIXED_CAM_QUAT_X[0],
    _FIXED_CAM_QUAT_X[1],
    _FIXED_CAM_QUAT_X[2],
)


@configclass
class FrankaRollingBallInterceptionDataCollectionEnvCfg(FrankaRollingBallInterceptionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        self.scene.fixed_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/fixed_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.4, 1.0, 1.5),
                rot=FIXED_CAM_QUAT,
                convention="world",
            ),
        )

        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.2, 0.0, -0.10),
                rot=(0.6830, -0.1830, -0.1830, 0.6830),
                convention="ros",
            ),
        )

        # Workaround for upstream Fabric localMatrix corruption on child-mounted
        # wrist_cam. Must run after all other reset events and before the first render.
        self.events.mark_wrist_cam_usd_dirty = EventTerm(
            func=mark_wrist_cam_usd_dirty,
            mode="reset",
        )
