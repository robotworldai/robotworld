# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedClassDecorator=false

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from reflexbench.fabric_fixes import mark_wrist_cam_usd_dirty

from .joint_pos_env_cfg import FrankaBallThrowingEnvCfg


FIXED_CAM_POS = (0.52, 1.35, 1.35)
FIXED_CAM_TARGET = (0.46, 0.0, 0.38)


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm <= 1e-8:
        return (1.0, 0.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quat_from_rot_matrix(
    m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float, float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return (
        (m[1][0] - m[0][1]) / s,
        (m[0][2] + m[2][0]) / s,
        (m[1][2] + m[2][1]) / s,
        0.25 * s,
    )


def _look_at_camera_quat_world(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    # CameraCfg convention="world": +X is forward and +Z is up.
    forward = _normalize((
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ))
    world_up = (0.0, 0.0, 1.0)
    right = _normalize(_cross(world_up, forward))
    up = _normalize(_cross(forward, right))
    rot = (
        (forward[0], right[0], up[0]),
        (forward[1], right[1], up[1]),
        (forward[2], right[2], up[2]),
    )
    return _quat_from_rot_matrix(rot)


FIXED_CAM_QUAT = _look_at_camera_quat_world(FIXED_CAM_POS, FIXED_CAM_TARGET)


@configclass
class FrankaBallThrowingDataCollectionEnvCfg(FrankaBallThrowingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        if hasattr(self.events, "print_pos"):
            self.events.print_pos = None

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
                pos=FIXED_CAM_POS,
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


@configclass
class FrankaBallThrowingDataCollectionEnvCfg_PLAY(FrankaBallThrowingDataCollectionEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
