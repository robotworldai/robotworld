# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass


def with_depth(camera_cfg_cls):
    """Return a variant of ``camera_cfg_cls`` whose cameras also render depth.

    Builds a subclass with each contained camera's ``data_types`` extended by
    ``"depth"``, leaving the original class untouched so registrations without
    depth keep their rgb-only render cost. Camera attribute names (and thus
    prim paths and observation term names) are unchanged.
    """
    instance = camera_cfg_cls()
    overrides = {}
    for attr_name in dir(instance):
        if attr_name.startswith("_"):
            continue
        attr_value = getattr(instance, attr_name)
        if isinstance(attr_value, CameraCfg) and "depth" not in attr_value.data_types:
            camera = copy.deepcopy(attr_value)
            camera.data_types = [*camera.data_types, "depth"]
            overrides[attr_name] = camera
    if not overrides:
        return camera_cfg_cls
    variant = type(f"{camera_cfg_cls.__name__}WithDepth", (camera_cfg_cls,), overrides)
    return configclass(variant)


@configclass
class OverShoulderLeftCameraCfg:
    """Left over-shoulder camera, matching DROID exterior_image_1_left placement.

    Mounted to the left of the robot workspace at (0.05, 0.57, 0.66).
    Look direction: (0.63, -0.48, -0.62) — looking right-and-down toward workspace.
    """
    over_shoulder_left_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )


@configclass
class OverShoulderRightCameraCfg:
    """Right over-shoulder camera, matching DROID exterior_image_2_left placement.

    Mirror of OverShoulderLeftCameraCfg across the XZ plane (Y → -Y).
    Mounted to the right of the robot workspace at (0.05, -0.57, 0.66).
    Look direction: (0.628, +0.490, -0.606) — looking left-and-down toward workspace.
    Up direction in world: (0.477, +0.372, +0.796) — Z component positive (upright image).

    Derivation: the correct XZ mirror requires det(R) = +1. The rotation matrix columns are
    the XZ-mirrored left-camera basis vectors with the right-vector sign corrected for
    right-handedness. Converting that matrix to quaternion gives (0.805, 0.399, -0.195, -0.393).
    """
    over_shoulder_right_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/over_shoulder_right_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"
        ),
    )


@configclass
class HeadCameraCfg:
    """Front-facing overhead camera, simulating an operator's head/eye view.

    Positioned 1.5 m in front of and 1.0 m above the robot, looking back toward
    the workspace.  Look direction: (-0.83, 0, -0.55) — straight-on frontal view.
    Rotation computed as pure Y-axis rotation mapping camera -Z to look direction:
    q = (0, sin(28.3°), 0, cos(28.3°)) = (0, 0.474, 0, 0.881).
    """
    head_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/head_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 1.0), rot=(0.0, 0.474, 0.0, 0.881), convention="opengl"
        ),
    )



################################################################################
# Egocentric cameras
################################################################################
@configclass
class EgocentricWideAngleCameraCfg:
    egocentric_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.5), rot=(0.653, 0.271, -0.271, -0.653), convention="opengl"
        ),
    )


################################################################################
# Egocentric mirrored, means the camera is looking at the robot from the front,
# Assuming the robot is at origin.
################################################################################
@configclass
class EgocentricMirroredWideAngleHighCameraCfg:
   egocentric_mirrored_wide_angle_high_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_high_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 1), rot=(0.653, 0.271, 0.271, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredWideAngleCameraCfg:
   egocentric_mirrored_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 0.5), rot=(0.653, 0.271, 0.271, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredCameraCfg:
   egocentric_mirrored_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_mirrored_camera",
    # height=720,
    # width=1280,
    height = 480,
    width = 864,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.29,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(1.5, 0.0, 1.0),
        rot=(0.653, 0.271, 0.271, 0.653),
        convention="opengl"
    ),
)
