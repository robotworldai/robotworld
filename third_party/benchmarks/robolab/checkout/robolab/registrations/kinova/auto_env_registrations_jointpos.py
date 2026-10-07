# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint-position RoboLab registration for the fixed-base Kinova Gen3."""

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR


def auto_register_kinova_envs(
    task_dirs=DEFAULT_TASK_SUBFOLDERS,
    task=None,
):
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.robots.kinova_gen3 import (
        KinovaGen3Cfg,
        KinovaJointPositionActionCfg,
        KinovaProprioceptionObservationCfg,
        KinovaWristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import (
        EgocentricMirroredWideAngleHighCameraCfg,
        OverShoulderLeftCameraCfg,
    )
    from robolab.variations.lighting import SphereLightCfg

    ViewportCameraCfg = generate_image_obs_from_cameras(
        [EgocentricMirroredWideAngleHighCameraCfg]
    )
    ImageObsCfg = generate_image_obs_from_cameras(
        [OverShoulderLeftCameraCfg, KinovaWristCameraCfg]
    )
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": KinovaProprioceptionObservationCfg(),
            "viewport_cam": ViewportCameraCfg(),
        }
    )

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_postfix="KinovaJointPosition",
        observations_cfg=ObservationCfg(),
        actions_cfg=KinovaJointPositionActionCfg(),
        robot_cfg=KinovaGen3Cfg,
        # The wrist camera is already attached through KinovaGen3Cfg. Adding its
        # wrapper here would try to spawn it before bracelet_link exists.
        camera_cfg=[
            OverShoulderLeftCameraCfg,
            EgocentricMirroredWideAngleHighCameraCfg,
        ],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
