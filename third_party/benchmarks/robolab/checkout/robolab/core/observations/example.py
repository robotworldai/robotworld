# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


def _image_observation_func():
    """Resolve the IsaacLab image observation function across versions.

    ``mdp.image`` on IsaacLab 2.3 (IsaacSim 5.1); ``mdp.observations.image``
    on IsaacLab 2.2 (IsaacSim 5.0).
    """
    image_func = getattr(mdp, "image", None)
    if image_func is not None:
        return image_func
    observations = getattr(mdp, "observations", None)
    image_func = getattr(observations, "image", None) if observations is not None else None
    if image_func is None:
        raise AttributeError("IsaacLab image observation function not found")
    return image_func


@configclass
class CameraObservationCfg:
    """Static camera observation configuration - example implementation."""
    @configclass
    class ImageObsCfg(ObsGroup):
        """Observations for policy."""
        egocentric_wide_angle_camera = ObsTerm(
            func=_image_observation_func(),
            params={
                "sensor_cfg": SceneEntityCfg("egocentric_wide_angle_camera"),
                "data_type": "rgb",
                "normalize": False,
                }
            )

        egocentric_mirrored_wide_angle_camera = ObsTerm(
            func=_image_observation_func(),
            params={
                "sensor_cfg": SceneEntityCfg("egocentric_mirrored_camera"),
                "data_type": "rgb",
                "normalize": False,
                }
            )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    image_obs: ImageObsCfg = ImageObsCfg()
