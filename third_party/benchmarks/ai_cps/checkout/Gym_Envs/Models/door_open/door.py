# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
import torch
import os


class Door(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "door",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """
        self._name = name

        self._usd_path = os.path.realpath( os.path.join(os.path.realpath(__file__), "..") ) + "/free_door_point_pull.usd"
        
        add_reference_to_stage(self._usd_path, prim_path)

        self._position = torch.tensor([-0.3, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1, 0.0, 0.0, 0.0]) if orientation is None else orientation

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )
