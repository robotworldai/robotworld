# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene_and_contact_object_list
from robolab.core.task.task import Task


@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class PlateBananaRubiksCubeTask(Task):
    scene, contact_object_list = import_scene_and_contact_object_list("test_plate_banana_rubiks_cube.usda")
    terminations = Terminations
    instruction: str = "Test object_on_top geometric gate on plate / bananas"
    episode_length_s: int = 1
