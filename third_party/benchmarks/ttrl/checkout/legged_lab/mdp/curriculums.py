# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:    
    from legged_lab.envs.base.tt_env import TTEnv


def modify_reward_weight(env: TTEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    global_sim_step_counter = env.sim_step_counter // env.cfg.sim.decimation

    if global_sim_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        
from typing import Sequence

def modify_reward_weight_linear(env: TTEnv, env_ids: Sequence[int], term_name: str, target_weight: float, start_step: int, end_step: int):
    """
    Continuously modifies a reward term's weight using linear interpolation 
    between steps [a, b]. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        target_weight: The final weight value to reach at step b.
        a: The step at which interpolation should begin.
        b: The step at which interpolation should end.
    """
    global_sim_step_counter = env.sim_step_counter // env.cfg.sim.decimation
    if global_sim_step_counter <= start_step:
        return
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    current_weight = term_cfg.weight
    if global_sim_step_counter >= end_step:
        term_cfg.weight = target_weight
    else:
        dw = (target_weight - current_weight) / (end_step - global_sim_step_counter)
        term_cfg.weight = current_weight + dw
    env.reward_manager.set_term_cfg(term_name, term_cfg)
