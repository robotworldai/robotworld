# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import asyncio

from omni.isaac.core.tasks import BaseTask


class RLTaskInterface(BaseTask):

    """This class provides a PyTorch RL-specific interface for setting up RL tasks.
    For a concrete implementation of the task, please refer to https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/omniisaacgymenvs/tasks/base/rl_task.py.
    """

    def __init__(self, name, env, offset=None) -> None:

        """Initializes RL parameters, cloner object, and buffers.

        Args:
            name (str): name of the task.
            env (VecEnvBase): an instance of the environment wrapper class to register task.
            offset (Optional[np.ndarray], optional): offset applied to all assets of the task. Defaults to None.
        """

        super().__init__(name=name, offset=offset)

        self._env = env
        self.control_frequency_inv = 1

        # initialize variables
        self._num_envs = None
        self._num_actions = None
        self._num_observations = None
        self._num_states = None
        self._num_agents = None

        self.action_space = None
        self.observation_space = None
        self.state_space = None

        self.obs_buf = None
        self.states_buf = None
        self.rew_buf = None
        self.reset_buf = None
        self.progress_buf = None
        self.extras = None

    def update_config(self, sim_config):
        """Optionally implemented by individual task classes to update values parsed from config file.

        Args:
            sim_config (SimConfig): Config object containing task and simulation parameters.
        """
        pass

    def initialize_views(self, scene):
        """Optionally implemented by individual task classes to initialize views used in the task.
            This API is required for the extension workflow, where tasks are expected to train on a pre-defined stage.

        Args:
            scene (Scene): Scene to remove existing views and initialize/add new views.
        """
        pass

    @property
    def num_envs(self):
        """Retrieves number of environments for task.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    @property
    def num_actions(self):
        """Retrieves dimension of actions.

        Returns:
            num_actions(int): Dimension of actions.
        """
        return self._num_actions

    @property
    def num_observations(self):
        """Retrieves dimension of observations.

        Returns:
            num_observations(int): Dimension of observations.
        """
        return self._num_observations

    @property
    def num_states(self):
        """Retrieves dimesion of states.

        Returns:
            num_states(int): Dimension of states.
        """
        return self._num_states

    @property
    def num_agents(self):
        """Retrieves number of agents for multi-agent environments.

        Returns:
            num_agents(int): Dimension of states.
        """
        return self._num_agents

    def get_states(self):
        """API for retrieving states buffer, used for asymmetric AC training.

        Returns:
            states_buf(torch.Tensor): States buffer.
        """
        return self.states_buf

    def get_extras(self):
        """API for retrieving extras data for RL.

        Returns:
            extras(dict): Dictionary containing extras data.
        """
        return self.extras

    def reset(self):
        """Resets all environments. This method is only called once at the beginning of training."""
        pass

    def pre_physics_step(self, actions):
        """Optionally implemented by individual task classes to process actions.

        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        pass

    def post_physics_step(self):
        """Processes RL required computations for observations, states, rewards, resets, and extras.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    async def pre_physics_step_async(self, actions):
        """Optionally implemented by individual task classes to process actions.
           Async method used in extension workflow.

        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        self.pre_physics_step(actions)

    async def post_physics_step_async(self):
        """Processes RL required computations for observations, states, rewards, resets, and extras.
            Async method used in extension workflow.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        return self.post_physics_step()
