import torch
import os
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env
from skrl.utils import set_seed
from skrl.resources.noises.torch import GaussianNoise


# set the seed for reproducibility
set_seed(42)


# Define the models (deterministic models) for the TD3 agent using mixins
# and programming with two approaches (torch functional and torch.nn.Sequential class).
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        # x = F.relu(self.linear_layer_1(inputs["states"]))
        # x = F.relu(self.linear_layer_2(x))
        return self.net(inputs["states"]), {}

class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}



# Load and wrap the Omniverse Isaac Gym environment]
omniisaacgymenvs_path = os.path.realpath( os.path.join(os.path.realpath(__file__), "..") ) 
env = load_omniverse_isaacgym_env(task_name="FrankaBallCatching", omniisaacgymenvs_path = omniisaacgymenvs_path)
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=2500, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# TRPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.trpo.html#spaces-and-models
# Instantiate the agent's models (function approximators).
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#spaces-and-models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic_1"] = DeterministicCritic(env.observation_space, env.action_space, device)
models["critic_2"] = DeterministicCritic(env.observation_space, env.action_space, device)
models["target_critic_1"] = DeterministicCritic(env.observation_space, env.action_space, device)
models["target_critic_2"] = DeterministicCritic(env.observation_space, env.action_space, device)




# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.trpo.html#configuration-and-hyperparameters
cfg_td3 = TD3_DEFAULT_CONFIG.copy()
cfg_td3["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
cfg_td3["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
cfg_td3["smooth_regularization_clip"] = 0.5
cfg_td3["batch_size"] = 16
cfg_td3["random_timesteps"] = 0
cfg_td3["learning_starts"] = 0
# logging to TensorBoard and write checkpoints each 16 and 80 timesteps respectively
cfg_td3["experiment"]["write_interval"] = 100
cfg_td3["experiment"]["checkpoint_interval"] = 1000

agent = TD3(models=models,
            memory=memory,
            cfg=cfg_td3,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 320000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
