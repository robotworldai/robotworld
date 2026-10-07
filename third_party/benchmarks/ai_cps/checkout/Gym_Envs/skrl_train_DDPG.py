import torch
import os
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env
from skrl.utils import set_seed


# set the seed for reproducibility
set_seed(42)



# Define the models (deterministic models) for the DDPG agent using mixins
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

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        # x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        # x = F.relu(self.linear_layer_2(x))
        # return torch.tanh(self.action_layer(x)), {}
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}




# Load and wrap the Omniverse Isaac Gym environment]
omniisaacgymenvs_path = os.path.realpath( os.path.join(os.path.realpath(__file__), "..") ) 
env = load_omniverse_isaacgym_env(task_name="FrankaCatching", omniisaacgymenvs_path = omniisaacgymenvs_path)
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_ddpg["critic"] = DeterministicCritic(env.observation_space, env.action_space, device)
models_ddpg["target_critic"] = DeterministicCritic(env.observation_space, env.action_space, device)


# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ddpg.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.5)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
# cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg_ddpg["gradient_steps"] = 1          # gradient steps
cfg_ddpg["batch_size"] = 32           # training batch size
cfg_ddpg["polyak"] = 0.005              # soft update hyperparameter (tau)
cfg_ddpg["discount_factor"] = 0.99      # discount factor (gamma)
cfg_ddpg["random_timesteps"] = 0      # random exploration steps
cfg_ddpg["learning_starts"] = 0       # learning starts after this many steps
cfg_ddpg["actor_learning_rate"] = 1e-3
cfg_ddpg["critic_learning_rate"] = 5e-3
# cfg_ddpg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward


# logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 100
cfg_ddpg["experiment"]["checkpoint_interval"] = 1000
# cfg_ddpg["experiment"]["experiment_name"] = ""

agent = DDPG(models=models_ddpg,
                  memory=memory,
                  cfg=cfg_ddpg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 320000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
