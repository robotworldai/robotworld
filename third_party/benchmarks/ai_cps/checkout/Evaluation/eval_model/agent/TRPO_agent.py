"""
Create PPO agent based on SKRL implementation
"""

import torch.nn as nn
import torch

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

# Define the models (stochastic and deterministic models) for the agent using mixins.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy_2_Layers(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Policy_3_Layers(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value_2_Layers(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Value_3_Layers(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# Create SKRL PPO agent
def create_skrl_trpo_agent(env, agent_path):

    device = env.device

    models_trpo_2_layer = {}
    models_trpo_2_layer["policy"] = Policy_2_Layers(env.observation_space, env.action_space, device)
    models_trpo_2_layer["value"] = Value_2_Layers(env.observation_space, env.action_space, device)

    models_trpo_3_layer = {}
    models_trpo_3_layer["policy"] = Policy_3_Layers(env.observation_space, env.action_space, device)
    models_trpo_3_layer["value"] = Value_3_Layers(env.observation_space, env.action_space, device)

    # Configs
    cfg_trpo = TRPO_DEFAULT_CONFIG.copy()
    cfg_trpo["state_preprocessor"] = RunningStandardScaler
    cfg_trpo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_trpo["value_preprocessor"] = RunningStandardScaler
    cfg_trpo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # no log to TensorBoard and write checkpoints 
    cfg_trpo["experiment"]["write_interval"] = 0
    cfg_trpo["experiment"]["checkpoint_interval"] = 0

    try: 
        # Initialize and load agent with 2 layers
        agent = TRPO(models=models_trpo_2_layer,
                memory=None,
                cfg=cfg_trpo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

        agent.load(agent_path)

    except:

        # Initialize and load agent with 3 layers
        agent = TRPO(models=models_trpo_3_layer,
                memory=None,
                cfg=cfg_trpo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

        agent.load(agent_path)

    return agent
        
    

    
