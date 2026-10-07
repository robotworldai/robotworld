"""
This is a copy from SKRL's implementation of loading oige environment, 
with modifications for generating testing oige environment
"""
import sys
import os
from contextlib import contextmanager

def _omegaconf_to_dict(config) -> dict:
    """Convert OmegaConf config to dict

    :param config: The OmegaConf config
    :type config: OmegaConf.Config

    :return: The config as dict
    :rtype: dict
    """
    # return config.to_container(dict)
    from omegaconf import DictConfig

    d = {}
    for k, v in config.items():
        d[k] = _omegaconf_to_dict(v) if isinstance(v, DictConfig) else v
    return d

def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration

    :param d: The dictionary to print
    :type d: dict
    :param indent: The indentation level (default: 0)
    :type indent: int, optional
    """
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print('  |   ' * indent + "  |-- {}: {}".format(key, value))

def load_oige_test_env(task_name: str = "",
                       omniisaacgymenvs_path: str = "",
                       num_envs: int = 1,
                       show_cfg: bool = True,
                       timeout: int = 30):
    """Load an Omniverse Isaac Gym environment, this is a slight modification of SKRL's implementation 

    :param task_name: The name of the task (default: "").
                      If not specified, the task name is taken from the command line argument (``task=TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param omniisaacgymenvs_path: The path to the ``omniisaacgymenvs`` directory (default: "").
                              If empty, the path will obtained from omniisaacgymenvs package metadata
    :type omniisaacgymenvs_path: str, optional
    :param show_cfg: Whether to print the configuration (default: True)
    :type show_cfg: bool, optional
    :param timeout: Seconds to wait for data when queue is empty in multi-threaded environment (default: 30)
    :type timeout: int, optional

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments
    :raises RuntimeError: The omniisaacgymenvs package is not installed or the path is wrong

    :return: Omniverse Isaac Gym environment
    :rtype: omni.isaac.gym.vec_env.vec_env_base.VecEnvBase or omni.isaac.gym.vec_env.vec_env_mt.VecEnvMT
    """
    import torch
    from hydra.types import RunMode
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_automatic_config_search_path, get_args_parser

    from omegaconf import OmegaConf

    from omni.isaac.gym.vec_env import VecEnvBase, TaskStopException

    import omniisaacgymenvs

    sys.argv.append("task={}".format(task_name))
    sys.argv.append("num_envs={}".format(num_envs))

    # get omniisaacgymenvs path from omniisaacgymenvs package metadata
    if omniisaacgymenvs_path == "":
        if not hasattr(omniisaacgymenvs, "__path__"):
            raise RuntimeError("omniisaacgymenvs package is not installed")
        omniisaacgymenvs_path = list(omniisaacgymenvs.__path__)[0]
    config_path = os.path.join(omniisaacgymenvs_path, "cfg")

    # set omegaconf resolvers
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda condition, a, b: a if condition else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

    # get hydra config without use @hydra.main
    config_file = "config"
    args = get_args_parser().parse_args()
    search_path = create_automatic_config_search_path(config_file, None, config_path)
    hydra_object = Hydra.create_main_hydra2(task_name='load_omniisaacgymenv', config_search_path=search_path)
    config = hydra_object.compose_config(config_file, args.overrides, run_mode=RunMode.RUN)

    cfg = {}
    cfg["task"] = _omegaconf_to_dict(config.task)
    cfg["task_name"] = config.task_name
    cfg["experiment"] = config.experiment
    cfg["num_envs"] = config.num_envs
    cfg["seed"] = config.seed
    cfg["torch_deterministic"] = config.torch_deterministic
    cfg["max_iterations"] = config.max_iterations
    cfg["physics_engine"] = config.physics_engine
    cfg["pipeline"] = config.pipeline
    cfg["sim_device"] = config.sim_device
    cfg["device_id"] = config.device_id
    cfg["rl_device"] = config.rl_device
    cfg["num_threads"] = config.num_threads
    cfg["solver_type"] = config.solver_type
    cfg["test"] = config.test
    cfg["checkpoint"] = config.checkpoint
    cfg["headless"] = config.headless

    # print config
    if show_cfg:
        print("\nOmniverse Isaac Gym environment ({})".format(config.task.name))
        _print_cfg(cfg)

    # internal classes
    class _OmniIsaacGymVecEnv(VecEnvBase):
        def step(self, actions):
            actions = torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions).to(self._task.device).clone()
            self._task.pre_physics_step(actions)

            for _ in range(self._task.control_frequency_inv):
                self._world.step(render=self._render)
                self.sim_frame_count += 1

            observations, rewards, dones, info = self._task.post_physics_step()

            return {"obs": torch.clamp(observations, -self._task.clip_obs, self._task.clip_obs).to(self._task.rl_device).clone()}, \
                rewards.to(self._task.rl_device).clone(), dones.to(self._task.rl_device).clone(), info.copy()

        def set_as_test(self):
            self._task.set_as_test()

        def set_initial_test_value(self, value):
            self._task.set_initial_test_value(value)

        def reset(self):
            self._task.reset()
            actions = torch.zeros((self.num_envs, self._task.num_actions), device=self._task.device)
            return self.step(actions)[0]

    # load environment
    sys.path.append(omniisaacgymenvs_path)
    from utils.task_util import initialize_task

    env = _OmniIsaacGymVecEnv(headless=config.headless)
    task = initialize_task(cfg, env, init_sim=True)

    return env