import os
import torch
from typing import Optional

from .load_oige import load_oige_test_env
from .agent.PPO_agent import create_skrl_ppo_agent
from .agent.TRPO_agent import create_skrl_trpo_agent
from skrl.envs.torch import wrap_env


class skrl_oige_model(object):
    """Testing environment model based on SKRL and Omniverse Isaac Gym Environments (OIGE)
    agent_path: the path to the agent parameters (checkpoint)
    oige_path: path to the OIGE environment;
    agent_type: type of DRL agent (PPO, DDPG, TRPO)
    task_name: the name of the task
    num_envs: the number of parallel running environments
    """

    def __init__(
        self,
        agent_path: str,
        oige_path: Optional[str] = None,
        agent_type: Optional[str] = None,
        task_name: Optional[str] = None,
        timesteps: Optional[int] = 10000,
        num_envs: Optional[int] = 1,
        headless: Optional[bool] = False,
        is_action_noise: Optional[bool] = False,
    ):

        # setup
        if oige_path is not None:
            self.oige_path = oige_path
        else:
            self.oige_path = os.path.realpath(
                os.path.join(os.path.realpath(__file__), "../../../Gym_Envs")
            )

        if agent_type is not None:
            self.agent_type = agent_type
        else:
            self.agent_type = "PPO"

        if task_name is not None:
            self.task_name = task_name
        else:
            self.task_name = "FrankaBallPushing"

        self.agent_path = agent_path
        self.timesteps = timesteps
        self.headless = headless

        # Load OIGE env with skrl wrapper
        self.num_envs = num_envs  # for testing, we use only 1 env for now
        env = load_oige_test_env(
            task_name=self.task_name,
            omniisaacgymenvs_path=self.oige_path,
            num_envs=self.num_envs,
        )
        self.env = wrap_env(env)
        self.env._env.set_as_test()
        # if action noise is required
        if is_action_noise is True:
            self.env._env.set_action_noise()

        # Load agent
        if self.agent_type is "PPO":

            self.agent = create_skrl_ppo_agent(self.env, self.agent_path)

        elif self.agent_type is "TRPO":

            self.agent = create_skrl_trpo_agent(self.env, self.agent_path)

        else:
            raise ValueError("Agent type unknown.")

        # Initialize agent
        # cfg_trainer = {"timesteps": self.timesteps, "headless": self.headless}
        self.agent.init()
        if self.num_envs == 1:
            self.agent.set_running_mode("eval")
        else:
            raise ValueError("Currently only one environment (agent) is supported")

    # close env
    def close_env(self):
        self.env.close()

    # Compute the trace w.r.t a given initial condition
    def compute_trace(self, initial_value):

        # set initial configuration
        self.env._env.set_initial_test_value(initial_value)

        # reset env
        states, infos = self.env.reset()

        # initialize trace
        trace = states

        # simulation loop
        for timestep in range(self.timesteps):

            # compute actions
            with torch.no_grad():
                actions = self.agent.act(
                    states, timestep=timestep, timesteps=self.timesteps
                )[0]

            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            # render scene
            if not self.headless:
                self.env.render()

            # record trace
            states.copy_(next_states)
            trace = torch.vstack([trace, states])

            # terminate simulation
            with torch.no_grad():
                if terminated.any() or truncated.any():
                    break

        return trace

    # Merge trace based on the task type
    def merge_trace(self, trace):

        if self.task_name is "FrankaBallPushing":

            # Ball hole distance
            ball_hole_distance = trace[:, 24:27].detach().cpu()
            ball_hole_distance = torch.norm(ball_hole_distance, p=2, dim=-1)
            ball_Z_pos = trace[:, 29].detach().cpu()

            # create index
            trace_length = list(ball_hole_distance.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_trace = torch.vstack((times, ball_hole_distance))
            indexed_trace = torch.transpose(indexed_trace, 0, 1).tolist()


        elif self.task_name is "FrankaBallBalancing":

            # Ball tool distance
            ball_tool_distance = trace[:, 21:23].detach().cpu()
            ball_tool_distance = torch.norm(ball_tool_distance, p=2, dim=-1)

            # create index
            trace_length = list(ball_tool_distance.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_trace = torch.vstack((times, ball_tool_distance))
            indexed_trace = torch.transpose(indexed_trace, 0, 1).tolist()

        elif self.task_name is "FrankaBallCatching":

            # Ball tool distance
            ball_tool_distance = trace[:, 21:23].detach().cpu()
            ball_tool_distance = torch.norm(ball_tool_distance, p=2, dim=-1)

            # create index
            trace_length = list(ball_tool_distance.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_trace = torch.vstack((times, ball_tool_distance))
            indexed_trace = torch.transpose(indexed_trace, 0, 1).tolist()

        elif self.task_name is "FrankaCubeStacking":

            # Cube distance
            cube_distance = trace[:, 25:27].detach().cpu()
            cube_distance = torch.norm(cube_distance, p=2, dim=-1)

            # Cube height
            cube_height_distance = trace[:, 27].detach().cpu()

            # create index
            trace_length = list(cube_distance.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_cube_distance = torch.vstack((times, cube_distance))
            indexed_cube_distance = torch.transpose(
                indexed_cube_distance, 0, 1
            ).tolist()

            indexed_cube_height_distance = torch.vstack((times, cube_height_distance))
            indexed_cube_height_distance = torch.transpose(
                indexed_cube_height_distance, 0, 1
            ).tolist()

            indexed_trace = {
                "distance_cube": indexed_cube_distance,
                "z_cube_distance": indexed_cube_height_distance,
            }

        elif self.task_name is "FrankaDoorOpen":

            # Ball tool distance
            handle_rot = trace[:, 21:25].detach().cpu()
            handle_yaw = torch.atan2(
                2.0
                * (
                    handle_rot[:, 0] * handle_rot[:, 3]
                    + handle_rot[:, 1] * handle_rot[:, 2]
                ),
                1.0
                - 2.0
                * (
                    handle_rot[:, 2] * handle_rot[:, 2]
                    + handle_rot[:, 3] * handle_rot[:, 3]
                ),
            )
            handle_yaw = torch.rad2deg(handle_yaw)

            # create index
            trace_length = list(handle_yaw.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_trace = torch.vstack((times, handle_yaw))
            indexed_trace = torch.transpose(indexed_trace, 0, 1).tolist()

        elif self.task_name is "FrankaPegInHole":

            # Ball tool distance
            tool_hole_distance = trace[:, 25:27].detach().cpu()
            tool_hole_distance = torch.norm(tool_hole_distance, p=2, dim=-1)
            # print(tool_hole_distance)

            # create index
            trace_length = list(tool_hole_distance.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_trace = torch.vstack((times, tool_hole_distance))
            indexed_trace = torch.transpose(indexed_trace, 0, 1).tolist()

        elif self.task_name is "FrankaPointReaching":

            # Ball tool distance
            finger_target_distance = trace[:, 24:27].detach().cpu()
            finger_target_distance = torch.norm(finger_target_distance, p=2, dim=-1)

            # create index
            trace_length = list(finger_target_distance.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_trace = torch.vstack((times, finger_target_distance))
            indexed_trace = torch.transpose(indexed_trace, 0, 1).tolist()

        elif self.task_name is "FrankaClothPlacing":

            # Cube distance
            cloth_target_distance = trace[:, 21:24].detach().cpu()
            cloth_target_distance = torch.norm(cloth_target_distance, p=2, dim=-1)

            # Cube height
            cloth_height = trace[:, 20].detach().cpu()

            # create index
            trace_length = list(cloth_target_distance.size())[0]
            times = torch.linspace(1, trace_length, steps=trace_length)

            # convert to list for computing robustness
            indexed_distance_cloth_target = torch.vstack((times, cloth_target_distance))
            indexed_distance_cloth_target = torch.transpose(
                indexed_distance_cloth_target, 0, 1
            ).tolist()

            indexed_cloth_height = torch.vstack((times, cloth_height))
            indexed_cloth_height = torch.transpose(
                indexed_cloth_height, 0, 1
            ).tolist()

            indexed_trace = {
                "distance_cloth_target": indexed_distance_cloth_target,
                "cloth_height": indexed_cloth_height,
            }


        else:

            raise ValueError("Task name unknown for merging the trace")

        return indexed_trace
