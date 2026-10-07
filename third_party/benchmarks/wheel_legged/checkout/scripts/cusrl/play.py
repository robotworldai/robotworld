# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from CusRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with CusRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="cusrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for playing.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Whether to run the agent in stochastic mode.",
)
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--keyboard_height", type=float, default=0.21, help="Initial body height in keyboard mode.")
parser.add_argument(
    "--keyboard_height_step", type=float, default=0.01, help="Body-height increment for each R/F key press."
)
parser.add_argument(
    "--command_range",
    type=float,
    default=0.1,
    help=(
        "Wheel-legged linear velocity limit used during play. "
        "Use the final vx_lim printed by training; old checkpoints trained with the broken curriculum use 0.1."
    ),
)
parser.add_argument(
    "--yaw_command_range",
    type=float,
    default=0.5,
    help=(
        "Wheel-legged yaw-rate limit used during play. "
        "Use the final wz_lim printed by training; a newly initialized curriculum starts at 0.5."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from dataclasses import replace

import gymnasium as gym

import cusrl
from cusrl.environment.isaaclab import TrainerCfg

from isaaclab.envs import DirectMARLEnvCfg  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: F401

import wheel_legged_robot.tasks  # noqa: F401


class KeyboardPlayerHook(cusrl.Player.Hook):
    def __init__(self, keyboard_controller):
        self.keyboard_controller = keyboard_controller

    def step(self, step: int, transition: dict):
        from rl_utils import camera_follow

        self.keyboard_controller.update()
        camera_follow(self.player.environment)

    def reset(self, indices):
        self.keyboard_controller.update()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: TrainerCfg):
    """Play with CusRL-RL agent."""
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    cusrl.set_global_seed(args_cli.seed)

    # modify environment configurations based on CLI args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "wheel_legged_commands"):
        command_ranges = env_cfg.commands.wheel_legged_commands.ranges
        command_ranges.lin_vel_x = (-args_cli.command_range, args_cli.command_range)
        command_ranges.ang_vel_yaw = (
            -args_cli.yaw_command_range,
            args_cli.yaw_command_range,
        )
        print(
            "[INFO] Wheel-legged play command ranges: "
            f"vx={command_ranges.lin_vel_x} m/s, "
            f"wz={command_ranges.ang_vel_yaw} rad/s"
        )
    env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        wheel_command_cfg = env_cfg.commands.wheel_legged_commands
        wheel_command_cfg.heading_command = False
        wheel_command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        wheel_command_cfg.debug_vis = False

    if args_cli.checkpoint is None:
        args_cli.checkpoint = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
    trial = cusrl.Trial(args_cli.checkpoint)
    if trial is not None:
        log_dir = trial.home
    else:
        # specify directory for logging videos
        log_dir = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
        log_dir = os.path.abspath(log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    keyboard_controller = None
    if args_cli.keyboard:
        from rl_utils import WheelLeggedKeyboardController

        keyboard_controller = WheelLeggedKeyboardController(
            env,
            vx_limit=args_cli.command_range,
            wz_limit=args_cli.yaw_command_range,
            default_height=args_cli.keyboard_height,
            height_step=args_cli.keyboard_height_step,
        )

    # 非键盘模式启用 command debug 可视化。
    cmd_term = env.unwrapped.command_manager._terms.get("wheel_legged_commands")
    if cmd_term is not None and not args_cli.keyboard and not args_cli.headless:
        cmd_term.set_debug_vis(True)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # create player from cusrl
    player = cusrl.Player(
        environment=cusrl.environment.IsaacLabEnvAdapter(env),
        agent=replace(agent_cfg.agent_factory, device=args_cli.device),
        checkpoint_path=trial,
        deterministic=not args_cli.stochastic,
    )

    export_model_dir = os.path.join(log_dir, "exported")
    player.agent.export(output_dir=export_model_dir, target_format="onnx", verbose=args_cli.verbose)
    player.agent.export(output_dir=export_model_dir, target_format="jit", verbose=args_cli.verbose)

    if args_cli.keyboard:
        player.register_hook(KeyboardPlayerHook(keyboard_controller))

    # run playing loop
    player.run_playing_loop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
