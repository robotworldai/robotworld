# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with CusRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CusRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="cusrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--run_name", type=str, default=None, help="Name of the run for logging.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for resuming training.")
parser.add_argument("--logger", type=str, default="tensorboard", help="Logger to use for training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--autocast", nargs="?", const=True, help="Datatype for automatic mixed precision.")
parser.add_argument("--compile", action="store_true", help="Whether to use `torch.compile` for optimization.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# set distributed to True if LOCAL_RANK is set
if os.environ.get("LOCAL_RANK") is not None:
    args_cli.distributed = True
    args_cli.device = f"cuda:{os.environ['LOCAL_RANK']}"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

import cusrl
from cusrl.environment.isaaclab import TrainerCfg

from isaaclab.envs import DirectMARLEnvCfg  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: F401

import wheel_legged_robot.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: TrainerCfg):
    """Train with CusRL agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    cusrl.set_global_seed(args_cli.seed)
    env_cfg.scene.num_envs = int(env_cfg.scene.num_envs / cusrl.utils.distributed.world_size())

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name is not None:
        log_dir = f"{log_dir}_{args_cli.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video and cusrl.utils.is_main_process():
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # create trainer from cusrl
    from dataclasses import replace
    from cusrl.template import TrainerHook
    from cusrl.utils.str_utils import format_float

    class DebugPrintHook(TrainerHook):
        """在终端输出中打印关键调试指标的 Hook。"""

        # 需要打印的指标及其显示名
        _METRICS = [
            ("Environment/cmd_vx_abs", "|vx|"),
            ("Environment/cmd_vx_limit", "vx_lim"),
            ("Environment/curriculum_score", "cur"),
            ("Environment/cmd_wz_limit", "wz_lim"),
            ("Environment/curriculum_angular_score", "cur_wz"),
            ("Environment/vel_x_abs", "|vel|"),
            ("Environment/vx_err_inst", "vx_err"),
            ("Environment/vx_tracking_gain", "gain"),
            ("Environment/vx_sign_match", "sign"),
            ("Environment/cmd_wz_abs", "|wz|"),
            ("Environment/omega_z_abs", "|ωz|"),
            ("Environment/wz_err_inst", "wz_err"),
            ("Environment/wz_tracking_gain", "gain_wz"),
            ("Environment/heading_err_abs", "hdg_err"),
            ("Environment/cmd_height", "cmd_h"),
            ("Environment/base_height", "height"),
            ("Environment/h_err_inst", "h_err"),
            ("Environment/theta0_L", "θ0_L"),
            ("Environment/theta0_R", "θ0_R"),
            ("Environment/L0_L", "L0_L"),
            ("Environment/L0_R", "L0_R"),
            ("Environment/rf0_pos", "rf0"),
            ("Environment/lf0_pos", "lf0"),
            ("Environment/tilt_angle", "tilt"),
            ("Environment/wheel_action_abs", "|a_w|"),
            ("Environment/wheel_action_clip_fraction", "clip_w"),
            ("Environment/wheel_vel_abs", "|ω_w|"),
        ]

        def pre_log_info(self, info: dict[str, float]):
            parts = []
            for key, label in self._METRICS:
                val = info.get(key)
                if val is not None:
                    parts.append(f"{label}={format_float(val, 6)}")
            if parts:
                print(" │ " + "  ".join(parts) + " │")

    trainer = cusrl.Trainer(
        environment=cusrl.environment.IsaacLabEnvAdapter(env),
        agent_factory=replace(
            agent_cfg.agent_factory,
            device=args_cli.device,
            autocast=args_cli.autocast,
            compile=args_cli.compile,
        ),
        logger_factory=cusrl.make_logger_factory(args_cli.logger, log_dir, add_datetime_prefix=False),
        num_iterations=agent_cfg.max_iterations,
        checkpoint_interval=agent_cfg.save_interval,
        checkpoint_path=args_cli.checkpoint,
        hooks=[DebugPrintHook()],
    )

    # run training
    trainer.run_training_loop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
