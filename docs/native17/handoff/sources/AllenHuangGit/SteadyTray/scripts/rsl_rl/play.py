# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL.

This script supports standard, adapter, and distillation policy configurations:
- Standard policies: Single policy network for all actions
- Adapter policies: Frozen base policy + trainable adapters + encoder (parameter-efficient fine-tuning)
- Distillation policies: Student encoder distilled from teacher encoder

The script automatically detects the configuration type based on the agent config
and uses the appropriate policy runner, environment wrapper, and observation handling.

Usage:
- Standard: python play.py --task=Template-G1-v0 --num_envs=2 --load_run=run_name
- Adapter: python play.py --task=Template-G1-Adapter-v0 --num_envs=2 --load_run=run_name
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--teacher", action="store_true", default=False, help="Run in teacher mode.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from base.on_policy_runner import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
import steadytray.tasks  # noqa: F401

# Import adapter policy runner and wrapper for parameter-efficient fine-tuning
try:
    from adapter.on_policy_runner import AdapterOnPolicyRunner
    from adapter.env_wrapper import AdapterRslRlVecEnvWrapper
    ADAPTER_POLICY_AVAILABLE = True
except ImportError as e:
    AdapterOnPolicyRunner = None
    AdapterRslRlVecEnvWrapper = None
    ADAPTER_POLICY_AVAILABLE = False

from steadytray.utils.parser_cfg import parse_env_cfg

def is_adapter_config(agent_cfg):
    """Check if the configuration is for adapter-based training.
    
    Adapter configs have policy.class_name in ["AdaptedActorCritic", "ResidualActorCritic"] 
    and use parameter-efficient fine-tuning with adapters (FiLM or Residual) and encoder.
    """
    return hasattr(agent_cfg, 'policy') and hasattr(agent_cfg.policy, 'class_name') and \
           agent_cfg.policy.class_name in ["AdaptedActorCritic", "ResidualActorCritic"]


def is_distillation_config(agent_cfg):
    """Check if the configuration is for distillation training.
    
    Distillation configs have algorithm.class_name == "Distillation" and use
    AdaptedStudentTeacher for online encoder distillation.
    """
    return hasattr(agent_cfg, 'algorithm') and hasattr(agent_cfg.algorithm, 'class_name') and \
           agent_cfg.algorithm.class_name == "Distillation"


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    # Check if we're using adapter or distillation configuration
    is_adapter = is_adapter_config(agent_cfg)
    is_distillation = is_distillation_config(agent_cfg)
    
    if is_adapter or is_distillation:
        if not ADAPTER_POLICY_AVAILABLE:
            raise ImportError("Adapter policy runner not available. Make sure adapter module is installed.")
        
        if is_distillation:
            print("[INFO] Using distillation configuration (online encoder distillation)")
            print("[INFO] Architecture: Trainable student encoder + Frozen teacher encoder + Shared frozen components")
            print("[INFO] Inference: Using student encoder for evaluation")
        else:
            print("[INFO] Using adapter-based policy configuration (parameter-efficient fine-tuning)")
            print("[INFO] Architecture: Frozen base policy + Trainable FiLM adapters + GRU encoder")
        
        # wrap around environment for rsl-rl (adapter/distillation version)
        env = AdapterRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model (adapter/distillation version)
        ppo_runner = AdapterOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference (adapter/distillation version)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_type = "distillation" if is_distillation else "adapter"
        
        # For adapter/distillation policy, we don't export since it's more complex
        policy_nn = None
    else:
        print("[INFO] Using standard policy configuration")
        # wrap around environment for rsl-rl (standard version)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model (standard version)
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference (standard version)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_type = "standard"

        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = ppo_runner.alg.actor_critic

        # export policy to onnx/jit (only for standard policy)
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
    
    dt = env.unwrapped.step_dt

    # reset environment
    obs, info = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping - different handling for distillation, adapter vs standard policy
            if policy_type == "distillation":
                # For distillation policy - observations are structured as a dictionary with specific keys
                # Use student encoder for inference (teacher is only for training)
                if isinstance(obs, dict):
                    student_encoder_obs = obs["student_encoder"]
                    policy_obs = obs["policy"]
                    # Call distillation policy with student encoder and policy observations
                    actions = policy.__call__(student_encoder_obs, policy_obs)
                else:
                    raise ValueError("Expected dictionary observations for distillation policy")
            elif policy_type == "adapter":
                # For adapter policy - observations are structured as a dictionary with specific keys
                if isinstance(obs, dict):
                    encoder_obs = obs["encoder"]
                    policy_obs = obs["policy"]
                    # Call adapter policy with encoder and policy observations
                    actions = policy.__call__(encoder_obs, policy_obs)
                else:
                    raise ValueError("Expected dictionary observations for adapter policy")
            else:
                # For standard policy - single observation tensor
                actions = policy.__call__(obs)
                
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
