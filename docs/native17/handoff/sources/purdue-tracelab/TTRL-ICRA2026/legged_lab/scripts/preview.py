import argparse

from isaaclab.app import AppLauncher
import legged_lab.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Preview a Legged Lab environment in Isaac Sim.")
parser.add_argument("--task", type=str, required=True, help="Name of the task to preview (e.g., 'Anymal-C-Flat').")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to display.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.utils import task_registry
from legged_lab.envs import * 
from legged_lab.utils.cli_args import update_rsl_rl_cfg
import torch

# !Recorder : BEGIN
import carb
from legged_lab.utils.data_recorder.frame_recorder import FrameRecorder, ensure_world_camera
# !Recorder : END

def main():
    """A simple script to preview a legged lab environment."""
    
    env_cfg, agent_cfg = task_registry.get_cfgs(name=args_cli.task)
    env_class = task_registry.get_task_class(name=args_cli.task)

    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    


    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed
    
    env = env_class(cfg=env_cfg, headless=False)
    
    # !Recorder : BEGIN
    camera_paths = [ensure_world_camera("/OmniverseKit_Persp")] # ! Image Recorder
    rgb_rec = FrameRecorder(
        camera_prim_paths=camera_paths,
        output_root="_rgb_recordings",
    )
    rgb_rec.start()
    env.sim.set_camera_view([-3.0, 6.0, 1.0], [-3.0, 0.0, 1.0])
    # !Recorder : END

    try:
        num_actions = env.num_actions
    except AttributeError:
        num_actions = 22 # ! Booster T1
        print(f"[WARN] env.num_actions not found. Falling back to dummy value: {num_actions}")

    dummy_actions = torch.zeros(args_cli.num_envs, num_actions, device=env.device)
    while simulation_app.is_running():
        env.step(dummy_actions)


    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()