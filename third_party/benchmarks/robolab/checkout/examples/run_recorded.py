# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""
Run pre-recorded episodes from HDF5 files for testing and validation.

This script replays previously recorded robot trajectories to verify task behavior,
subtask progress tracking, and environment consistency.

Usage:
    Basic usage with default task:
    $ python run_recorded.py

    Specify a custom task:
    $ python run_recorded.py --task RubiksCubeOrBananaTask

    Run headless (no rendering):
    $ python run_recorded.py --task MyTask --headless

    Replay an episode straight from an eval output folder:
    $ python run_recorded.py --task MyTask --recorded-data-folder output/<run_folder> --file run_0.hdf5 --episode 2

Requirements:
    - Recorded data must exist at: <recorded-data-folder>/<task>/<file>
      (default: examples/recorded_data/<task>/data.hdf5)
    - Task must be registered in the environment factory

Output:
    Results are saved to: output/playback_output/
    - Episode logs: <task_name>/log_<episode>.json
    - Summary: results.json and episode_results.json
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import json
import sys
import traceback

from isaaclab.app import AppLauncher
from robolab.constants import PACKAGE_DIR, set_output_dir # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--task", '-t', type=str, default="RubiksCubeAndBananaTask", help="Task name to run.")
parser.add_argument("--recorded-data-folder", '--dir', type=str, default=os.path.join(PACKAGE_DIR, 'examples', 'recorded_data'), help="Recorded data folder to run.")
parser.add_argument("--file", type=str, default="data.hdf5",
                    help="HDF5 filename inside <recorded-data-folder>/<task>/ (e.g. run_0.hdf5 to replay an eval "
                         "output directly).")
parser.add_argument("--episode", type=int, default=0,
                    help="Demo index within the HDF5 file to replay (demo_<episode>; in multi-env recordings "
                         "demo_i is env i's episode).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--enable-subtask", "--enable_subtask", dest="enable_subtask", action="store_true", default=True,
                    help="Enable subtask progress checking (default: on; kept for backward compatibility).")
parser.add_argument("--disable-subtask", "--disable_subtask", dest="enable_subtask", action="store_false",
                    help="Disable subtask progress checking.")
parser.add_argument("--env-config", choices=["recorded", "current"], default="recorded",
                    help="Which env config to replay with: 'recorded' overlays the env_cfg.json saved next to the "
                         "recording (faithful playback, default); 'current' rebuilds it from the current repo's "
                         "task definitions.")
parser.add_argument("--validate-states", action="store_true",
                    help="Debug tool: compare the sim state against the recorded per-step states and report drift.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from episodes import run_prerecorded_episode_hdf5 # noqa
from robolab.core.environments.config import parse_env_cfg # noqa
from robolab.core.replay import apply_recorded_env_cfg, load_recorded_env_cfg # noqa
from robolab.core.environments.runtime import create_env, end_episode # noqa
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.constants import get_timestamp # noqa
from robolab.core.logging.results import dump_results_to_file, get_all_env_events, summarize_experiment_results # noqa
from robolab.core.logging.results import init_experiment, update_experiment_results # noqa
import robolab.constants # noqa

# Run automatic factory generation before main
auto_register_droid_envs()

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = True
robolab.constants.DEBUG = False
robolab.constants.RECORD_IMAGE_DATA = False

def resolve_replay_env_cfg(task_env: str, hdf5_path: str):
    """Resolve what to hand ``create_env``: the recorded env config or the env name.

    With ``--env-config recorded`` (the default), the ``env_cfg.json`` saved
    next to the recording is overlaid onto a freshly built config so playback
    uses exactly the recorded values. Falls back to the current repo's config
    (returning the env name) when the sidecar is missing or ``--env-config
    current`` was requested, with a warning that behavior may diverge.
    """
    loaded = load_recorded_env_cfg(hdf5_path) if args_cli.env_config == "recorded" else None
    if loaded is None:
        reason = ("--env-config current was requested" if args_cli.env_config == "current"
                  else f"no env_cfg.json found next to {hdf5_path}")
        print(f"\033[93mWARNING: replaying with the current repo's env config, not the one the episode "
              f"was recorded with ({reason}). If task or scene definitions changed since recording, "
              "the env config is not the same as before and behavior may diverge.\033[0m")
        return task_env

    recorded_cfg, sidecar_path = loaded
    env_cfg = parse_env_cfg(task_env, device=args_cli.device, seed=0, num_envs=args_cli.num_envs,
                            env_spacing=None, eye=None, lookat=None, use_fabric=True)
    skipped = apply_recorded_env_cfg(env_cfg, recorded_cfg)
    print(f"Restored recorded env config from {sidecar_path}")
    if skipped:
        print(f"\033[93mWARNING: {len(skipped)} recorded config field(s) no longer match the current "
              "config schema and were kept at their current values, so the env config is not exactly "
              "the same as recorded and behavior may diverge:\n  " + "\n  ".join(skipped) + "\033[0m")
    return env_cfg


def main():
    """Main function."""
    task = args_cli.task
    output_dir = os.path.join(PACKAGE_DIR, "output", "playback_"+os.path.basename(args_cli.recorded_data_folder) + "_" + task)
    os.makedirs(output_dir, exist_ok=True)


    # Check if task is a folder inside the recorded_data folder
    if os.path.isdir(os.path.join(args_cli.recorded_data_folder, task)):
        hdf5_path = os.path.join(args_cli.recorded_data_folder, task, args_cli.file)
    else:
        raise ValueError(f"Task {task} not found in {args_cli.recorded_data_folder}")
    if not os.path.isfile(hdf5_path):
        raise ValueError(f"Recorded data file not found: {hdf5_path}")

    task_envs = get_envs(task=task)
    print(f"Running {len(task_envs)} environments: {task_envs}")

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        scene = resolve_replay_env_cfg(task_env, hdf5_path)

        env, env_cfg = create_env(scene,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True)

        # Can be used to loop through multiple episodes; run identifiers (logs,
        # exported run_<i>.hdf5, episode ids) follow the selected demo index.
        for i in [args_cli.episode]:

            run_name = task_env + f"_run{i}"
            print(f"Running {run_name}: '{env_cfg.instruction}'")

            env_results, msgs = run_prerecorded_episode_hdf5(env,
                hdf5_path=hdf5_path,
                episode=i,
                save_videos=args_cli.save_videos,
                headless=args_cli.headless,
                validate_states=args_cli.validate_states)

            # Write v2 per-env event logs
            per_env_events = get_all_env_events(env) or []
            for eid in range(args_cli.num_envs):
                events = per_env_events[eid] if eid < len(per_env_events) else []
                log_obj = {
                    "schema_version": 2,
                    "task": task_env,
                    "env_id": eid,
                    "run": i,
                    "events": events,
                }
                log_path = os.path.join(scene_output_dir, f"log_{i}_env{eid}.json")
                dump_results_to_file(log_path, log_obj, append=False)

            # Emit one run_summary per env (each env is an independent episode)
            for r in env_results:
                episode_id = i * args_cli.num_envs + r['env_id']

                run_summary = {
                    "env_name": task_env,
                    "run": i,
                    "episode": episode_id,
                    "env_id": r['env_id'],
                    "success": r['success'],
                    "step": r['step'],
                    "instruction": env_cfg.instruction,
                }

                if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING:
                    if len(msgs) > 0 and msgs[-1] is not None:
                        subtask_info = msgs[-1]
                        run_summary["score"] = subtask_info.get("score", None)
                        run_summary["reason"] = subtask_info.get("info", None)

                episode_results = update_experiment_results(run_summary=run_summary, episode_results=episode_results, episode_results_file=episode_results_file)

        env.close()

    summarize_experiment_results(episode_results)
    simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
