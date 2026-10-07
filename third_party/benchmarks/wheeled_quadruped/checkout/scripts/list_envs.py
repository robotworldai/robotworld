# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the environments registered by this project.

The script iterates over all registered gym environments and prints, in a table,
the ones that belong to the ``wheeled_quadruped`` package (task ids that contain
``Wheeled-Quadruped``) together with their entry point and config entry point.

Mirrors the upstream Isaac Lab ``scripts/environments/list_envs.py`` pattern: the
simulation app is launched (headless) first so that importing the task package -
which pulls in Isaac Lab config classes - succeeds.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List the wheeled-quadruped environments registered by this project.")
parser.add_argument(
    "--keyword", type=str, default="Wheeled-Quadruped", help="Keyword used to filter the registered task ids."
)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import wheeled_quadruped.tasks  # noqa: F401


def main():
    """Print all environments registered by the wheeled_quadruped package."""
    # build the table
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Wheeled Quadruped Environments"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all matching environment names
    for task_spec in gym.registry.values():
        if args_cli.keyword in task_spec.id:
            # add details to table (config entry point may be absent on some registrations)
            table.add_row([
                index + 1,
                task_spec.id,
                task_spec.entry_point,
                task_spec.kwargs.get("env_cfg_entry_point", ""),
            ])
            # increment count
            index += 1

    if index == 0:
        print(f"[WARNING] No registered task ids contain the keyword '{args_cli.keyword}'.")
    else:
        print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
