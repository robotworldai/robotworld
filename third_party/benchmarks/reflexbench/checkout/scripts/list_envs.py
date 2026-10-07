# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
List registered ReflexBench environments.

The script iterates over registered environments and prints a table with
environment name, entry point, and config entry point.

Use ``--all`` to include environments registered by other installed packages.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List ReflexBench environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
parser.add_argument(
    "--all",
    action="store_true",
    help="List environments from all installed packages.",
)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import reflexbench.tasks  # noqa: F401


def main():
    """Print registered environments with an environment config entry point."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available ReflexBench Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        kwargs = getattr(task_spec, "kwargs", None) or {}
        if "env_cfg_entry_point" not in kwargs:
            continue
        # Keep the default output scoped to the six benchmark tasks.
        if not args_cli.all:
            name = task_spec.id
            prefixes = (
                "ConveyorBeltPickAndPlace-",
                "BallCatching-",
                "WhackAMole-",
                "RollingBallInterception-",
                "BallThrowing-",
                "RotatingPegInsertion-",
            )
            if not name.startswith(prefixes):
                continue
        if args_cli.keyword is not None and args_cli.keyword not in task_spec.id:
            continue
        config = kwargs.get("env_cfg_entry_point", "")
        table.add_row([index + 1, task_spec.id, task_spec.entry_point, config])
        index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close the app
        simulation_app.close()
