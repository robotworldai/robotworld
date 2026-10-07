#!/usr/bin/env python3
"""CLI entry point for the MuJoCo soccer experiment."""

try:
    from mujoco_soccer.cli import main
except ModuleNotFoundError as exc:
    if exc.name != "mujoco_soccer":
        raise
    from exp.mujoco_soccer.cli import main


if __name__ == "__main__":
    main()
