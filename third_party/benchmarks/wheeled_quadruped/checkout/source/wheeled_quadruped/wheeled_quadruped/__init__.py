"""Wheeled quadruped Isaac Lab extension package.

Importing this package registers all gym tasks (balance and velocity) as a side
effect, so a plain ``import wheeled_quadruped`` is enough to make the tasks
available through :func:`gymnasium.make`.
"""

from . import tasks  # noqa: F401
