# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-task language prompts for VLA data collection / LeRobot conversion.

Keyed by the task slug produced by :func:`_task_slug` in both
``collect_vla_data.py`` and ``convert_to_lerobot.py``.  Accessed via
:func:`resolve_prompt_from_task`, which normalizes any Gym id (e.g.
``ConveyorBeltPickAndPlace-Franka-DataCollection-v0``) into the same slug.

Keep the wording short, action-driven, and English-only; VLA tokenizers
handle these best.  If a slug is missing, we fall back to
``LEGACY_DEFAULT_PROMPT`` so both scripts degrade gracefully.
"""

from __future__ import annotations

# Legacy default used as both the argparse default in collect_vla_data.py
# and the sentinel for "user did not customize the prompt" when deciding
# whether to auto-fill from the task lookup.
LEGACY_DEFAULT_PROMPT: str = "manipulate the object"


#
# Keys match the slug produced by ``_task_slug`` in ``collect_vla_data.py``
# and ``convert_to_lerobot.py`` (suffixes + robot tokens stripped, then joined
# by underscore).  We also register the "spelled-out" variant (e.g.
# ``conveyor_belt_pick_and_place``) as a synonym for humans who call the resolver
# directly with task profile names.
_CONVEYOR_BELT_PICK_AND_PLACE = (
    "pick up the cube from the moving conveyor and place it into the bin"
)
_WHACK_A_MOLE = "hit the mole that pops up on the board with the closed gripper"
_BALL_CATCHING = (
    "hold the container with the gripper and catch the thrown ball in the container"
)
_ROLLING_BALL_INTERCEPTION = "catch the ball rolling down the ramp with the container"
_ROTATING_PEG_INSERTION = "insert the peg into the hole on the rotating disc"
_BALL_THROWING = "throw the ball into the bin"

TASK_PROMPTS: dict[str, str] = {
    # slug as produced by `_task_slug(gym_id)`:
    "conveyorbeltpickandplace": _CONVEYOR_BELT_PICK_AND_PLACE,
    "whackamole": _WHACK_A_MOLE,
    "ballcatching": _BALL_CATCHING,
    "rollingballinterception": _ROLLING_BALL_INTERCEPTION,
    "rotatingpeginsertion": _ROTATING_PEG_INSERTION,
    "ballthrowing": _BALL_THROWING,
    # human-readable profile names (also accepted):
    "conveyor_belt_pick_and_place": _CONVEYOR_BELT_PICK_AND_PLACE,
    "whack_a_mole": _WHACK_A_MOLE,
    "ball_catching": _BALL_CATCHING,
    "rolling_ball_interception": _ROLLING_BALL_INTERCEPTION,
    "rotating_peg_insertion": _ROTATING_PEG_INSERTION,
    "ball_throwing": _BALL_THROWING,
}


def _slugify(task_name: str) -> str:
    """Normalize a task/gym id into a lookup slug.

    Mirrors ``_task_slug`` in ``collect_vla_data.py`` / ``convert_to_lerobot.py``:
    strips known suffixes and robot tokens, then joins with underscores.
    Also accepts already-slugified inputs like ``conveyor_belt_pick_and_place``.
    """
    if not task_name:
        return ""
    normalized = task_name.lower().strip()

    # Already a slug (no dashes) -> just return as-is (after normalizing underscores).
    if "-" not in normalized and "_" in normalized:
        return normalized

    for suffix in ("-datacollection-v0", "-play-v0", "-ik-abs-v0", "-ik-rel-v0", "-v0"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    tokens = [
        token for token in normalized.split("-")
        if token and token not in {"franka", "panda", "ur5", "ur10", "xarm", "kinova"}
    ]
    if not tokens:
        return normalized.replace("-", "_")
    return "_".join(tokens)


def resolve_prompt_from_task(
    task_name: str | None,
    default: str = LEGACY_DEFAULT_PROMPT,
) -> str:
    """Return the prompt registered for *task_name*, else *default*."""
    if not task_name:
        return default
    slug = _slugify(task_name)
    return TASK_PROMPTS.get(slug, default)


def resolve_prompt(
    task_name: str | None,
    user_prompt: str | None,
    legacy_default: str = LEGACY_DEFAULT_PROMPT,
) -> str:
    """Decide which prompt to use.

    Priority:
      1. Non-empty *user_prompt* that differs from the legacy default.
      2. Task-specific prompt from :data:`TASK_PROMPTS`.
      3. *legacy_default*.

    This means the user can still override via ``--prompt "my custom text"``,
    while leaving ``--prompt`` untouched (or equal to the legacy default)
    triggers per-task auto-fill.
    """
    if user_prompt is not None and user_prompt != "" and user_prompt != legacy_default:
        return user_prompt
    return resolve_prompt_from_task(task_name, default=legacy_default)
