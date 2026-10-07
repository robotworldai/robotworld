# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility rule for recordings made before the frame contract (docs/frames.md).

A recorded demo without a ``robot_root_pose`` group predates the frame
contract. Every such recording was produced with a Franka-family robot, whose
root sits at the env origin with identity rotation — so its root pose is the
identity, and the demo's robot-centric channels (recorded env-locally at the
time) are already valid robot-root values.

This module is the single home of that assumption. Consumers that need the
robot's root pose from a recording must call :func:`demo_robot_root_pose`
rather than re-implementing the fallback.
"""

from __future__ import annotations

import numpy as np


def demo_robot_root_pose(demo_group, num_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-step robot root pose for a recorded demo.

    Args:
        demo_group: An open ``data/<demo>`` HDF5 group (or any mapping with the
            same layout).
        num_steps: Length of the identity fallback used for pre-contract demos.
            Defaults to the demo's action count, or 1 if the demo has no
            ``actions`` dataset.

    Returns:
        ``(position, orientation)`` with shapes ``(num_steps, 3)`` and
        ``(num_steps, 4)``: env-local position in meters and ``(w, x, y, z)``
        quaternion, per docs/frames.md.
    """
    if "robot_root_pose" in demo_group:
        group = demo_group["robot_root_pose"]
        return np.asarray(group["position"]), np.asarray(group["orientation"])
    if num_steps is None:
        num_steps = int(demo_group["actions"].shape[0]) if "actions" in demo_group else 1
    position = np.zeros((num_steps, 3), dtype=np.float32)
    orientation = np.zeros((num_steps, 4), dtype=np.float32)
    orientation[:, 0] = 1.0  # identity quaternion (w, x, y, z)
    return position, orientation
