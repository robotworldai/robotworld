# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export. Use ``robolab.registrations.droid.auto_env_registrations_bg_variations`` instead."""
import warnings

from robolab.registrations.droid.auto_env_registrations_bg_variations import *  # noqa: F401, F403
from robolab.registrations.droid.auto_env_registrations_bg_variations import auto_register_droid_envs_bg_variations  # noqa: F401

warnings.warn(
    "robolab.registrations.droid_jointpos.auto_env_registrations_bg_variations is deprecated; "
    "import from robolab.registrations.droid.auto_env_registrations_bg_variations instead.",
    DeprecationWarning,
    stacklevel=2,
)
