# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export. Use ``robolab.registrations.droid.auto_env_registrations_abs_ik`` instead."""
import warnings

from robolab.registrations.droid.auto_env_registrations_abs_ik import *  # noqa: F401, F403
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs  # noqa: F401

warnings.warn(
    "robolab.registrations.droid_jointpos.auto_env_registrations_abs_ik is deprecated; "
    "import from robolab.registrations.droid.auto_env_registrations_abs_ik instead.",
    DeprecationWarning,
    stacklevel=2,
)
