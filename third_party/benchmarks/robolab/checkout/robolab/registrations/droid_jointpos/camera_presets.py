# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export. Use ``robolab.registrations.droid.camera_presets`` instead."""
import warnings

from robolab.registrations.droid.camera_presets import *  # noqa: F401, F403
from robolab.registrations.droid.camera_presets import (  # noqa: F401
    LEFT_RIGHT,
    WRIST,
    WRIST_LEFT,
    WRIST_LEFT_RIGHT,
    WRIST_LEFT_RIGHT_HEAD,
    WRIST_RIGHT,
)

warnings.warn(
    "robolab.registrations.droid_jointpos.camera_presets is deprecated; "
    "import from robolab.registrations.droid.camera_presets instead.",
    DeprecationWarning,
    stacklevel=2,
)
