# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify isaaclab is installed and importable.

Replaces the legacy scripts/check_isaaclab.py. The acceptable version is
pinned by pyproject.toml — no need to re-assert it here.
"""

from importlib.metadata import PackageNotFoundError, version


def test_isaaclab_installed():
    try:
        version("isaaclab")
    except PackageNotFoundError as e:
        raise AssertionError("isaaclab is not installed") from e

    import isaaclab  # noqa: F401
