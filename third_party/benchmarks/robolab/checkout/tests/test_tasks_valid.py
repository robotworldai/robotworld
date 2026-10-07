# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify every task definition in robolab/tasks/benchmark/ is valid.

Parametrized over each discovered task file so failures name the offender.
Replaces the legacy scripts/check_tasks_valid.py.
"""

import os

import pytest

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
from robolab.core.task.task import verify_task_valid
from robolab.core.task.task_utils import find_task_files, load_task_from_file


def _all_task_files():
    files = []
    for subfolder in DEFAULT_TASK_SUBFOLDERS:
        folder = os.path.join(TASK_DIR, subfolder)
        if os.path.isdir(folder):
            files.extend(find_task_files(folder))
    return files


_TASK_FILES = _all_task_files()


def test_task_files_discovered():
    assert len(_TASK_FILES) >= 100, (
        f"Expected ≥100 task files under {TASK_DIR}, found {len(_TASK_FILES)}"
    )


@pytest.mark.parametrize("task_file", _TASK_FILES, ids=lambda p: os.path.basename(p))
def test_task_valid(task_file):
    task_class = load_task_from_file(task_file, allow_multiple=False)
    valid, error = verify_task_valid(task_class)
    assert valid, f"{task_file}: {error}"


def test_no_duplicate_task_class_names():
    name_to_files = {}
    for task_file in _TASK_FILES:
        try:
            task_class = load_task_from_file(task_file, allow_multiple=False)
        except Exception:
            continue
        name = getattr(task_class, "__name__", str(task_class))
        name_to_files.setdefault(name, []).append(task_file)
    duplicates = {name: files for name, files in name_to_files.items() if len(files) > 1}
    assert not duplicates, f"Duplicate task class names: {duplicates}"
