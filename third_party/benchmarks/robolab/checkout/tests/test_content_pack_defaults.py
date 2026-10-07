# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from policies.volo.registration import VOLO_TASK_SUBFOLDERS
from robolab.constants import DEFAULT_TASK_SUBFOLDERS
from robolab.core.environments.factory import EnvFactory


def test_robovolo_content_pack_is_discoverable_by_volo():
    assert "robovolo" in VOLO_TASK_SUBFOLDERS
    assert all(subfolder in VOLO_TASK_SUBFOLDERS for subfolder in DEFAULT_TASK_SUBFOLDERS)


def test_standard_registrations_skip_the_content_pack():
    assert "robovolo" not in DEFAULT_TASK_SUBFOLDERS


def test_explicit_task_resolves_from_scoped_content_pack(tmp_path, monkeypatch):
    task_file = tmp_path / "robovolo" / "sample.py"
    task_file.parent.mkdir()
    task_file.write_text("class SampleTask:\n    pass\n")

    monkeypatch.setattr(
        "robolab.core.task.task_utils.get_task_class_name_from_file",
        lambda path: "SampleTask",
    )
    factory = EnvFactory(task_dir=tmp_path)
    resolved = []

    def fake_create_env_cfg(task, **kwargs):
        resolved.append((task, kwargs))
        return object

    monkeypatch.setattr(factory, "create_env_cfg", fake_create_env_cfg)
    generated = factory.auto_discover_and_create_cfgs(
        tasks="SampleTask",
        task_subdirs=["robovolo"],
        add_tags="volo",
        env_postfix="DroidJointPosition",
    )

    assert generated == {"SampleTask": object}
    assert Path(resolved[0][0]) == task_file
    assert resolved[0][1] == {
        "tags": "volo",
        "env_prefix": "",
        "env_postfix": "DroidJointPosition",
    }
