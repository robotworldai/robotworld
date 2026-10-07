# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation rules for robot contact_gripper declarations (labels and alias groups)."""

import pytest

from robolab.core.sensors.contact_sensor_utils import validate_contact_grippers


def test_single_arm_declaration_passes():
    declaration = {"gripper": "{ENV_REGEX_NS}/robot/finger"}
    assert validate_contact_grippers(declaration) == declaration


def test_bimanual_group_passes_and_gets_no_sensors():
    concrete = validate_contact_grippers(
        {"left": "/robot/l_finger", "right": "/robot/r_finger", "gripper": ["left", "right"]}
    )
    assert set(concrete) == {"left", "right"}


def test_missing_gripper_entry_rejected():
    with pytest.raises(ValueError, match="must declare a 'gripper' entry"):
        validate_contact_grippers({"left": "/robot/l_finger", "right": "/robot/r_finger"})


def test_group_with_unknown_member_rejected():
    with pytest.raises(ValueError, match="not concrete gripper labels"):
        validate_contact_grippers({"left": "/robot/l_finger", "gripper": ["left", "rigth"]})


def test_nested_group_rejected():
    with pytest.raises(ValueError, match="not concrete gripper labels"):
        validate_contact_grippers(
            {"left": "/robot/l_finger", "hands": ["left"], "gripper": ["hands"]}
        )


def test_empty_group_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        validate_contact_grippers({"gripper": []})
