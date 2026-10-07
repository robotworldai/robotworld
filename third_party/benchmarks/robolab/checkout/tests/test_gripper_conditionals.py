# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Truth-table coverage for robot-owned and bimanual gripper conditionals."""

import math
from types import SimpleNamespace

import torch

from robolab.core.task import conditionals


class _JointWorld:
    def __init__(self, env, positions):
        self.env = env
        self._positions = torch.tensor(positions, dtype=torch.float32)

    def get_joint_names(self, _robot_name):
        return ["left_joint", "right_joint"]

    def get_joint_positions(self, _robot_name, env_id=None):
        return self._positions if env_id is None else self._positions[env_id]

    def resolve_contact_bodies(self, body):
        names = [body] if isinstance(body, str) else list(body)
        resolved = []
        for name in names:
            members = ["left", "right"] if name == "gripper" else [name]
            for member in members:
                if member not in resolved:
                    resolved.append(member)
        return resolved


def _env(positions):
    cfg = SimpleNamespace(
        contact_gripper={"left": "/left", "right": "/right", "gripper": ["left", "right"]},
        gripper_closure_cfg={
            "left": ("left_joint", 0.0, 2.0),
            "right": ("right_joint", 0.0, 2.0),
        },
        contact_object_list=["target", "wrong", "table"],
    )
    return SimpleNamespace(cfg=cfg, num_envs=len(positions), device=torch.device("cpu"))


def test_robot_owned_gripper_ranges_and_aliases(monkeypatch):
    positions = [[0.0, 0.0], [0.0, 2.0], [1.6, 0.0], [2.0, 2.0]]
    env = _env(positions)
    world = _JointWorld(env, positions)
    monkeypatch.setattr(conditionals, "get_world", lambda _env: world)

    assert torch.equal(
        conditionals.gripper_fully_closed(env),
        torch.tensor([False, True, True, True]),
    )


def test_legacy_finger_joint_contract_is_preserved(monkeypatch):
    env = SimpleNamespace(
        cfg=SimpleNamespace(contact_gripper={"gripper": "/finger"}),
        num_envs=2,
        device=torch.device("cpu"),
    )

    class _LegacyWorld:
        def __init__(self):
            self.env = env
            self.positions = torch.tensor([[0.0], [math.pi / 4]])

        def get_joint_names(self, _robot_name):
            return ["finger_joint"]

        def get_joint_positions(self, _robot_name, env_id=None):
            return self.positions if env_id is None else self.positions[env_id]

        def resolve_contact_bodies(self, _body):
            return ["gripper"]

    monkeypatch.setattr(conditionals, "get_world", lambda _env: _LegacyWorld())
    assert torch.equal(conditionals.gripper_fully_closed(env), torch.tensor([False, True]))
    assert conditionals.gripper_fully_closed(env, gripper_joint_name="finger_joint", env_id=1)


def test_wrong_object_requires_closure_on_contacting_hand(monkeypatch):
    env = _env([[0.0, 1.0]])  # right is slightly closed; left is open
    world = _JointWorld(env, [[0.0, 1.0]])
    contacts = {"left": ["wrong"], "right": ["target"]}
    world.get_objects_in_contact_with = lambda label, _candidates, env_id: contacts[label]
    monkeypatch.setattr(conditionals, "get_world", lambda _env: world)

    assert conditionals.get_wrong_object_grabbed(env, "target", env_id=0) is None
    contacts["right"] = ["wrong"]
    assert conditionals.get_wrong_object_grabbed(env, "target", env_id=0) == "wrong"
