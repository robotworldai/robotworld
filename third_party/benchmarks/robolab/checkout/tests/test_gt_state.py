# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ground-truth state exporter (no simulator required)."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
from robolab.core.logging.recorder_manager import RobolabRecorderManager
from robolab.eval.gt_state import GroundTruthStateExporter

NUM_ENVS = 2


class _FakeSubtaskTerm(SubtaskCompletionRecorderTerm):
    """Real type (so isinstance-based lookup works) without the heavy init."""

    def __init__(self, num_envs: int):
        self.infos = [
            {"status": 0, "completed": eid, "total": 3, "info": f"env{eid}", "score": 0.5 * eid}
            for eid in range(num_envs)
        ]
        conditions = {"banana": [(lambda env, env_id: env_id == 1, 1.0)]}
        self.subtask_state_machines = [
            SimpleNamespace(
                subtasks=[SimpleNamespace(conditions=conditions, logical="all")],
                conditionals_state_machine=None,
                current_subtask_index=0,
            )
            for _ in range(num_envs)
        ]


def _make_manager(term) -> RobolabRecorderManager:
    manager = RobolabRecorderManager(None, None)
    manager._terms = {"subtask": term} if term is not None else {}
    return manager


class _FakeWorld:
    """Per-env poses so cross-env leakage is detectable."""

    def __init__(self):
        self.contact_queries: list[int] = []

    def get_body(self, name):
        return object()

    def get_pose(self, name, is_relative=True, env_id=None):
        pos = torch.tensor([1.0 * env_id, 0.0, 0.1 * (env_id + 1)])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return pos, quat

    def get_velocity(self, name, env_id=None):
        return torch.zeros(6)

    def get_articulation(self, name):
        body_pos = torch.zeros(NUM_ENVS, 1, 3)
        body_pos[1, 0, 0] = 5.0
        return SimpleNamespace(
            data=SimpleNamespace(
                body_names=["base_link"],
                body_pos_w=body_pos,
                body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * NUM_ENVS),
                joint_names=["finger_joint"],
                # Env 0 open, env 1 fully closed.
                joint_pos=torch.tensor([[0.0], [np.pi / 4]]),
            )
        )

    def get_objects_in_contact_with(self, body, candidates, env_id=None):
        self.contact_queries.append(env_id)
        return ["banana"] if env_id == 1 else []


def _make_env(term) -> SimpleNamespace:
    return SimpleNamespace(
        recorder_manager=_make_manager(term),
        scene=SimpleNamespace(env_origins=torch.zeros(NUM_ENVS, 3)),
    )


def _make_exporter(monkeypatch, term) -> tuple[GroundTruthStateExporter, _FakeWorld]:
    world = _FakeWorld()
    monkeypatch.setattr("robolab.core.world.world_state.get_world", lambda env: world)
    env_cfg = SimpleNamespace(contact_object_list=["banana", "table", "robot"])
    return GroundTruthStateExporter(_make_env(term), env_cfg), world


def test_get_term_returns_matching_term_or_none():
    term = _FakeSubtaskTerm(NUM_ENVS)
    assert _make_manager(term).get_term(SubtaskCompletionRecorderTerm) is term
    assert _make_manager(None).get_term(SubtaskCompletionRecorderTerm) is None


def test_fixtures_are_excluded_from_scene_objects(monkeypatch):
    exporter, _ = _make_exporter(monkeypatch, _FakeSubtaskTerm(NUM_ENVS))
    assert exporter._object_names == ["banana"]


def test_export_all_is_per_env(monkeypatch):
    exporter, world = _make_exporter(monkeypatch, _FakeSubtaskTerm(NUM_ENVS))

    states = exporter.export_all([0, 1])

    assert set(states) == {0, 1}
    # Object pose comes from each env's own row, not env 0's.
    np.testing.assert_allclose(states[0]["objects"]["banana"]["pos"], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(states[1]["objects"]["banana"]["pos"], [1.0, 0.0, 0.2])
    # The snapshot is raw: no derived lift/grasp fields (those are the
    # consumer's business, e.g. the VoLo metadata mixin).
    assert set(states[0]["objects"]["banana"]) == {"pos", "quat", "vel"}
    assert set(states[0]["robot"]) == {"ee_pos", "ee_quat", "gripper_closedness", "objects_in_contact"}
    # Robot state: env 1's EE offset and closed gripper, env 0 open.
    np.testing.assert_allclose(states[1]["robot"]["ee_pos"], [5.0, 0.0, 0.0])
    assert states[0]["robot"]["gripper_closedness"] == 0.0
    assert states[1]["robot"]["gripper_closedness"] == 1.0
    # Contacts are reported per env regardless of gripper closure.
    assert world.contact_queries == [0, 1]
    assert states[0]["robot"]["objects_in_contact"] == []
    assert states[1]["robot"]["objects_in_contact"] == ["banana"]
    # Subtask block reflects each env's own recorder info and live conditions.
    assert states[0]["subtask"]["completed"] == 0
    assert states[1]["subtask"]["completed"] == 1
    assert states[0]["subtask"]["all_subtask_conditions"] == {"subtask_0": False}
    assert states[1]["subtask"]["all_subtask_conditions"] == {"subtask_0": True}
    # Step counter advances once per export_all call, not per env.
    assert states[0]["step"] == states[1]["step"] == 1
    assert exporter.export_all([0])[0]["step"] == 2


def test_missing_recorder_yields_empty_subtask_state(monkeypatch):
    exporter, _ = _make_exporter(monkeypatch, None)

    state = exporter.export_all([0])[0]

    assert state["subtask"]["completed"] == 0
    assert state["subtask"]["conditions"] == []


def test_group_object_keys_resolution(monkeypatch):
    exporter, _ = _make_exporter(monkeypatch, _FakeSubtaskTerm(NUM_ENVS))

    def stacked(env=None, env_id=None, objects=None):
        return True

    # Real scene object: the group name is the key.
    assert exporter._group_object_keys("banana", [], fallback="x") == ["banana"]
    # Generic group name: objects recovered from the partial's keywords.
    cond_list = [(partial(stacked, objects=["red_block", "blue_block"]), 1.0)]
    assert exporter._group_object_keys("conditions", cond_list, fallback="x") == ["red_block", "blue_block"]
    # Nothing to recover: fall back to the provided label.
    assert exporter._group_object_keys("conditions", [], fallback="subtask_0:conditions") == ["subtask_0:conditions"]
