# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for APIs used by the external RoboVoLo content pack."""

from __future__ import annotations

from functools import partial

from robolab.core.task import conditionals


def test_pick_and_place_grouped_builds_parallel_destination_ladders():
    subtask = conditionals.pick_and_place_grouped(
        groups=[
            {"object": ["lemon", "lime"], "container": "bowl"},
            {"object": "can", "container": "bin"},
        ],
        logical="all",
        score=0.8,
    )

    assert subtask.name == "pick_and_place_grouped"
    assert subtask.logical == "all"
    assert subtask.score == 0.8
    assert set(subtask.conditions) == {"lemon", "lime", "can"}

    for object_name, container in (("lemon", "bowl"), ("lime", "bowl"), ("can", "bin")):
        ladder = subtask.conditions[object_name]
        assert len(ladder) == 4
        assert [score for _, score in ladder] == [0.25, 0.25, 0.25, 0.25]
        assert all(isinstance(func, partial) for func, _ in ladder)
        assert ladder[-1][0].keywords == {
            "object": object_name,
            "container": container,
            "tolerance": 0.01,
        }


def test_pick_and_place_grouped_preserves_choose_configuration():
    subtask = conditionals.pick_and_place_grouped(
        groups=[{"object": ["a", "b"], "container": "bin"}],
        logical="choose",
        K=1,
    )

    assert subtask.logical == "choose"
    assert subtask.K == 1
