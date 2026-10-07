# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every task scene authors its ground plane at its locked height.

The task table is a dynamic rigid body resting on the ground plane, so the
authored ground height sets the tabletop height: changing it drops or raises
the work surface and silently invalidates every recording made in that scene
(the robot replays actions aimed at the old heights). Ground heights are
therefore locked per scene rather than freely editable.

New scenes must use CANONICAL_GROUND_Z. LEGACY_GROUND_Z lists scenes that
predate the canonical height and keep their original -0.65 ground to preserve
replay compatibility with existing recordings. Floor-standing robots follow
the per-scene ground automatically via the ``root_z_above_ground`` robot
label (see robolab.core.environments.scene_fixture), so legacy heights are
fully supported.
"""

import glob
import os

import pytest
from pxr import Usd, UsdGeom

from robolab.constants import SCENE_DIR

CANONICAL_GROUND_Z = -0.697

# Scenes recorded against a -0.65 ground before the canonical height existed.
# Their tabletop height (and thus every recording made in them) depends on it.
LEGACY_GROUND_Z = {
    "banana_crate.usda": -0.65,
    "bananas_5_grey_bin.usda": -0.65,
    "bananas_5_in_crate.usda": -0.65,
    "bottles_crate.usda": -0.65,
    "conditionals_test_scene.usda": -0.65,
    "mug_banana_ketchup_bowl_rubiks3_bin.usda": -0.65,
    "mugs2_bananas2_ketchup_rubiks3_bin.usda": -0.65,
    "mugs4_measuringcup_drill_bowl.usda": -0.65,
    "mugs4_measuringcup_drill_bowl_v2.usda": -0.65,
    "rubiks_cube_2_bowl.usda": -0.65,
    "rubiks_cube_banana_bowl.usda": -0.65,
}


def _scene_files() -> list[str]:
    return sorted(glob.glob(os.path.join(SCENE_DIR, "*.usd*")))


def _authored_ground_z(usd_path: str) -> float | None:
    stage = Usd.Stage.Open(usd_path)
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        return None
    ground = stage.GetPrimAtPath(default_prim.GetPath().AppendChild("GroundPlane"))
    if not ground.IsValid():
        return None
    transform = UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return float(transform.ExtractTranslation()[2])


def test_every_scene_authors_locked_ground():
    scene_files = _scene_files()
    assert scene_files, f"no scenes found under {SCENE_DIR}"
    missing = []
    wrong = []
    for usd_path in scene_files:
        name = os.path.basename(usd_path)
        expected_z = LEGACY_GROUND_Z.get(name, CANONICAL_GROUND_Z)
        ground_z = _authored_ground_z(usd_path)
        if ground_z is None:
            missing.append(name)
        elif ground_z != pytest.approx(expected_z, abs=1e-6):
            wrong.append(f"{name}: {ground_z} (expected {expected_z})")
    assert not missing, f"scenes without /GroundPlane: {missing}"
    assert not wrong, f"scenes with wrong ground height: {wrong}"
