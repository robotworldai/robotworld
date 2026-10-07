# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "assets" / "robots" / "kinova_gen3_robotiq_2f85"
USD_LAYERS = (
    ASSET_DIR / "kinova_gen3_7dof_robotiq_2f85.usd",
    ASSET_DIR / "configuration" / "kinova_gen3_7dof_robotiq_2f85_base.usd",
    ASSET_DIR / "configuration" / "kinova_gen3_7dof_robotiq_2f85_physics.usd",
    ASSET_DIR / "configuration" / "kinova_gen3_7dof_robotiq_2f85_robot.usd",
    ASSET_DIR / "configuration" / "kinova_gen3_7dof_robotiq_2f85_sensor.usd",
)


def test_usd_asset_is_self_contained():
    assert all(path.is_file() for path in USD_LAYERS)
    assert not (ASSET_DIR / "meshes").exists()


def test_usd_layers_have_no_machine_specific_paths():
    for path in USD_LAYERS:
        contents = path.read_bytes()
        assert b"/home/" not in contents
        assert b"package://" not in contents


def test_asset_licenses_are_included():
    assert (ASSET_DIR / "LICENSE-KINOVA").is_file()
    assert (ASSET_DIR / "LICENSE-ROBOTIQ").is_file()
