# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in depth camera flag (registration-time)."""

from robolab.core.observations.observation_utils import generate_image_obs_from_cameras
from robolab.variations.camera import OverShoulderLeftCameraCfg, with_depth


def test_with_depth_returns_variant_and_preserves_original():
    depth_cls = with_depth(OverShoulderLeftCameraCfg)

    assert depth_cls is not OverShoulderLeftCameraCfg
    assert depth_cls().over_shoulder_left_camera.data_types == ["rgb", "depth"]
    # The module-level class must stay rgb-only so standard registrations
    # keep their render cost.
    assert OverShoulderLeftCameraCfg().over_shoulder_left_camera.data_types == ["rgb"]


def test_with_depth_is_idempotent():
    depth_cls = with_depth(OverShoulderLeftCameraCfg)
    assert with_depth(depth_cls) is depth_cls


def test_depth_cameras_get_matching_observation_terms():
    rgb_obs = generate_image_obs_from_cameras([OverShoulderLeftCameraCfg])()
    for term_suffix in ("depth", "pos", "quat", "K"):
        assert not hasattr(rgb_obs, f"over_shoulder_left_camera_{term_suffix}")

    depth_obs = generate_image_obs_from_cameras([with_depth(OverShoulderLeftCameraCfg)])()
    assert hasattr(depth_obs, "over_shoulder_left_camera")
    assert hasattr(depth_obs, "over_shoulder_left_camera_depth")
    assert depth_obs.over_shoulder_left_camera_depth.params["data_type"] == "depth"
    # Depth-enabled cameras also expose calibration through the obs pipeline.
    for term_suffix in ("pos", "quat", "K"):
        assert hasattr(depth_obs, f"over_shoulder_left_camera_{term_suffix}")
