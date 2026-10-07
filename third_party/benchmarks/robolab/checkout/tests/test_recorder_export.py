# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for HDF5 export of non-tensor recorder leaves.

On the IsaacSim 5.1 / IsaacLab 2.3 stack some ``initial_state`` leaves arrive as
Python lists (or lists of tensors) instead of a single stacked tensor. The
streaming exporter used to call ``value.cpu()`` unconditionally, so export
crashed with ``AttributeError: 'list' object has no attribute 'cpu'`` during the
per-env reset export of a policy-driven eval (never hit by the no-policy runner,
which is why the suite missed it). ``_append_to_dataset`` must coerce such
leaves and still write correct data.
"""

import h5py
import numpy as np
import torch

from robolab.core.logging.streaming_hdf5_handler import StreamingHDF5DatasetFileHandler as Handler


def test_append_to_dataset_coerces_nontensor_leaves(tmp_path):
    path = tmp_path / "episode.hdf5"
    # Nested dict mirroring initial_state/<group>/<entity>/<field>, mixing a
    # normal tensor leaf, a plain Python list leaf, and a list-of-tensors leaf.
    value = {
        "articulation": {"robot": {"joint_position": torch.zeros(1, 7)}},
        "rigid_object": {"banana": {"root_pose": [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]]}},
        "cameras": {"wrist": {"position": [torch.tensor([1.0, 2.0, 3.0])]}},
    }

    with h5py.File(path, "w") as f:
        group = f.create_group("demo_0")
        Handler._append_to_dataset(group, "initial_state", value, {})

    with h5py.File(path, "r") as f:
        assert f["demo_0/initial_state/articulation/robot/joint_position"].shape == (1, 7)
        rp = f["demo_0/initial_state/rigid_object/banana/root_pose"][()]
        assert rp.shape == (1, 7)
        np.testing.assert_allclose(rp[0, :3], [0.1, 0.2, 0.3], rtol=1e-5)
        pos = f["demo_0/initial_state/cameras/wrist/position"][()]
        assert pos.shape == (1, 3)
        np.testing.assert_allclose(pos[0], [1.0, 2.0, 3.0], rtol=1e-5)
