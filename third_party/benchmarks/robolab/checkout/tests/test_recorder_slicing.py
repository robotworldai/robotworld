# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for recorder env-slicing across torch dtypes.

Reproduces the crash seen on the IsaacSim 5.1 / torch-cu128 stack when
recording with ``--enable-subtask``: the subtask status codes are a uint16
tensor and, when resident on CUDA, ``tensor[env_ids]`` raises
``RuntimeError: "index_cuda" not implemented for 'UInt16'``. The other tests
never exercised this because they don't enable subtask recording, so the
failure only showed up on OSMO. ``_slice_to_envs`` must slice such tensors
without raising and preserve their dtype.
"""

import pytest
import torch

from robolab.core.logging.recorder_manager import _slice_to_envs


@pytest.mark.parametrize("dtype", [torch.uint16, torch.float32, torch.int64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_slice_to_envs_preserves_values_and_dtype(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    env_ids = torch.tensor([0, 2], device=device)
    # Nested dict mirrors the real recorder payload (e.g. subtask/status).
    value = {
        "subtask": {
            "status": torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]],
                                    dtype=dtype, device=device),
        },
    }

    sliced = _slice_to_envs(value, env_ids)

    out = sliced["subtask"]["status"]
    assert out.dtype == dtype
    assert out.shape == (2, 2)
    assert out.cpu().tolist() == [[10, 11], [30, 31]]
