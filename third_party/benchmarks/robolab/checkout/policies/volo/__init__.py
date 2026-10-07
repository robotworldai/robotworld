# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoLo inference-proxy backend.

Wraps existing policy backends (cosmos3, pi0 family) with the extra simulator
metadata a VoLo orchestrator proxy consumes: depth images, camera calibration,
front RGB, episode IDs, and opt-in ground-truth state.

Concrete clients live in :mod:`policies.volo.client` (imports the backend
client libraries, e.g. openpi-client); the backend-independent mixin lives in
:mod:`policies.volo.metadata`.
"""
