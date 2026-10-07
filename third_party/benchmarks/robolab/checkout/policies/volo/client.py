# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete VoLo proxy clients: existing backends plus orchestrator metadata.

The wire format of each client is a strict superset of its backend's, so a
VoLo proxy can forward the request to an unmodified policy server after
stripping the extra keys.
"""

from policies.cosmos3.client import Cosmos3Client
from policies.pi0_family.client import Pi0DroidJointposClient
from policies.volo.metadata import OrchestratorMetadataMixin


class VoloCosmos3Client(OrchestratorMetadataMixin, Cosmos3Client):
    """Cosmos3 backend served through a VoLo inference proxy."""


class VoloPi0Client(OrchestratorMetadataMixin, Pi0DroidJointposClient):
    """Pi0-family backend served through a VoLo inference proxy.

    Beyond the mixin's metadata, forwards full-resolution copies of both
    camera images (the standard request only carries 224x224 resizes) and the
    end-effector pose from the proprioceptive observation.
    """

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        extracted = super()._extract_observation(raw_obs, env_id=env_id)
        robot_state = raw_obs["proprio_obs"]
        for key in ("ee_pos", "ee_quat"):
            value = robot_state.get(key) if hasattr(robot_state, "get") else None
            if value is not None:
                extracted[key] = value[env_id].clone().detach().cpu().numpy()
        return extracted

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = super()._pack_request(extracted_obs, instruction)
        request["observation/exterior_image_1_left_raw"] = extracted_obs["right_image"]
        request["observation/wrist_image_left_raw"] = extracted_obs["wrist_image"]
        for key in ("ee_pos", "ee_quat"):
            if key in extracted_obs:
                request[f"observation/{key}"] = extracted_obs[key]
        return request
