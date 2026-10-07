"""Coordinate-chain verification for the OpenWAM XPolicyLab adapter (no GPU).

Checks, element-wise on random valid poses:
  1. Adapter obs->EEF20 == official openwam.dataloader RoboDojo contract.
  2. EEF20 -> native ee dicts -> EEF20 round trip is the identity.
  3. world -> base -> world pose round trip is the identity.
  4. Prompt template matches training ``format_prompt_for_inference``.

The adapter defaults to the OpenWAM source vendored at
``policy/OpenWAM/OpenWAM/`` and prepends it to sys.path. Run from a checkout
where ``XPolicyLab`` is importable:

  PYTHONPATH=<repo_root> python verify_frames_contract.py
"""

from __future__ import annotations

import sys

import numpy as np


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    pos = rng.uniform(-0.6, 0.6, size=3) + np.array([0.0, -0.2, 0.9])
    return np.concatenate([pos, quat])


def main() -> int:
    from XPolicyLab.policy.OpenWAM.model import Model

    model = Model(
        {
            "action_type": "ee",
            "env_cfg_type": "arx_x5",
            "allow_dummy_policy": True,
        }
    )

    import openwam

    print(f"[openwam] imported from {openwam.__file__}")

    # Training-side contract (official OpenWAM). The standalone
    # benchmarks/robodojo frames helpers were removed from official main;
    # the adapter already uses these same modules.
    from openwam.dataloader.robodojo_contract import arx_x5_calibration as pinned_calibration
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference as pinned_prompt
    from openwam.dataloader.utils.poses import (
        arms_to_eef20 as pinned_arms_to_eef20,
        env_relative_world_to_robot_base as pinned_world_to_base,
    )

    rng = np.random.default_rng(7)
    calib = pinned_calibration()
    n = 200
    max_state_diff = 0.0
    max_roundtrip_diff = 0.0
    max_pose_rt_diff = 0.0

    for _ in range(n):
        left_pose, right_pose = _random_pose(rng), _random_pose(rng)
        left_grip = rng.uniform(0.0, 1.0, size=1)
        right_grip = rng.uniform(0.0, 1.0, size=1)
        state = {
            "left_ee_pose": left_pose,
            "left_ee_joint_state": left_grip,
            "right_ee_pose": right_pose,
            "right_ee_joint_state": right_grip,
        }

        # 1. adapter (canonical chain) vs pinned benchmarks chain
        adapter_eef20 = model._state_to_eef20(state)
        pinned_eef20 = pinned_arms_to_eef20(
            pinned_world_to_base(
                left_pose,
                calib["arms"]["left"]["base_pos_relative_to_env_origin"],
                calib["arms"]["left"]["base_quat_wxyz"],
            ),
            left_grip,
            pinned_world_to_base(
                right_pose,
                calib["arms"]["right"]["base_pos_relative_to_env_origin"],
                calib["arms"]["right"]["base_quat_wxyz"],
            ),
            right_grip,
        )
        max_state_diff = max(max_state_diff, float(np.abs(adapter_eef20 - pinned_eef20).max()))

        # 2. EEF20 -> native dicts -> EEF20 round trip
        native = model._eef20_chunk_to_native(adapter_eef20[None, :])[0]
        recovered = model._state_to_eef20(native)
        max_roundtrip_diff = max(max_roundtrip_diff, float(np.abs(recovered - adapter_eef20).max()))

        # 3. world -> base -> world pose round trip (rotation-invariant compare)
        world_again = np.concatenate([native["left_ee_pose"], native["right_ee_pose"]])
        orig = np.concatenate([left_pose, right_pose])
        for off in (0, 7):
            p0, q0 = orig[off : off + 3], orig[off + 3 : off + 7]
            p1, q1 = world_again[off : off + 3], world_again[off + 3 : off + 7]
            dq = min(np.abs(q1 - q0).max(), np.abs(q1 + q0).max())  # q ~ -q
            max_pose_rt_diff = max(max_pose_rt_diff, float(np.abs(p1 - p0).max()), float(dq))

    # 4. prompt template byte-for-byte
    probe = "stack the bowls on the plate"
    prompt_ok = model._format_prompt(probe) == pinned_prompt(probe)

    print(f"[1] adapter vs pinned EEF20      max_abs = {max_state_diff:.3e}")
    print(f"[2] EEF20 <-> native round trip  max_abs = {max_roundtrip_diff:.3e}")
    print(f"[3] world <-> base round trip    max_abs = {max_pose_rt_diff:.3e}")
    print(f"[4] prompt template byte-equal   {prompt_ok}")

    # float32 cast in the adapter state path bounds [1]/[2] at ~1e-7 relative.
    ok = max_state_diff < 1e-6 and max_roundtrip_diff < 1e-5 and max_pose_rt_diff < 1e-5 and prompt_ok
    print("FRAMES CONTRACT " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
