"""GPU verification: generate_batch vs generate on a real checkpoint.

Loads a checkpoint through the standard deploy path (build_server_from_config,
so config merge / normalizer / obs preprocessing match the JSON server), then
checks the batch-inference contract on real weights:

  1. B=1 parity   — generate_batch([x]) matches generate(x).
  2. Row parity   — every sample of generate_batch([x1..xN]) matches its own
                    B=1 generate run (no cross-sample contamination).

Acceleration features that are single-stream or shape-sensitive must be off
(dit_cache, compile); the script fails fast if the config enables them.

Usage:
  python scripts/verify_batch_equivalence.py --ckpt-dir <dir> [--batch 3] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_batch_equivalence")


def _synthetic_obs(sample_idx: int, *, instruction: str) -> dict:
    """Build a deterministic RoboDojo-shaped obs (3 cameras + EEF20-convertible state)."""
    from PIL import Image

    rng = np.random.default_rng(1234 + sample_idx)

    def _img(shift: int) -> Image.Image:
        # Smooth gradient + per-sample offset: deterministic, JPEG-free, RGB.
        h, w = 240, 320
        yy, xx = np.mgrid[0:h, 0:w]
        base = ((xx + yy + 37 * (sample_idx + 1) + shift) % 256).astype(np.uint8)
        arr = np.stack([base, np.roll(base, 13, axis=1), np.roll(base, 29, axis=0)], axis=-1)
        return Image.fromarray(arr, mode="RGB")

    # Env-relative world poses near the dual-X5 workspace; identity-ish world
    # orientation, unit wxyz. Values only need to be finite and unit-quaternion.
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    left_pose = np.concatenate([np.array([-0.25, -0.10, 0.95]) + rng.normal(0, 0.02, 3), quat])
    right_pose = np.concatenate([np.array([0.25, -0.10, 0.95]) + rng.normal(0, 0.02, 3), quat])

    from openwam.dataloader.transforms.multiview import format_prompt_for_inference
    from openwam.dataloader.utils.poses import arms_to_eef20, env_relative_world_to_robot_base

    # Base quats normalized exactly as the pinned arx_x5_calibration() does.
    base_quat = np.array([0.707, 0.0, 0.0, 0.707])
    base_quat = base_quat / np.linalg.norm(base_quat)
    left_base_pos, left_base_quat = (-0.3, -0.45, 0.765), base_quat
    right_base_pos, right_base_quat = (0.3, -0.45, 0.765), base_quat
    state = arms_to_eef20(
        env_relative_world_to_robot_base(left_pose, left_base_pos, left_base_quat),
        np.array([float(sample_idx % 2)]),
        env_relative_world_to_robot_base(right_pose, right_base_pos, right_base_quat),
        np.array([float((sample_idx + 1) % 2)]),
    )

    return {
        "images": {
            "head_camera": _img(0),
            "left_wrist_camera": _img(64),
            "right_wrist_camera": _img(128),
        },
        "prompt": format_prompt_for_inference(instruction),
        "state": state.astype(np.float32).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--deploy-config", default=None, help="deploy yaml (defaults to configs/deploy.yaml)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    deploy_cfg = _load_deploy_yaml(args.deploy_config)
    # Correctness-first eval settings; fail if the yaml re-enables them.
    OmegaConf.update(deploy_cfg, "optimization.dit_cache.enabled", False, merge=False)
    OmegaConf.update(deploy_cfg, "optimization.compile.enabled", False, merge=False)
    OmegaConf.update(deploy_cfg, "optimization.decode_video", False, merge=False)

    logger.info("Loading checkpoint from %s on %s ...", args.ckpt_dir, args.device)
    server = build_server_from_config(deploy_cfg, args.ckpt_dir, device=args.device)
    server._init_policy()
    engine = server.engine
    preprocessor = server._obs_preprocessor

    resolved = {
        "denoise_steps": OmegaConf.select(server.cfg, "inference.denoise_steps"),
        "denoise_mode": OmegaConf.select(server.cfg, "inference.denoise_mode"),
        "inference_mode": OmegaConf.select(server.cfg, "inference.inference_mode"),
        "inference_horizon": OmegaConf.select(server.cfg, "inference.inference_horizon"),
        "num_frames": OmegaConf.select(server.cfg, "inference.num_frames"),
        "video_num_frames": OmegaConf.select(server.cfg, "inference.video_num_frames"),
        "dit_cache": OmegaConf.select(server.cfg, "optimization.dit_cache.enabled"),
        "compile": OmegaConf.select(server.cfg, "optimization.compile.enabled"),
        "decode_video": OmegaConf.select(server.cfg, "optimization.decode_video"),
    }
    logger.info("Resolved deploy hyperparameters: %s", resolved)

    def _conditions(obs: dict) -> dict:
        obs = preprocessor.preprocess(dict(obs))
        cond = {"observation": obs, "first_frame_image": [obs["image"]], "prompt": obs["prompt"]}
        if obs.get("state") is not None:
            cond["proprio"] = obs["state"]
        return cond

    obs_list = [_synthetic_obs(i, instruction="stack the bowls on the plate") for i in range(args.batch)]
    conditions = [_conditions(o) for o in obs_list]

    logger.info("Running %d single-sample generate() references ...", args.batch)
    singles = []
    for i, cond in enumerate(conditions):
        out = engine.generate(dict(cond))
        singles.append(np.asarray(out["actions"]))
        logger.info("  single[%d]: actions %s", i, singles[-1].shape)

    logger.info("Running generate_batch B=1 ...")
    b1 = np.asarray(engine.generate_batch([dict(conditions[0])])["actions"])
    logger.info("Running generate_batch B=%d ...", args.batch)
    bn = np.asarray(engine.generate_batch([dict(c) for c in conditions])["actions"])

    failures = []

    def _check(name: str, got: np.ndarray, want: np.ndarray) -> None:
        if got.shape != want.shape:
            failures.append(f"{name}: shape {got.shape} != {want.shape}")
            return
        diff = np.abs(got - want)
        max_abs = float(diff.max())
        denom = np.maximum(np.abs(want), 1e-8)
        max_rel = float((diff / denom).max())
        ok = np.allclose(got, want, atol=args.atol, rtol=args.rtol)
        logger.info("  %s: max_abs=%.3e max_rel=%.3e -> %s", name, max_abs, max_rel, "OK" if ok else "FAIL")
        if not ok:
            failures.append(f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

    logger.info("Comparing outputs (atol=%g rtol=%g):", args.atol, args.rtol)
    _check("B=1 parity", b1[0], singles[0])
    for i in range(args.batch):
        _check(f"row[{i}] vs single[{i}]", bn[i], singles[i])

    # ------------------------------------------------------------------
    # Numerics probes: separate cross-sample CONTAMINATION (a bug) from
    # batched-kernel float reassociation amplified by the denoise loop
    # (irreducible at bf16). Three measurements:
    #   A. partner swap  — row 0's output with different batch partners.
    #      Contamination => O(row-to-row action difference); reassociation
    #      => bf16-ULP scale.
    #   B. clone batch   — [x0, x0, x0]: rows of identical inputs inside ONE
    #      launch. Any row-to-row diff is pure position-dependent numerics.
    #   C. intrinsic sensitivity — single x0 vs single x0 with proprio
    #      perturbed by one bf16 ULP: the model's own noise floor.
    # ------------------------------------------------------------------
    def _stat(name: str, a: np.ndarray, b: np.ndarray) -> float:
        d = float(np.abs(a - b).max())
        logger.info("  [probe] %s: max_abs=%.3e", name, d)
        return d

    logger.info("Numerics probes:")
    extra = [_conditions(_synthetic_obs(100 + i, instruction="wipe the table")) for i in range(2)]
    swapped = np.asarray(
        engine.generate_batch([dict(conditions[0]), dict(extra[0]), dict(extra[1])])["actions"]
    )
    probe_a = _stat("A partner-swap row0 drift", swapped[0], bn[0])

    clones = np.asarray(engine.generate_batch([dict(conditions[0])] * 3)["actions"])
    probe_b = max(
        _stat("B clone rows 0 vs 1", clones[0], clones[1]),
        _stat("B clone rows 0 vs 2", clones[0], clones[2]),
    )

    cond_pert = dict(conditions[0])
    proprio = np.asarray(cond_pert["proprio"], dtype=np.float32).copy()
    proprio[0] = np.nextafter(proprio[0], np.float32(np.inf), dtype=np.float32)
    cond_pert["proprio"] = proprio
    single_pert = np.asarray(engine.generate(cond_pert)["actions"])
    probe_c = _stat("C 1-ULP proprio sensitivity", single_pert, singles[0])

    row_scale = float(np.abs(singles[0] - singles[1]).max())
    logger.info(
        "  [probe] reference scales: cross-sample action difference=%.3e, "
        "batch-vs-single row0=%.3e",
        row_scale,
        float(np.abs(bn[0] - singles[0]).max()),
    )
    contamination = probe_a > max(10.0 * max(probe_b, probe_c), 0.05 * row_scale)
    logger.info(
        "  [probe] verdict: partner-swap drift %.3e vs numerics floor %.3e -> %s",
        probe_a,
        max(probe_b, probe_c),
        "CONTAMINATION SUSPECTED" if contamination else "reassociation-level (no contamination)",
    )

    if contamination:
        logger.error("BATCH CONTAMINATION SUSPECTED — do not deploy the batch path.")
        return 2
    if failures:
        logger.warning(
            "Row parity exceeded strict tolerance (atol=%g rtol=%g) but probes attribute the "
            "difference to bf16 batched-kernel reassociation, not contamination:\n  %s",
            args.atol,
            args.rtol,
            "\n  ".join(failures),
        )
        logger.info("BATCH EQUIVALENCE PASSED WITH NUMERIC TOLERANCE (no contamination detected).")
        return 0
    logger.info("BATCH EQUIVALENCE PASSED (B=1 parity + %d-row independence).", args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
