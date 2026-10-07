# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whack-a-Mole event handlers: popup schedule, press detection, mole position management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# -- Constants ----------------------------------------------------------------

NUM_MOLES = 5
NUM_WINDOWS = 1
MOLE_NAMES = ["mole_0", "mole_1", "mole_2", "mole_3", "mole_4"]

# Mole XY positions relative to the environment origin (quincunx layout).
MOLE_XY_POSITIONS = [
    (0.55, 0.00),   # Center
    (0.55, 0.10),   # Front  (+y)
    (0.55, -0.10),  # Back   (-y)
    (0.45, 0.00),   # Left   (-x, closer to robot)
    (0.65, 0.00),   # Right  (+x, farther from robot)
]

TABLE_HEIGHT = 0.20
BOARD_THICKNESS = 0.02
BOARD_SURFACE_Z = TABLE_HEIGHT + BOARD_THICKNESS
MOLE_DOWN_Z = BOARD_SURFACE_Z - 0.005
MOLE_UP_Z = BOARD_SURFACE_Z + 0.010

# Press detection thresholds
PRESS_XY_THRESHOLD = 0.035   # EE within 3.5 cm of mole centre (XY)
PRESS_Z_MAX = BOARD_SURFACE_Z + 0.045
PRESS_Z_MIN = BOARD_SURFACE_Z - 0.010
PRESS_DWELL_STEPS = 2        # 2 control steps approximately 80 ms at 25 Hz
POST_HIT_COMPLETE_STEPS = 3  # keep a few frames after a valid hit before ending

GRIPPER_CLOSED_WIDTH_THRESHOLD = 0.03
MOLE_IDLE_COLOR = (0.95, 0.85, 0.20)
MOLE_ACTIVE_COLOR = (0.95, 0.18, 0.18)


# -- Helpers ------------------------------------------------------------------

def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if env_ids.dim() == 0:
        return env_ids.unsqueeze(0)
    return env_ids


def _ensure_mole_materials(env: ManagerBasedEnv) -> None:
    """Cache each mole's spawn-time shader diffuseColor input per (env, mole).

    We intentionally do NOT create new materials or swap material bindings.
    Instead we write directly to the shader inputs that already drive the
    cylinder's appearance, so the RTX renderer does not treat a state change
    as a material re-binding (which otherwise triggers a visible temporal
    fade from yellow to red under the path tracer).
    """
    if hasattr(env, "_wam_mole_shader_inputs"):
        return

    try:
        import omni.usd
        from pxr import Usd, UsdGeom, UsdShade
    except Exception:
        env._wam_mole_shader_inputs = None
        return

    stage = omni.usd.get_context().get_stage()
    inputs_map: dict[tuple[int, int], list] = {}

    for env_id in range(env.num_envs):
        for mole_id in range(NUM_MOLES):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Mole{mole_id}")
            if not prim.IsValid():
                continue
            shader_inputs: list = []
            for child in Usd.PrimRange(prim):
                if not child.IsA(UsdGeom.Gprim):
                    continue
                binding_api = UsdShade.MaterialBindingAPI(child)
                material, _ = binding_api.ComputeBoundMaterial()
                if not material:
                    continue
                surface_output = material.GetSurfaceOutput()
                if not surface_output:
                    continue
                connection = surface_output.GetConnectedSource()
                if not connection:
                    continue
                shader = UsdShade.Shader(connection[0])
                for name in ("diffuseColor", "baseColor"):
                    inp = shader.GetInput(name)
                    if inp:
                        shader_inputs.append(inp)
            if shader_inputs:
                inputs_map[(env_id, mole_id)] = shader_inputs

    env._wam_mole_shader_inputs = inputs_map if inputs_map else None


def _set_mole_visual_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mole_ids: torch.Tensor,
    is_active: bool,
) -> None:
    if len(env_ids) == 0:
        return

    _ensure_mole_materials(env)
    inputs_map = getattr(env, "_wam_mole_shader_inputs", None)
    if not inputs_map:
        return

    try:
        from pxr import Gf
    except Exception:
        return

    color = Gf.Vec3f(*(MOLE_ACTIVE_COLOR if is_active else MOLE_IDLE_COLOR))

    env_ids_list = env_ids.to(dtype=torch.int64).cpu().tolist()
    mole_ids_list = mole_ids.to(dtype=torch.int64).cpu().tolist()
    for env_id, mole_id in zip(env_ids_list, mole_ids_list, strict=False):
        inputs = inputs_map.get((env_id, mole_id))
        if not inputs:
            continue
        for inp in inputs:
            inp.Set(color)


def _set_all_moles_inactive(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    env_ids = _resolve_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    tiled_env_ids = env_ids.repeat_interleave(NUM_MOLES)
    tiled_mole_ids = torch.arange(NUM_MOLES, device=env.device, dtype=torch.int64).repeat(len(env_ids))
    _set_mole_visual_state(env, tiled_env_ids, tiled_mole_ids, is_active=False)


def _ensure_whack_a_mole_buffers(env: ManagerBasedEnv) -> None:
    """Lazily initialise all task-specific tensors on the env."""
    if hasattr(env, "_wam_initialised"):
        return

    N = env.num_envs
    D = env.device

    # Episode-level state
    env.task_phase = torch.zeros(N, dtype=torch.int32, device=D)
    env.episode_timer = torch.zeros(N, dtype=torch.float32, device=D)

    # Popup schedule (generated on reset)
    env.popup_mole_ids = torch.zeros(N, NUM_WINDOWS, dtype=torch.int32, device=D)
    env.popup_start_times = torch.zeros(N, NUM_WINDOWS, dtype=torch.float32, device=D)
    env.popup_durations = torch.zeros(N, NUM_WINDOWS, dtype=torch.float32, device=D)

    # Window tracking
    env.current_window_idx = torch.zeros(N, dtype=torch.int32, device=D)
    env.active_mole_id = torch.full((N,), -1, dtype=torch.int32, device=D)
    env.window_active = torch.zeros(N, dtype=torch.bool, device=D)

    # Scoring
    env.valid_hits = torch.zeros(N, dtype=torch.int32, device=D)
    env.total_popups = torch.zeros(N, dtype=torch.int32, device=D)
    env.success_rate = torch.zeros(N, dtype=torch.float32, device=D)

    # Press detection
    env.press_dwell_counter = torch.zeros(N, dtype=torch.int32, device=D)
    env.window_hit_registered = torch.zeros(N, dtype=torch.bool, device=D)
    env.post_hit_complete_counter = torch.zeros(N, dtype=torch.int32, device=D)

    # Mole visual state (z-height per mole)
    env.mole_heights = torch.full((N, NUM_MOLES), MOLE_DOWN_Z, dtype=torch.float32, device=D)

    # Mole reference positions in env-local frame (set once during reset)
    env.mole_positions_local = torch.zeros(N, NUM_MOLES, 3, dtype=torch.float32, device=D)

    # Per-window diagnostic stats (reset at window activation)
    env._wam_win_min_xy = torch.full((N,), float("inf"), dtype=torch.float32, device=D)
    env._wam_win_min_z = torch.full((N,), float("inf"), dtype=torch.float32, device=D)
    env._wam_win_max_z = torch.full((N,), float("-inf"), dtype=torch.float32, device=D)
    env._wam_win_in_xy_ticks = torch.zeros(N, dtype=torch.int32, device=D)
    env._wam_win_in_z_ticks = torch.zeros(N, dtype=torch.int32, device=D)
    env._wam_win_both_ticks = torch.zeros(N, dtype=torch.int32, device=D)
    env._wam_win_total_ticks = torch.zeros(N, dtype=torch.int32, device=D)
    env._wam_win_max_dwell = torch.zeros(N, dtype=torch.int32, device=D)

    # Track EE and mole positions at closest XY approach for diagnostics
    env._wam_win_ee_at_closest = torch.zeros(N, 3, dtype=torch.float32, device=D)
    env._wam_win_mole_xy_ref = torch.zeros(N, 2, dtype=torch.float32, device=D)

    env._wam_initialised = True


# -- Reset events -------------------------------------------------------------

def reset_task_phase(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset all whack-a-mole state and generate a new popup schedule."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_whack_a_mole_buffers(env)

    N = len(env_ids)
    D = env.device

    # -- zero-out state --
    env.task_phase[env_ids] = 0
    env.episode_timer[env_ids] = 0.0
    env.current_window_idx[env_ids] = 0
    env.active_mole_id[env_ids] = -1
    env.window_active[env_ids] = False
    env.valid_hits[env_ids] = 0
    env.total_popups[env_ids] = 0
    env.success_rate[env_ids] = 0.0
    env.press_dwell_counter[env_ids] = 0
    env.window_hit_registered[env_ids] = False
    env.post_hit_complete_counter[env_ids] = 0
    env.mole_heights[env_ids] = MOLE_DOWN_Z

    # -- store fixed mole XY reference positions --
    for i, (x, y) in enumerate(MOLE_XY_POSITIONS):
        env.mole_positions_local[env_ids, i, 0] = x
        env.mole_positions_local[env_ids, i, 1] = y
        env.mole_positions_local[env_ids, i, 2] = BOARD_SURFACE_Z

    # -- generate random popup schedule --
    env.popup_mole_ids[env_ids] = torch.randint(
        0, NUM_MOLES, (N, NUM_WINDOWS), device=D, dtype=torch.int32,
    )

    # Window durations: fixed 2.0 s
    durations = torch.full((N, NUM_WINDOWS), 2.0, device=D)
    env.popup_durations[env_ids] = durations

    # Gaps between consecutive windows: fixed 0.3 s
    gaps = torch.full((N, NUM_WINDOWS), 0.3, device=D)

    # Initial delay before first window: fixed 0.0 s (first mole pops immediately)
    initial_delay = torch.full((N,), 0.0, device=D)

    # Cumulative start times
    start_times = torch.zeros(N, NUM_WINDOWS, device=D)
    start_times[:, 0] = initial_delay
    for i in range(1, NUM_WINDOWS):
        start_times[:, i] = start_times[:, i - 1] + durations[:, i - 1] + gaps[:, i]
    env.popup_start_times[env_ids] = start_times


def reset_moles(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Teleport all 5 moles to their retracted (down) positions."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_whack_a_mole_buffers(env)

    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    for i, mole_name in enumerate(MOLE_NAMES):
        mole: RigidObject = env.scene[mole_name]
        root_state = mole.data.default_root_state.clone()[env_ids]
        x, y = MOLE_XY_POSITIONS[i]
        root_state[:, 0] = env.scene.env_origins[env_ids, 0] + x
        root_state[:, 1] = env.scene.env_origins[env_ids, 1] + y
        root_state[:, 2] = env.scene.env_origins[env_ids, 2] + MOLE_DOWN_Z
        root_state[:, 3:7] = quat_identity
        root_state[:, 7:] = 0.0
        mole.write_root_state_to_sim(root_state, env_ids=env_ids)

    _set_all_moles_inactive(env, env_ids)


# -- Interval events ---------------------------------------------------------

def update_mole_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Teleport every mole to its current z-height each control step."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_whack_a_mole_buffers(env)

    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    for i, mole_name in enumerate(MOLE_NAMES):
        mole: RigidObject = env.scene[mole_name]
        root_state = mole.data.root_state_w[env_ids].clone()
        x, y = MOLE_XY_POSITIONS[i]
        root_state[:, 0] = env.scene.env_origins[env_ids, 0] + x
        root_state[:, 1] = env.scene.env_origins[env_ids, 1] + y
        root_state[:, 2] = env.scene.env_origins[env_ids, 2] + env.mole_heights[env_ids, i]
        root_state[:, 3:7] = quat_identity
        root_state[:, 7:] = 0.0
        mole.write_root_state_to_sim(root_state, env_ids=env_ids)


def check_popup_schedule(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Main interval event: advance timers, manage popup windows, detect presses."""
    env_ids = _resolve_env_ids(env, env_ids)
    _ensure_whack_a_mole_buffers(env)

    dt = getattr(env, "_interval_event_dt", env.step_dt)

    # Only process envs still in progress (task_phase == 0)
    in_progress = env.task_phase[env_ids] == 0
    active_ids = env_ids[in_progress]
    if len(active_ids) == 0:
        return

    # Advance episode timer
    env.episode_timer[active_ids] += dt
    timer = env.episode_timer[active_ids]

    # -- Handle episodes where all windows are exhausted ------------------
    all_done = env.current_window_idx[active_ids] >= NUM_WINDOWS
    if all_done.any():
        done_ids = active_ids[all_done]
        env.task_phase[done_ids] = 4
        env.active_mole_id[done_ids] = -1
        env.window_active[done_ids] = False
        env.mole_heights[done_ids] = MOLE_DOWN_Z
        env.success_rate[done_ids] = env.valid_hits[done_ids].float() / env.total_popups[done_ids].float().clamp(min=1)
        _set_all_moles_inactive(env, done_ids)

    still_running = active_ids[~all_done]
    if len(still_running) == 0:
        return

    timer = env.episode_timer[still_running]
    win_idx = env.current_window_idx[still_running].long()

    # Gather per-env window parameters via advanced indexing
    start_t = env.popup_start_times[still_running].gather(1, win_idx.unsqueeze(1)).squeeze(1)
    dur = env.popup_durations[still_running].gather(1, win_idx.unsqueeze(1)).squeeze(1)
    mole_ids = env.popup_mole_ids[still_running].gather(1, win_idx.unsqueeze(1)).squeeze(1)
    end_t = start_t + dur

    # -- Handle missed windows (timer jumped past entire window) ---------
    missed = (~env.window_active[still_running]) & (timer >= end_t)
    if missed.any():
        missed_ids = still_running[missed]
        env.window_active[missed_ids] = False
        env.active_mole_id[missed_ids] = -1
        env.current_window_idx[missed_ids] += 1

    # -- Window activation ------------------------------------------------
    should_activate = (~env.window_active[still_running]) & (timer >= start_t) & (timer < end_t)
    if should_activate.any():
        act_ids = still_running[should_activate]
        act_moles = mole_ids[should_activate].long()

        env.window_active[act_ids] = True
        env.active_mole_id[act_ids] = act_moles.int()
        env.press_dwell_counter[act_ids] = 0
        env.window_hit_registered[act_ids] = False
        env.total_popups[act_ids] += 1

        # Reset per-window diagnostic stats
        env._wam_win_min_xy[act_ids] = float("inf")
        env._wam_win_min_z[act_ids] = float("inf")
        env._wam_win_max_z[act_ids] = float("-inf")
        env._wam_win_in_xy_ticks[act_ids] = 0
        env._wam_win_in_z_ticks[act_ids] = 0
        env._wam_win_both_ticks[act_ids] = 0
        env._wam_win_total_ticks[act_ids] = 0
        env._wam_win_max_dwell[act_ids] = 0
        env._wam_win_ee_at_closest[act_ids] = 0.0
        env._wam_win_mole_xy_ref[act_ids] = 0.0

        # Pop up the target mole
        env.mole_heights[act_ids, act_moles] = MOLE_UP_Z
        _set_mole_visual_state(env, act_ids, act_moles, is_active=True)

    # -- Press detection for active windows -------------------------------
    is_active = env.window_active[still_running]
    if is_active.any():
        active_envs = still_running[is_active]
        _advance_post_hit_completion(env, active_envs)

        detect_envs = active_envs[
            env.window_active[active_envs] & (~env.window_hit_registered[active_envs])
        ]
        if len(detect_envs) > 0:
            _detect_press(env, detect_envs)

    # -- Window expiration ------------------------------------------------
    should_expire = env.window_active[still_running] & (timer >= end_t)
    if should_expire.any():
        _handle_window_expiry(env, still_running[should_expire])


# -- Internal helpers ---------------------------------------------------------

def _advance_post_hit_completion(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Finish a single-button episode a few control steps after a valid hit."""
    if len(env_ids) == 0 or not hasattr(env, "post_hit_complete_counter"):
        return

    pending = env.window_hit_registered[env_ids] & (env.post_hit_complete_counter[env_ids] > 0)
    if not pending.any():
        return

    pending_ids = env_ids[pending]
    env.post_hit_complete_counter[pending_ids] -= 1
    complete = env.post_hit_complete_counter[pending_ids] <= 0
    if not complete.any():
        return

    done_ids = pending_ids[complete]
    active_mid = env.active_mole_id[done_ids].long()
    valid_moles = active_mid >= 0
    if valid_moles.any():
        vm_ids = done_ids[valid_moles]
        vm_moles = active_mid[valid_moles]
        env.mole_heights[vm_ids, vm_moles] = MOLE_DOWN_Z
        _set_mole_visual_state(env, vm_ids, vm_moles, is_active=False)

    env.task_phase[done_ids] = 4
    env.window_active[done_ids] = False
    env.active_mole_id[done_ids] = -1
    env.current_window_idx[done_ids] = NUM_WINDOWS
    env.press_dwell_counter[done_ids] = 0
    env.success_rate[done_ids] = (
        env.valid_hits[done_ids].float() / env.total_popups[done_ids].float().clamp(min=1)
    )


def _detect_press(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Geometric press detection: check if EE is over the active mole and held."""
    if len(env_ids) == 0:
        return

    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_pos_local = ee_pos_w - env.scene.env_origins[env_ids]

    # Active mole XY in local frame
    active_mid = env.active_mole_id[env_ids].long()
    batch_idx = torch.arange(len(env_ids), device=env.device)
    mole_xy = env.mole_positions_local[env_ids][batch_idx, active_mid, :2]  # (N, 2)

    # Distance checks (geometric only; gripper-closed is no longer required)
    xy_dist = torch.norm(ee_pos_local[:, :2] - mole_xy, dim=1)
    ee_z = ee_pos_local[:, 2]
    in_xy = xy_dist < PRESS_XY_THRESHOLD
    in_z = (ee_z < PRESS_Z_MAX) & (ee_z > PRESS_Z_MIN)
    is_pressing = in_xy & in_z

    # Update per-window diagnostic stats
    closer = xy_dist < env._wam_win_min_xy[env_ids]
    if closer.any():
        c_ids = env_ids[closer]
        env._wam_win_ee_at_closest[c_ids] = ee_pos_local[closer]
        env._wam_win_mole_xy_ref[c_ids] = mole_xy[closer]
    env._wam_win_min_xy[env_ids] = torch.minimum(env._wam_win_min_xy[env_ids], xy_dist)
    env._wam_win_min_z[env_ids] = torch.minimum(env._wam_win_min_z[env_ids], ee_z)
    env._wam_win_max_z[env_ids] = torch.maximum(env._wam_win_max_z[env_ids], ee_z)
    env._wam_win_total_ticks[env_ids] += 1
    if in_xy.any():
        env._wam_win_in_xy_ticks[env_ids[in_xy]] += 1
    if in_z.any():
        env._wam_win_in_z_ticks[env_ids[in_z]] += 1
    if is_pressing.any():
        env._wam_win_both_ticks[env_ids[is_pressing]] += 1

    # Dwell counter
    if is_pressing.any():
        env.press_dwell_counter[env_ids[is_pressing]] += 1
    not_pressing = ~is_pressing
    if not_pressing.any():
        np_ids = env_ids[not_pressing]
        no_hit_yet = ~env.window_hit_registered[np_ids]
        if no_hit_yet.any():
            env.press_dwell_counter[np_ids[no_hit_yet]] = 0

    # Track peak dwell so we can diagnose near-misses after reset
    env._wam_win_max_dwell[env_ids] = torch.maximum(
        env._wam_win_max_dwell[env_ids], env.press_dwell_counter[env_ids]
    )

    # Valid hit: dwell threshold met and not already registered
    hit_mask = (
        is_pressing
        & (env.press_dwell_counter[env_ids] >= PRESS_DWELL_STEPS)
        & (~env.window_hit_registered[env_ids])
    )
    if hit_mask.any():
        hit_ids = env_ids[hit_mask]
        env.valid_hits[hit_ids] += 1
        env.window_hit_registered[hit_ids] = True
        env.post_hit_complete_counter[hit_ids] = POST_HIT_COMPLETE_STEPS
        # Retract the mole immediately on hit
        hit_moles = env.active_mole_id[hit_ids].long()
        env.mole_heights[hit_ids, hit_moles] = MOLE_DOWN_Z
        _set_mole_visual_state(env, hit_ids, hit_moles, is_active=False)


def _handle_window_expiry(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Score the expired window, retract mole, advance to next window."""
    # Diagnostic dump for windows that expired without a registered hit
    not_hit = ~env.window_hit_registered[env_ids]
    if not_hit.any():
        nh_ids = env_ids[not_hit]
        for eid in nh_ids.tolist():
            win = int(env.current_window_idx[eid].item())
            mid = int(env.active_mole_id[eid].item())
            min_xy = float(env._wam_win_min_xy[eid].item())
            min_z = float(env._wam_win_min_z[eid].item())
            max_z = float(env._wam_win_max_z[eid].item())
            in_xy_t = int(env._wam_win_in_xy_ticks[eid].item())
            in_z_t = int(env._wam_win_in_z_ticks[eid].item())
            both_t = int(env._wam_win_both_ticks[eid].item())
            total_t = int(env._wam_win_total_ticks[eid].item())
            max_dw = int(env._wam_win_max_dwell[eid].item())

            reasons = []
            if min_xy >= PRESS_XY_THRESHOLD:
                reasons.append(f"xy_never_in (min={min_xy:.4f} >= {PRESS_XY_THRESHOLD})")
            if max_z < PRESS_Z_MIN or min_z > PRESS_Z_MAX or in_z_t == 0:
                reasons.append(
                    f"z_never_in (range=[{min_z:.4f},{max_z:.4f}] vs [{PRESS_Z_MIN:.3f},{PRESS_Z_MAX:.3f}])"
                )
            if in_xy_t > 0 and in_z_t > 0 and both_t == 0:
                reasons.append("xy_and_z_never_simultaneous")
            if both_t > 0 and max_dw < PRESS_DWELL_STEPS:
                reasons.append(f"dwell_short (max={max_dw} < {PRESS_DWELL_STEPS})")
            if not reasons:
                reasons.append("unknown")

            ee_c = env._wam_win_ee_at_closest[eid].tolist()
            mole_ref = env._wam_win_mole_xy_ref[eid].tolist()
            print(
                f"[WAM-MISS] env={eid} win={win} mole={mid} ticks={total_t} "
                f"in_xy={in_xy_t} in_z={in_z_t} both={both_t} max_dwell={max_dw} "
                f"min_xy={min_xy:.4f} z=[{min_z:.4f},{max_z:.4f}] "
                f"ee_local=({ee_c[0]:.4f},{ee_c[1]:.4f},{ee_c[2]:.4f}) "
                f"mole_xy=({mole_ref[0]:.4f},{mole_ref[1]:.4f}) | " + "; ".join(reasons),
                flush=True,
            )

    # Retract the active mole
    active_mid = env.active_mole_id[env_ids].long()
    valid_moles = active_mid >= 0
    if valid_moles.any():
        vm_ids = env_ids[valid_moles]
        vm_moles = active_mid[valid_moles]
        env.mole_heights[vm_ids, vm_moles] = MOLE_DOWN_Z
        _set_mole_visual_state(env, vm_ids, vm_moles, is_active=False)

    # Advance to next window
    env.window_active[env_ids] = False
    env.active_mole_id[env_ids] = -1
    env.current_window_idx[env_ids] += 1
    env.press_dwell_counter[env_ids] = 0
    env.window_hit_registered[env_ids] = False
    env.post_hit_complete_counter[env_ids] = 0
