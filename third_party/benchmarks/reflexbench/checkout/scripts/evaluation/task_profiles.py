"""Task-specific profiles for evaluation.

Each profile contains:
- gym_id: Registered Gym environment name (uses DataCollection-v0 so that
  evaluation shares exactly the same scene, cameras and reset events as the
  data-collection pipeline, including the ``mark_wrist_cam_usd_dirty`` Fabric
  workaround)
- action_scale: Scaling factor for arm joint-relative actions
- success_type: How to determine episode success
- phase_names: Human-readable names for task phases (diagnostics)
- has_ik_override: Whether to disable env's built-in IK override
- WhackAMole-specific popup timing overrides
- ConveyorBeltPickAndPlace box ranges for object_in_box check

Helper functions:
- check_success(): Evaluate success for a given task profile
- apply_whack_a_mole_eval_overrides(): Override popup timing for eval
- install_phase_tracker(): Patch _reset_idx to capture phase at termination
"""

from __future__ import annotations

import torch


# ============================================================
#  Task Profiles Registry
# ============================================================

TASK_PROFILES: dict[str, dict] = {
    "conveyor_belt_pick_and_place": {
        "gym_id": "ConveyorBeltPickAndPlace-Franka-DataCollection-v0",
        "action_scale": 0.1,
        "has_ik_override": True,
        "short_circuit_success": True,
        "success_type": "object_in_box",
        "phase_names": {
            0: "approach", 1: "lift", 2: "transport", 3: "release", 4: "return", 5: "done",
        },
        "box_x_range": (-0.2, 0.2),
        "box_y_range": (-0.6, -0.3),
        "box_z_range": (0.0, 0.08),
    },
    "ball_catching": {
        "gym_id": "BallCatching-Franka-DataCollection-v0",
        "action_scale": 0.1,
        "has_ik_override": False,
        "short_circuit_success": True,
        "success_type": "task_phase_4",
        "phase_names": {
            0: "wait", 1: "launch", 2: "predict", 3: "position", 4: "catch",
        },
    },
    "rolling_ball_interception": {
        "gym_id": "RollingBallInterception-Franka-DataCollection-v0",
        "action_scale": 0.1,
        "has_ik_override": False,
        "short_circuit_success": True,
        "success_type": "task_phase_4",
        "phase_names": {
            0: "wait", 1: "roll", 2: "predict", 3: "position", 4: "catch",
        },
    },
    "whack_a_mole": {
        "gym_id": "WhackAMole-Franka-DataCollection-v0",
        "action_scale": 0.2,
        "has_ik_override": False,
        "short_circuit_success": False,
        "success_type": "whack_a_mole",
        "popup_duration_s": 2.0,
        "popup_gap_s": 0.3,
        "popup_initial_delay_s": 0.0,
        "eval_episode_length_s": 3.0,
        "phase_names": {
            0: "wait popup", 1: "move", 2: "strike", 3: "return", 4: "done",
        },
    },
    "ball_throwing": {
        "gym_id": "BallThrowing-Franka-DataCollection-v0",
        "action_scale": 0.1,
        "has_ik_override": False,
        "short_circuit_success": True,
        "success_type": "task_phase_2",
        "phase_names": {
            0: "swing", 1: "release", 2: "in box",
        },
    },
    "rotating_peg_insertion": {
        "gym_id": "RotatingPegInsertion-Franka-DataCollection-v0",
        "action_scale": 0.1,
        "has_ik_override": False,
        "short_circuit_success": True,
        "success_type": "task_phase_4",
        "phase_names": {
            0: "pre-align", 1: "track", 2: "wait alignment", 3: "insert", 4: "done",
        },
    },
}


def resolve_profile(name_or_gym_id: str) -> dict | None:
    """Look up a profile by short name or gym_id.  Returns None if not found."""
    if name_or_gym_id in TASK_PROFILES:
        return TASK_PROFILES[name_or_gym_id]
    for profile in TASK_PROFILES.values():
        if profile["gym_id"] == name_or_gym_id:
            return profile
    return None


# ============================================================
#  Success Checking
# ============================================================

def check_success(manager_env, task_profile: dict, env_id: int = 0) -> bool:
    """Evaluate success for a single env according to the task profile."""
    st = task_profile["success_type"]
    if st == "object_in_box":
        obj = manager_env.scene["object"]
        local = (
            obj.data.root_pos_w[env_id, :3]
            - manager_env.scene.env_origins[env_id, :3]
        )
        bx = task_profile["box_x_range"]
        by = task_profile["box_y_range"]
        bz = task_profile["box_z_range"]
        return bool(
            bx[0] <= local[0].item() <= bx[1]
            and by[0] <= local[1].item() <= by[1]
            and bz[0] <= local[2].item() <= bz[1]
        )
    elif st == "task_phase_4":
        return (
            hasattr(manager_env, "task_phase")
            and manager_env.task_phase[env_id].item() == 4
        )
    elif st == "task_phase_2":
        return (
            hasattr(manager_env, "task_phase")
            and manager_env.task_phase[env_id].item() == 2
        )
    elif st == "whack_a_mole":
        return (
            hasattr(manager_env, "valid_hits")
            and manager_env.valid_hits[env_id].item() > 0
        )
    return False


# ============================================================
#  WhackAMole Eval Overrides
# ============================================================

def apply_whack_a_mole_eval_overrides(manager_env, task_profile: dict | None) -> None:
    """Override popup timing for WhackAMole evaluation."""
    if not task_profile or task_profile.get("success_type") != "whack_a_mole":
        return
    if not hasattr(manager_env, "popup_durations") or not hasattr(
        manager_env, "popup_start_times"
    ):
        return

    popup_duration_s = float(task_profile.get("popup_duration_s", 1.0))
    popup_gap_s = float(task_profile.get("popup_gap_s", 0.3))
    popup_initial_delay_s = float(task_profile.get("popup_initial_delay_s", 1.0))

    manager_env.popup_durations.fill_(popup_duration_s)
    start_times = torch.zeros_like(manager_env.popup_start_times)
    start_times[:, 0] = popup_initial_delay_s
    for i in range(1, start_times.shape[1]):
        start_times[:, i] = start_times[:, i - 1] + popup_duration_s + popup_gap_s
    manager_env.popup_start_times.copy_(start_times)


# ============================================================
#  Phase Tracker
# ============================================================

def install_phase_tracker(manager_env) -> bool:
    """Patch _reset_idx to capture task_phase at termination time."""
    manager_env._phase_at_term = -1
    if not hasattr(manager_env, "_reset_idx"):
        print("  [WARN] _reset_idx not found - phase_at_term unavailable")
        return False
    original_reset_idx = manager_env._reset_idx

    def _patched_reset_idx(env_ids, *args, **kwargs):
        if len(env_ids) > 0 and hasattr(manager_env, "task_phase"):
            manager_env._phase_at_term = manager_env.task_phase[env_ids[0]].item()
        return original_reset_idx(env_ids, *args, **kwargs)

    manager_env._reset_idx = _patched_reset_idx
    print("  Phase tracker installed")
    return True


# ============================================================
#  WhackAMole Debug Diagnostics
# ============================================================

def debug_whack_a_mole_press(manager_env, step_idx: int, env_id: int = 0) -> None:
    """Print press-detection diagnostics for WhackAMole."""
    if not hasattr(manager_env, "window_active"):
        return
    if not manager_env.window_active[env_id]:
        return

    from isaaclab.sensors import FrameTransformer

    robot = manager_env.scene["robot"]
    ee_frame: FrameTransformer = manager_env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[env_id, 0, :]
    ee_pos_local = ee_pos_w - manager_env.scene.env_origins[env_id]

    active_mid = int(manager_env.active_mole_id[env_id].item())
    if active_mid < 0:
        return

    mole_xy = manager_env.mole_positions_local[env_id, active_mid, :2]
    mole_z = manager_env.mole_heights[env_id, active_mid].item()

    xy_dist = torch.norm(ee_pos_local[:2] - mole_xy).item()
    ee_z = ee_pos_local[2].item()

    finger_ids = robot.find_joints(["panda_finger.*"])[0]
    finger_pos = robot.data.joint_pos[env_id, finger_ids]
    finger_width = finger_pos.sum().item()

    BOARD_SURFACE_Z = 0.22
    XY_THR = 0.035
    Z_MAX = BOARD_SURFACE_Z + 0.018
    Z_MIN = BOARD_SURFACE_Z - 0.010
    GRIP_THR = 0.03

    ok_xy = xy_dist < XY_THR
    ok_z = Z_MIN < ee_z < Z_MAX
    ok_grip = finger_width < GRIP_THR
    dwell = manager_env.press_dwell_counter[env_id].item()
    hit_reg = manager_env.window_hit_registered[env_id].item()
    valid_hits = manager_env.valid_hits[env_id].item()
    win_idx = manager_env.current_window_idx[env_id].item()
    timer = manager_env.episode_timer[env_id].item()

    print(
        f"  [WAM-DBG] step={step_idx:>4} t={timer:.2f}s win={win_idx} mole={active_mid} "
        f"| ee_xy=({ee_pos_local[0].item():.4f},{ee_pos_local[1].item():.4f}) "
        f"mole_xy=({mole_xy[0].item():.4f},{mole_xy[1].item():.4f}) "
        f"xy_d={xy_dist:.4f} {'OK' if ok_xy else 'FAR':>3} "
        f"| ee_z={ee_z:.4f} [{Z_MIN:.3f},{Z_MAX:.3f}] "
        f"{'OK' if ok_z else ('HI' if ee_z >= Z_MAX else 'LO'):>2} "
        f"| grip_w={finger_width:.4f} {'CLOSED' if ok_grip else 'OPEN':>6} "
        f"| dwell={dwell} hit={hit_reg} total_hits={valid_hits}"
    )


# ============================================================
#  Smoothness Metrics
# ============================================================

def calculate_smoothness(actions_list: list) -> tuple[float, float]:
    """Compute (avg_diff, avg_jerk) from a list of action arrays."""
    import numpy as np

    if len(actions_list) < 3:
        return 0.0, 0.0
    actions = np.array(actions_list)
    arm_actions = actions[:, :7] if actions.shape[-1] >= 7 else actions
    diff1 = np.linalg.norm(np.diff(arm_actions, axis=0), axis=1)
    diff2 = np.linalg.norm(np.diff(arm_actions, n=2, axis=0), axis=1)
    return float(np.mean(diff1)), float(np.mean(diff2))
