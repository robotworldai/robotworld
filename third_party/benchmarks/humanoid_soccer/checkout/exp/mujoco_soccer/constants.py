from __future__ import annotations

import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = REPO_ROOT / "source/whole_body_tracking/soccer/assets/unitree_description/mjcf/g1_actuator.xml"
DEFAULT_MOTION_PATH = REPO_ROOT / "motions/soccer-standard"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "exp/results"
BALL_RADIUS = 0.11

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

JOINT_NAMES = MUJOCO_JOINT_NAMES

ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

ISAACLAB_TO_MUJOCO_REINDEX = np.array(
    [ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES],
    dtype=np.int32,
)
MUJOCO_TO_ISAACLAB_REINDEX = np.array(
    [MUJOCO_JOINT_NAMES.index(name) for name in ISAACLAB_JOINT_NAMES],
    dtype=np.int32,
)

BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


def _joint_map(default: float = 0.0) -> dict[str, float]:
    return {name: default for name in JOINT_NAMES}


JOINT_STIFFNESS = _joint_map()
JOINT_DAMPING = _joint_map()
JOINT_EFFORT_LIMIT = _joint_map()

for joint_name in JOINT_NAMES:
    if "_hip_pitch_joint" in joint_name or "_hip_yaw_joint" in joint_name:
        JOINT_STIFFNESS[joint_name] = STIFFNESS_7520_14
        JOINT_DAMPING[joint_name] = DAMPING_7520_14
        JOINT_EFFORT_LIMIT[joint_name] = 88.0
    elif "_hip_roll_joint" in joint_name:
        JOINT_STIFFNESS[joint_name] = STIFFNESS_7520_22
        JOINT_DAMPING[joint_name] = DAMPING_7520_22
        JOINT_EFFORT_LIMIT[joint_name] = 139.0
    elif "_knee_joint" in joint_name:
        JOINT_STIFFNESS[joint_name] = STIFFNESS_7520_22
        JOINT_DAMPING[joint_name] = DAMPING_7520_22
        JOINT_EFFORT_LIMIT[joint_name] = 139.0
    elif "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
        JOINT_STIFFNESS[joint_name] = 2.0 * STIFFNESS_5020
        JOINT_DAMPING[joint_name] = 2.0 * DAMPING_5020
        JOINT_EFFORT_LIMIT[joint_name] = 50.0
    elif joint_name in {"waist_roll_joint", "waist_pitch_joint"}:
        JOINT_STIFFNESS[joint_name] = 2.0 * STIFFNESS_5020
        JOINT_DAMPING[joint_name] = 2.0 * DAMPING_5020
        JOINT_EFFORT_LIMIT[joint_name] = 50.0
    elif joint_name == "waist_yaw_joint":
        JOINT_STIFFNESS[joint_name] = STIFFNESS_7520_14
        JOINT_DAMPING[joint_name] = DAMPING_7520_14
        JOINT_EFFORT_LIMIT[joint_name] = 88.0
    elif "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
        JOINT_STIFFNESS[joint_name] = STIFFNESS_4010
        JOINT_DAMPING[joint_name] = DAMPING_4010
        JOINT_EFFORT_LIMIT[joint_name] = 5.0
    else:
        JOINT_STIFFNESS[joint_name] = STIFFNESS_5020
        JOINT_DAMPING[joint_name] = DAMPING_5020
        JOINT_EFFORT_LIMIT[joint_name] = 25.0

DEFAULT_JOINT_POS_ARRAY = np.array([DEFAULT_JOINT_POS.get(name, 0.0) for name in JOINT_NAMES], dtype=np.float32)
JOINT_STIFFNESS_ARRAY = np.array([JOINT_STIFFNESS[name] for name in JOINT_NAMES], dtype=np.float32)
JOINT_DAMPING_ARRAY = np.array([JOINT_DAMPING[name] for name in JOINT_NAMES], dtype=np.float32)
JOINT_EFFORT_LIMIT_ARRAY = np.array([JOINT_EFFORT_LIMIT[name] for name in JOINT_NAMES], dtype=np.float32)
ACTION_SCALE_ARRAY = 0.25 * JOINT_EFFORT_LIMIT_ARRAY / np.maximum(JOINT_STIFFNESS_ARRAY, 1.0e-6)
DEFAULT_JOINT_POS_ISAACLAB = DEFAULT_JOINT_POS_ARRAY[MUJOCO_TO_ISAACLAB_REINDEX]
ACTION_SCALE_ISAACLAB = ACTION_SCALE_ARRAY[MUJOCO_TO_ISAACLAB_REINDEX]
