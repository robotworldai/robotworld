from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from .constants import BALL_RADIUS, BODY_NAMES, ISAACLAB_TO_MUJOCO_REINDEX, JOINT_NAMES, TRACKED_BODY_NAMES
from .math_utils import normalize_quat
from .schemas import MotionClip


def first_worldbody_with_robot(root: ET.Element) -> ET.Element:
    for worldbody in root.findall("worldbody"):
        if worldbody.find("./body[@name='pelvis']") is not None:
            return worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    return worldbody


def make_experiment_xml(base_xml: Path, output_dir: Path, goal_width: float = 2.0) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(base_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        mesh_dir = (base_xml.parent / compiler.attrib.get("meshdir", ".")).resolve()
        compiler.set("meshdir", str(mesh_dir))

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", option.attrib.get("timestep", "0.002"))
    option.set("gravity", option.attrib.get("gravity", "0 0 -9.81"))

    worldbody = first_worldbody_with_robot(root)
    for body_name in ("soccer_ball", "soccer_goal_marker"):
        for old in worldbody.findall(f"./body[@name='{body_name}']"):
            worldbody.remove(old)

    ball = ET.SubElement(worldbody, "body", {"name": "soccer_ball", "pos": f"0 0 {BALL_RADIUS}"})
    ET.SubElement(ball, "freejoint", {"name": "soccer_ball_joint"})
    ET.SubElement(
        ball,
        "geom",
        {
            "name": "soccer_ball_geom",
            "type": "sphere",
            "size": f"{BALL_RADIUS}",
            "mass": "0.1",
            "friction": "0.8 0.01 0.0001",
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.01",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "1 1 1 1",
        },
    )

    half_goal_width = 0.5 * float(goal_width)
    goal = ET.SubElement(worldbody, "body", {"name": "soccer_goal_marker", "mocap": "true", "pos": f"0 0 {BALL_RADIUS}"})
    ET.SubElement(
        goal,
        "geom",
        {
            "name": "soccer_goal_center_geom",
            "type": "sphere",
            "size": "0.08",
            "rgba": "1 0 0 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        goal,
        "geom",
        {
            "name": "soccer_goal_width_geom",
            "type": "capsule",
            "fromto": f"0 {-half_goal_width} 0 0 {half_goal_width} 0",
            "size": "0.025",
            "rgba": "1 0 0 0.75",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for endpoint_name, endpoint_y in (("left", -half_goal_width), ("right", half_goal_width)):
        ET.SubElement(
            goal,
            "geom",
            {
                "name": f"soccer_goal_{endpoint_name}_post_geom",
                "type": "sphere",
                "pos": f"0 {endpoint_y} 0",
                "size": "0.06",
                "rgba": "1 0.85 0 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    generated = output_dir / "g1_soccer_experiment.xml"
    tree.write(generated, encoding="utf-8", xml_declaration=True)
    return generated


class MujocoSoccer:
    def __init__(self, xml_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.joint_qpos_addr = np.array(
            [self.model.jnt_qposadr[self._joint_id(name)] for name in JOINT_NAMES], dtype=np.int32
        )
        self.joint_dof_addr = np.array(
            [self.model.jnt_dofadr[self._joint_id(name)] for name in JOINT_NAMES], dtype=np.int32
        )
        self.actuator_ids = np.array([self._actuator_id(name) for name in JOINT_NAMES], dtype=np.int32)
        self.body_ids = {name: self._body_id(name) for name in BODY_NAMES}
        self.tracked_body_ids = np.array([self._body_id(name) for name in TRACKED_BODY_NAMES], dtype=np.int32)
        self.tracked_body_motion_ids = np.array([BODY_NAMES.index(name) for name in TRACKED_BODY_NAMES], dtype=np.int32)
        self.pelvis_body_id = self._body_id("pelvis")
        self.ball_body_id = self._body_id("soccer_ball")
        self.goal_body_id = self._body_id("soccer_goal_marker")
        self.goal_mocap_id = int(self.model.body_mocapid[self.goal_body_id])
        if self.goal_mocap_id < 0:
            raise RuntimeError("soccer_goal_marker must be a mocap body")
        self.ball_joint_id = self._joint_id("soccer_ball_joint")
        self.ball_qpos_addr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_dof_addr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.ball_geom_id = self._geom_id("soccer_ball_geom")
        excluded_contact_bodies = {0, self.ball_body_id, self.goal_body_id}
        self.robot_geom_ids = {
            idx
            for idx in range(self.model.ngeom)
            if self.model.geom_bodyid[idx] not in excluded_contact_bodies
        }
        self.foot_geom_ids = {
            idx
            for idx in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, idx))
            and ("foot" in name and "collision" in name)
        }

    def _joint_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if idx < 0:
            raise KeyError(f"MuJoCo joint not found: {name}")
        return idx

    def _body_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise KeyError(f"MuJoCo body not found: {name}")
        return idx

    def _geom_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if idx < 0:
            raise KeyError(f"MuJoCo geom not found: {name}")
        return idx

    def _actuator_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise KeyError(f"MuJoCo actuator not found: {name}")
        return idx

    def reset(self, motion: MotionClip, robot_xy: np.ndarray, ball_xyz: np.ndarray, ball_vel_xy: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        first_pelvis = motion.body_pos_w[0, 0].astype(np.float64)
        translation = np.array([robot_xy[0] - first_pelvis[0], robot_xy[1] - first_pelvis[1], 0.0], dtype=np.float64)

        self.data.qpos[:3] = first_pelvis + translation
        self.data.qpos[3:7] = normalize_quat(motion.body_quat_w[0, 0].astype(np.float64))
        self.data.qvel[:3] = motion.body_lin_vel_w[0, 0]
        self.data.qvel[3:6] = motion.body_ang_vel_w[0, 0]
        self.data.qpos[self.joint_qpos_addr] = motion.joint_pos[0][ISAACLAB_TO_MUJOCO_REINDEX]
        self.data.qvel[self.joint_dof_addr] = motion.joint_vel[0][ISAACLAB_TO_MUJOCO_REINDEX]

        self.data.qpos[self.ball_qpos_addr : self.ball_qpos_addr + 3] = ball_xyz
        self.data.qpos[self.ball_qpos_addr + 3 : self.ball_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.ball_dof_addr : self.ball_dof_addr + 3] = np.array([ball_vel_xy[0], ball_vel_xy[1], 0.0])
        self.data.qvel[self.ball_dof_addr + 3 : self.ball_dof_addr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return translation

    def set_goal_marker(self, goal_world: np.ndarray, goal_normal: np.ndarray) -> None:
        goal_normal = np.asarray(goal_normal, dtype=np.float64).reshape(2)
        norm = np.linalg.norm(goal_normal)
        if norm < 1.0e-9:
            goal_normal = np.array([1.0, 0.0], dtype=np.float64)
        else:
            goal_normal = goal_normal / norm
        yaw = math.atan2(float(goal_normal[1]), float(goal_normal[0]))
        self.data.mocap_pos[self.goal_mocap_id] = np.asarray(goal_world, dtype=np.float64).reshape(3)
        self.data.mocap_quat[self.goal_mocap_id] = np.array(
            [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_addr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.joint_dof_addr].copy()

    @property
    def pelvis_pos(self) -> np.ndarray:
        return self.data.xpos[self.pelvis_body_id].copy()

    @property
    def pelvis_quat(self) -> np.ndarray:
        return self.data.xquat[self.pelvis_body_id].copy()

    @property
    def pelvis_ang_vel_b(self) -> np.ndarray:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis_body_id, vel, 1)
        return vel[:3].copy()

    @property
    def base_ang_vel(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    @property
    def ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].copy()

    @property
    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self.ball_dof_addr : self.ball_dof_addr + 3].copy()

    def has_ball_robot_contact(self) -> bool:
        return self._has_ball_contact_with(self.robot_geom_ids)

    def has_ball_foot_contact(self) -> bool:
        return self._has_ball_contact_with(self.foot_geom_ids)

    def _has_ball_contact_with(self, geom_ids: set[int]) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.ball_geom_id and contact.geom2 in geom_ids:
                return True
            if contact.geom2 == self.ball_geom_id and contact.geom1 in geom_ids:
                return True
        return False

    def body_lin_vel_w(self, body_ids: np.ndarray) -> np.ndarray:
        velocities = np.zeros((len(body_ids), 3), dtype=np.float64)
        object_velocity = np.zeros(6, dtype=np.float64)
        for row, body_id in enumerate(body_ids):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), object_velocity, 0)
            velocities[row] = object_velocity[3:6]
        return velocities

    def set_pd_action(
        self,
        action: np.ndarray,
        default_joint_pos: np.ndarray,
        action_scale: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        effort_limit: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != len(JOINT_NAMES):
            raise ValueError(f"Expected {len(JOINT_NAMES)} actions, got {action.shape[0]}")
        action_mujoco = action[ISAACLAB_TO_MUJOCO_REINDEX]
        target = default_joint_pos + action_mujoco * action_scale
        torque = stiffness * (target - self.joint_pos) - damping * self.joint_vel
        torque = np.clip(torque, -effort_limit, effort_limit)
        self.data.ctrl[self.actuator_ids] = torque
        return target
