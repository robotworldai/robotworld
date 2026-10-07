from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .constants import (
    ACTION_SCALE_ISAACLAB,
    BODY_NAMES,
    DEFAULT_JOINT_POS_ISAACLAB,
    JOINT_NAMES,
    MUJOCO_JOINT_NAMES,
)
from .math_utils import parse_csv_floats
from .schemas import ExperimentConfig, MotionClip


def parse_metadata_names(meta: dict[str, str], key: str) -> list[str]:
    value = meta.get(key, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_metadata_array(meta: dict[str, str], key: str, fallback: np.ndarray) -> np.ndarray:
    value = meta.get(key)
    if not value:
        return fallback.astype(np.float32)
    parsed = np.array(parse_csv_floats(value), dtype=np.float32)
    if parsed.shape != fallback.shape:
        return fallback.astype(np.float32)
    joint_names = parse_metadata_names(meta, "joint_names")
    if len(joint_names) == len(MUJOCO_JOINT_NAMES) and all(name in joint_names for name in MUJOCO_JOINT_NAMES):
        return np.array([parsed[joint_names.index(name)] for name in MUJOCO_JOINT_NAMES], dtype=np.float32)
    return parsed.astype(np.float32)


@dataclasses.dataclass
class PolicyStep:
    action: np.ndarray
    ref_joint_pos: np.ndarray
    ref_joint_vel: np.ndarray
    ref_ang_vel: np.ndarray


def _motion_reference_at(motion: MotionClip, time_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = min(time_step, motion.length - 1)
    return (
        motion.joint_pos[t].astype(np.float32),
        motion.joint_vel[t].astype(np.float32),
        motion.body_ang_vel_w[t, BODY_NAMES.index("torso_link")].astype(np.float32),
    )


class ReferencePolicy:
    def __init__(self, default_joint_pos: np.ndarray, action_scale: np.ndarray):
        self.default_joint_pos = default_joint_pos
        self.action_scale = np.maximum(action_scale, 1.0e-6)
        self.metadata: dict[str, str] = {}
        self.obs_dim: int | None = None
        self.ref_joint_pos = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self.ref_joint_vel = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self.ref_ang_vel = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.ref_joint_pos.fill(0.0)
        self.ref_joint_vel.fill(0.0)
        self.ref_ang_vel.fill(0.0)

    def step(self, obs: np.ndarray, time_step: int, motion: MotionClip) -> PolicyStep:
        del obs
        self.ref_joint_pos, self.ref_joint_vel, self.ref_ang_vel = _motion_reference_at(motion, time_step)
        action = (self.ref_joint_pos - self.default_joint_pos) / self.action_scale
        return PolicyStep(
            action=action.astype(np.float32),
            ref_joint_pos=self.ref_joint_pos,
            ref_joint_vel=self.ref_joint_vel,
            ref_ang_vel=self.ref_ang_vel,
        )


class OnnxPolicy:
    def __init__(self, path: Path, rnn_num_layers: int, rnn_hidden_dim: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for --policy/--policy-dir. Install it or run with --reference-only."
            ) from exc

        self.path = path
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.obs_dim = self._obs_dim()
        self.h_name = "h_in" if "h_in" in self.input_names else None
        self.c_name = "c_in" if "c_in" in self.input_names else None
        self.recurrent = self.h_name is not None and self.c_name is not None
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.h: np.ndarray | None = None
        self.c: np.ndarray | None = None
        self.ref_joint_pos = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self.ref_joint_vel = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self.ref_ang_vel = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.ref_joint_pos.fill(0.0)
        self.ref_joint_vel.fill(0.0)
        self.ref_ang_vel.fill(0.0)
        self.action.fill(0.0)
        if not self.recurrent:
            self.h = None
            self.c = None
            return
        h_shape = self._input_shape(self.h_name, (self.rnn_num_layers, 1, self.rnn_hidden_dim))
        c_shape = self._input_shape(self.c_name, (self.rnn_num_layers, 1, self.rnn_hidden_dim))
        self.h = np.zeros(h_shape, dtype=np.float32)
        self.c = np.zeros(c_shape, dtype=np.float32)

    def _input_shape(self, name: str | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
        if name is None:
            return fallback
        for inp in self.session.get_inputs():
            if inp.name != name:
                continue
            shape = []
            for dim, fallback_dim in zip(inp.shape, fallback, strict=False):
                shape.append(fallback_dim if dim is None or isinstance(dim, str) else int(dim))
            if len(shape) == len(fallback):
                return tuple(shape)
        return fallback

    def _obs_dim(self) -> int | None:
        obs_name = "obs" if "obs" in self.input_names else self.input_names[0]
        for inp in self.session.get_inputs():
            if inp.name != obs_name or len(inp.shape) == 0:
                continue
            dim = inp.shape[-1]
            if dim is None or isinstance(dim, str):
                return None
            return int(dim)
        return None

    def step(self, obs: np.ndarray, time_step: int, motion: MotionClip) -> PolicyStep:
        feeds: dict[str, np.ndarray] = {}
        if "obs" in self.input_names:
            feeds["obs"] = obs.astype(np.float32).reshape(1, -1)
        else:
            feeds[self.input_names[0]] = obs.astype(np.float32).reshape(1, -1)
        if "time_step" in self.input_names:
            feeds["time_step"] = np.array([[time_step]], dtype=np.float32)
        if self.recurrent:
            if self.h is None or self.c is None:
                self.reset()
            feeds[self.h_name] = self.h
            feeds[self.c_name] = self.c
        outputs = self.session.run(None, feeds)
        output_map = dict(zip(self.output_names, outputs, strict=False))
        if self.recurrent:
            self.h = output_map.get("h_out", outputs[1]).astype(np.float32)
            self.c = output_map.get("c_out", outputs[2]).astype(np.float32)
        self.action = np.asarray(output_map.get("actions", outputs[0]), dtype=np.float32).reshape(-1)
        self.ref_joint_pos, self.ref_joint_vel, self.ref_ang_vel = _motion_reference_at(motion, time_step)
        return PolicyStep(
            action=self.action,
            ref_joint_pos=self.ref_joint_pos,
            ref_joint_vel=self.ref_joint_vel,
            ref_ang_vel=self.ref_ang_vel,
        )


class PolicyBank:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.cache: dict[Path, OnnxPolicy] = {}
        self.single_policy: Path | None = cfg.policy
        if cfg.policy_dir is not None:
            policy_files = sorted(cfg.policy_dir.glob("*.onnx"))
            if not policy_files:
                raise FileNotFoundError(f"No .onnx policy files found in {cfg.policy_dir}")
            self.policy_files = policy_files
        else:
            self.policy_files = []

        if cfg.preload_policies:
            for policy_file in self.policy_files:
                self._load(policy_file)

    def find_for_motion(self, motion: MotionClip) -> OnnxPolicy | ReferencePolicy:
        if self.cfg.reference_only or (self.cfg.policy is None and self.cfg.policy_dir is None):
            return ReferencePolicy(DEFAULT_JOINT_POS_ISAACLAB, ACTION_SCALE_ISAACLAB)
        if self.single_policy is not None:
            return self._load(self.single_policy)
        matches = [path for path in self.policy_files if motion.name in path.stem]
        if not matches:
            raise FileNotFoundError(
                f"No ONNX policy in {self.cfg.policy_dir} matches motion '{motion.name}'. "
                "Export with --export_motion_name all or pass --policy for a single policy."
            )
        if len(matches) > 1:
            matches = sorted(matches, key=lambda p: len(p.stem))
        return self._load(matches[0])

    def _load(self, path: Path) -> OnnxPolicy:
        if path not in self.cache:
            self.cache[path] = OnnxPolicy(path, self.cfg.rnn_num_layers, self.cfg.rnn_hidden_dim)
        return self.cache[path]


def validate_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} actions, got {action.shape[0]}")
    return action
