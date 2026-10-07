"""Public RoboDojo TASK_ENV boundary; no upstream mutation or LLM API."""
from environment.benchmarks.action_contracts import action_contract, describe_tools
import base64
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from environment.robots.arx_x5.embodiment_docs import build_arx_x5_docs
from environment.runtime.image_history import ObservationHistory
from .docs import eef_docs_from_joint_docs
from .eef_executor import EefExecutor
from .spec import RoboDojoActionSpec
from .types import Observation
from environment.runtime.events import EventLog
from environment.benchmarks.operating_brief import operating_brief


def numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def action_spec(task_env):
    manager = task_env.robot_manager
    labels, low, high, limits, sides = [], [], [], [], []
    mounts = {}
    hz = float(task_env.obs_manager.collect_freq)
    interval = float(task_env.obs_manager.collect_interval)
    if hz <= 0 or interval <= 0:
        raise ValueError("Invalid observation/control rate")
    for index, robot in enumerate(manager.robot_list):
        if robot.type != "target":
            continue
        side = robot.arm_name.removesuffix("_arm")
        sides.append(side)
        mounts[side] = robot.entity_origin_pose
        asset = manager.robot_key[index]
        bounds = numpy(asset.data.soft_joint_pos_limits[0, robot.arm_joint_indices])
        velocities = numpy(asset.data.joint_vel_limits[0, robot.arm_joint_indices])
        labels.extend(f"{side}_{name}" for name in robot.arm_joints_name)
        low.extend(bounds[:, 0]); high.extend(bounds[:, 1])
        for (lo, hi), velocity in zip(bounds, velocities, strict=True):
            step = min(float(velocity) / hz, 0.05)
            limits.append(step if 0 < step < hi - lo else None)
        labels.append(f"{side}_gripper")
        low.append(0.0); high.append(1.0)
        # RoboProbe derives this from RoboDojo's declared clamp, not a model parameter.
        import inspect
        from env.robot_manager.control_manager import MetaControl
        source = inspect.getsource(MetaControl.get_action)
        import re
        found = re.search(r"gripper_eps\s*=\s*([0-9.]+)", source)
        if not found:
            raise ValueError("Cannot identify RoboDojo gripper clamp declaration")
        limits.append(min(1.0, float(found[1]) * interval, 0.25))
    if sides != ["left", "right"] or len(labels) != 14:
        raise ValueError("This adapter requires dual 6-DoF ARX X5 in left/right order")
    return RoboDojoActionSpec(tuple(labels), np.array(low), np.array(high), hz,
                             eef_docs_from_joint_docs(build_arx_x5_docs(mounts)), tuple(limits))


class RoboDojoAdapter:
    def __init__(self, task_env, *, output_dir, decode_image=None, spec=None,
                 image_history=4, history_interval=2, max_actions=32, record_video=False):
        if image_history < 0 or history_interval < 1:
            raise ValueError("image_history must be nonnegative; history_interval must be positive")
        if int(getattr(task_env, "num_envs", 1)) != 1:
            raise ValueError("One Codex conversation requires one environment")
        self.env = task_env
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(self.output / "events/environment.jsonl")
        self.decode_image = decode_image
        self.sequence = 0
        self.image_history = image_history
        self.history_interval = history_interval
        self.visual_history = ObservationHistory(image_history, history_interval)
        self._image_frames = self.visual_history.frames
        self.observation = None
        self.deadline = None
        self.executor = EefExecutor(spec or action_spec(task_env), self._plan)
        self.executor._max_llm_calls = max_actions
        self.instructions = self.executor._system_message()+"\n"+action_contract("robodojo")+"\n"+operating_brief("robodojo")
        self.instructions += (
            f"\nEach observation includes current camera images and up to {image_history} historical observations, "
            f"sampled every {history_interval} observation rounds. Image labels identify their observation and environment step. "
            "Observation rounds are not equal durations; compare environment steps to judge motion. "
            "Historical images are past states, not additional current views. "
            "The simulator advances when actions execute, not while you reason.")
        self.instructions += "\n" + self.visual_history.instructions
        self.instructions += (
            "\nThere is no give_up tool. Use move_eef to attempt the task within the remaining budget; "
            "the environment determines success, failure and episode termination.")
        self.video = None
        if record_video:
            from .video import ContinuousVideo
            self.video = ContinuousVideo(self.output / "video", self.executor.action_spec.control_hz)
        self.task_instruction = "Follow the official task instruction in the observation."

    def _plan(self, *, arm, target_pose):
        manager = self.env.robot_manager
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            return {"status": "Unavailable"}
        return planner.plan_path(manager.get_joint(robot, env_idx_list=[0])[0], target_pose,
                                 real_robot_pose=copy.deepcopy(robot.entity_origin_pose))

    def tool_spec(self):
        return describe_tools([self.executor.tool_spec()], action_contract("robodojo"))[0]

    def ended(self):
        return bool(self.env.is_episode_end())

    @property
    def state(self):
        return {'native_steps':int(self.env.take_action_cnt[0])}

    def official_success(self):
        # RoboDojo's success starts True, so it is not an outcome until termination.
        return bool(self.env.success[0]) if self.ended() else None

    def observe_content(self):
        raw = self.env.get_obs()
        images = {}
        for name, camera in raw.get("vision", {}).items():
            color = camera.get("color")
            if color is None:
                continue
            if not isinstance(color, np.ndarray) or color.ndim != 3:
                if self.decode_image is None:
                    raise ValueError("Encoded observation needs the benchmark shared decode_image_bit helper")
                color = self.decode_image(color)
            images[name] = np.asarray(color, dtype=np.uint8)
        used = int(self.env.take_action_cnt[0])
        if self.video:
            self.video.write(images, used)
        remaining = max(0, int(self.env.step_lim) - used)
        self.events.write("observation", {"observation": self.sequence, "env_step": used,
                          "state": raw["state"], "frames": f"frames/{self.sequence}"})
        self.observation = Observation(images=images, state=raw["state"],
                                       instruction=raw.get("instruction") or raw.get("instructions"),
                                       step=self.sequence, remaining_steps=remaining)
        text = self.executor._state_block(self.observation)
        text += f"\nEnv steps remaining before the episode ends: {remaining}"
        text += f"\nObservation: {self.sequence}; env step: {used}; env steps remaining: {remaining}"
        text += "\n" + self.executor._arrival_text(self.observation)
        parts = [{"type": "inputText", "text": text}]
        frame_dir = self.output / "frames" / str(self.sequence)
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "observation.txt").write_text(text, encoding="utf-8")
        image_parts = []
        for name, image in images.items():
            # Never convert RGB to BGR at the model boundary.
            rgb = Image.fromarray(image)
            safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
            rgb.save(frame_dir / f"{safe_name}.png")
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=95)
            image_parts.extend([{"type": "inputText", "text": f"camera '{name}' (step {self.sequence}):"},
                          {"type": "inputImage", "imageUrl": "data:image/jpeg;base64," +
                           base64.b64encode(buffer.getvalue()).decode("ascii")}])
        parts.extend(self.visual_history.append(self.sequence, used, image_parts))
        (frame_dir / "history.json").write_text(json.dumps(self.visual_history.included, indent=2) + "\n")
        self.sequence += 1
        return parts

    def move_eef(self, arguments):
        if self.ended():
            raise RuntimeError("Cannot execute after episode end")
        if self.observation is None:
            self.observe_content()
        if not isinstance(arguments, dict):
            outcome = self.executor._repair("Arguments must be an object")
        else:
            outcome = self.executor.execute_plan(arguments, self.observation)
        trace = {"planned_waypoints": 0, "executed_waypoints": 0,
                 "tool_result": outcome.tool_result}
        if outcome.chunk is not None:
            trace.update(outcome.chunk.meta.get("trace", {}))
            for action in outcome.chunk.actions:
                if self.ended():
                    break
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    trace["interrupted"] = "timeout"
                    break
                self.events.write("action_requested", {"observation": self.sequence - 1,
                                  "env_step_before": int(self.env.take_action_cnt[0]),
                                  "action": dict(action.data)})
                self.env.take_action(dict(action.data))
                trace["executed_waypoints"] += 1
                # Preserve official video flushing at every waypoint.
                raw = self.env.get_obs()
                self.events.write("action_completed", {"observation": self.sequence - 1,
                                  "env_step": int(self.env.take_action_cnt[0]),
                                  "state": raw["state"]})
                if self.video:
                    images = {}
                    for name, camera in raw.get("vision", {}).items():
                        color = camera.get("color")
                        if color is not None:
                            if not isinstance(color, np.ndarray) or color.ndim != 3:
                                color = self.decode_image(color)
                            images[name] = np.asarray(color, dtype=np.uint8)
                    self.video.write(images, int(self.env.take_action_cnt[0]))
        content = [{"type": "inputText", "text": outcome.tool_result}]
        content.append({"type": "inputText", "text":
                        f"Actually executed {trace['executed_waypoints']} of {trace['planned_waypoints']} waypoints."})
        content.extend(self.observe_content())
        return {"success": outcome.chunk is not None, "contentItems": content}, trace
