"""Sensor-only bridge to the existing native BEHAVIOR worker, not RoboDojo EEF."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import time

from PIL import Image
from environment.benchmarks.behavior_1k.budgets import configure_protocol, get_budget
from environment.runtime.image_history import ObservationHistory


INSTRUCTIONS = """Control R1Pro from RGB and robot proprioception using observe and step only.
No object ground truth, predicate/Q-score hints, skills, shell, files, reset or simulator internals.
step selects one arm. delta_position is a displacement in robot-base axes, in metres,
norm <=0.04. delta_rotation is a left-multiplied axis-angle delta in those axes,
in radians, norm <=0.15. Observed quaternions use XYZW.
One absolute IK target is computed per command and held, NOT repeated as a delta each tick.
Other-arm pose and trunk reset pose are held. Trunk movement is unavailable.
gripper: 0=closed, 1=open; finger joints are measured, opening fraction is commanded.
base=[forward,left,yaw] is native normalized control, each in [-0.2,0.2].
Do not combine base motion with nonzero arm pose deltas. Base axes: x forward, y left, z up.
steps is 1..15 at 30Hz; reasoning and observe do not advance physics.
IK is not collision-aware and may fail near singularities. Inspect measured errors and RGB.
An unreached target is not a rejection: physics and the gripper may have advanced.
Use the latest observation_id; do not replay commands whose execution is uncertain.
Budget: 600 wall seconds, 96 tool calls, 40 motion requests, 500 native steps.
Only the original checker success=true establishes completion. give_up ends this attempt.
This is a restricted fixed-trunk diagnostic, not the unrestricted BEHAVIOR challenge."""

PUBLIC_FIELDS = (
    "observation_id", "protocol", "robot", "base_world_pose", "commands",
    "native_steps", "physics_events", "success", "ended", "instruction",
)


def load_protocol(worker_dir, budget_profile="legacy"):
    path = Path(worker_dir) / "eef_protocol.py"
    spec = importlib.util.spec_from_file_location("behavior_native_eef_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return configure_protocol(module, budget_profile)


class BehaviorAdapter:
    require_isolated_thread = True
    instructions = INSTRUCTIONS
    developer_instructions = (
        "Use only observe/step for robot interaction and give_up to stop. "
        "BEHAVIOR actions use base-relative EEF deltas, NOT RoboDojo move_eef coordinates. "
        "Do not access files, shell, web, skills, reset or hidden simulator state."
    )
    read_only_tools = {"observe"}

    def __init__(self, simulator_dir, process, protocol, *, request_timeout=90, budget_profile="legacy"):
        self.simulator = Path(simulator_dir).resolve()
        self.process = process
        self.protocol = protocol
        self.request_timeout = request_timeout
        self.sequence = 0
        self.uncertain = False
        self.state = None
        self.feedback = None
        self.deadline = None
        self.task_instruction = ""
        self.visual_history = ObservationHistory()
        self.observation_round = 0
        self.budget = get_budget(budget_profile)
        if self.budget["wall_seconds"] is None:
            self.instructions = INSTRUCTIONS.replace(
                "Budget: 600 wall seconds, 96 tool calls, 40 motion requests, 500 native steps.",
                f"Budget: {self.budget['native_steps']} native control steps at 30Hz "
                f"({self.budget['native_steps'] / 30:.2f} simulation seconds). "
                "There is no 600-second wall-clock cutoff and no 40-action limit. "
                "Every step response reports remaining control steps; choose steps <= min(15, remaining)."
            ).replace("give_up ends this attempt.",
                "Do not end merely because success looks difficult or a target is not yet visible. "
                "Explore from RGB, approach objects, attempt manipulation, and change strategy after failures. "
                "Continue using observe/step until the checker succeeds or the control-step budget ends. "
                "Do not consume the budget with no-op actions; take meaningful bounded actions. "
                "There is no give_up tool. Infrastructure failures are handled by the controller.")
            self.developer_instructions = self.developer_instructions.replace(
                "and give_up to stop", "until success or the control-step budget is exhausted")
        self.instructions += "\n" + self.visual_history.instructions

    def wait_json(self, path, seconds):
        end = time.monotonic() + seconds
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError("BEHAVIOR worker exited before acknowledgement")
            if time.monotonic() >= end:
                raise TimeoutError("BEHAVIOR acknowledgement missing; never replay")
            time.sleep(.03)
        return json.loads(path.read_text())

    def start(self):
        self.wait_json(self.simulator / "ipc/ready.json", 1200)
        self.request("observe")
        goal = self.state["instruction"]
        self.task_instruction = goal if isinstance(goal, str) else json.dumps(goal)

    def request(self, tool, arguments=None):
        if self.uncertain:
            raise RuntimeError("Prior execution is uncertain; refuse further commands")
        if tool != "close" and self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Episode deadline exceeded before command submission")
        self.sequence += 1
        ipc = self.simulator / "ipc"
        path = ipc / ("request-%05d.json" % self.sequence)
        data = json.dumps({"tool": tool, "arguments": arguments or {}}, allow_nan=False)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(data + "\n")
        temporary.replace(path)
        try:
            seconds = self.request_timeout
            if tool != "close" and self.deadline is not None:
                seconds = max(0, min(seconds, self.deadline - time.monotonic()))
            reply = self.wait_json(ipc / ("response-%05d.json" % self.sequence), seconds)
            if reply["ok"] and tool != "close":
                raw = reply["observation"]
                self.state = {key: raw[key] for key in PUBLIC_FIELDS}
                self.image = raw["image"]
                self.feedback = reply.get("feedback")
            return reply
        except BaseException:
            self.uncertain = True
            raise

    def tool_specs(self):
        return [dict(type="function", **copy.deepcopy(spec)) for spec in self.protocol.TOOLS]

    def tool_handlers(self):
        return {"observe": lambda args: self.call("observe", args),
                "step": lambda args: self.call("step", args)}

    def content(self):
        path = (self.simulator / self.image).resolve(strict=True)
        if not path.is_relative_to(self.simulator / "observations") or path.stat().st_size > 8 * 1024**2:
            raise RuntimeError("Camera file outside the observation boundary")
        with Image.open(path) as picture:
            if picture.size != (256, 296) or picture.mode != "RGB":
                raise RuntimeError("Unexpected BEHAVIOR camera mosaic")
        label = ("Sensor observation_id " + self.state["observation_id"] +
                 ": top-left=head, top-right=left wrist, bottom-left=right wrist; "
                 "each native RGB is 128x128.")
        public = {"observation": self.state, "feedback": self.feedback}
        if self.budget["wall_seconds"] is None:
            public["budget"] = dict(control_steps_limit=self.budget["native_steps"],
                                    control_steps_remaining=self.budget["native_steps"] - self.state["native_steps"],
                                    wall_clock_task_timeout=None)
        image_parts = [
            {"type": "inputText", "text": label},
            {"type": "inputImage", "imageUrl": "data:image/png;base64," +
             base64.b64encode(path.read_bytes()).decode()},
        ]
        parts = self.visual_history.append(self.observation_round, self.state["native_steps"], image_parts)
        history_dir = self.simulator.parent / "image-history"
        history_dir.mkdir(exist_ok=True)
        (history_dir / f"{self.observation_round:05d}.json").write_text(
            json.dumps(self.visual_history.snapshot(), indent=2) + "\n")
        self.observation_round += 1
        return [
            {"type": "inputText", "text": json.dumps(
                public, allow_nan=False)},
            *parts,
        ]

    def observe_content(self):
        self.request("observe")
        return self.content()

    def call(self, name, arguments):
        before = self.state["native_steps"]
        try:
            if name == "observe":
                if arguments != {}:
                    raise ValueError("observe takes no arguments")
            elif name == "step":
                arguments = self.protocol.check_budget(
                    arguments, self.state["observation_id"],
                    self.state["commands"], self.state["native_steps"])
            else:
                raise ValueError("Unknown BEHAVIOR tool")
        except ValueError as error:
            return ({"success": False, "contentItems": [{"type": "inputText", "text": str(error)}]},
                    {"executed_native_steps": 0, "validation_error": str(error)})
        raw = self.request(name, arguments)
        if not raw["ok"]:
            return ({"success": False, "contentItems": [{"type": "inputText", "text": raw["error"]}]},
                    {"executed_native_steps": 0, "validation_error": raw["error"]})
        return ({"success": True, "contentItems": self.content()},
                {"executed_native_steps": self.state["native_steps"] - before,
                 "observation_id": self.state["observation_id"], "feedback": self.feedback})

    def ended(self):
        return self.state is not None and bool(self.state["ended"])

    def official_success(self):
        return None if self.uncertain or self.state is None else bool(self.state["success"])

    def close(self):
        return self.request("close")
