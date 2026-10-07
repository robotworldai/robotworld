"""EEF execution extracted from RoboProbe, without its Agent/HTTP layer.

Normal (non-perturbed) reference behavior is retained. See SOURCES.json.
"""
from typing import Any, Mapping
import numpy as np
from environment.robots.arx_x5 import pose
from .types import Action, ActionChunk, ActionSpace, JOINT_CHANNELS, Observation
from .spec import RoboDojoActionSpec, MotionOutcome, InfrastructureFailure

_LEGACY_KEYS = ('xyz', 'quat', 'arm', 'left', 'right')

def named_state(labels, state):
    return dict(zip(labels, ActionSpace().encode(state), strict=True))

def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }

class EefExecutor:
    def __init__(self, action_spec, planner):
        self.action_spec = action_spec
        self._planner = planner
        self._max_duration_s = 10.0
        self._gripper_dwell_step = 0.18
        self._negate_xyz = False
        self._jitter_sphere_per_move = False
        self._jitter_sphere_r = 0.0
        self._jitter_sphere_mean_r = 0.0
        self._last_targets = {}
        self._gripper_goals = {}
        limits = self._declared_step_limits()
        located = ActionSpace(JOINT_CHANNELS).offsets()
        self._arm_step_limits = {}
        self._gripper_step_limits = {}
        for arm in pose.ARMS:
            start, size = located[f'{arm}_arm_joint_state']
            self._arm_step_limits[arm] = limits[start:start + size]
            start, _ = located[f'{arm}_ee_joint_state']
            self._gripper_step_limits[arm] = min(float(limits[start]), self._gripper_dwell_step)

    def _system_message(self) -> str:
        return (
            "You are controlling a real robot embodiment named "
            "'robodojo-arx-x5'. You receive RGB camera images, the current "
            "world-frame grasp-point state of both arms in the same 14 dimensions "
            "move_eef takes, arm joint angles as context you cannot command, "
            "and a task instruction. Move with move_eef by "
            "naming only the world-frame dimensions you want to change. "
            "Cartesian targets must be estimated from RGB; no depth or "
            "world-coordinate query is available. Respond with exactly one tool "
            "call per turn. After each motion the next observation reports how "
            "far the grasp point ended up from what you asked for, so check it "
            "before assuming a motion landed. Two budgets run down at once and "
            f"whichever empties first ends the episode: {self._max_llm_calls} "
            "LLM calls, one per turn, and the environment's own step limit, "
            "reported with each observation as the env steps remaining. A "
            "motion spends env steps in proportion to how far it travels, so a "
            "small correction is cheap and a long reach is not."
            f"\n\nEmbodiment notes:\n{self.action_spec.docs.strip()}"
        )

    def _jitter_offset(self):
        return 0.0, 0.0, 0.0

    def execute_plan(self, arguments, observation):
        return self._handle_motion('move_eef', arguments, observation)

    def tool_spec(self):
        function = self._build_tools(self.action_spec)[0]['function']
        return {'type': 'function', 'name': function['name'],
                'description': function['description'], 'inputSchema': function['parameters']}

    @property
    def _max_steps(self) -> int:
        return max(1, int(np.ceil(self._max_duration_s * self.action_spec.control_hz)))

    def _build_tools(self, action_spec: RoboDojoActionSpec) -> list[dict[str, Any]]:
        terminal_tools = []
        move = _function(
            "move_eef",
            (
                "Move to absolute world-frame grasp-point targets. Name only the "
                "dimensions you intend to change; every unnamed dimension holds "
                "its observed value, so lowering one hand 2 cm is a single "
                "entry. Positions are metres. Orientation is three angles in "
                "degrees measured from a straight-down grasp, not absolute "
                "world angles: at "
                "pitch_deg=roll_deg=yaw_deg=0 the outward tool axis points at "
                "world -z. pitch_deg turns about world +x (0 is straight down, "
                "+90 is horizontal pointing at world +y), roll_deg about world "
                "+y, and yaw_deg about world +z, which is the jaw rotation of a "
                "straight-down grasp. Naming dimensions of both arms moves them "
                "simultaneously along time-aligned paths. The motion plays at a "
                "fixed safe speed, so how long it takes follows from how far it "
                f"goes; a move needing over {self._max_duration_s:g}s is refused "
                "and has to be split. A gripper value named alongside a pose "
                "moves the jaw only after the arm has arrived, so naming both "
                "in one call is a safe way to grasp at a pose. "
                f"Per-dimension bounds: {self._bounds_text()}."
            ),
            {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Map of dimension name to value. Valid names: "
                            + ", ".join(pose.labels())
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "What you observe right now and why you chose this "
                            "motion, in one or two plain sentences."
                        ),
                    },
                },
                "required": ["targets", "note"],
            },
        )
        return [move, *terminal_tools]

    def _axis_bounds(self, label: str) -> tuple[float, float]:
        low, high = pose.bounds(label)
        _, _, axis = label.partition("_")
        if self._negate_xyz and axis in pose.POSITION_AXES:
            return (-high, -low)
        return (low, high)

    def _bounds_text(self) -> str:
        return ", ".join(
            f"{label}: [{low:.4g}, {high:.4g}]"
            for label, (low, high) in (
                (label, self._axis_bounds(label)) for label in pose.labels()
            )
        )

    def _to_model_xyz(self, values: dict[str, float]) -> dict[str, float]:
        if not self._negate_xyz:
            return values
        out = dict(values)
        for axis in pose.POSITION_AXES:
            out[axis] = -float(out[axis])
        return out

    def _from_model_xyz(self, values: dict[str, float]) -> dict[str, float]:
        # Same transform both ways: observing -p and commanding -p recovers p.
        return self._to_model_xyz(values)

    def _state_block(self, observation: Observation) -> str:
        """Render the 14 Cartesian dimensions, then joints as read-only context.

        The Cartesian vector leads and is emitted in ``pose.labels()`` order, so
        it is dimension-for-dimension the space ``move_eef`` targets are
        absolute in. That alignment is the whole point: an absolute target only
        says where to go, so the model needs its starting point named the same
        way to reason about a correction at all.

        Joint angles follow as context the model cannot command. Their two
        gripper dimensions are the same quantity as ``<arm>_gripper`` above, so
        they are left out rather than repeated under a second name.
        """
        measured = {arm: self._measured_values(observation, arm) for arm in pose.ARMS}
        lines = [
            f"Instruction: {observation.instruction or ''}",
            "World-frame grasp-point state, named as move_eef takes them "
            "(metres; degrees from the straight-down reference):",
        ]
        for label in pose.labels():
            arm, _, axis = label.partition("_")
            if axis == "gripper":
                lines.append(f"{label}={self._measured_gripper(observation, arm):.2f}")
            elif measured[arm] is None:
                # Never silently drop a dimension: a missing line reads as
                # "nothing to say about it" when it means the reference the
                # model is about to aim from is not there.
                lines.append(f"{label}=unavailable")
            elif axis in pose.ANGLE_AXES:
                lines.append(f"{label}={measured[arm][axis]:.1f}")
            else:
                lines.append(f"{label}={measured[arm][axis]:.4f}")
        joints = named_state(self.action_spec.labels, observation.state)
        for arm in pose.ARMS:
            joints.pop(f"{arm}_gripper", None)
        lines.append("Arm joint angles, radians, for context; not commandable:")
        lines.extend(f"{label}={value:.4f}" for label, value in joints.items())
        return "\n".join(lines)

    def _arrival_text(self, observation: Observation) -> str:
        """Report how far the previous motion ended up from what was requested."""
        pending, self._last_targets = self._last_targets, {}
        parts = []
        for arm, target in sorted(pending.items()):
            measured = self._measured_pose(observation, arm)
            if measured is None:
                continue
            # Both sides are stored as flange poses because that is what the
            # planner takes, but the model reasons in grasp points, and the
            # two distances differ whenever the orientation missed as well.
            reached = pose.pose_to_values(measured)
            wanted = pose.pose_to_values(target)
            offset = float(
                np.linalg.norm(
                    [reached[axis] - wanted[axis] for axis in pose.POSITION_AXES]
                )
            )
            angle = pose.orientation_error_deg(measured[3:7], target[3:7])
            parts.append(f"{arm} {offset * 1000:.0f} mm and {angle:.1f} deg away")
        if not parts:
            return ""
        return (
            "Arrival check for the previous move_eef target: "
            + "; ".join(parts)
            + "."
        )

    @staticmethod
    def _measured_pose(observation: Observation, arm: str) -> np.ndarray | None:
        raw = np.asarray(
            observation.state.get(f"{arm}_ee_pose", []), dtype=np.float64
        ).reshape(-1)
        if raw.size < 7 or not np.isfinite(raw[:7]).all():
            return None
        if float(np.linalg.norm(raw[3:7])) <= 1e-8:
            return None
        return raw[:7]

    def _measured_values(
        self, observation: Observation, arm: str
    ) -> dict[str, float] | None:
        measured = self._measured_pose(observation, arm)
        if measured is None:
            return None
        return self._to_model_xyz(pose.pose_to_values(measured))

    @staticmethod
    def _measured_gripper(observation: Observation, arm: str) -> float:
        raw = np.asarray(
            observation.state.get(f"{arm}_ee_joint_state", [0.0]), dtype=np.float64
        ).reshape(-1)
        return float(raw[0]) if raw.size else 0.0

    def _handle_motion(
        self,
        name: str,
        arguments: Mapping[str, Any],
        observation: Observation,
    ) -> MotionOutcome:
        del name
        legacy = [key for key in _LEGACY_KEYS if key in arguments]
        if legacy:
            return self._repair(
                f"move_eef no longer takes {legacy[0]!r}. Name world-frame "
                'dimensions instead, for example {"targets": {"left_z": 0.90, '
                '"left_yaw_deg": -90}}.'
            )
        by_arm, error = self._parse_targets(arguments.get("targets"))
        if error:
            return self._repair(error)
        assert by_arm is not None

        paths: dict[str, np.ndarray] = {}
        target_poses: dict[str, np.ndarray] = {}
        requested_poses: dict[str, np.ndarray] = {}
        if self._jitter_sphere_per_move:
            self._resample_sphere_jitter()
        jx, jy, jz = self._jitter_offset()
        for arm in sorted(by_arm):
            axes = {
                axis: value
                for axis, value in by_arm[arm].items()
                if axis != "gripper"
            }
            if not axes:
                continue
            measured = self._measured_values(observation, arm)
            if measured is None:
                raise InfrastructureFailure(
                    f"RoboDojo reported no usable {arm}_ee_pose; an absolute "
                    "Cartesian target cannot be built from a missing flange pose"
                )
            # measured is already in the model frame; fold overrides there, then
            # map positions back to the world before CuRobo sees them.
            merged = {**measured, **axes}
            world = self._from_model_xyz(merged)
            requested_pose = pose.values_to_pose(world)
            executed = dict(world)
            if jx or jy or jz:
                executed["x"] = float(executed["x"]) + jx
                executed["y"] = float(executed["y"]) + jy
                executed["z"] = float(executed["z"]) + jz
            target_pose = pose.values_to_pose(executed)
            path, failure = self._plan_arm(arm, target_pose)
            if failure is not None:
                return MotionOutcome(
                    chunk=None,
                    tool_result=f"{arm} arm {failure.tool_result} Neither arm moved.",
                    repairable=failure.repairable,
                )
            assert path is not None
            paths[arm] = path
            # Arrival checks the model-named pose; the planner flew to the
            # jittered one. Keeping those distinct is what makes the bias
            # visible as residual error on the next turn.
            requested_poses[arm] = requested_pose
            target_poses[arm] = target_pose

        grippers = {
            arm: (self._measured_gripper(observation, arm), overrides["gripper"])
            for arm, overrides in by_arm.items()
            if overrides.get("gripper") is not None
        }
        # The jaw moves only once the arm has arrived. Closing it while the
        # hand is still travelling sweeps the object out of the way before the
        # grasp, which is the most common complaint in the recorded give_up
        # reasons. Sequencing costs the jaw's own waypoints on top of the
        # arm's, which is why the derivation adds the two phases.
        arm_steps = self._arm_steps(paths)
        # A stationary/single-point plan still needs a hold sample before a
        # simultaneous jaw change. The reference otherwise indexes an empty
        # resampled path when gripper_steps > 0. No upstream file is changed.
        if paths and arm_steps == 0:
            arm_steps = 1
        gripper_steps = self._gripper_steps(grippers)
        steps = arm_steps + gripper_steps
        if steps > self._max_steps:
            return self._repair(
                f"this motion needs {steps / self.action_spec.control_hz:.1f}s at "
                f"the safe speed, over the {self._max_duration_s:g}s playout cap; "
                "split it into smaller motions"
            )
        if steps == 0:
            # Every named dimension already holds the value it was asked for.
            arm_steps = steps = 1
        # The count is never clamped to the planner's own point count.
        # Resampling interpolates along the planned polyline, so asking for
        # more waypoints than CuRobo returned adds points on that same
        # collision-checked path; clamping instead would be the compression
        # this derivation exists to avoid, only in miniature.
        synchronized = {
            arm: self._resample_path(path, arm_steps) for arm, path in paths.items()
        }
        actions = []
        for index in range(steps):
            data = self._held_action_data(observation)
            for arm in by_arm:
                if arm in synchronized:
                    # Both arms travel over the same waypoints and arrive
                    # together, then hold while the jaws move.
                    data[f"{arm}_arm_joint_state"] = synchronized[arm][
                        min(index, arm_steps - 1)
                    ]
                if arm in grippers and index >= arm_steps:
                    data[f"{arm}_ee_joint_state"][:] = grippers[arm][1]
            actions.append(
                Action(
                    data=data,
                    meta={"chunk_final": True} if index == steps - 1 else {},
                )
            )

        self._last_targets = dict(requested_poses)
        # Latched after the waypoints are built, so the arm travels to the
        # release pose still holding the previous grip and only lets go on
        # arrival.
        for arm, (_, target) in grippers.items():
            self._gripper_goals[arm] = float(target)
        trace = {
            "tool": "move_eef",
            "mode": "bimanual" if len(by_arm) == 2 else "single_arm",
            "arms": {
                arm: {
                    "named": {
                        f"{arm}_{axis}": value
                        for axis, value in sorted(by_arm[arm].items())
                    },
                    "target_pose": (
                        requested_poses[arm].round(6).tolist()
                        if arm in requested_poses
                        else None
                    ),
                    "executed_pose": (
                        target_poses[arm].round(6).tolist()
                        if arm in target_poses
                        else None
                    ),
                    "source_waypoints": len(paths[arm]) if arm in paths else 0,
                }
                for arm in sorted(by_arm)
            },
            "plan_status": "Success",
            "planned_waypoints": steps,
            "arm_waypoints": arm_steps,
            "gripper_waypoints": gripper_steps,
            "normalized_from_parallel_calls": bool(
                arguments.get("_normalized_from_parallel_calls")
            ),
            "jitter_dx": jx,
            "jitter_dy": jy,
            "jitter_dz": jz,
            "jitter_sphere_r": self._jitter_sphere_r,
            "jitter_sphere_mean_r": self._jitter_sphere_mean_r,
            "jitter_sphere_per_move": self._jitter_sphere_per_move,
        }
        return MotionOutcome(
            chunk=ActionChunk(
                actions=actions,
                control_hz=self.action_spec.control_hz,
                meta={"trace": trace},
            ),
            tool_result=self._accepted_text(steps, gripper_steps, target_poses),
        )

    def _accepted_text(
        self,
        steps: int,
        gripper_steps: int,
        target_poses: Mapping[str, np.ndarray],
    ) -> str:
        text = f"Accepted: playing {steps} waypoints"
        text += f" ({steps / self.action_spec.control_hz:.1f}s)."
        if gripper_steps:
            text += (
                f" The last {gripper_steps} move the gripper, once the arm has "
                "arrived."
            )
        if target_poses:
            text += " The next observation reports how close the grasp point landed."
        return text

    def _parse_targets(
        self, raw: Any
    ) -> tuple[dict[str, dict[str, float]] | None, str | None]:
        if not isinstance(raw, Mapping) or not raw:
            return None, (
                "move_eef.targets must be a non-empty object of dimension "
                "name to value"
            )
        valid = pose.labels()
        by_arm: dict[str, dict[str, float]] = {}
        for raw_label, raw_value in raw.items():
            label = str(raw_label)
            if label not in valid:
                return None, (
                    f"unknown dimension {label!r}; valid names: {', '.join(valid)}"
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                return None, (
                    f"value for {label!r} must be a finite number, got {raw_value!r}"
                )
            value = float(raw_value)
            if not np.isfinite(value):
                return None, (
                    f"value for {label!r} must be a finite number, got {raw_value!r}"
                )
            low, high = self._axis_bounds(label)
            if not low <= value <= high:
                return None, (
                    f"target for {label!r} is outside [{low:.4g}, {high:.4g}]"
                )
            arm, _, axis = label.partition("_")
            by_arm.setdefault(arm, {})[axis] = value
        return by_arm, None

    def _plan_arm(
        self, arm: str, target_pose: np.ndarray
    ) -> tuple[np.ndarray | None, MotionOutcome | None]:
        planned = self._planner(arm=arm, target_pose=target_pose)
        if planned.get("status") != "Success" or planned.get("position") is None:
            status = str(planned.get("status", "Fail"))
            return None, MotionOutcome(
                chunk=None,
                tool_result=(
                    f"pose unreachable (planner status: {status}). The robot did "
                    "not move; choose another pose."
                ),
                repairable=False,
            )
        path = np.asarray(planned["position"], dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 6 or not np.isfinite(path).all():
            raise InfrastructureFailure(
                f"EEF planner returned path shape {path.shape}; expected finite (N, 6)"
            )
        if len(path) == 0:
            raise InfrastructureFailure("EEF planner returned an empty joint path")
        return path, None

    def _declared_step_limits(self) -> np.ndarray:
        """The per-step budget for each of the 14 joint-space dimensions.

        The embodiment's declaration governs. A dimension that declares nothing
        is one whose whole range already fits inside a step, so its range is the
        only bound left to apply.
        """
        labels = self.action_spec.labels
        declared = self.action_spec.max_step or (None,) * len(labels)
        span = self.action_spec.high - self.action_spec.low
        limits = np.asarray(
            [
                span[index] if entry is None else entry
                for index, entry in enumerate(declared)
            ],
            dtype=np.float64,
        )
        if not np.isfinite(limits).all() or bool(np.any(limits <= 0.0)):
            offenders = [
                label for label, limit in zip(labels, limits, strict=True) if limit <= 0.0
            ]
            raise InfrastructureFailure(
                "every action dimension needs a positive per-step budget to "
                f"derive a waypoint count from; {offenders} have none"
            )
        return limits

    def _arm_steps(self, paths: Mapping[str, np.ndarray]) -> int:
        """Waypoints needed to keep every joint under its declared per-step limit.

        Resampling is uniform over the planned path, so the travel one env step
        asks for is the path's travel divided by the count. A CuRobo plan can
        double back, so the per-joint total variation is the honest measure of
        that travel; end-to-end displacement would under-count a detour. Each
        joint is measured against its own limit, since a declaration is
        per-dimension and nothing says the six move alike.
        """
        steps = 0
        for arm, path in paths.items():
            if len(path) < 2:
                continue
            travel = np.abs(np.diff(path, axis=0)).sum(axis=0)
            needed = np.ceil(travel / self._arm_step_limits[arm]).max()
            steps = max(steps, int(needed))
        return steps

    def _gripper_steps(self, grippers: Mapping[str, tuple[float, float]]) -> int:
        """Env steps the jaw's setpoint is held for, scaled to its travel."""
        steps = 0
        for arm, (start, target) in grippers.items():
            travel = abs(target - start)
            steps = max(
                steps, int(np.ceil(travel / self._gripper_step_limits[arm]))
            )
        return steps

    @staticmethod
    def _resample_path(path: np.ndarray, steps: int) -> np.ndarray:
        """Resample to ``steps`` waypoints ending on the planned target.

        The planner's first point is where the arm already is, so it is not a
        waypoint: commanding it spends an env step standing still, and at
        ``steps == 1`` it would be the only thing commanded and the arm would
        never set off. Sampling starts one interval in, as upstream's
        interpolation does, which also makes every interval exactly the travel
        the step count was derived from.
        """
        source = np.linspace(0.0, 1.0, len(path))
        target = np.linspace(0.0, 1.0, steps + 1)[1:]
        return np.stack(
            [np.interp(target, source, path[:, joint]) for joint in range(6)],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _repair(message: str) -> MotionOutcome:
        return MotionOutcome(chunk=None, tool_result=message, repairable=True)

    def _held_action_data(self, observation: Observation) -> dict[str, np.ndarray]:
        """Hold the arms where they are and the jaws on what they were told.

        The jaw cannot hold by echoing its own reading. RoboDojo reports
        ``<arm>_ee_joint_state`` from the previous command after its
        20%-of-range clamp, not from the finger position, so re-sending it
        parks the jaw at the width it already has: the position error, and with
        it the grip force, goes to zero and the object slides out of a closed
        hand during the next transfer. Re-sending the value the model last
        asked for keeps the clamp a full 20% ahead of the fingers, which is the
        same squeeze the closing phase applied. Until the model has named a
        jaw, the reading is all there is.
        """
        data = {
            name: np.asarray(observation.state[name], dtype=np.float32).copy()
            for name in (
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            )
        }
        for arm, goal in self._gripper_goals.items():
            data[f"{arm}_ee_joint_state"][:] = goal
        return data
