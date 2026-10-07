"""ReflexBench integration; upstream dynamics, events, scoring and observations stay native."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

WORLD = Path(__file__).resolve().parents[3]
ROOT = WORLD / "third_party/benchmarks/reflexbench"
TASK = "BallCatching-Franka-IK-Abs-v0"
PREFIX = "reflexbench.tasks.manager_based.ball_catching"


def setup(source: Path, output: Path) -> None:
    """Called after AppLauncher; compatibility is outside the read-only checkout."""
    compat_path = WORLD / "third_party/benchmarks/robolab/compat/isaac601.py"
    spec = importlib.util.spec_from_file_location("world_reflexbench_isaac601", compat_path)
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    compat.install()

    # Lab split its sim.utils file into submodules after the 2.2 release.
    import isaaclab.sim.utils as sim_utils
    if "isaaclab.sim.utils.prims" not in sys.modules:
        alias = types.ModuleType("isaaclab.sim.utils.prims")
        alias.clone = sim_utils.clone
        sys.modules[alias.__name__] = alias

    # Avoid importing unrelated task registries / planning dependencies.
    package = Path(source) / "checkout/source/reflexbench/reflexbench"
    for suffix in ("", "tasks", "tasks.manager_based", "tasks.manager_based.ball_catching",
                   "tasks.manager_based.ball_catching.config", "tasks.manager_based.ball_catching.config.franka"):
        name = "reflexbench" + ("." + suffix if suffix else "")
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(package.joinpath(*suffix.split("."))) if suffix else str(package)]
            sys.modules[name] = module
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "reflexbench-boundary.json").write_text(json.dumps({
        "upstream_task": TASK, "runtime": "Isaac Sim 6.0.1 / IsaacLab 2.2 experimental",
        "native_environment_class": PREFIX + ".ball_catching_env.BallCatchingEnv",
        "native_interval_events_per_substep": False,
        "native_observations_include_predicted_intercept": True,
        "future_launch_parameters_exposed": False,
        "reset_policy": "one initial reset, stop on first native termination; no autoreset",
        "changes": ["single environment", "seed", "byte-identical local official asset paths",
                    "API aliases", "recording API compatibility only when no recorder terms",
                    "disable global texture wait for state-only actor; review rendering verified separately"],
    }, indent=2) + "\n")


def build(task_id: str, seed: int, output: Path):
    if task_id not in ("T03", TASK):
        raise ValueError(f"Unsupported ReflexBench task: {task_id}")
    module = importlib.import_module(PREFIX + ".config.franka.ik_abs_env_cfg")
    cfg = module.FrankaBallCatchingEnvCfg_IK_Abs()
    cfg.scene.num_envs = 1
    cfg.seed = seed
    # Kit 110's global assets-loading callback can remain pending after local USD
    # composition. This actor uses state only; review frames are checked separately.
    # Preserve all native reset events and physics, bypass only this render wait.
    cfg.wait_for_textures = False
    if not hasattr(cfg, 'num_rerenders_on_reset'):
        # ReflexBench's newer step uses the count form of this render-only API.
        cfg.num_rerenders_on_reset = int(cfg.rerender_on_reset)
    mapping = {
        "robot": ROOT / "assets/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
        "bucket": ROOT / "assets/Isaac/Props/Mugs/SM_Mug_A2.usd",
        "plane": ROOT / "assets/Isaac/Environments/Grid/default_environment.usd",
    }
    for name, path in mapping.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing original asset {path}; run scripts/eval/reflexbench.sh assets")
        getattr(cfg.scene, name).spawn.usd_path = str(path)
    cfg.viewer.eye = (2.8, 2.0, 2.2)
    cfg.viewer.lookat = (0.7, 0.0, 0.55)
    return cfg


def make_env(cfg):
    native = importlib.import_module(PREFIX + ".ball_catching_env").BallCatchingEnv

    class SingleEpisodeBallCatchingEnv(native):
        """Retain the custom upstream step order, suppress reset only after initial reset."""
        _world_reset_completed = False

        def reset(self, *args, **kwargs):
            if self._world_reset_completed:
                raise RuntimeError("A scored ReflexBench run is a single episode")
            result = super().reset(*args, **kwargs)
            self._world_reset_completed = True
            return result

        def _reset_idx(self, env_ids):
            if not self._world_reset_completed:
                return super()._reset_idx(env_ids)
            # Native step still computes done/reward and records terminal state.
            # Returning here prevents resetting launcher, task phase and physics.
            return None

    env = SingleEpisodeBallCatchingEnv(cfg=cfg, render_mode="rgb_array")
    recorder = env.recorder_manager
    if not hasattr(recorder, "record_post_physics_decimation_step"):
        if recorder.active_terms:
            raise RuntimeError("Lab 2.2 recorder compatibility supports no active native recorder terms")
        recorder.record_post_physics_decimation_step = lambda: None
    return env


def success(env) -> bool:
    """Exact upstream terminal predicate, never equate touching the ball with success."""
    terminations = importlib.import_module(PREFIX + ".mdp.terminations")
    return bool(terminations.task_completed(env)[0].item())


def probe_action(task_id, sim):
    """Hold measured EEF pose; a zero quaternion is not a valid absolute IK probe."""
    from isaaclab.utils.math import subtract_frame_transforms
    robot = sim.env.scene["robot"]
    ee = sim.env.scene["ee_frame"]
    pos, quat = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         ee.data.target_pos_w[:, 0], ee.data.target_quat_w[:, 0])
    return pos[0].tolist() + quat[0].tolist() + [1.0]


def instruction(task_id: str, env) -> str:
    return """Catch the randomly launched ball in the cup carried by the Franka arm and retain it
until the native environment reports task_completed (task_phase == 4). This is ReflexBench's
original BallCatching-Franka-IK-Abs-v0 state-observation task, not a pure-vision benchmark.
The native environment times out after 4 seconds: 100 control steps at 25 Hz, each comprising
4 physics steps at 100 Hz. The ball launches after a randomized 0.3–0.8 s delay at 3.4 m/s,
with randomized elevation 50–65 degrees and yaw offset -15–15 degrees. Future launch timer,
pitch and yaw are private. Use actual post-launch feedback; do not presume a fixed trajectory.

Action is an 8-vector: [x, y, z, qw, qx, qy, qz, gripper]. XYZ are an ABSOLUTE end-effector
pose in the robot root frame (metres), quaternion is WXYZ, normalized. A damped least-squares
IK controller maps it to joints; these values are not normalized joint deltas. Gripper > 0
opens to 0.015 m per finger; gripper <= 0 closes to 0.0. The commanded panda_hand tool offset
is [0,0,0.107] m; the native measured ee frame offset is [0,0,0.1034] m. Do not send a zero
quaternion. Keep a sensible orientation while moving toward the observed intercept.

The native policy observation comprises joint_pos (relative to robot defaults), ee_position,
ball_position, ball_velocity, gripper_width (sum of finger positions / 0.08, clamped), task_phase
(phase / 4), launch_detected, predicted_intercept_pos, predicted_intercept_time. Positions and
velocity are expressed in robot root coordinates. The intercept quantities are PROVIDED BY
THE NATIVE ENVIRONMENT; they are not your own estimate. Their pre-launch zero values are not
a target. Use the recorded observation layout rather than guessing flattened-vector indices.

The cup is a gravity-disabled native object updated to follow the end effector; the upstream
task uses a virtual capture zone and damping. Preserve this mechanism; it is not natural
gripper grasping. Successful contact alone does not complete the task: keep the ball contained
until phase 4. A launched ball below 0.03 m terminates as failure. No dense reward is defined.
You may use coding_control to write your own feedback program returning these same native
actions each control tick. It receives only the declared observations and cannot directly
write simulator state or access an author controller. Simulation advances only during action
execution; model wall-clock deliberation is not an asynchronous real-time latency benchmark.
"""
