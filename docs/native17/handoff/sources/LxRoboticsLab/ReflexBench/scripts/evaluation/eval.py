#!/usr/bin/env python3
# Copyright (c) 2022-2026, ReflexBench Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate policies under simulated inference latency via a remote policy server.

Time model
~~~~~~~~~~
This script models **two independent frequencies** in the deployment loop:

1. **Control frequency** (``--ctrl_freq``, Hz) - how often the robot
   consumes an action.  One ``env.step()`` advances the sim by
   ``decimation`` physics sub-steps; ``ctrl_freq = sim_freq / decimation``.
   Example: ``--ctrl_freq 100`` -> one action is executed every 10 ms.

2. **Inference frequency** (``--infer_freq``, Hz) - how often a new
   action chunk is produced by the policy.  Internally stored as an
   **inference period** in milliseconds: ``infer_period_ms = 1000 /
   infer_freq``.  Example: ``--infer_freq 25`` -> one chunk every 40 ms.

Internally all durations use ms and are converted to physics sub-steps::

    sim_steps = round(infer_period_ms / (sim_dt * 1000))

``sim_dt`` is the physics timestep (e.g. 0.01 s for 100 Hz).  Because the
inference period is expressed in ms it is independent of ``ctrl_freq``;
``--infer_freq 25`` always means 40 ms of world time between chunks,
whether the robot controls at 25 Hz or 100 Hz.

Typical deployment pattern
--------------------------
Control is usually **faster** than inference: e.g. ``--ctrl_freq 100``
(10 ms per action) + ``--infer_freq 25`` (40 ms between chunks) +
``--execution_horizon 8``.  Each 40 ms window consumes 4 actions out of
an 8-action chunk, leaving headroom for chunk boundaries to stay
aligned.  If ``--execution_horizon`` is smaller than the number of
control steps per inference period, the extra steps fall back to a
freeze-arm action (sync) or the previous chunk (async).

Simulated Synchronous Inference (``--inference_mode sync``)
    1. At sim time *t*, capture O_t.  Pause sim, call policy (blocking).
    2. Advance physics by ``infer_period_ms`` with a **freeze-arm**
       action (latency penalty).  Sub-control-step remainders are
       handled via ``advance_sim`` when available.
    3. Execute the **just-returned** action chunk - one action per
       control step, up to ``execution_horizon`` steps.
    4. Repeat from 1.

Simulated Asynchronous Inference (``--inference_mode async``)
    1. At sim time *t*, capture O_t.  Pause sim, call policy (blocking).
    2. Advance physics by ``infer_period_ms`` using the **previous
       cycle's** action chunk (pipelined).
    3. Store the just-returned actions for the *next* cycle.
    4. Repeat from 1.
    Actions are always one-cycle stale, reproducing real-world
    pipelined deployment.
    With ``--async_time_align_actions``, the just-returned chunk is promoted
    with a start index derived from elapsed inference time, so stale actions
    at the beginning of the chunk are skipped when it is later executed.

Both modes use ``--infer_freq <Hz>`` as the fixed inference frequency.
Pass ``--use_real_latency`` to replace the fixed value with the measured
wall-clock inference time converted to sim time by the measured real-time
factor (RTF).

How to use
----------
1. Start a policy server that implements the API in POLICY_SERVER.md
   (GET /info, POST /predict, optional POST /reset). Example::

       python your_policy_server.py   # e.g. listen on http://localhost:8000

2. From the project root, run eval with Isaac Sim (headless or with UI)::

       python scripts/evaluation/eval.py --task <TASK> --server_url <URL> [options]

   Or from scripts/evaluation/::

       python eval.py --task <TASK> --server_url <URL> [options]

3. Required: ``--task`` (e.g. ConveyorBeltPickAndPlace-Franka-v0),
   ``--server_url`` (e.g. http://localhost:8000).

4. Recommended options (pass both frequencies explicitly):
   ``--ctrl_freq <Hz>``, ``--infer_freq <Hz>``, ``--inference_mode sync|async``,
   ``--use_real_latency`` (override fixed period with measured wall-clock),
   ``--obs_mode state|vla``, ``--control joint_pos|ik_abs|ik_rel``,
   ``--state_format auto|joint|eef_pose|both``,
   ``--orientation_rep quat|euler`` (default quat, same as collect),
   ``--num_episodes``, ``--output results.json``.  With ``--state_format auto``
   (default) state follows control (joint_pos -> joint, ik_abs|ik_rel -> eef_pose),
   matching data collection.

Usage examples
--------------
    # Sync, control 100 Hz + inference 10 Hz (100 ms period), state obs, 50 episodes
    python eval.py --task ConveyorBeltPickAndPlace-Franka-v0 \\
        --server_url http://localhost:8000 \\
        --ctrl_freq 100 --inference_mode sync --infer_freq 10 \\
        --num_episodes 50 --output results_sync.json

    # Async, real latency, VLA + IK absolute (state_format=auto -> eef_pose, orient=quat like collect)
    python eval.py --task ConveyorBeltPickAndPlace-Franka-DataCollection-v0 \\
        --server_url http://localhost:8000 \\
        --ctrl_freq 100 --inference_mode async --use_real_latency \\
        --control ik_abs --obs_mode vla --task_description "pick up the object" \\
        --num_episodes 20 --output results_async.json

    # Control 25 Hz + inference 25 Hz (matched frequencies), joint pos
    python eval.py --task ConveyorBeltPickAndPlace-Franka-v0 \\
        --server_url http://localhost:8000 \\
        --ctrl_freq 25 --infer_freq 25 --control joint_pos
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime

from isaaclab.app import AppLauncher

# ------------------------- CLI ------------------------- #

parser = argparse.ArgumentParser(
    description="Evaluate policies with simulated inference latency.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

g_task = parser.add_argument_group("Task")
g_task.add_argument("--task", type=str, default=None, help="Registered Gym task name (auto-derived from --task_profile if omitted, or vice versa)")
g_task.add_argument("--num_envs", type=int, default=1)
g_task.add_argument("--num_episodes", type=int, default=100, help="Total episodes to evaluate")
g_task.add_argument(
    "--task_profile",
    type=str,
    default=None,
    help="Task profile name from task_profiles.py (e.g. conveyor_belt_pick_and_place). "
    "Enables task-specific success criteria, action_scale, camera config, etc.",
)

g_inf = parser.add_argument_group("Inference")
g_inf.add_argument(
    "--backend",
    type=str,
    default="server",
    choices=["server", "local"],
    help="Inference backend: server (HTTP policy server) or local (load ACT model directly). Default: server",
)
g_inf.add_argument("--server_url", type=str, default=None, help="Policy server base URL (required for --backend server)")
g_inf.add_argument("--model_path", type=str, default=None, help="Path to pretrained ACT model (required for --backend local)")
g_inf.add_argument(
    "--temporal_ensemble_coeff",
    type=float,
    default=None,
    help="Temporal ensemble coefficient for ACT (local backend only, None=disabled)",
)
g_inf.add_argument(
    "--execution_horizon",
    type=int,
    default=None,
    help="Max actions to execute per inference. "
    "Local: default=model chunk_size. Server: default=/info action_horizon. "
    "If the server returns fewer steps than this cap, only the returned steps are executed.",
)
g_inf.add_argument(
    "--inference_mode",
    type=str,
    default="async",
    choices=["sync", "async"],
    help="sync = pause + penalty; async = pipeline. Default: async",
)
g_inf.add_argument(
    "--infer_freq",
    type=float,
    default=20.0,
    help="Inference frequency in Hz (default: 20, i.e. one chunk every 50 ms). "
    "Example: 25 = 40 ms per inference.",
)
g_inf.add_argument(
    "--use_real_latency",
    action="store_true",
    help="Measure actual wall-clock inference time (ms) per cycle and use "
    "RTF-converted sim-time latency as the inference period, overriding --infer_freq.",
)
g_inf.add_argument(
    "--async_time_align_actions",
    action="store_true",
    help="Async mode only: when a newly inferred action chunk becomes executable, "
    "start from the action index closest to the elapsed inference time instead "
    "of always starting from index 0.",
)
g_inf.add_argument(
    "--async_chunk_steps",
    type=int,
    default=None,
    help="Async mode only: max number of control steps to execute from each "
    "action chunk before switching to the latest completed pending chunk. "
    "Default: fixed-period async switches at the latency-window cadence; "
    "server --use_real_latency async switches when a pending chunk is ready.",
)

g_ctrl = parser.add_argument_group("Control & representation")
g_ctrl.add_argument(
    "--control",
    type=str,
    default="joint_pos",
    choices=["joint_pos", "ik_abs", "ik_rel"],
    help="Robot control mode (joint_pos: 7+1, ik_abs: 6+1, ik_rel: 6+1). Default: joint_pos",
)
g_ctrl.add_argument(
    "--state_format",
    type=str,
    default="auto",
    choices=["auto", "joint", "eef_pose", "both"],
    help="Proprioception format: auto (follow control, same as collect) | joint | eef_pose | both. Default: auto",
)
g_ctrl.add_argument(
    "--orientation_rep",
    type=str,
    default="quat",
    choices=["quat", "euler"],
    help="EEF orientation in state: quat (w,x,y,z, 4D) or euler (roll,pitch,yaw, 3D). Default: quat (same as collect)",
)
g_ctrl.add_argument(
    "--action_scale",
    type=float,
    default=1.0,
    help="Scale factor for arm/pose actions before env.step (gripper unchanged). Default: 1.0",
)

g_obs = parser.add_argument_group("Observation")
g_obs.add_argument(
    "--obs_mode",
    type=str,
    default="vla",
    choices=["state", "vla"],
    help="state = flat vector; vla = images + proprioception. Default: vla",
)
g_obs.add_argument("--task_description", type=str, default="manipulate the object")
g_obs.add_argument(
    "--image_history",
    type=int,
    default=1,
    help="Number of recent camera frames sent to the policy per inference "
    "(oldest -> newest). E.g. 2 sends the current frame and the previous one. "
    "Default: 1 (current frame only). Server payload uses nested lists when >1.",
)
g_obs.add_argument(
    "--image_history_stride",
    type=int,
    default=1,
    help="Real-time frame interval for --image_history. "
    "1 = consecutive frames (t-1,t), 2 = skip one frame (t-2,t). "
    "History is sampled from frames recorded as sim/control time advances, not from inference calls.",
)
g_obs.add_argument("--cam_names", type=str, default="", help="Comma-separated; empty = auto-detect")
g_obs.add_argument("--ee_frame", type=str, default="ee_frame")
g_obs.add_argument("--arm_joints", type=str, default="panda_joint.*")
g_obs.add_argument("--finger_joints", type=str, default="panda_finger.*")

g_out = parser.add_argument_group("Output")
g_out.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
g_out.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Directory to save per-episode videos for ALL envs. Omit to disable.",
)
g_out.add_argument("--video_fps", type=int, default=30, help="Video FPS (default: 30)")
g_out.add_argument(
    "--video_cam",
    type=str,
    default=None,
    help="Scene camera name for per-env video (auto-detect if omitted). "
    "Requires a task with cameras (e.g. DataCollection variant).",
)

g_sim = parser.add_argument_group("Sim")
g_sim.add_argument("--disable_fabric", action="store_true", default=False)
g_sim.add_argument(
    "--ctrl_freq",
    type=float,
    default=None,
    help="Override robot control frequency (Hz). Adjusts decimation while keeping "
    "physics dt unchanged. Default: use task config (e.g. 50 Hz).",
)
g_sim.add_argument(
    "--debug_whack",
    action="store_true",
    default=False,
    help="Print WhackAMole press-detection diagnostics each control step.",
)
g_sim.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate --infer_freq and derive the internal inference period (ms).
if args_cli.infer_freq <= 0:
    parser.error("--infer_freq must be positive")
if args_cli.image_history <= 0:
    parser.error("--image_history must be positive")
if args_cli.image_history_stride <= 0:
    parser.error("--image_history_stride must be positive")
if args_cli.async_chunk_steps is not None and args_cli.async_chunk_steps <= 0:
    parser.error("--async_chunk_steps must be positive when set")
args_cli.infer_period_ms = 1000.0 / args_cli.infer_freq

# Validate backend requirements
if args_cli.backend == "server" and not args_cli.server_url:
    parser.error("--server_url is required when --backend server")
if args_cli.backend == "local" and not args_cli.model_path:
    parser.error("--model_path is required when --backend local")

# Local backend always needs cameras (ACT uses images)
if args_cli.backend == "local":
    args_cli.enable_cameras = True
if args_cli.obs_mode == "vla" or args_cli.video_dir:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------- Post-launcher imports --------------- #

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reflexbench.tasks  # noqa: F401, E402
from reflexbench.reset_utils import clone_obs_buf, refresh_manager_observations, seed_everything  # noqa: E402
from policy_client import PolicyClient, encode_images_base64  # noqa: E402
from task_profiles import (  # noqa: E402
    TASK_PROFILES,
    apply_whack_a_mole_eval_overrides,
    calculate_smoothness,
    check_success as _tp_check_success,
    install_phase_tracker,
    resolve_profile,
)

# Resolve --task <-> --task_profile (bidirectional auto-derivation).
# Users may pass either; the other is filled in automatically. If both are given,
# exact registered task/profile mismatches are rejected while variant overrides
# such as --task ...-IK-Abs-v0 with a DataCollection-v0-bound profile remain valid.
_GYM_ID_TO_PROFILE = {cfg["gym_id"]: key for key, cfg in TASK_PROFILES.items()}

if args_cli.task is None and args_cli.task_profile is None:
    parser.error("At least one of --task or --task_profile must be specified")

if args_cli.task is None:
    # profile -> gym_id
    if args_cli.task_profile not in TASK_PROFILES:
        parser.error(
            f"Unknown --task_profile '{args_cli.task_profile}'. "
            f"Available: {list(TASK_PROFILES.keys())}"
        )
    args_cli.task = TASK_PROFILES[args_cli.task_profile]["gym_id"]
    print(f"[INFO] --task auto-derived from profile: {args_cli.task}")
elif args_cli.task_profile is None:
    # gym_id -> profile (exact match on DataCollection-v0 variants)
    if args_cli.task in _GYM_ID_TO_PROFILE:
        args_cli.task_profile = _GYM_ID_TO_PROFILE[args_cli.task]
        print(f"[INFO] --task_profile auto-derived from task: {args_cli.task_profile}")
    else:
        print(
            f"[INFO] No registered profile matches --task {args_cli.task}; "
            f"running without task_profile (success checks, phase names and "
            f"task-specific overrides will be skipped). Pass --task_profile "
            f"explicitly to enable them."
        )
else:
    if args_cli.task_profile not in TASK_PROFILES:
        parser.error(
            f"Unknown --task_profile '{args_cli.task_profile}'. "
            f"Available: {list(TASK_PROFILES.keys())}"
        )
    exact_profile = _GYM_ID_TO_PROFILE.get(args_cli.task)
    if exact_profile is not None and exact_profile != args_cli.task_profile:
        parser.error(
            f"--task {args_cli.task} maps to --task_profile '{exact_profile}', "
            f"but got '{args_cli.task_profile}'."
        )

# Control mode -> action format (same as collect_vla_data.py for eval/collect consistency)
_CONTROL_TO_ACTION_FMT: dict[str, str] = {
    "joint_pos": "abs_joint",
    "ik_abs": "abs_eef_pose",
    "ik_rel": "rel_eef_pose",
}


# ------------------- Local ACT Model ---------------------- #


def _ensure_lerobot_importable():
    """Ensure lerobot is importable."""
    try:
        import lerobot  # noqa: F401
        return
    except ImportError:
        pass
    src = os.environ.get("LEROBOT_SRC", os.path.expanduser("~/lerobot/src"))
    if os.path.isdir(src):
        sys.path.insert(0, src)
    else:
        raise ImportError(
            "Cannot import lerobot. Install via pip or set LEROBOT_SRC "
            "to the path containing the lerobot package (e.g. /path/to/lerobot/src)."
        )


def load_act_model(model_path: str, temporal_ensemble_coeff: float | None = None):
    """Load ACT policy, preprocessor, postprocessor from a pretrained path."""
    print("[INFO] Loading ACT model ...")
    _ensure_lerobot_importable()
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.realpath(os.path.expanduser(model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    policy = ACTPolicy.from_pretrained(model_path)

    if temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff, policy.config.chunk_size
        )
        print(f"  Temporal Ensemble enabled: coeff={temporal_ensemble_coeff}")

    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(
        f"  Model loaded on {device}, chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}"
    )
    return policy, preprocessor, postprocessor, device


# ------------------- Video Writer ---------------------- #


class AsyncVideoWriter:
    """Background-thread video writer that manages per-env streams.

    All encoding and disk I/O happens on a single daemon thread.
    The main loop only calls non-blocking ``put_frame`` / ``finish_episode``.
    While recording, each env writes to a temp file; ``finish_episode``
    renames it to include the episode index and success/fail tag.
    """

    _SENTINEL = object()

    def __init__(self, video_dir: str, fps: int = 30, name: str = "cam"):
        self.video_dir = video_dir
        self.fps = fps
        # Unique tag used in tmp filenames so multiple writers sharing the
        # same ``video_dir`` (e.g. one writer per camera) do not collide on
        # ``_rec_env{env_id}.mp4``. Without this, concurrent writers open
        # the same path with ``av.open(..., mode="w")`` and trample each
        # other's bytes; the first writer to call ``os.replace`` wins the
        # filename and the second writer's payload is silently lost.
        self.name = name
        os.makedirs(video_dir, exist_ok=True)

        self._q: queue.Queue = queue.Queue(maxsize=2000)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- public API (called from main thread, non-blocking) --

    def start_episode(self, env_id: int) -> None:
        """Begin recording a new episode for *env_id*."""
        self._q.put(("START", env_id, None, None))

    def put_frame(self, env_id: int, frame: np.ndarray) -> None:
        """Enqueue one RGB frame for *env_id*. Drops if queue is full."""
        try:
            self._q.put_nowait(("FRAME", env_id, frame, None))
        except queue.Full:
            pass

    def finish_episode(self, env_id: int, ep_idx: int, success: bool, suffix: str = "") -> None:
        """Finalize video for *env_id*, rename with ep_idx and result."""
        self._q.put(("END", env_id, ep_idx, success, suffix))

    def close(self) -> None:
        """Flush all remaining items and stop the background thread."""
        self._q.put(self._SENTINEL)
        self._thread.join(timeout=60)

    # -- background thread --

    def _loop(self) -> None:
        import av as _av
        from fractions import Fraction

        # env_id -> (container, stream, frame_count, temp_path)
        writers: dict[int, tuple[_av.container.OutputContainer, _av.stream.Stream, int, str]] = {}

        def _open(env_id: int) -> None:
            # Include ``self.name`` so co-located writers (one per camera)
            # never share a tmp path; see __init__ for rationale.
            tmp = os.path.join(
                self.video_dir, f"_rec_{self.name}_env{env_id}.mp4"
            )
            container = _av.open(tmp, mode="w")
            stream = container.add_stream("libx264", rate=self.fps)
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, self.fps)
            writers[env_id] = (container, stream, 0, tmp)

        def _write(env_id: int, frame_rgb: np.ndarray) -> None:
            if env_id not in writers:
                return
            container, stream, count, tmp = writers[env_id]
            h, w = frame_rgb.shape[:2]
            if count == 0:
                stream.width = w
                stream.height = h
            vf = _av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            vf.pts = count
            vf.time_base = Fraction(1, self.fps)
            for packet in stream.encode(vf):
                container.mux(packet)
            writers[env_id] = (container, stream, count + 1, tmp)

        def _close(env_id: int) -> None:
            if env_id not in writers:
                return
            container, stream, _, tmp = writers.pop(env_id)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            return tmp

        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                for eid in list(writers):
                    tmp = _close(eid)
                    if tmp is not None and os.path.isfile(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                break

            if len(item) == 5:
                tag, env_id, arg1, arg2, arg3 = item
            else:
                tag, env_id, arg1, arg2 = item
                arg3 = ""

            if tag == "START":
                if env_id in writers:
                    tmp = _close(env_id)
                    if tmp is not None and os.path.isfile(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                _open(env_id)

            elif tag == "FRAME":
                _write(env_id, arg1)

            elif tag == "END":
                tmp = _close(env_id)
                if tmp is not None:
                    ep_idx, success = arg1, arg2
                    suffix = arg3 or ""
                    label = "success" if success else "fail"
                    name = f"episode_{ep_idx:04d}_{label}"
                    if suffix:
                        name += f"_{suffix}"
                    final = os.path.join(
                        self.video_dir, f"{name}.mp4"
                    )
                    try:
                        os.replace(tmp, final)
                    except OSError:
                        pass


# ----------------------- Evaluator --------------------- #


class Evaluator:
    """Run evaluation under simulated inference latency.

    Two independent frequencies drive the eval loop:
      * ``ctrl_freq`` (Hz): how often an action is consumed by the env.
      * ``infer_freq`` (Hz): how often a new action chunk is produced.
        Internally stored as ``infer_period_ms = 1000 / infer_freq``.
    """

    def __init__(self, args: argparse.Namespace):
        # Task
        self.task: str = args.task
        self.num_envs: int = args.num_envs
        self.num_episodes: int = args.num_episodes

        # Task profile (optional)
        self.task_profile: dict | None = None
        if args.task_profile:
            self.task_profile = resolve_profile(args.task_profile)
            if self.task_profile is None:
                raise ValueError(
                    f"Unknown task profile '{args.task_profile}'. "
                    f"Available: {list(TASK_PROFILES.keys())}"
                )

        # Backend
        self.backend: str = args.backend
        self.server_url: str | None = args.server_url
        self.model_path: str | None = args.model_path
        self.temporal_ensemble_coeff: float | None = args.temporal_ensemble_coeff
        self.execution_horizon_arg: int | None = args.execution_horizon

        # Inference
        self.inference_mode: str = args.inference_mode
        # Inference period in ms (internal SoT), derived from --infer_freq in
        # the CLI post-parse block above.
        self.infer_period_ms: float = args.infer_period_ms
        self.use_real_latency: bool = args.use_real_latency
        self.async_time_align_actions: bool = args.async_time_align_actions
        self.async_chunk_steps: int | None = args.async_chunk_steps

        # Control & representation (aligned with collect: action/state/rotation follow controller)
        self.control_mode: str = args.control
        self.action_fmt: str = _CONTROL_TO_ACTION_FMT[self.control_mode]
        if getattr(args, "state_format", "auto") == "auto":
            self.state_fmt = "joint" if self.control_mode == "joint_pos" else "eef_pose"
        else:
            self.state_fmt = args.state_format
        self.orient_rep: str = args.orientation_rep
        # action_scale: prefer task_profile if set, else CLI
        if self.task_profile and "action_scale" in self.task_profile:
            self.action_scale: float = self.task_profile["action_scale"]
        else:
            self.action_scale = args.action_scale

        # Observation
        self.obs_mode: str = args.obs_mode
        self.task_description: str = args.task_description
        self.cam_arg: str = args.cam_names
        self.ee_frame_name: str = args.ee_frame
        self.arm_pattern: str = args.arm_joints
        self.finger_pattern: str = args.finger_joints
        self.needs_eef: bool = self.state_fmt in ("eef_pose", "both")
        self.image_history: int = args.image_history
        self.image_history_stride: int = args.image_history_stride

        # Sim override
        self.ctrl_freq_override: float | None = args.ctrl_freq
        self.seed: int = args.seed
        self.debug_whack: bool = args.debug_whack

        # Output - bare filename auto-placed in eval_results/<profile>/
        self.output_path: str | None = args.output
        if self.output_path and os.sep not in self.output_path and '/' not in self.output_path:
            profile_slug = args.task_profile or "default"
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _project_root = os.path.abspath(os.path.join(_script_dir, "../.."))
            self.output_path = os.path.join(
                _project_root, "scripts", "act_training", "eval_results",
                profile_slug, self.output_path
            )
        self.video_dir: str | None = args.video_dir
        self.video_fps: int = args.video_fps
        self.video_cam_arg: str | None = args.video_cam

        # Populated by setup()
        self.env: gym.Env = None  # type: ignore[assignment]
        self.menv: ManagerBasedRLEnv = None  # type: ignore[assignment]
        self.client: PolicyClient | None = None
        self.env_dt: float = 0.0
        self.action_horizon: int = 1
        self.cam_names: list[str] = []
        self.arm_ids: list[int] = []
        self.finger_ids: list[int] = []

        # Local backend model objects (populated by setup)
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._local_device = None
        # joint_pos + (local ACT | server): absolute 7+1 targets, converted per-step
        self._raw_abs_chunks: np.ndarray | None = None   # current inference abs actions
        self._prev_abs_chunks: np.ndarray | None = None  # previous cycle (for async)
        self._action_chunk_lengths: np.ndarray | None = None  # current per-env valid H
        self._prev_action_chunk_lengths: np.ndarray | None = None  # previous per-env valid H
        self._last_action_align_ms: float = 0.0
        self._warned_async_action_offset_exhausted: bool = False
        self._warned_async_chunk_steps_not_longer: bool = False
        self._rtf_wall_t0: float | None = None
        self._rtf_wall_t_end: float | None = None
        self._rtf_sim_elapsed_s: float = 0.0
        self._rtf_last_value: float = 1.0
        self._latency_wall_ms: list[float] = []
        self._latency_sim_ms: list[float] = []
        self._latency_rtf: list[float] = []
        # Per-env past camera frames for multi-frame policy inputs (cam -> list per env)
        self._image_histories: dict[str, list[list[np.ndarray]]] = {}
        self._warned_local_image_history: bool = False

        # Per-env episode tracking
        self._ep_return: list[float] = []
        self._ep_length: list[int] = []
        self._max_phase: list[int] = []
        self._ep_actions: list[list] = []  # per-env action history for smoothness
        self._success_at_term: list[bool] = []
        self._phase_at_term: list[int] = []
        self._has_term_snapshot_hook: bool = False
        self.results: list[dict] = []

    # --------------- Setup --------------- #

    def setup(self) -> None:
        # Seed
        seed_everything(self.seed)

        # --- Env config ---
        env_cfg = parse_env_cfg(
            self.task,
            device=args_cli.device,
            num_envs=self.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )

        # Override control mode (arm action space)
        self._apply_control_mode(env_cfg)

        # Task profile: DataCollectionEnvCfg already carries both cameras and the
        # wrist_cam Fabric-sync workaround, so no runtime scene patching needed.
        if self.task_profile:
            tp = self.task_profile

            # WhackAMole episode length override
            if tp.get("success_type") == "whack_a_mole" and "eval_episode_length_s" in tp:
                env_cfg.episode_length_s = float(tp["eval_episode_length_s"])
                print(f"[INFO] WhackAMole episode length override: {env_cfg.episode_length_s}s")

            # IK override from profile
            if tp.get("has_ik_override"):
                pass  # handled below after env creation

        # Disable observation noise for eval
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            if hasattr(env_cfg.observations.policy, "enable_corruption"):
                env_cfg.observations.policy.enable_corruption = False

        # Suppress print_pos event if present
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "print_pos"):
            env_cfg.events.print_pos = None

        # render_interval = 1 for camera rendering every physics step
        env_cfg.sim.render_interval = 1

        # Override robot control frequency by adjusting decimation only.
        if self.ctrl_freq_override is not None:
            new_dec = max(1, round(1.0 / (env_cfg.sim.dt * self.ctrl_freq_override)))
            old_freq = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
            env_cfg.decimation = new_dec
            actual_freq = 1.0 / (env_cfg.sim.dt * new_dec)
            print(
                f"[INFO] Control freq override: {old_freq:.1f} Hz -> {actual_freq:.1f} Hz "
                f"(decimation {new_dec}, physics dt={env_cfg.sim.dt})"
            )

        self.sim_dt: float = env_cfg.sim.dt
        self.sim_freq: float = 1.0 / self.sim_dt
        self.decimation: int = env_cfg.decimation
        self.env_dt: float = self.sim_dt * self.decimation
        self.ctrl_freq: float = 1.0 / self.env_dt
        self.env = gym.make(self.task, cfg=env_cfg)
        self.menv = self.env.unwrapped  # type: ignore[assignment]

        # IK control changes the action dimension; disable the env's
        # built-in IK override which assumes joint_pos (8D) actions.
        if self.control_mode in ("ik_abs", "ik_rel"):
            self.menv.enable_ik_override = False
        if self.task_profile and self.task_profile.get("has_ik_override"):
            self.menv.enable_ik_override = False
            print("  IK override DISABLED - policy controls the full episode")

        # Phase tracker
        self._has_phase_tracker = install_phase_tracker(self.menv)

        # Video: one writer per camera (fixed_cam, wrist_cam, etc.)
        self._video_writers: dict[str, AsyncVideoWriter] = {}  # cam_name -> writer
        self._video_actual_dir: str | None = None
        if self.video_dir:
            video_cams = self._resolve_video_cams()
            if video_cams:
                task_safe = re.sub(r"[<>:\"/\\|?*]", "_", self.task).strip(" .") or "task"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                subdir = f"{task_safe}_{timestamp}"
                self._video_actual_dir = os.path.join(self.video_dir, subdir)
                for cam in video_cams:
                    self._video_writers[cam] = AsyncVideoWriter(
                        self._video_actual_dir, self.video_fps, name=cam
                    )
                print(f"[INFO] Video recording cameras: {video_cams}")
            else:
                print("[WARN] No scene camera found for per-env video. "
                      "Use a task with cameras (e.g. DataCollection variant) "
                      "or specify --video_cam. Video disabled.")

        # Enable per-physics-step video frame buffering inside the env
        if self._video_writers:
            if hasattr(self.menv, "enable_substep_recording"):
                for cam in self._video_writers:
                    self.menv.enable_substep_recording(cam)

        # --- Inference backend setup ---
        if self.backend == "server":
            self.client = PolicyClient(self.server_url)
            info = self.client.get_info()
            info_h = int(info.get("action_horizon", 1))
            if self.execution_horizon_arg is not None:
                self.action_horizon = self.execution_horizon_arg
            else:
                self.action_horizon = info_h
            if self.execution_horizon_arg is not None and self.execution_horizon_arg != info_h:
                print(
                    f"[INFO] --execution_horizon={self.action_horizon} caps execution; "
                    f"server /info reports action_horizon={info_h}. "
                    "Each /predict response is truncated to min(returned_H, cap)."
                )
        elif self.backend == "local":
            policy, preprocessor, postprocessor, device = load_act_model(
                self.model_path, self.temporal_ensemble_coeff
            )
            self._policy = policy
            self._preprocessor = preprocessor
            self._postprocessor = postprocessor
            self._local_device = device
            chunk_size = policy.config.chunk_size
            self.action_horizon = self.execution_horizon_arg or chunk_size
            print(f"  Execution horizon: {self.action_horizon} (model chunk_size={chunk_size})")

        if self.inference_mode == "async" and not self.use_real_latency:
            lat_sim = round(self.infer_period_ms / (self.sim_dt * 1000.0))
            lat_ctrl = lat_sim // self.decimation + (1 if lat_sim % self.decimation else 0)
            if self.action_horizon < lat_ctrl:
                print(
                    f"[WARN] action_horizon ({self.action_horizon}) < control steps "
                    f"per inference period ({lat_ctrl}). Some steps will reuse the last action."
                )
            if self.async_time_align_actions:
                start_idx = self._elapsed_to_action_start_idx(self.infer_period_ms)
                if start_idx >= self.action_horizon:
                    print(
                        f"[WARN] --async_time_align_actions start index ({start_idx}) "
                        f">= action_horizon ({self.action_horizon}); async execution "
                        "will fall back to freeze actions for that chunk."
                    )

        # VLA sensors (always needed for local backend; conditional for server)
        robot = self.menv.scene["robot"]
        self.arm_ids = list(robot.find_joints([self.arm_pattern])[0])
        self.finger_ids = list(robot.find_joints([self.finger_pattern])[0])
        if self.obs_mode == "vla" or self.backend == "local":
            self.cam_names = self._detect_cameras()
            self._image_histories = {
                cam: [[] for _ in range(self.num_envs)] for cam in self.cam_names
            }

        # Per-env tracking
        self._ep_return = [0.0] * self.num_envs
        self._ep_length = [0] * self.num_envs
        self._max_phase = [0] * self.num_envs
        self._ep_actions = [[] for _ in range(self.num_envs)]
        self._success_at_term = [False] * self.num_envs
        self._phase_at_term = [-1] * self.num_envs
        self._has_term_snapshot_hook = self._install_termination_snapshot_hook()

        self._print_config()

    def _apply_control_mode(self, env_cfg) -> None:
        """Override env_cfg.actions.arm_action to match the desired control mode."""
        if self.control_mode == "joint_pos":
            return

        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        env_cfg.scene.robot.spawn.activate_contact_sensors = True

        if self.control_mode == "ik_abs":
            env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.107]
                ),
            )
            print("[INFO] Control mode: IK absolute pose (action dim = 6 + 1 gripper)")
        elif self.control_mode == "ik_rel":
            env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=True,
                    ik_method="dls",
                ),
                scale=1.0,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, 0.107]
                ),
            )
            print("[INFO] Control mode: IK relative pose (action dim = 6 + 1 gripper)")

    @staticmethod
    def _quat_to_euler(q: torch.Tensor) -> torch.Tensor:
        """Quaternion (w,x,y,z) -> euler (roll, pitch, yaw).  (..., 4) -> (..., 3)."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return torch.stack([roll, pitch, yaw], dim=-1)

    def _detect_cameras(self) -> list[str]:
        if self.cam_arg:
            return [
                n.strip()
                for n in self.cam_arg.split(",")
                if n.strip() and n.strip() in self.menv.scene.keys()
            ]
        found: list[str] = []
        for key in self.menv.scene.keys():
            ent = self.menv.scene[key]
            if (
                hasattr(ent, "data")
                and hasattr(ent.data, "output")
                and isinstance(ent.data.output, dict)
                and "rgb" in ent.data.output
            ):
                found.append(key)
        return found

    def _resolve_video_cams(self) -> list[str]:
        """Pick scene cameras for per-env video capture. Returns all cameras with rgb output."""
        if self.video_cam_arg:
            requested = [c.strip() for c in self.video_cam_arg.split(",") if c.strip()]
            valid = [c for c in requested if c in self.menv.scene.keys()]
            invalid = [c for c in requested if c not in self.menv.scene.keys()]
            if invalid:
                print(f"[WARN] --video_cam cameras not in scene: {invalid}")
            return valid
        found: list[str] = []
        for key in self.menv.scene.keys():
            ent = self.menv.scene[key]
            if (
                hasattr(ent, "data")
                and hasattr(ent.data, "output")
                and isinstance(ent.data.output, dict)
                and "rgb" in ent.data.output
            ):
                found.append(key)
        return found

    def _print_config(self) -> None:
        lat_sim = round(self.infer_period_ms / (self.sim_dt * 1000.0))
        print(f"\n{'=' * 60}")
        print(f"  Task            : {self.task}")
        print(f"  Backend         : {self.backend}")
        if self.task_profile:
            print(f"  Task profile    : {self.task_profile.get('success_type', '?')}")
        print(f"  Inference mode  : {self.inference_mode}")
        if self.inference_mode == "async":
            if self.async_chunk_steps is not None:
                async_chunk_label = str(self.async_chunk_steps)
            elif self.use_real_latency and self.backend == "server":
                async_chunk_label = "pending-ready"
            else:
                async_chunk_label = "latency window"
            print(f"  Async time align: {self.async_time_align_actions}")
            print(f"  Async chunk steps: {async_chunk_label}")
        if self.use_real_latency:
            if self.inference_mode == "async" and self.backend == "server":
                print("  Async scheduler : realtime Future/pending chunks")
                print("  Infer latency   : wall /predict latency * measured RTF")
            else:
                print("  Infer period    : wall inference time * measured RTF")
        else:
            infer_hz = 1000.0 / self.infer_period_ms if self.infer_period_ms > 0 else float('inf')
            print(
                f"  Infer period    : {self.infer_period_ms:.1f} ms "
                f"(= {infer_hz:.2f} Hz, {lat_sim} sim steps)"
            )
        print(f"  Action horizon  : {self.action_horizon}")
        print(f"  Control mode    : {self.control_mode}")
        print(f"  Action format   : {self.action_fmt}")
        print(f"  State format    : {self.state_fmt} (orient={self.orient_rep})")
        print(f"  Action scale    : {self.action_scale}")
        print(f"  Obs mode        : {self.obs_mode}")
        if self.obs_mode == "vla" or self.backend == "local":
            print(f"  Image history   : {self.image_history}")
            print(f"  Image hist stride: {self.image_history_stride}")
        print(f"  Num envs        : {self.num_envs}")
        print(f"  Num episodes    : {self.num_episodes}")
        print(f"  Seed            : {self.seed}")
        print(f"  Sim freq        : {self.sim_freq:.0f} Hz (sim_dt={self.sim_dt:.4f}s)")
        print(f"  Ctrl freq       : {self.ctrl_freq:.1f} Hz (decimation={self.decimation})")
        if self.backend == "server":
            print(f"  Server          : {self.server_url}")
        else:
            print(f"  Model path      : {self.model_path}")
        if self.cam_names:
            print(f"  Cameras         : {self.cam_names}")
        if self._video_writers and self._video_actual_dir:
            print(f"  Video           : {self._video_actual_dir} @ {self.video_fps} FPS (cams={list(self._video_writers.keys())})")
        print(f"{'=' * 60}\n")

    def _install_termination_snapshot_hook(self) -> bool:
        if not hasattr(self.menv, "_reset_idx"):
            return False

        original_reset_idx = self.menv._reset_idx

        def _patched_reset_idx(env_ids, *args, **kwargs):
            if isinstance(env_ids, torch.Tensor):
                if env_ids.ndim == 0:
                    env_id_list = [int(env_ids.item())]
                else:
                    env_id_list = [int(eid) for eid in env_ids.tolist()]
            else:
                env_id_list = [int(eid) for eid in env_ids]

            tc = getattr(self.menv, "_last_step_task_completed", None)
            term = getattr(self.menv, "reset_terminated", None)
            time_outs = getattr(self.menv, "reset_time_outs", None)
            has_phase = hasattr(self.menv, "task_phase")

            for eid in env_id_list:
                if has_phase:
                    self._phase_at_term[eid] = int(self.menv.task_phase[eid].item())

                if self.task_profile:
                    self._success_at_term[eid] = bool(
                        _tp_check_success(self.menv, self.task_profile, eid)
                    )
                elif tc is not None:
                    tc_eid = tc[eid]
                    self._success_at_term[eid] = (
                        bool(tc_eid.item()) if hasattr(tc_eid, "item") else bool(tc_eid)
                    )
                elif has_phase:
                    self._success_at_term[eid] = self._phase_at_term[eid] >= 4
                elif term is not None and time_outs is not None:
                    term_eid = term[eid]
                    timeout_eid = time_outs[eid]
                    is_term = bool(term_eid.item()) if hasattr(term_eid, "item") else bool(term_eid)
                    is_timeout = bool(timeout_eid.item()) if hasattr(timeout_eid, "item") else bool(timeout_eid)
                    self._success_at_term[eid] = is_term and not is_timeout
                else:
                    self._success_at_term[eid] = False

            return original_reset_idx(env_ids, *args, **kwargs)

        self.menv._reset_idx = _patched_reset_idx
        return True

    def _clone_obs(self, obs_buf) -> dict | torch.Tensor:
        """Return a copy of obs_buf (dict of tensors or single tensor)."""
        return clone_obs_buf(obs_buf)

    def _apply_post_reset_task_overrides(self) -> None:
        if self.task_profile:
            apply_whack_a_mole_eval_overrides(self.menv, self.task_profile)

    def _refresh_obs_after_reset(self):
        return self._clone_obs(refresh_manager_observations(self.menv))

    def _reset_env_for_episode(self):
        obs, info = self.env.reset()
        if self.task_profile:
            self._apply_post_reset_task_overrides()
            obs = self._refresh_obs_after_reset()
        # Episode-boundary policy state reset (local backend only).
        # Must happen at the START of the first episode so any in-policy
        # buffers (ACT's _action_queue / temporal_ensembler.ensembled_actions)
        # do not carry over from a previously-loaded checkpoint or prior run.
        self._reset_policy_state()
        return obs, info

    def _reset_policy_state(self) -> None:
        """Clear in-policy per-episode state (local backend).

        ACTPolicy.reset() wipes either ``_action_queue`` (used by
        ``select_action``) or ``temporal_ensembler.ensembled_actions``
        (used when ``temporal_ensemble_coeff`` is set). Our ``_local_infer``
        currently calls ``predict_action_chunk`` directly, which is
        stateless, so this is effectively a no-op today. It is still
        correct to call at episode boundaries only (NOT per inference)
        so that if the inference path ever switches to ``select_action``
        or ``temporal_ensembler.update`` the semantics hold automatically.
        """
        if self.backend == "local" and self._policy is not None:
            self._policy.reset()

    def _capture_camera_images(self) -> dict[str, list[np.ndarray]]:
        """Capture current RGB frames for all envs and configured cameras."""
        images: dict[str, list[np.ndarray]] = {}
        for cam in self.cam_names:
            raw = self.menv.scene[cam].data.output["rgb"]
            images[cam] = [
                raw[i, ..., :3].cpu().numpy().astype(np.uint8)
                for i in range(self.num_envs)
            ]
        return images

    def _build_image_sequences(
        self, current: dict[str, list[np.ndarray]]
    ) -> dict[str, list[list[np.ndarray]]]:
        """Build per-env image sequences of length ``image_history`` (oldest first)."""
        sequences: dict[str, list[list[np.ndarray]]] = {}
        for cam in self.cam_names:
            env_seqs: list[list[np.ndarray]] = []
            past_by_env = self._image_histories.get(
                cam, [[] for _ in range(self.num_envs)]
            )
            for eid in range(self.num_envs):
                past = past_by_env[eid]
                cur = current[cam][eid]
                # For image_history > 1, the buffer is updated whenever sim time
                # advances, so it already represents real-time frames up to now.
                seq = list(past) if self.image_history > 1 else [cur]
                if not seq:
                    seq = [cur]
                selected = [
                    seq[-1 - i * self.image_history_stride]
                    if len(seq) > i * self.image_history_stride
                    else seq[0]
                    for i in range(self.image_history - 1, -1, -1)
                ]
                env_seqs.append(selected)
            sequences[cam] = env_seqs
        return sequences

    def _commit_image_history(self, current: dict[str, list[np.ndarray]]) -> None:
        """Append frames captured after real sim-time advances."""
        if self.image_history <= 1:
            return
        max_frames = (self.image_history - 1) * self.image_history_stride + 1
        for cam in self.cam_names:
            if cam not in self._image_histories:
                self._image_histories[cam] = [[] for _ in range(self.num_envs)]
            for eid in range(self.num_envs):
                hist = self._image_histories[cam][eid]
                hist.append(current[cam][eid].copy())
                if len(hist) > max_frames:
                    self._image_histories[cam][eid] = hist[-max_frames:]

    def _seed_image_history(self, env_ids: list[int] | None = None) -> None:
        """Initialize history for new episodes with the current camera frame."""
        if self.image_history <= 1 or not self.cam_names:
            return
        current = self._capture_camera_images()
        ids = range(self.num_envs) if env_ids is None else env_ids
        for cam in self.cam_names:
            if cam not in self._image_histories:
                self._image_histories[cam] = [[] for _ in range(self.num_envs)]
            for eid in ids:
                self._image_histories[cam][eid] = [current[cam][eid].copy()]

    def _clear_image_history(self, env_ids: list[int]) -> None:
        """Drop stored camera history for envs that just reset."""
        if self.image_history <= 1 or not self._image_histories:
            return
        for cam in self._image_histories:
            for eid in env_ids:
                self._image_histories[cam][eid] = []

    def _build_obs(
        self,
        obs,
        current_frames: dict[str, list[np.ndarray]] | None = None,
    ) -> dict:
        """Package env observation for the policy server."""
        payload: dict = {
            "num_envs": self.num_envs,
            "step_ids": list(self._ep_length),
        }

        if self.obs_mode == "state":
            state = obs["policy"] if isinstance(obs, dict) else obs
            payload["type"] = "state"
            payload["state"] = state.cpu().numpy().tolist()
            payload["control_mode"] = self.control_mode
            payload["action_format"] = self.action_fmt
            payload["state_format"] = self.state_fmt
            payload["orientation_rep"] = self.orient_rep
            return payload

        # VLA mode: proprioception + images
        payload["type"] = "vla"
        robot = self.menv.scene["robot"]

        finger_mean = robot.data.joint_pos[:, self.finger_ids].mean(dim=-1, keepdim=True)
        gripper_state = (finger_mean > 0.035).float().cpu().numpy()

        proprio: dict = {"gripper_state": gripper_state.tolist()}

        if self.state_fmt in ("joint", "both"):
            joint_pos = robot.data.joint_pos[:, self.arm_ids].cpu().numpy()
            proprio["joint_positions"] = joint_pos.tolist()

        if self.needs_eef and self.ee_frame_name in self.menv.scene.keys():
            from isaaclab.utils.math import subtract_frame_transforms

            ee = self.menv.scene[self.ee_frame_name]
            pos_b, quat_b = subtract_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                ee.data.target_pos_w[:, 0, :],
                ee.data.target_quat_w[:, 0, :],
            )
            proprio["eef_pos"] = pos_b.cpu().numpy().tolist()
            if self.orient_rep == "euler":
                orient = self._quat_to_euler(quat_b).cpu().numpy()
            else:
                orient = quat_b.cpu().numpy()
            proprio["eef_orient"] = orient.tolist()

        payload["proprioception"] = proprio
        payload["control_mode"] = self.control_mode
        payload["action_format"] = self.action_fmt
        payload["state_format"] = self.state_fmt
        payload["orientation_rep"] = self.orient_rep

        if current_frames is None:
            current_frames = self._capture_camera_images()
        sequences = self._build_image_sequences(current_frames)
        images: dict[str, list] = {}
        for cam in self.cam_names:
            if self.image_history <= 1:
                images[cam] = encode_images_base64(
                    [seq[-1] for seq in sequences[cam]]
                )
            else:
                images[cam] = [
                    encode_images_base64(seq) for seq in sequences[cam]
                ]
        payload["images"] = images
        if self.image_history > 1:
            payload["image_history"] = self.image_history
            payload["image_history_stride"] = self.image_history_stride
        payload["task_description"] = self.task_description

        return payload

    # --------- Phase / episode tracking -------- #

    def _track_phase(self) -> None:
        """Snapshot max task_phase *before* the step that may trigger auto-reset."""
        if not hasattr(self.menv, "task_phase"):
            return
        for eid in range(self.num_envs):
            ph = int(self.menv.task_phase[eid].item())
            if ph > self._max_phase[eid]:
                self._max_phase[eid] = ph

    # ----------- Video capture ----------- #

    def _capture_frame(self) -> None:
        """Record camera frames after sim time advances and enqueue videos."""
        current_frames = None
        if self.image_history > 1 and self.cam_names:
            current_frames = self._capture_camera_images()
            self._commit_image_history(current_frames)

        if not self._video_writers:
            return
        for cam_name, writer in self._video_writers.items():
            for eid in range(self.num_envs):
                if current_frames is not None and cam_name in current_frames:
                    frame = current_frames[cam_name][eid]
                else:
                    rgb = self.menv.scene[cam_name].data.output["rgb"]
                    frame = rgb[eid, ..., :3].cpu().numpy().astype(np.uint8)
                writer.put_frame(eid, frame)

    def _process_dones(
        self,
        rew: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> list[int]:
        """Record finished episodes. Returns list of env IDs that just reset.

        Also applies task-profile ``short_circuit_success``: envs that are not
        yet done but satisfy ``check_success`` are force-reset here so the
        episode ends at the exact step of success. WhackAMole sets
        ``short_circuit_success=False`` so valid_hits can accumulate across
        the full episode.
        """
        dones = terminated | truncated
        has_phase = hasattr(self.menv, "task_phase")
        reset_ids: list[int] = []

        # Per-env done flag after possibly injecting a short-circuit reset.
        done_flags = [bool(dones[eid].item()) for eid in range(self.num_envs)]
        if (
            self.task_profile
            and self.task_profile.get("short_circuit_success", True)
            and self._has_term_snapshot_hook
        ):
            for eid in range(self.num_envs):
                if done_flags[eid]:
                    continue
                if not _tp_check_success(self.menv, self.task_profile, eid):
                    continue
                # Force-reset this env. The patched _reset_idx hook will
                # snapshot _success_at_term[eid]=True and _phase_at_term[eid]
                # before the real reset runs.
                reset_env_ids = torch.tensor(
                    [eid], device=self.menv.device, dtype=torch.long
                )
                self.menv._reset_idx(reset_env_ids)
                done_flags[eid] = True

        for eid in range(self.num_envs):
            self._ep_return[eid] += rew[eid].item()
            self._ep_length[eid] += 1

            if not done_flags[eid]:
                continue

            # Determine success: task_profile check_success > env signal > phase > term
            if self._has_term_snapshot_hook:
                success = self._success_at_term[eid]
            elif self.task_profile:
                success = _tp_check_success(self.menv, self.task_profile, eid)
            else:
                tc = getattr(self.menv, "_last_step_task_completed", None)
                if tc is not None:
                    success = bool(tc[eid].item()) if hasattr(tc[eid], "item") else bool(tc[eid])
                elif has_phase:
                    success = self._max_phase[eid] >= 4
                else:
                    success = bool(terminated[eid]) and not bool(truncated[eid])

            # Smoothness
            avg_diff, avg_jerk = calculate_smoothness(self._ep_actions[eid])

            # Phase at termination (from phase tracker patch)
            phase_at_term = (
                self._phase_at_term[eid]
                if self._has_term_snapshot_hook
                else getattr(self.menv, "_phase_at_term", -1)
            )

            ep_idx = len(self.results)
            end_reason = "success" if success else ("timeout" if bool(truncated[eid]) else "terminated")
            self.results.append({
                "episode": ep_idx,
                "success": success,
                "return": round(self._ep_return[eid], 4),
                "length": self._ep_length[eid],
                "max_phase": self._max_phase[eid] if has_phase else -1,
                "terminated_phase": phase_at_term,
                "end_reason": end_reason,
                "avg_diff": round(avg_diff, 4),
                "avg_jerk": round(avg_jerk, 4),
            })

            n = len(self.results)
            sr = sum(r["success"] for r in self.results) / n * 100
            tag = "OK" if success else "FAIL"
            print(
                f"  [{tag}] ep {n}: len={self._ep_length[eid]}  "
                f"ret={self._ep_return[eid]:.1f}  Ph:{self._max_phase[eid]}->{phase_at_term}  "
                f"Jk:{avg_jerk:.4f}  |  SR={sr:.1f}% ({n}/{self.num_episodes})"
            )

            if self._video_writers:
                for cam_name, writer in self._video_writers.items():
                    writer.finish_episode(eid, ep_idx, success, suffix=cam_name)
                if not self._done_enough():
                    for writer in self._video_writers.values():
                        writer.start_episode(eid)

            self._ep_return[eid] = 0.0
            self._ep_length[eid] = 0
            self._max_phase[eid] = 0
            self._ep_actions[eid] = []
            if self._has_term_snapshot_hook:
                self._success_at_term[eid] = False
                self._phase_at_term[eid] = -1
            reset_ids.append(eid)

        if reset_ids:
            if self.client is not None:
                self.client.notify_reset(reset_ids)
            self._clear_image_history(reset_ids)
            # Local backend: episode-boundary policy state reset. Note that
            # ACT's temporal ensembler buffer is GLOBAL (not per-env), so any
            # episode boundary on any env invalidates the buffer for all
            # envs. This is acceptable for num_envs=1 (the supported config
            # for temporal ensemble); for num_envs>1 with temporal ensemble
            # enabled, the buffer would get clipped whenever ANY env resets
            # - the upstream LeRobot ensembler does not support per-env
            # state, so this is a known limitation, not a new one.
            self._reset_policy_state()
        return reset_ids

    def _done_enough(self) -> bool:
        return len(self.results) >= self.num_episodes

    # ----------- Common inference call ----------- #

    def _build_local_obs(
        self,
        env_id: int = 0,
        current_frames: dict[str, list[np.ndarray]] | None = None,
    ) -> dict:
        """Build observation dict for local ACT model (single env)."""
        if self.image_history > 1 and not self._warned_local_image_history:
            print(
                "[WARN] --image_history > 1 is supported for server backend only; "
                "local ACT backend uses the latest frame."
            )
            self._warned_local_image_history = True

        robot = self.menv.scene["robot"]
        if current_frames is None:
            current_frames = self._capture_camera_images()
        sequences = self._build_image_sequences(current_frames)
        fixed_rgb = sequences["fixed_cam"][env_id][-1]
        wrist_rgb = sequences["wrist_cam"][env_id][-1]
        joint_pos = robot.data.joint_pos[env_id, self.arm_ids].cpu().numpy()
        finger_pos = robot.data.joint_pos[env_id, self.finger_ids].mean().item()
        gripper_state = np.array([1.0 if finger_pos > 0.03 else 0.0], dtype=np.float32)
        state = np.concatenate([joint_pos, gripper_state]).astype(np.float32)

        dev = self._local_device
        return {
            "observation.images.fixed_cam": (
                torch.from_numpy(fixed_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev)
            ),
            "observation.images.wrist_cam": (
                torch.from_numpy(wrist_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev)
            ),
            "observation.state": (
                torch.from_numpy(state).float().unsqueeze(0).to(dev)
            ),
        }

    def _local_infer(self) -> tuple[torch.Tensor, float]:
        """Run local ACT model inference for all envs.

        Returns **raw absolute** actions (N, H, D) as numpy on CPU +
        wall-clock ms.  The caller must use ``_convert_abs_action()``
        per step to get env-ready relative actions with the *actual*
        current joint position (more accurate than pre-computing).
        """
        all_chunks = []
        t0 = time.monotonic()
        current_frames = self._capture_camera_images()
        for eid in range(self.num_envs):
            obs_dict = self._build_local_obs(eid, current_frames=current_frames)
            obs = self._preprocessor(obs_dict)
            with torch.no_grad():
                raw_chunk = self._policy.predict_action_chunk(obs)
            chunk = self._postprocessor(raw_chunk).cpu().numpy()
            if chunk.ndim == 3:
                chunk = chunk.squeeze(0)
            all_chunks.append(chunk[:self.action_horizon])
        wall_ms = (time.monotonic() - t0) * 1000.0

        # Stack into (N, H, D)
        max_h = max(c.shape[0] for c in all_chunks)
        action_dim = all_chunks[0].shape[-1]
        actions_np = np.zeros((self.num_envs, max_h, action_dim), dtype=np.float32)
        for eid, chunk in enumerate(all_chunks):
            actions_np[eid, :chunk.shape[0]] = chunk

        # Store raw absolute chunks (will be converted per-step)
        self._raw_abs_chunks = actions_np
        self._action_chunk_lengths = None
        period_ms = self._real_latency_period_ms(wall_ms)
        self._last_action_align_ms = self._convert_real_latency_to_sim_ms(wall_ms)

        # Return a (N, H, 8) tensor - placeholder filled per-step via _convert_abs_action
        result = torch.zeros(
            (self.num_envs, max_h, 8), device=self.menv.device, dtype=torch.float32
        )
        return result, period_ms

    def _convert_abs_action(self, h: int) -> torch.Tensor:
        """Convert raw absolute action at chunk index *h* to env-ready relative action.

        Uses the **actual** current joint position for accurate abs -> rel conversion,
        matching the original eval_act_sync_chunk.py behaviour.
        """
        return self._convert_abs_action_from(self._raw_abs_chunks, h)

    def _convert_abs_action_from(self, abs_chunks: np.ndarray, h: int) -> torch.Tensor:
        """Convert absolute action from arbitrary chunk array at index *h*."""
        robot = self.menv.scene["robot"]
        env_acts = []
        for eid in range(self.num_envs):
            act = abs_chunks[eid, h]
            current_joint = robot.data.joint_pos[eid, self.arm_ids].cpu().numpy()
            joint_rel = (act[:7] - current_joint) / self.action_scale
            gripper = 1.0 if float(act[7]) > 0.5 else -1.0
            env_act = np.concatenate([joint_rel, np.array([gripper], dtype=np.float32)])
            env_acts.append(env_act)
        return torch.tensor(np.array(env_acts), device=self.menv.device, dtype=torch.float32)

    def _normalize_server_actions(self, raw_actions) -> tuple[np.ndarray, np.ndarray]:
        """Convert server actions to padded (N, H, D) plus per-env valid lengths."""

        def _expected_action_dim() -> int:
            if self.control_mode == "joint_pos":
                return 8
            return int(self.env.action_space.shape[-1])

        def _coerce_env_chunk(chunk) -> np.ndarray:
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.ndim == 1:
                if arr.size == 0:
                    return np.zeros((0, _expected_action_dim()), dtype=np.float32)
                arr = arr[np.newaxis, :]
            if arr.ndim != 2:
                raise ValueError(
                    f"Each env action chunk must have shape (H, D), got {arr.shape}"
                )
            return np.ascontiguousarray(arr[: self.action_horizon], dtype=np.float32)

        try:
            actions_np = np.asarray(raw_actions, dtype=np.float32)
        except ValueError:
            actions_np = None

        if actions_np is not None:
            if actions_np.ndim == 2:
                if actions_np.shape[0] == self.num_envs:
                    actions_np = actions_np[:, np.newaxis, :]
                elif self.num_envs == 1:
                    actions_np = actions_np[np.newaxis, :, :]
                else:
                    raise ValueError(
                        f"Server actions shape {actions_np.shape} is ambiguous for "
                        f"num_envs={self.num_envs}; expected (N,H,D), ragged list, or (N,D)."
                    )
            if actions_np.ndim != 3:
                raise ValueError(
                    f"Server actions must have shape (N,H,D), got {actions_np.shape}"
                )
            if actions_np.shape[0] != self.num_envs:
                raise ValueError(
                    f"Server actions batch N={actions_np.shape[0]} != num_envs={self.num_envs}"
                )
            h_use = min(actions_np.shape[1], self.action_horizon)
            actions_np = np.ascontiguousarray(actions_np[:, :h_use, :], dtype=np.float32)
            lengths = np.full((self.num_envs,), h_use, dtype=np.int64)
            return actions_np, lengths

        if not isinstance(raw_actions, (list, tuple)) or len(raw_actions) != self.num_envs:
            raise ValueError(
                f"Ragged server actions must be a list of length num_envs={self.num_envs}"
            )

        chunks = [_coerce_env_chunk(chunk) for chunk in raw_actions]
        dims = [chunk.shape[-1] for chunk in chunks if chunk.shape[0] > 0]
        action_dim = dims[0] if dims else _expected_action_dim()
        for eid, chunk in enumerate(chunks):
            if chunk.shape[-1] != action_dim:
                raise ValueError(
                    f"Server actions for env {eid} have action_dim={chunk.shape[-1]}, "
                    f"expected {action_dim}"
                )

        lengths = np.array([chunk.shape[0] for chunk in chunks], dtype=np.int64)
        max_h = int(lengths.max()) if lengths.size else 0
        actions_np = np.zeros((self.num_envs, max_h, action_dim), dtype=np.float32)
        for eid, chunk in enumerate(chunks):
            if chunk.shape[0] > 0:
                actions_np[eid, : chunk.shape[0], :] = chunk
        return actions_np, lengths

    def _select_chunk_action(
        self,
        actions: torch.Tensor,
        h: int,
        abs_chunks: np.ndarray | None = None,
        lengths: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Select step h, freezing only envs whose returned chunk is exhausted."""
        if h >= actions.shape[1]:
            return self._build_freeze_action()

        if lengths is None:
            valid = np.ones((self.num_envs,), dtype=bool)
        else:
            valid = h < lengths
        if not bool(np.any(valid)):
            return self._build_freeze_action()

        if abs_chunks is not None:
            selected = self._convert_abs_action_from(abs_chunks, h)
        else:
            selected = actions[:, h, :]

        if bool(np.all(valid)):
            return selected

        action = self._build_freeze_action()
        valid_ids = torch.tensor(np.nonzero(valid)[0], device=self.menv.device, dtype=torch.long)
        action[valid_ids] = selected[valid_ids]
        return action

    def _decode_server_result(
        self, result: dict
    ) -> tuple[torch.Tensor, float, np.ndarray | None, np.ndarray | None]:
        """Convert a server /predict response into executable chunk storage."""
        wall_ms = float(result.get("wall_latency_s", 0.0)) * 1000.0
        server_ms = float(result.get("server_latency_s", 0.0)) * 1000.0
        # For chunk time alignment, prefer the policy server's model latency.
        # It reflects when the action sequence was produced, while wall latency
        # may include client transport/serialization overhead.
        raw_align_ms = server_ms if server_ms > 0 else wall_ms
        self._last_action_align_ms = self._convert_real_latency_to_sim_ms(raw_align_ms)

        actions_np, action_lengths = self._normalize_server_actions(result["actions"])

        # joint_pos: server returns absolute joint targets + gripper; env.step
        # consumes relative deltas generated lazily from the actual current q.
        if self.control_mode == "joint_pos":
            if actions_np.shape[0] != self.num_envs:
                raise ValueError(
                    f"Server actions batch N={actions_np.shape[0]} != num_envs={self.num_envs}"
                )
            if actions_np.shape[-1] < 8:
                raise ValueError(
                    f"joint_pos expects action_dim>=8 (7 joints + gripper), got {actions_np.shape[-1]}"
                )
            abs_chunks = np.ascontiguousarray(actions_np[:, :, :8], dtype=np.float32)
            placeholder = torch.zeros(
                (self.num_envs, abs_chunks.shape[1], 8),
                device=self.menv.device,
                dtype=torch.float32,
            )
            return placeholder, wall_ms, abs_chunks, action_lengths

        # IK modes: keep env-relative semantics + optional scale.
        actions_np = np.ascontiguousarray(actions_np, dtype=np.float32)
        actions = torch.tensor(actions_np, device=self.menv.device, dtype=torch.float32)
        if self.action_scale != 1.0 and actions.shape[-1] > 1:
            actions = actions.clone()
            actions[..., :-1] *= self.action_scale
        return actions, wall_ms, None, action_lengths

    def _infer(self, obs) -> tuple[torch.Tensor, float]:
        """Unified inference: routes to server or local backend.

        Returns:
            actions: ``(N, H, action_dim)`` tensor on device.
            period_ms: inference period to advance sim for, in milliseconds.
        """
        if self.backend == "local":
            # NOTE: policy.reset() is intentionally NOT called here. It is
            # invoked at episode boundaries only (see _reset_env_for_episode
            # and _process_dones -> _reset_policy_state). Calling it every
            # inference would wipe ACT's temporal ensembler buffer each
            # cycle, defeating the purpose of temporal ensembling.
            return self._local_infer()

        # Server backend
        current_frames = None
        if self.obs_mode == "vla":
            current_frames = self._capture_camera_images()
        payload = self._build_obs(obs, current_frames=current_frames)

        result = self.client.predict(payload)
        actions, wall_ms, abs_chunks, action_lengths = self._decode_server_result(result)
        self._raw_abs_chunks = abs_chunks.copy() if abs_chunks is not None else None
        self._action_chunk_lengths = action_lengths.copy()
        period_ms = self._real_latency_period_ms(wall_ms)
        return actions, period_ms

    # ======================================================= #
    #          Simulated Synchronous Inference                #
    # ======================================================= #

    def _period_to_sim_steps(self, period_ms: float) -> int:
        """Convert an inference period in ms to number of physics sub-steps."""
        return max(0, round(period_ms / (self.sim_dt * 1000.0)))

    def _start_rtf_timer(self) -> None:
        """Start wall-clock timing used for real-time-factor accounting."""
        self._rtf_wall_t0 = time.monotonic()
        self._rtf_wall_t_end = None
        self._rtf_sim_elapsed_s = 0.0
        self._rtf_last_value = 1.0

    def _record_sim_steps(self, sim_steps: int) -> None:
        """Record simulation time advanced by this evaluator loop."""
        if sim_steps <= 0:
            return
        self._rtf_sim_elapsed_s += float(sim_steps) * self.sim_dt
        if self._rtf_wall_t0 is None:
            return
        wall_elapsed = time.monotonic() - self._rtf_wall_t0
        if wall_elapsed > 1e-6:
            self._rtf_last_value = self._rtf_sim_elapsed_s / wall_elapsed

    def _record_control_step(self) -> None:
        """Record one env.step(), which advances all envs by one control step."""
        self._record_sim_steps(self.decimation)

    def _current_rtf(self) -> float:
        """Return the latest measured sim-time / wall-time ratio."""
        if self._rtf_wall_t0 is None or self._rtf_sim_elapsed_s <= 0.0:
            return 1.0
        return max(self._rtf_last_value, 1e-6)

    def _rtf_wall_elapsed_s(self) -> float:
        """Wall-clock duration covered by RTF accounting."""
        if self._rtf_wall_t0 is None:
            return 0.0
        t_end = self._rtf_wall_t_end if self._rtf_wall_t_end is not None else time.monotonic()
        return max(0.0, t_end - self._rtf_wall_t0)

    def _convert_real_latency_to_sim_ms(self, wall_ms: float) -> float:
        """Map real wall-clock latency to the equivalent sim-time latency."""
        if not self.use_real_latency:
            return wall_ms
        return wall_ms * self._current_rtf()

    def _real_latency_period_ms(self, wall_ms: float) -> float:
        """Return the inference period used by the scheduler."""
        if not self.use_real_latency:
            return self.infer_period_ms
        rtf = self._current_rtf()
        sim_ms = wall_ms * rtf
        self._latency_wall_ms.append(wall_ms)
        self._latency_sim_ms.append(sim_ms)
        self._latency_rtf.append(rtf)
        return sim_ms

    def _elapsed_to_action_start_idx(self, period_ms: float) -> int:
        """Map elapsed inference time to the nearest action index in a chunk."""
        if not self.async_time_align_actions:
            return 0
        env_dt_ms = self.env_dt * 1000.0
        if env_dt_ms <= 0:
            return 0
        return max(0, math.floor(period_ms / env_dt_ms + 0.5))

    def _build_freeze_action(self) -> torch.Tensor:
        """Build a freeze-arm action: target = current joint pos, keep gripper.

        For relative joint_pos action space, this means zero arm delta.
        The gripper dimension is set to maintain current state.
        """
        robot = self.menv.scene["robot"]
        # Zero relative arm movement = freeze in place
        act_shape = self.env.action_space.shape
        freeze = torch.zeros(act_shape, device=self.menv.device)
        # Keep gripper at current state: open (1.0) or closed (-1.0)
        finger_mean = robot.data.joint_pos[:, self.finger_ids].mean(dim=-1)
        gripper_cmd = torch.where(finger_mean > 0.03, torch.tensor(1.0, device=self.menv.device),
                                  torch.tensor(-1.0, device=self.menv.device))
        freeze[:, -1] = gripper_cmd
        return freeze

    def _run_sync(self) -> None:
        """Pause -> infer -> freeze-arm during inference period -> execute H actions.

        During the inference period (``infer_period_ms`` of world time), the
        arm is frozen at its current position (target = current joint pos)
        while physics continues. After that penalty window, the action chunk
        is executed - one action per control step, up to ``execution_horizon``
        steps.
        """
        obs, _ = self._reset_env_for_episode()
        if self._video_writers:
            for eid in range(self.num_envs):
                for writer in self._video_writers.values():
                    writer.start_episode(eid)
        self._seed_image_history()
        dec = self.decimation
        has_advance = hasattr(self.menv, "advance_sim")

        while not self._done_enough():
            # 1. Pause sim - capture observation and infer
            self._track_phase()
            acts, period_ms = self._infer(obs)

            # 2. Inference period: freeze arm (target=current), world continues
            freeze = self._build_freeze_action()
            lat_sim = self._period_to_sim_steps(period_ms)
            full_ctrl = lat_sim // dec
            remainder = lat_sim % dec

            for _ in range(full_ctrl):
                self._track_phase()
                obs, rew, term, trunc, _info = self.env.step(freeze)
                self._record_control_step()
                self._capture_frame()
                resets = self._process_dones(rew, term, trunc)
                if resets:
                    if self.task_profile:
                        self._apply_post_reset_task_overrides()
                        obs = self._refresh_obs_after_reset()
                    self._seed_image_history(resets)
                if self._done_enough():
                    return

            if remainder > 0:
                if has_advance:
                    self.menv.advance_sim(remainder, freeze)
                    self._record_sim_steps(remainder)
                    obs = self._clone_obs(self.menv.obs_buf)
                    self._capture_frame()
                else:
                    self._track_phase()
                    obs, rew, term, trunc, _info = self.env.step(freeze)
                    self._record_control_step()
                    self._capture_frame()
                    resets = self._process_dones(rew, term, trunc)
                    if resets:
                        if self.task_profile:
                            self._apply_post_reset_task_overrides()
                            obs = self._refresh_obs_after_reset()
                        self._seed_image_history(resets)
                    if self._done_enough():
                        return

            # 3. Execute chunk: one action per control step
            H = acts.shape[1]
            for h in range(H):
                self._track_phase()
                action = self._select_chunk_action(
                    acts, h, self._raw_abs_chunks, self._action_chunk_lengths
                )
                # Track actions for smoothness metrics
                for eid in range(self.num_envs):
                    self._ep_actions[eid].append(action[eid].cpu().numpy())
                obs, rew, term, trunc, _info = self.env.step(action)
                self._record_control_step()
                self._capture_frame()
                resets = self._process_dones(rew, term, trunc)
                if resets:
                    if self.task_profile:
                        self._apply_post_reset_task_overrides()
                        obs = self._refresh_obs_after_reset()
                    self._seed_image_history(resets)
                    if h < H - 1:
                        for eid in resets:
                            if self._action_chunk_lengths is not None:
                                self._action_chunk_lengths[eid] = min(
                                    int(self._action_chunk_lengths[eid]), h + 1
                                )
                            acts[eid, h + 1 :, :] = 0.0
                            # Clear abs cache too: _convert_abs_action reads
                            # self._raw_abs_chunks directly, not `acts`, so zeroing
                            # only `acts` still leaves stale abs targets.
                            if self._raw_abs_chunks is not None:
                                self._raw_abs_chunks[eid, h + 1 :, :] = 0.0
                if self._done_enough():
                    return

    # ======================================================= #
    #         Simulated Asynchronous Inference                #
    # ======================================================= #

    def _run_async_realtime(self) -> None:
        """Run true async server inference while control advances at ctrl_freq."""
        obs, _ = self._reset_env_for_episode()
        if self._video_writers:
            for eid in range(self.num_envs):
                for writer in self._video_writers.values():
                    writer.start_episode(eid)
        self._seed_image_history()

        active_actions: torch.Tensor | None = None
        active_abs_chunks: np.ndarray | None = None
        active_action_lengths: np.ndarray | None = None
        active_action_idx = 0
        active_steps_used = 0

        pending_actions: torch.Tensor | None = None
        pending_abs_chunks: np.ndarray | None = None
        pending_action_lengths: np.ndarray | None = None
        pending_action_start_idx = 0

        inflight = None
        inflight_reset_ids: set[int] = set()

        def _submit_inference(obs_snapshot):
            current_frames = None
            if self.obs_mode == "vla":
                current_frames = self._capture_camera_images()
            payload = self._build_obs(obs_snapshot, current_frames=current_frames)
            return self.client.predict_async(payload)

        def _fill_freeze_for_resets(
            actions: torch.Tensor | None,
            abs_chunks: np.ndarray | None,
            lengths: np.ndarray | None,
            resets: set[int] | list[int],
            start_idx: int = 0,
        ) -> None:
            if actions is None or not resets or start_idx >= actions.shape[1]:
                return
            robot = self.menv.scene["robot"]
            finger_mean = robot.data.joint_pos[:, self.finger_ids].mean(dim=-1)
            current_q = robot.data.joint_pos[:, self.arm_ids].cpu().numpy()
            for eid in resets:
                gripper_open = bool((finger_mean[eid] > 0.03).item())
                actions[eid, start_idx:, :].zero_()
                actions[eid, start_idx:, -1] = 1.0 if gripper_open else -1.0
                if lengths is not None:
                    lengths[eid] = actions.shape[1]
                if abs_chunks is not None:
                    abs_chunks[eid, start_idx:, :7] = current_q[eid, :]
                    abs_chunks[eid, start_idx:, 7] = 1.0 if gripper_open else 0.0

        def _promote_pending() -> bool:
            nonlocal active_actions, active_abs_chunks, active_action_lengths
            nonlocal active_action_idx, active_steps_used
            nonlocal pending_actions, pending_abs_chunks, pending_action_lengths
            nonlocal pending_action_start_idx

            if pending_actions is None:
                return False
            active_actions = pending_actions
            active_abs_chunks = pending_abs_chunks
            active_action_lengths = pending_action_lengths
            active_action_idx = pending_action_start_idx
            active_steps_used = 0
            pending_actions = None
            pending_abs_chunks = None
            pending_action_lengths = None
            pending_action_start_idx = 0
            return True

        def _should_switch_active() -> bool:
            if active_actions is None:
                return True
            if active_action_idx >= active_actions.shape[1]:
                return True
            if self.async_chunk_steps is not None:
                return active_steps_used >= self.async_chunk_steps
            # Default true-async behavior: switch at a control boundary when a
            # newer chunk is ready, but execute at least one action per active chunk.
            return pending_actions is not None and active_steps_used > 0

        def _select_action() -> torch.Tensor:
            nonlocal active_action_idx, active_steps_used

            if _should_switch_active():
                _promote_pending()
            if active_actions is None or active_action_idx >= active_actions.shape[1]:
                return self._build_freeze_action()
            action = self._select_chunk_action(
                active_actions,
                active_action_idx,
                active_abs_chunks,
                active_action_lengths,
            )
            active_action_idx += 1
            active_steps_used += 1
            return action

        while not self._done_enough():
            self._track_phase()

            if inflight is not None and inflight.done():
                result = inflight.result()
                new_actions, wall_ms, new_abs_chunks, new_action_lengths = (
                    self._decode_server_result(result)
                )
                self._real_latency_period_ms(wall_ms)
                _fill_freeze_for_resets(
                    new_actions,
                    new_abs_chunks,
                    new_action_lengths,
                    inflight_reset_ids,
                )
                align_ms = self._last_action_align_ms if self.async_time_align_actions else 0.0
                next_action_start_idx = self._elapsed_to_action_start_idx(align_ms)
                if (
                    self.async_time_align_actions
                    and next_action_start_idx >= new_actions.shape[1]
                    and not self._warned_async_action_offset_exhausted
                ):
                    print(
                        f"[WARN] Async time-aligned start index ({next_action_start_idx}) "
                        f">= chunk length ({new_actions.shape[1]}) "
                        f"(align_ms={align_ms:.1f}, control_dt={self.env_dt * 1000.0:.1f} ms). "
                        "This chunk has no remaining actions and will execute freeze actions."
                    )
                    self._warned_async_action_offset_exhausted = True
                pending_actions = new_actions
                pending_abs_chunks = new_abs_chunks
                pending_action_lengths = new_action_lengths
                pending_action_start_idx = next_action_start_idx
                inflight = None
                inflight_reset_ids = set()

            if inflight is None:
                inflight = _submit_inference(obs)
                inflight_reset_ids = set()

            action = _select_action()

            for eid in range(self.num_envs):
                self._ep_actions[eid].append(action[eid].cpu().numpy())

            obs, rew, term, trunc, _info = self.env.step(action)
            self._record_control_step()
            self._capture_frame()
            resets = self._process_dones(rew, term, trunc)
            if resets:
                if self.task_profile:
                    self._apply_post_reset_task_overrides()
                    obs = self._refresh_obs_after_reset()
                self._seed_image_history(resets)
                reset_set = set(resets)
                inflight_reset_ids.update(reset_set)
                _fill_freeze_for_resets(
                    active_actions,
                    active_abs_chunks,
                    active_action_lengths,
                    reset_set,
                    active_action_idx,
                )
                _fill_freeze_for_resets(
                    pending_actions,
                    pending_abs_chunks,
                    pending_action_lengths,
                    reset_set,
                )

    def _run_async(self) -> None:
        """Pause -> infer -> execute *previous* chunk for one inference period.

        Each cycle advances sim by ``infer_period_ms`` worth of physics
        sub-steps.  The actions executed are always from the *previous*
        inference (one cycle stale), reproducing real-world pipelined
        deployment.

        First cycle has no previous actions -> freeze-arm actions.
        """
        if self.use_real_latency and self.backend == "server":
            self._run_async_realtime()
            return
        if self.use_real_latency:
            print(
                "[WARN] True realtime async is only implemented for server backend; "
                "falling back to period-based async."
            )
        obs, _ = self._reset_env_for_episode()
        if self._video_writers:
            for eid in range(self.num_envs):
                for writer in self._video_writers.values():
                    writer.start_episode(eid)
        self._seed_image_history()
        dec = self.decimation
        has_advance = hasattr(self.menv, "advance_sim")

        active_actions: torch.Tensor | None = None
        active_abs_chunks: np.ndarray | None = None
        active_action_lengths: np.ndarray | None = None
        active_action_idx = 0
        active_steps_used = 0
        active_step_limit: int | None = None

        pending_actions: torch.Tensor | None = None
        pending_abs_chunks: np.ndarray | None = None
        pending_action_lengths: np.ndarray | None = None
        pending_action_start_idx = 0

        def _promote_pending(limit: int) -> bool:
            nonlocal active_actions, active_abs_chunks, active_action_lengths
            nonlocal active_action_idx, active_steps_used, active_step_limit
            nonlocal pending_actions, pending_abs_chunks, pending_action_lengths
            nonlocal pending_action_start_idx

            if pending_actions is None:
                return False
            active_actions = pending_actions
            active_abs_chunks = pending_abs_chunks
            active_action_lengths = pending_action_lengths
            active_action_idx = pending_action_start_idx
            active_steps_used = 0
            active_step_limit = limit
            pending_actions = None
            pending_abs_chunks = None
            pending_action_lengths = None
            pending_action_start_idx = 0
            return True

        def _active_exhausted(for_remainder: bool = False) -> bool:
            if active_actions is None:
                return True
            if active_action_idx >= active_actions.shape[1]:
                return self.async_chunk_steps is not None
            if self.async_chunk_steps is not None:
                return active_steps_used >= self.async_chunk_steps
            if for_remainder:
                return False
            return active_step_limit is not None and active_steps_used >= active_step_limit

        def _fill_freeze_for_resets(
            actions: torch.Tensor | None,
            abs_chunks: np.ndarray | None,
            lengths: np.ndarray | None,
            resets: set[int] | list[int],
            start_idx: int = 0,
        ) -> None:
            if actions is None or not resets or start_idx >= actions.shape[1]:
                return
            robot = self.menv.scene["robot"]
            finger_mean = robot.data.joint_pos[:, self.finger_ids].mean(dim=-1)
            current_q = robot.data.joint_pos[:, self.arm_ids].cpu().numpy()
            for eid in resets:
                gripper_open = bool((finger_mean[eid] > 0.03).item())
                actions[eid, start_idx:, :].zero_()
                actions[eid, start_idx:, -1] = 1.0 if gripper_open else -1.0
                if lengths is not None:
                    lengths[eid] = actions.shape[1]
                if abs_chunks is not None:
                    abs_chunks[eid, start_idx:, :7] = current_q[eid, :]
                    abs_chunks[eid, start_idx:, 7] = 1.0 if gripper_open else 0.0

        def _select_active_action(limit: int, consume: bool = True) -> torch.Tensor:
            nonlocal active_action_idx, active_steps_used

            if _active_exhausted(for_remainder=not consume):
                _promote_pending(limit)
            if _active_exhausted(for_remainder=not consume):
                return self._build_freeze_action()
            action = self._select_chunk_action(
                active_actions,
                active_action_idx,
                active_abs_chunks,
                active_action_lengths,
            )
            if consume:
                active_action_idx += 1
                active_steps_used += 1
            return action

        while not self._done_enough():
            # 1. Pause sim - capture observation and infer
            self._track_phase()
            new_actions, period_ms = self._infer(obs)
            new_abs_chunks = (
                self._raw_abs_chunks.copy()
                if self._raw_abs_chunks is not None else None
            )
            new_action_lengths = (
                self._action_chunk_lengths.copy()
                if self._action_chunk_lengths is not None else None
            )

            # 2. Execute PREVIOUS chunk during the inference period
            lat_sim = self._period_to_sim_steps(period_ms)
            full_ctrl = lat_sim // dec
            remainder = lat_sim % dec
            # Must call env.step() at least once per cycle for episode
            # bookkeeping (termination checks, episode_length_buf, resets).
            if full_ctrl == 0:
                full_ctrl = 1
                remainder = 0
            if (
                self.async_chunk_steps is not None
                and self.async_chunk_steps <= full_ctrl
                and not self._warned_async_chunk_steps_not_longer
            ):
                print(
                    f"[WARN] --async_chunk_steps={self.async_chunk_steps} is not larger "
                    f"than this latency window ({full_ctrl} control steps). It will not "
                    "extend chunk execution for such cycles."
                )
                self._warned_async_chunk_steps_not_longer = True
            chunk_step_limit = (
                full_ctrl
                if self.async_chunk_steps is None
                else self.async_chunk_steps
            )
            # Track envs that reset anywhere in this cycle so we can scrub
            # active, pending and just-inferred chunks before they are reused.
            # Otherwise the freshly
            # reset env would start its new episode by executing ~1 infer
            # period of actions computed from the PREVIOUS episode's
            # pre-reset observation.
            resets_this_cycle: set[int] = set()

            for _ in range(full_ctrl):
                self._track_phase()
                action = _select_active_action(chunk_step_limit, consume=True)

                # Track actions for smoothness
                for eid in range(self.num_envs):
                    self._ep_actions[eid].append(action[eid].cpu().numpy())

                obs, rew, term, trunc, _info = self.env.step(action)
                self._record_control_step()
                self._capture_frame()
                resets = self._process_dones(rew, term, trunc)
                if resets:
                    if self.task_profile:
                        self._apply_post_reset_task_overrides()
                        obs = self._refresh_obs_after_reset()
                    self._seed_image_history(resets)
                    resets_this_cycle.update(resets)
                    _fill_freeze_for_resets(
                        active_actions,
                        active_abs_chunks,
                        active_action_lengths,
                        resets,
                        active_action_idx,
                    )
                    _fill_freeze_for_resets(
                        pending_actions,
                        pending_abs_chunks,
                        pending_action_lengths,
                        resets,
                    )
                if self._done_enough():
                    return

            if remainder > 0:
                action = _select_active_action(chunk_step_limit, consume=False)
                if has_advance:
                    self.menv.advance_sim(remainder, action)
                    self._record_sim_steps(remainder)
                    obs = self._clone_obs(self.menv.obs_buf)
                    self._capture_frame()
                else:
                    self._track_phase()
                    obs, rew, term, trunc, _info = self.env.step(action)
                    self._record_control_step()
                    self._capture_frame()
                    resets = self._process_dones(rew, term, trunc)
                    if resets:
                        if self.task_profile:
                            self._apply_post_reset_task_overrides()
                            obs = self._refresh_obs_after_reset()
                        self._seed_image_history(resets)
                        resets_this_cycle.update(resets)
                        _fill_freeze_for_resets(
                            active_actions,
                            active_abs_chunks,
                            active_action_lengths,
                            resets,
                            active_action_idx,
                        )
                        _fill_freeze_for_resets(
                            pending_actions,
                            pending_abs_chunks,
                            pending_action_lengths,
                            resets,
                        )
                    if self._done_enough():
                        return

            # 3. Store current result for the next cycle
            # Invalidate per-env slices of the just-computed chunk for envs
            # that reset during this cycle: `new_actions` was produced from
            # the PRE-reset observation and would otherwise be executed as
            # `prev_actions` at the start of the next (already-new) episode
            # for ~1 inference period. Replace the stale chunk with a
            # physically-correct freeze: arm delta=0 (rel) / target=current
            # joint (abs), gripper = maintain current open/close state.
            # NOTE: naive zero-fill is WRONG for `_raw_abs_chunks` because
            # those are absolute joint targets: abs=0 produces a huge
            # negative rel delta (joint -> 0 attractor), worse than stale.
            _fill_freeze_for_resets(
                new_actions,
                new_abs_chunks,
                new_action_lengths,
                resets_this_cycle,
            )
            align_ms = self._last_action_align_ms if self.async_time_align_actions else 0.0
            next_action_start_idx = self._elapsed_to_action_start_idx(align_ms)
            if (
                self.async_time_align_actions
                and next_action_start_idx >= new_actions.shape[1]
                and not self._warned_async_action_offset_exhausted
            ):
                print(
                    f"[WARN] Async time-aligned start index ({next_action_start_idx}) "
                    f">= chunk length ({new_actions.shape[1]}) "
                    f"(align_ms={align_ms:.1f}, control_dt={self.env_dt * 1000.0:.1f} ms). "
                    "This chunk has no remaining actions and will execute freeze actions."
                )
                self._warned_async_action_offset_exhausted = True
            # Keep the freshest completed chunk ready. If inference is faster
            # than the configured chunk execution window, older pending chunks
            # are intentionally superseded instead of building a stale queue.
            pending_actions = new_actions
            pending_action_start_idx = next_action_start_idx
            pending_abs_chunks = new_abs_chunks
            pending_action_lengths = new_action_lengths

    # ======================================================= #
    #                    Entry point                          #
    # ======================================================= #

    def run(self) -> None:
        self.setup()
        try:
            self._start_rtf_timer()
            if self.inference_mode == "sync":
                self._run_sync()
            else:
                self._run_async()
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            self._rtf_wall_t_end = time.monotonic()
            for writer in self._video_writers.values():
                writer.close()
            self._report()
            if self.client is not None:
                self.client.close()
            self.env.close()

    # ----------- Reporting ----------- #

    def _report(self) -> None:
        n = len(self.results)
        if n == 0:
            print("[WARN] No completed episodes.")
            return

        successes = sum(r["success"] for r in self.results)
        returns = [r["return"] for r in self.results]
        lengths = [r["length"] for r in self.results]
        avg_diff_vals = [r["avg_diff"] for r in self.results if "avg_diff" in r]
        avg_jerk_vals = [r["avg_jerk"] for r in self.results if "avg_jerk" in r]
        wall_elapsed_s = self._rtf_wall_elapsed_s()
        sim_elapsed_s = self._rtf_sim_elapsed_s
        measured_rtf = sim_elapsed_s / wall_elapsed_s if wall_elapsed_s > 1e-6 else 0.0
        latency_metrics = {
            "samples": len(self._latency_wall_ms),
            "mean_wall_ms": round(float(np.mean(self._latency_wall_ms)), 4)
            if self._latency_wall_ms else 0.0,
            "mean_sim_ms": round(float(np.mean(self._latency_sim_ms)), 4)
            if self._latency_sim_ms else 0.0,
            "mean_rtf_used": round(float(np.mean(self._latency_rtf)), 4)
            if self._latency_rtf else 0.0,
        }

        summary = {
            "config": {
                "task": self.task,
                "backend": self.backend,
                "inference_mode": self.inference_mode,
                "infer_period_ms": self.infer_period_ms,
                "infer_freq_hz": (
                    round(1000.0 / self.infer_period_ms, 4)
                    if self.infer_period_ms > 0 else None
                ),
                "use_real_latency": self.use_real_latency,
                "async_time_align_actions": self.async_time_align_actions,
                "async_chunk_steps": self.async_chunk_steps,
                "action_horizon": self.action_horizon,
                "action_scale": self.action_scale,
                "obs_mode": self.obs_mode,
                "image_history": self.image_history,
                "image_history_stride": self.image_history_stride,
                "num_envs": self.num_envs,
                "num_episodes": self.num_episodes,
                "seed": self.seed,
                "sim_freq_hz": round(self.sim_freq, 2),
                "ctrl_freq_hz": round(self.ctrl_freq, 2),
                "sim_dt": self.sim_dt,
                "env_dt": self.env_dt,
                "task_profile": self.task_profile.get("success_type") if self.task_profile else None,
                "model_path": self.model_path,
            },
            "metrics": {
                "total_episodes": n,
                "successes": successes,
                "success_rate": round(successes / n, 4),
                "mean_return": round(float(np.mean(returns)), 4),
                "std_return": round(float(np.std(returns)), 4),
                "mean_length": round(float(np.mean(lengths)), 2),
                "std_length": round(float(np.std(lengths)), 2),
                "mean_avg_diff": round(float(np.mean(avg_diff_vals)), 4) if avg_diff_vals else 0,
                "mean_avg_jerk": round(float(np.mean(avg_jerk_vals)), 4) if avg_jerk_vals else 0,
                "sim_elapsed_s": round(sim_elapsed_s, 4),
                "wall_elapsed_s": round(wall_elapsed_s, 4),
                "real_time_factor": round(measured_rtf, 4),
                "real_latency_conversion": latency_metrics,
            },
            "episodes": self.results,
        }

        # Phase histogram for failures
        has_phase = any(r.get("max_phase", -1) >= 0 for r in self.results)
        if has_phase:
            fail_phases = [r["max_phase"] for r in self.results if not r["success"]]
            phase_hist: dict[int, int] = {}
            for p in fail_phases:
                phase_hist[p] = phase_hist.get(p, 0) + 1
            summary["metrics"]["fail_phase_histogram"] = phase_hist

        print(f"\n{'=' * 60}")
        print(f"  Episodes    : {n}")
        print(f"  Success     : {successes}/{n} = {successes / n * 100:.1f}%")
        print(f"  Return      : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  Length      : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(
            f"  RTF         : {measured_rtf:.3f} "
            f"(sim={sim_elapsed_s:.2f}s, wall={wall_elapsed_s:.2f}s)"
        )
        if self.use_real_latency and self._latency_wall_ms:
            print(
                "  Latency     : "
                f"wall={latency_metrics['mean_wall_ms']:.1f}ms -> "
                f"sim={latency_metrics['mean_sim_ms']:.1f}ms "
                f"(mean RTF={latency_metrics['mean_rtf_used']:.3f})"
            )
        if avg_jerk_vals:
            print(f"  Smoothness  : diff={np.mean(avg_diff_vals):.4f}  jerk={np.mean(avg_jerk_vals):.4f}")
        if has_phase and phase_hist:
            phase_names = self.task_profile.get("phase_names", {}) if self.task_profile else {}
            parts = [
                f"P{p}({phase_names.get(p, '?')}):{c}"
                for p, c in sorted(phase_hist.items())
            ]
            print(f"  Fail phases : {', '.join(parts)}")
        print(f"{'=' * 60}\n")

        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[INFO] Results -> {self.output_path}")


# --------------------- main ---------------------------- #


def main():
    Evaluator(args_cli).run()
    simulation_app.close()


if __name__ == "__main__":
    main()
