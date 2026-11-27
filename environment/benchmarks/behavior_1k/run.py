"""Run main's episode runner against the existing local BEHAVIOR Docker worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tomllib
import traceback

from environment.benchmarks.behavior_1k.adapter import BehaviorAdapter, load_protocol
from environment.runtime.codex_session import CodexSession
from environment.runtime.episode_runner import run_episode
from environment.runtime.request_audit import RequestAudit, azure_authorization
from environment.benchmarks.behavior_1k.tasks import selection, verify_template
from environment.benchmarks.behavior_1k.budgets import PROFILES, get_budget
from environment.runtime.image_history import POLICY as IMAGE_HISTORY_POLICY


WORLD = Path(__file__).resolve().parents[3]
PRIVATE = Path("/root/robotworld-behavior-private")
SCORABLE = {"agent_completed", "give_up", "action_budget", "tool_budget", "timeout", "environment_end"}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(worker, record_video=False, task_selected=False, parallel=False):
    paths = [worker / n for n in ("run_native.sh", "eef_worker.py", "eef_protocol.py")]
    paths += [Path(__file__), Path(__file__).with_name("adapter.py")]
    paths += [Path(__file__).with_name("tasks.py")]
    paths += [Path(__file__).with_name("budgets.py")]
    if task_selected:
        paths += [Path(__file__).with_name("task_worker.py")]
    paths += [WORLD / "environment/runtime" / n for n in
              ("episode_runner.py", "codex_session.py", "request_audit.py", "api_retry.py", "image_history.py")]
    if record_video:
        paths += [Path(__file__).with_name(n) for n in ("run_video.sh", "record_worker.py")]
        paths += [WORLD / "environment/benchmarks/robodojo/video.py"]
    if parallel:
        paths += [Path(__file__).with_name(n) for n in ("run_parallel.sh", "parallel_runtime.py", "run_four_parallel.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def model_overrides():
    # Per-process restrictions only; existing provider and authentication remain untouched.
    features = ("shell_tool", "unified_exec", "view_image", "multi_agent", "multi_agent_v2",
                "apps", "plugins", "remote_plugin", "browser_use", "computer_use",
                "image_generation", "memories", "hooks", "skill_search", "code_mode",
                "code_mode_host", "unbounded_connection_retries", "goals",
                "default_mode_request_user_input")
    return [*(f"features.{name}=false" for name in features),
            "features.skip_host_skill_discovery=true", "mcp_servers={}",
            "project_doc_max_bytes=0", 'sandbox_mode="read-only"',
            "agents.enabled=false", "tools.experimental_request_user_input.enabled=false",
            "tools.update_plan.enabled=false",
            'web_search="disabled"', 'developer_instructions=""',
            'model_reasoning_effort="medium"']


def gateway_checks(adapter):
    first = adapter.state
    second = adapter.request("observe")["observation"]
    if first["physics_events"] != second["physics_events"]:
        raise RuntimeError("observe advanced physics")
    reply = adapter.request("step", dict(
        observation_id=first["observation_id"], arm="left", delta_position=[0, 0, 0],
        delta_rotation=[0, 0, 0], gripper=1, base=[0, 0, 0], steps=1))
    after = adapter.request("observe")["observation"]
    if reply["ok"] or "Stale" not in reply["error"] or after["native_steps"] != 0:
        raise RuntimeError("Stale command was not rejected before motion")
    if after["physics_events"] != first["physics_events"]:
        raise RuntimeError("Stale validation advanced physics")
    return {"observe_nonstepping": True, "stale_rejected_without_motion": True}


def physical_smoke(adapter):
    checks = []
    fingers = adapter.state["robot"]["left"]["measured_finger_joints"]
    initial_image = hashlib.sha256((adapter.simulator / adapter.image).read_bytes()).hexdigest()
    for arm, delta, grip in (("left", [.01, 0, 0], 1), ("right", [.01, 0, 0], 1),
                             ("left", [0, 0, 0], 0), ("left", [0, 0, 0], 1)):
        response, trace = adapter.call("step", dict(
            observation_id=adapter.state["observation_id"], arm=arm,
            delta_position=delta, delta_rotation=[0, 0, 0], gripper=grip,
            base=[0, 0, 0], steps=15))
        if not response["success"] or trace["executed_native_steps"] != 15:
            raise RuntimeError("Native physical smoke command failed")
        if any(delta):
            if not trace["feedback"]["reached"] or trace["feedback"]["measured_motion_m"] <= .002:
                raise RuntimeError("EEF did not achieve real motion")
        else:
            now = adapter.state["robot"]["left"]["measured_finger_joints"]
            motion = sum((a-b)**2 for a, b in zip(fingers, now, strict=True))**.5
            if motion <= .001:
                raise RuntimeError("Finger joints did not move")
            trace["finger_motion"] = motion
            fingers = now
        checks.append(trace)
    image = hashlib.sha256((adapter.simulator / adapter.image).read_bytes()).hexdigest()
    if image == initial_image or adapter.state["physics_events"] != 240:
        raise RuntimeError("Missing image change or incorrect physics cadence")
    return {"checks": checks, "physics_steps": 240, "rgb_changed": True}


def verify_video(directory, steps):
    capture = json.loads((directory / "capture.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if (steps <= 0 or manifest["frames_per_camera"] != steps or manifest["encoding_errors"] or
            manifest["env_steps"] != list(range(1, steps+1)) or manifest["fps"] != 30 or
            [f["native_step"] for f in capture["frames"]] != list(range(1, steps+1)) or
            any(f["physics_callbacks"] != 4 for f in capture["frames"])):
        raise RuntimeError("Video frame/physics cadence mismatch")
    streams = {}
    for name in ("head_640x480", "head", "left_wrist", "right_wrist"):
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height", "-of", "json",
            str(directory / (name + ".mp4"))], text=True))["streams"][0]
        expected = (640, 480) if name == "head_640x480" else (128, 128)
        if (int(probe["nb_read_frames"]) != steps or probe["avg_frame_rate"] != "30/1" or
                (probe["width"], probe["height"]) != expected):
            raise RuntimeError("Encoded video differs from captured frames")
        streams[name] = probe
    return dict(streams=streams, frames=steps, duration_seconds=steps/30,
                continuous=True, extra_physics_steps=0)


def legacy_limit_crossing(adapter):
    # An acceptance-only hold test; agent runs always start from a fresh reset.
    for _ in range(38):
        reply, trace = adapter.call("step", dict(
            observation_id=adapter.state["observation_id"], arm="left", delta_position=[0, 0, 0],
            delta_rotation=[0, 0, 0], gripper=1, base=[0, 0, 0], steps=15))
        if not reply["success"] or trace["executed_native_steps"] != 15 or adapter.ended():
            raise RuntimeError("Extended gateway stopped at an obsolete limit")
    if adapter.state["commands"] != 42 or adapter.state["native_steps"] != 630:
        raise RuntimeError("Extended gateway control counts differ")
    return dict(commands=42, native_steps=630, physics_steps=adapter.state["physics_events"],
                old_limits_crossed=True, model_inference=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gateway", "agent"))
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--accepted-smoke", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--aihub-direct", action="store_true")
    parser.add_argument("--task", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--budget-profile", choices=tuple(PROFILES), default="legacy")
    parser.add_argument("--managed-batch", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    worker = args.worker_dir.resolve()
    run = args.run.resolve()
    task = selection(args.task or 2)
    budget = get_budget(args.budget_profile)
    long_run = budget["wall_seconds"] is None
    allocation = None
    if args.managed_batch:
        from environment.benchmarks.behavior_1k.parallel_runtime import assignment
        if not long_run or not args.record_video or args.task is None:
            parser.error("Managed parallel runs require task, recording and a control-step budget profile")
        allocation = assignment(args.managed_batch, args.gpu)
        if allocation["task"] != args.task:
            parser.error("Task does not match the assigned GPU lane")
    elif args.gpu != 0:
        parser.error("Nonzero GPU requires a managed parallel batch")
    if long_run and args.task is None:
        parser.error("Control-step budget profiles require the task-selected worker")
    if run.parent != PRIVATE or not run.name.startswith(f"native-task{task['feishu_id']}-scaffold-") or run.exists():
        parser.error("Use a fresh private native-task<ID>-scaffold-* output directory")
    if args.task is not None and not args.record_video:
        parser.error("Task-selected worker currently requires --record-video")
    verify_template(PRIVATE / "data", task, json.loads((PRIVATE / "partial/receipt.json").read_text()))
    hashes = source_hashes(worker, args.record_video, args.task is not None, allocation is not None)
    protocol = load_protocol(worker, args.budget_profile)
    if protocol.VERSION != "behavior-r1pro-eef-v1":
        parser.error("Only the verified R1Pro EEF protocol is supported")
    if args.mode == "agent":
        if not args.aihub_direct:
            parser.error("Use --aihub-direct: the legacy local provider drops current tool RGB")
        azure_authorization()  # Validate local credential before spending time loading the simulator.
        if not args.manifest or not args.accepted_smoke:
            parser.error("Agent mode requires the source-build manifest and accepted gateway smoke")
        accepted = json.loads((args.accepted_smoke / "result.json").read_text())
        if not (accepted.get("accepted") and accepted.get("mode") == "gateway" and
                accepted.get("source_hashes") == hashes and
                accepted.get("selection") == task and
                accepted.get("budget_profile") == args.budget_profile and
                accepted.get("gpu_allocation") == allocation and
                (not args.record_video or accepted.get("video_valid"))):
            parser.error("A passing source-matched gateway smoke is required")
        CodexSession.verify_build(args.manifest)
    os.umask(0o077)
    run.mkdir()
    (run / "sources").mkdir()
    for index, path in enumerate(hashes):
        shutil.copy2(path, run / "sources" / (str(index) + "-" + Path(path).name))
    result = {"mode": args.mode, "accepted": False, "model_score": None,
              "model_turn_started": False, "source_hashes": hashes, "status": "preflight",
              "record_video": args.record_video,
              "api_route": "aihub-azure-direct" if args.aihub_direct else "configured-provider",
              "base_main_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=WORLD, text=True).strip(),
              "observation": "sensor_only", "skills": [], "native_sim": "5.1.0.0",
              "tools": ["observe", "step"] + (["give_up"] if budget["allow_give_up"] else []),
              "budgets": budget, "budget_profile": args.budget_profile,
              "early_stop_policy": "continue_same_thread" if long_run else "allow_give_up",
              "gpu_allocation": allocation,
              "task": task["activity"], "feishu_id": task["feishu_id"], "split": "train", "instance": 0,
              "selection": task,
              "exact_feishu_instance_verified": False,
              "image_history": IMAGE_HISTORY_POLICY}
    save(run / "result.json", result)
    process = adapter = log = None
    try:
        overrides = model_overrides()
        if args.mode == "agent":
            cfg = tomllib.loads((Path.home() / ".codex/config.toml").read_text())
            model = cfg["model"]
            if args.aihub_direct and model not in {"gpt-6-astra-azure", "gpt-6-astra"}:
                raise ValueError("AI Hub direct route requires GPT-6 Astra; do not silently switch models")
            result["model"] = model
            # Only registers tools, no inference or simulator side effects.
            with CodexSession(args.manifest, run / "registration", config_overrides=overrides) as session:
                params = dict(ephemeral=True, approvalPolicy="never", environments=[], model=model,
                              baseInstructions=BehaviorAdapter.instructions,
                              dynamicTools=[dict(type="function", **s) for s in protocol.TOOLS],
                              config={"features.shell_tool": False, "features.code_mode": False})
                session.rpc("thread/start", params)
            result["registration_passed"] = True
        sim = run / "simulator"
        command = ["bash", str(worker / "run_native.sh"), str(sim), "--eef-agent"]
        worker_env = os.environ.copy()
        worker_env.pop("BEHAVIOR_FEISHU_ID", None)
        worker_env.pop("BEHAVIOR_BUDGET_PROFILE", None)
        worker_env.pop("BEHAVIOR_MANAGED_BATCH", None)
        worker_env.pop("BEHAVIOR_HOST_GPU", None)
        if args.task is not None:
            worker_env["BEHAVIOR_FEISHU_ID"] = str(args.task)
            worker_env["BEHAVIOR_BUDGET_PROFILE"] = args.budget_profile
        if args.record_video:
            ffmpeg = (WORLD / "var/venv/bin/ffmpeg").resolve(strict=True)
            worker_env.update(BEHAVIOR_NATIVE_WORKER=str(worker), BEHAVIOR_VIDEO_FFMPEG=str(ffmpeg))
            command[1] = str(Path(__file__).with_name("run_video.sh"))
            if allocation:
                worker_env.update(BEHAVIOR_MANAGED_BATCH=str(args.managed_batch.resolve()),
                                  BEHAVIOR_HOST_GPU=str(args.gpu))
                command[1] = str(Path(__file__).with_name("run_parallel.sh"))
            result["ffmpeg_sha256"] = hashlib.sha256(ffmpeg.read_bytes()).hexdigest()
        save(run / "command.json", command)
        log = (run / "launcher.log").open("w")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=worker_env)
        adapter = BehaviorAdapter(sim, process, protocol, budget_profile=args.budget_profile)
        adapter.start()
        native_ready = json.loads((sim / "native-result.json").read_text())
        native_selection = native_ready["selection"]
        if args.task is not None and native_selection != task:
            raise RuntimeError("Worker loaded a different task selection")
        if long_run and native_ready.get("budgets") != budget:
            raise RuntimeError("Worker budget does not match the host")
        result["gateway"] = gateway_checks(adapter)
        result["status"] = "environment_ready"
        save(run / "result.json", result)
        if args.mode == "gateway":
            result["physical_smoke"] = physical_smoke(adapter)
            if long_run:
                result["legacy_limit_crossing"] = legacy_limit_crossing(adapter)
        else:
            with RequestAudit(cfg, run / "request-audit.json", set(result["tools"]),
                              current_image=lambda: (adapter.simulator / adapter.image).read_bytes(),
                              aihub_direct=args.aihub_direct, max_requests=budget["max_requests"],
                              max_request_bytes=budget["request_bytes"], max_retries=budget["api_retries"],
                              image_window=adapter.visual_history.snapshot) as audit:
                episode = run_episode(adapter, manifest=args.manifest, output_dir=run / "episode",
                                      model=model, config_overrides=overrides + audit.overrides,
                                      max_actions=budget["motion_requests"], max_tools=budget["tool_calls"],
                                      timeout_s=budget["wall_seconds"], allow_give_up=budget["allow_give_up"],
                                      continue_on_completion=long_run,
                                      idle_timeout_s=budget["model_idle_seconds"],
                                      on_interrupt=audit.controller_interrupt)
            result["request_boundary_valid"] = audit.valid
            result["episode"] = episode
            result["model_turn_started"] = episode["model_turn_started"]
        result["status"] = "completed"
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
    finally:
        if adapter is not None and process.poll() is None and not adapter.uncertain and adapter.state:
            try:
                result["checker"] = adapter.close()["checker"]
                process.wait(timeout=90)
            except BaseException:
                result["cleanup_error"] = traceback.format_exc()
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result["forced_cleanup"] = True
        if log is not None:
            log.close()
        result["source_unchanged"] = source_hashes(worker, args.record_video, args.task is not None,
                                                  allocation is not None) == hashes
        if process is not None:
            result["simulator_exit"] = process.returncode
            native = run / "simulator/native-result.json"
            clean = run / "simulator/exit-summary.json"
            report = json.loads(native.read_text()) if native.exists() else {}
            exit_report = json.loads(clean.read_text()) if clean.exists() else {}
            result["worker"] = report
            result["execution_uncertain"] = adapter.uncertain if adapter else True
            result["environment_valid"] = bool(
                process.returncode == 0 and exit_report.get("process_exit_clean") and
                report.get("status") == "passed" and report.get("reset_verified") and
                report.get("checker_executed") and not result.get("forced_cleanup") and
                not result["execution_uncertain"] and result["source_unchanged"])
            result["accepted"] = bool(result["status"] == "completed" and result["environment_valid"])
            if args.record_video:
                try:
                    result["video"] = verify_video(run / "simulator/video", report["native_steps"])
                    result["video_valid"] = True
                except Exception:
                    result["video_valid"] = False
                    result["video_error"] = traceback.format_exc()
                result["accepted"] = result["accepted"] and result["video_valid"]
            if args.mode == "agent":
                ep = run / "episode/episode.json"
                episode = json.loads(ep.read_text()) if ep.exists() else {}
                result["model_turn_started"] = episode.get("model_turn_started", False)
                valid = (result["accepted"] and result["model_turn_started"] and
                         result.get("request_boundary_valid", False) and
                         episode.get("termination") in SCORABLE)
                if long_run:
                    exhausted = report.get("native_steps") == budget["native_steps"]
                    successful = report.get("checker", {}).get("success") is True
                    result["control_step_budget_exhausted"] = exhausted
                    # Operational guards and early native truncation are not full-budget model scores.
                    valid = valid and (exhausted or successful)
                result["score_valid"] = bool(valid)
                if valid:
                    result["model_score"] = report["checker"]["q_score"]
                    result["success"] = report["checker"]["success"]
        save(run / "result.json", result)
        print(json.dumps({k: result.get(k) for k in
                          ("status", "accepted", "mode", "model_score", "success", "score_valid")}))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
