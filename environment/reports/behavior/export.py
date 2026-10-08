"""Export an allowlisted, static review bundle. Never serve the private run root."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

TITLES = {1: "多房间杂货搬运与冰箱收尾", 2: "儿童房书桌整理与物品归位",
          3: "取菜、切菜并关闭冰箱", 4: "多类蔬菜按规则分拣"}
CAMERAS = ("head_640x480", "head", "left_wrist", "right_wrist")
PUBLIC_ARGUMENTS = ("observation_id", "arm", "base", "delta_position", "delta_rotation", "gripper", "steps")
PUBLIC_FEEDBACK = ("reached", "position_error_m", "orientation_error_rad", "measured_motion_m",
                   "target_position_base", "measured_position_base", "collision_aware")


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def public_text(text):
    # Model-authored explanations are not trusted to be safe to publish verbatim.
    text = str(text)
    text = re.sub(r"(?i)Bearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}", "[credential redacted]", text)
    text = re.sub(r"/(?:root|dockerdata|apdcephfs_[^/\s]+)(?:/[^\s\"']*)?", "[local path]", text)
    text = re.sub(r"(?i)(api[_-]?key|authorization|token|password)\s*[:=]\s*\S+", r"\1=[redacted]", text)
    return text


def copy_public(source, destination, boundary):
    if source.is_symlink() or not source.resolve().is_relative_to(boundary.resolve()):
        raise ValueError("Refuse asset outside selected run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def checker(value):
    return pick(value or {}, ("q_score", "success", "terminated", "truncated"))


def export_task(run, output, number):
    result = read(run / "result.json", {})
    episode = read(run / "episode/episode.json", {})
    native = read(run / "simulator/native-result.json", {})
    task = {"id": number, "title": TITLES[number], "activity": result.get("task"),
            "status": result.get("status", "queued"), "score": result.get("model_score"),
            "score_valid": result.get("score_valid", False), "success": result.get("success"),
            "termination": episode.get("termination"), "action_calls": episode.get("action_calls", 0),
            "tool_calls": episode.get("tool_calls", 0), "native_steps": native.get("native_steps", 0),
            "checks": {k: result.get(k) for k in
                       ("environment_valid", "request_boundary_valid", "video_valid")},
            "selection": pick(result.get("selection", {}),
                              ("scene", "instance", "split", "seed", "robot", "gpu_dynamics")),
            "budgets": result.get("budgets", {}), "checker": checker(result.get("checker")),
            "events": [], "videos": {}, "observations": [], "instructions": []}
    task["prompt"] = read(run / "episode/prompt.json", {})
    task["prompt"] = {k: public_text(v) for k, v in task["prompt"].items()
                      if k in ("baseInstructions", "developerInstructions")}
    base = output / "tasks" / str(number)
    native_actions = []
    path = run / "simulator/actions.jsonl"
    if path.exists():
        native_actions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_input = {}
    for row_number, row in enumerate(native_actions, 1):
        key = str(row["request"]["observation_id"])
        if key in by_input:
            raise ValueError("Duplicate native action observation ID")
        by_input[key] = (row_number, row)
    observation_cache = {}

    def observation(identifier):
        identifier = str(identifier)
        if not identifier.isdigit():
            raise ValueError("Invalid observation ID")
        if identifier in observation_cache:
            return observation_cache[identifier]
        folder = run / "simulator/observations" / f"{int(identifier):06d}"
        state = read(folder / "state.json")
        if state is None:
            raise ValueError("Acknowledged observation missing on disk")
        obs = pick(state, ("observation_id", "native_steps", "physics_events", "robot", "base_world_pose"))
        obs["images"] = {}
        for camera in ("head", "left_wrist", "right_wrist", "mosaic"):
            source = folder / f"{camera}.png"
            if source.is_file():
                relative = Path("tasks") / str(number) / "observations" / identifier / source.name
                copy_public(source, output / relative, run)
                obs["images"][camera] = relative.as_posix()
        if not task["instructions"]:
            goal = state.get("instruction", [])
            task["instructions"] = [public_text(x) for x in (goal if isinstance(goal, list) else [goal])]
        task["observations"].append(obs)
        observation_cache[identifier] = obs
        return obs

    cumulative = 0
    for index, event in enumerate(episode.get("events", []), 1):
        execution = event["execution"]
        steps = execution.get("executed_native_steps", 0)
        arguments = pick(event.get("arguments", {}), PUBLIC_ARGUMENTS)
        row = {"number": index, "kind": event["tool"], "arguments": arguments,
               "start_step": cumulative, "end_step": cumulative + steps,
               "executed_steps": steps, "feedback": pick(execution.get("feedback") or {}, PUBLIC_FEEDBACK),
               "source": f"episode.json events[{index-1}]", "checker": None}
        if execution.get("validation_error"):
            row["error"] = public_text(execution["validation_error"])
        if event["tool"] == "step" and steps:
            match = by_input.get(str(arguments.get("observation_id")))
            if match is None:
                raise ValueError("Acknowledged action has no native record")
            line, action = match
            if pick(action["request"], PUBLIC_ARGUMENTS) != arguments:
                raise ValueError("Native action arguments differ from tool call")
            row.update(native_line=line, checker=checker(action.get("checker")),
                       native_action=action["native_action"])
        identifier = execution.get("observation_id")
        if identifier is not None:
            obs = observation(identifier)
            if obs["native_steps"] != cumulative + steps:
                raise ValueError("Observation and acknowledged step counts differ")
            row["observation_id"] = str(identifier)
        row["video_time"] = max(0, (cumulative + steps - 1) / 30)
        cumulative += steps
        task["events"].append(row)
    if episode.get("give_up_arguments"):
        task["stop_reason"] = {k: public_text(v) for k, v in episode["give_up_arguments"].items()
                               if k in ("reason", "hindsight")}
    if episode.get("agent_messages"):
        task["messages"] = [public_text(x) for x in episode["agent_messages"]]
    task["native_steps"] = max(task["native_steps"], cumulative)
    task["sim_seconds"] = task["native_steps"] / 30
    task["unreached"] = sum(e["feedback"].get("reached") is False for e in task["events"])
    if result.get("video_valid"):
        for camera in CAMERAS:
            source = run / "simulator/video" / f"{camera}.mp4"
            if source.exists():
                relative = Path("tasks") / str(number) / "video" / source.name
                copy_public(source, output / relative, run)
                task["videos"][camera] = relative.as_posix()
    audit = read(run / "request-audit.json", [])
    task["api_summary"] = {"requests": len(audit),
                           "completed": sum(r.get("terminal_event") == "response.completed" for r in audit),
                           "violations": sum(bool(r.get("violations")) for r in audit)}
    if audit:
        first = audit[0].get("started_at")
        last = audit[-1].get("controller_interrupt_time") or audit[-1].get("finished_at")
        task["api_span_seconds"] = max(0, last-first) if first and last else None
    write(base / "trace.json", task)
    task["download"] = f"tasks/{number}/trace.json"
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = read(args.batch / "status.json")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for number in TITLES:
        item = next((x for x in batch["tasks"] if x["feishu_id"] == number), {})
        run = args.batch.parent / f"native-task{number}-scaffold-{args.batch.name}-agent"
        task = export_task(run, output, number)
        task["gateway_accepted"] = item.get("gateway", {}).get("accepted", False)
        task["queue_status"] = item.get("status", "queued")
        tasks.append(task)
    for name in ("index.html", "style.css", "app.js"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    write(output / "data.json", {"generated_at": datetime.now(timezone.utc).isoformat(),
          "batch_status": batch["status"], "model": "GPT-6 Astra",
          "harness": "Codex + sensor-only EEF", "simulator": "Isaac Sim 5.1 / OmniGibson 3.9.3",
          "scope": "原生训练实例 0；固定躯干、无高层 skill 的诊断回合，不等同于完整官方评测。",
          "tasks": tasks})
    print(json.dumps({"output": str(output), "tasks": len(tasks),
                      "scored": sum(t["score_valid"] for t in tasks)}))


if __name__ == "__main__":
    main()
