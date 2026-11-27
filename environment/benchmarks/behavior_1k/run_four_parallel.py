"""Four independent GPU lanes, each gateway then agent, one owned Docker daemon."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from environment.benchmarks.behavior_1k.run import PRIVATE, WORLD, save, source_hashes
from environment.benchmarks.behavior_1k.budgets import DEFAULT_BATCH_PROFILE, get_budget
from environment.benchmarks.behavior_1k.tasks import selection, verify_template
from environment.benchmarks.behavior_1k.parallel_runtime import BASE, BatchDaemon, discover_gpus
from environment.runtime.codex_session import CodexSession
from environment.runtime.request_audit import azure_authorization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--budget-profile", choices=("steps5000", "steps10000"), default=DEFAULT_BATCH_PROFILE)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != PRIVATE or not output.name.startswith("first-four-parallel-") or output.exists():
        parser.error("Use a fresh private first-four-parallel-* directory")
    worker, manifest = args.worker_dir.resolve(), args.manifest.resolve()
    azure_authorization()
    CodexSession.verify_build(manifest)
    lanes = discover_gpus(args.gpus)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 32:
        parser.error("Four-lane startup needs at least 32 available CPUs")
    per_lane = min(64, len(cpus) // 4)
    for number, lane in enumerate(lanes, 1):
        lane.update(task=number, cpus=",".join(map(str, cpus[(number-1)*per_lane:number*per_lane])))
    receipt = json.loads((PRIVATE / "partial/receipt.json").read_text())
    for number in range(1, 5):
        verify_template(PRIVATE / "data", selection(number), receipt)
    subprocess.run([sys.executable, str(worker / "partial_assets.py"), "--verify"], check=True)
    os.umask(0o077)
    output.mkdir()
    save(output / "gpu-assignments.json", lanes)
    budget = get_budget(args.budget_profile)
    hashes = source_hashes(worker, True, True, True)
    status = dict(status="preparing", concurrency=4, started_at=datetime.now(timezone.utc).isoformat(),
                  budget_profile=args.budget_profile, budgets=budget, source_hashes=hashes, tasks=[])
    save(output / "status.json", status)
    for lane in lanes:
        folder = output / f"lane{lane['task']}"
        folder.mkdir()
        subprocess.run(["cp", "-a", "--reflink=auto", str(BASE / "behavior51-venv/lib/python3.11/site-packages/isaacsim/apps"),
                        str(folder / "apps")], check=True)
        for relative in ("global/warp_cache", "local/cache/shadercache", "local/cache/nv_shadercache"):
            destination = folder / "appdata" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = PRIVATE / "appdata" / relative
            if source.exists():
                subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(destination)], check=True)
        status["tasks"].append(dict(feishu_id=lane["task"], activity=selection(lane["task"])["activity"],
                                    gpu=lane["index"], status="pending", model_score=None, score_valid=False))
    running = {}

    def start(number, mode):
        if source_hashes(worker, True, True, True) != hashes:
            raise RuntimeError("Source changed; no further lanes started")
        item = status["tasks"][number-1]
        run = PRIVATE / f"native-task{number}-scaffold-{output.name}-{mode}"
        item.update(status=mode, current_run=str(run))
        cmd = [sys.executable, "-m", "environment.benchmarks.behavior_1k.run", mode,
               "--task", str(number), "--record-video", "--budget-profile", args.budget_profile,
               "--worker-dir", str(worker), "--run", str(run), "--managed-batch", str(output),
               "--gpu", str(item["gpu"])]
        if mode == "agent":
            cmd += ["--aihub-direct", "--manifest", str(manifest), "--accepted-smoke", item["gateway"]["run"]]
        save(output / f"task{number}-{mode}-command.json", cmd)
        log = (output / f"task{number}-{mode}.log").open("w")
        proc = subprocess.Popen(cmd, cwd=WORLD, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        running[number] = dict(process=proc, log=log, start=time.monotonic(), mode=mode, run=run)
        save(output / "status.json", status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with BatchDaemon(output):
            try:
                status["status"] = "running"
                for lane in lanes:
                    start(lane["task"], "gateway")
                while running:
                    for number, job in list(running.items()):
                        proc = job["process"]
                        if proc.poll() is None:
                            if time.monotonic() - job["start"] > budget["launcher_seconds"] + 300:
                                raise TimeoutError(f"Lane {number} exceeded operational watchdog")
                            continue
                        job["log"].close()
                        del running[number]
                        path = job["run"] / "result.json"
                        result = json.loads(path.read_text()) if path.exists() else {}
                        item = status["tasks"][number-1]
                        item[job["mode"]] = dict(run=str(job["run"]), exit_code=proc.returncode,
                                                  accepted=result.get("accepted", False), error=result.get("error"))
                        if proc.returncode or not result.get("accepted"):
                            item.update(status=job["mode"] + "_failed", score_valid=False)
                        elif job["mode"] == "gateway":
                            start(number, "agent")
                        else:
                            item.update(status="completed" if result.get("score_valid") else "agent_invalid",
                                        model_score=result.get("model_score"), score_valid=result.get("score_valid", False),
                                        success=result.get("success"), video_valid=result.get("video_valid"),
                                        termination=result.get("episode", {}).get("termination"),
                                        action_calls=result.get("episode", {}).get("action_calls"),
                                        native_steps=result.get("worker", {}).get("native_steps"))
                        save(output / "status.json", status)
                    time.sleep(2)
            finally:
                # Cleanup only lane processes owned by this supervisor. Daemon cleans their containers.
                for job in running.values():
                    proc = job["process"]
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=40)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    job["log"].close()
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
    else:
        status["status"] = "completed"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(output / "status.json", status)
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
