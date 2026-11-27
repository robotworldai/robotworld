"""Bounded serial diagnostic: native gateway then fresh Codex episode per task."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from environment.benchmarks.behavior_1k.run import PRIVATE, WORLD, save, source_hashes
from environment.benchmarks.behavior_1k.tasks import TASKS, selection, verify_template
from environment.runtime.codex_session import CodexSession
from environment.runtime.request_audit import azure_authorization
from environment.benchmarks.behavior_1k.budgets import DEFAULT_BATCH_PROFILE, PROFILES, get_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--budget-profile", choices=tuple(PROFILES), default=DEFAULT_BATCH_PROFILE)
    args = parser.parse_args()
    budget = get_budget(args.budget_profile)
    output = args.output.resolve()
    if output.parent != PRIVATE or not output.name.startswith("first-four-") or output.exists():
        parser.error("Use a fresh private first-four-* directory")
    worker, manifest = args.worker_dir.resolve(), args.manifest.resolve()
    azure_authorization()
    CodexSession.verify_build(manifest)
    receipt = json.loads((PRIVATE / "partial/receipt.json").read_text())
    for number in TASKS:
        verify_template(PRIVATE / "data", selection(number), receipt)
    os.umask(0o077)
    output.mkdir()
    hashes = source_hashes(worker, True, True)
    status = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                  profile=f"Codex + sensor-only EEF, fixed trunk, {args.budget_profile}",
                  budget_profile=args.budget_profile, budgets=budget,
                  selections=[selection(n) for n in TASKS], tasks=[], source_hashes=hashes)
    save(output / "status.json", status)
    for number in TASKS:
        if source_hashes(worker, True, True) != hashes:
            raise RuntimeError("Source changed during batch; remaining tasks not started")
        item = dict(feishu_id=number, activity=TASKS[number]["activity"], status="gateway",
                    model_score=None, score_valid=False)
        status["tasks"].append(item)
        gateway = PRIVATE / f"native-task{number}-scaffold-{output.name}-gateway"
        agent = PRIVATE / f"native-task{number}-scaffold-{output.name}-agent"
        for mode, run in (("gateway", gateway), ("agent", agent)):
            item.update(status=mode, current_run=str(run))
            save(output / "status.json", status)
            command = [sys.executable, "-m", "environment.benchmarks.behavior_1k.run", mode,
                       "--task", str(number), "--record-video", "--worker-dir", str(worker),
                       "--run", str(run), "--budget-profile", args.budget_profile]
            if mode == "agent":
                command += ["--aihub-direct", "--accepted-smoke", str(gateway), "--manifest", str(manifest)]
            save(output / f"task{number}-{mode}-command.json", command)
            with (output / f"task{number}-{mode}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=WORLD, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=budget["launcher_seconds"] + 300)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    item.update(status="infrastructure_timeout", score_valid=False)
                    # The worker may have a separate session. Never launch another run until inspected.
                    status["status"] = "stopped_for_cleanup_inspection"
                    save(output / "status.json", status)
                    return 1
            path = run / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            item[mode] = dict(run=str(run), exit_code=code, accepted=result.get("accepted", False),
                              error=result.get("error"))
            if code or not result.get("accepted"):
                item.update(status=f"{mode}_failed", score_valid=False)
                break
            if mode == "agent":
                item.update(status="completed", model_score=result.get("model_score"),
                            score_valid=result.get("score_valid", False), success=result.get("success"),
                            checker=result.get("checker"), video_valid=result.get("video_valid"),
                            termination=result.get("episode", {}).get("termination"),
                            action_calls=result.get("episode", {}).get("action_calls"),
                            native_steps=result.get("worker", {}).get("native_steps"))
            save(output / "status.json", status)
        save(output / "status.json", status)
    status.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    save(output / "status.json", status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
