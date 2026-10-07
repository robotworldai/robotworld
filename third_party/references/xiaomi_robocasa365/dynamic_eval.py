# Copyright (C) 2026 Xiaomi Corporation.
"""Dynamically schedule RoboCasa365 rollouts across local GPU workers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import entry


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_evaluation_args(arguments: list[str]) -> argparse.Namespace:
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    parsed = entry.parse_args(arguments)
    entry.validate_args(parsed)
    return parsed


def load_manifest(queue_dir: Path) -> dict[str, Any]:
    manifest_path = queue_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Queue manifest does not exist: {manifest_path}")
    return read_json(manifest_path)


def output_dir_from_config(config: dict[str, Any]) -> Path:
    return Path(config["save_root_dir"]) / config["run_id"]


def initialize_queue(queue_dir: Path, evaluation_args: list[str]) -> None:
    args = parse_evaluation_args(evaluation_args)

    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    task_to_index, selected_tasks = entry.select_tasks(args, TASK_SET_REGISTRY)
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    queue_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("pending", "running", "results", "errors", "logs"):
        (queue_dir / directory).mkdir()

    jobs = []
    for env_name in selected_tasks:
        task_index = task_to_index[env_name]
        for episode in range(args.num_trials):
            global_episode_index = task_index * args.num_trials + episode
            job = {
                "id": f"{global_episode_index:08d}",
                "env_name": env_name,
                "task_index": task_index,
                "episode": episode,
                "global_episode_index": global_episode_index,
            }
            jobs.append(job)
            write_json_atomic(queue_dir / "pending" / f"{job['id']}.json", job)

    manifest = {
        "version": 1,
        "config": vars(args),
        "selected_tasks": selected_tasks,
        "jobs": jobs,
    }
    write_json_atomic(queue_dir / "manifest.json", manifest)
    output_dir_from_config(manifest["config"]).mkdir(parents=True, exist_ok=True)
    logging.info(
        "Initialized %d rollout jobs for %d tasks in %s",
        len(jobs),
        len(selected_tasks),
        queue_dir,
    )


def claim_job(queue_dir: Path, worker_id: str) -> tuple[Path, dict[str, Any]] | None:
    pending_dir = queue_dir / "pending"
    running_dir = queue_dir / "running"
    for pending_path in sorted(pending_dir.glob("*.json")):
        running_path = running_dir / f"{pending_path.stem}.{worker_id}.json"
        try:
            pending_path.rename(running_path)
        except FileNotFoundError:
            continue
        return running_path, read_json(running_path)
    return None


def run_worker(queue_dir: Path, worker_id: str, server_addr: str, server_port: int) -> None:
    manifest = load_manifest(queue_dir)
    args = argparse.Namespace(**manifest["config"])
    entry.validate_args(args)
    output_dir = output_dir_from_config(manifest["config"])

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    client = entry.EvalClient(
        model_path=args.model_path,
        host=server_addr,
        port=server_port,
        robot_type=args.robot_type,
        crop_ratio=args.crop_ratio,
    )
    completed = 0
    failed_job = None
    try:
        while True:
            claim = claim_job(queue_dir, worker_id)
            if claim is None:
                break
            running_path, job = claim
            logging.info(
                "Worker %s claimed rollout %s (%s episode %d)",
                worker_id,
                job["id"],
                job["env_name"],
                job["episode"],
            )
            try:
                stats = entry.evaluate_task(
                    env_name=job["env_name"],
                    task_index=job["task_index"],
                    args=args,
                    client=client,
                    gym=gym,
                    get_task_horizon=get_task_horizon,
                    convert_action=convert_action,
                    output_dir=output_dir,
                    episode_indices=[job["episode"]],
                    show_progress=False,
                    write_task_stats=False,
                )
                write_json_atomic(
                    queue_dir / "results" / f"{job['id']}.json",
                    {"job": job, "worker_id": worker_id, "stats": stats},
                )
                completed += 1
            except Exception as error:
                failed_job = job
                write_json_atomic(
                    queue_dir / "errors" / f"{job['id']}.json",
                    {
                        "job": job,
                        "worker_id": worker_id,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                logging.exception("Worker %s failed rollout %s", worker_id, job["id"])
                break
            finally:
                running_path.unlink(missing_ok=True)
    finally:
        client.close()

    logging.info("Worker %s completed %d rollouts", worker_id, completed)
    if failed_job is not None:
        raise RuntimeError(f"Worker {worker_id} failed rollout {failed_job['id']}")


def validate_result(result: dict[str, Any], expected_job: dict[str, Any], seed: int) -> bool:
    if result.get("job") != expected_job:
        return False
    stats = result.get("stats", {})
    episodes = stats.get("episodes", [])
    if stats.get("env_name") != expected_job["env_name"] or len(episodes) != 1:
        return False
    episode = episodes[0]
    return (
        episode.get("episode") == expected_job["episode"]
        and episode.get("global_episode_index") == expected_job["global_episode_index"]
        and episode.get("seed") == seed + expected_job["global_episode_index"]
    )


def merge_results(queue_dir: Path) -> None:
    manifest = load_manifest(queue_dir)
    config = manifest["config"]
    args = argparse.Namespace(**config)
    expected_jobs = {job["id"]: job for job in manifest["jobs"]}
    result_paths = sorted((queue_dir / "results").glob("*.json"))
    results = [read_json(path) for path in result_paths]
    actual_ids = [result.get("job", {}).get("id") for result in results]

    duplicate_ids = sorted(job_id for job_id, count in Counter(actual_ids).items() if count > 1)
    unexpected_ids = sorted(set(actual_ids) - set(expected_jobs), key=str)
    missing_ids = sorted(set(expected_jobs) - set(actual_ids))
    invalid_ids = sorted(
        result.get("job", {}).get("id")
        for result in results
        if result.get("job", {}).get("id") in expected_jobs
        and not validate_result(
            result,
            expected_jobs[result["job"]["id"]],
            args.seed,
        )
    )
    global_indices = [
        result.get("job", {}).get("global_episode_index") for result in results
    ]
    duplicate_global_indices = sorted(
        index for index, count in Counter(global_indices).items() if count > 1
    )
    if duplicate_ids or unexpected_ids or missing_ids or invalid_ids or duplicate_global_indices:
        error_files = sorted(path.name for path in (queue_dir / "errors").glob("*.json"))
        raise RuntimeError(
            "Cannot merge incomplete or inconsistent rollout results: "
            f"missing={missing_ids[:20]}, unexpected={unexpected_ids[:20]}, "
            f"duplicate_ids={duplicate_ids[:20]}, invalid={invalid_ids[:20]}, "
            f"duplicate_global_indices={duplicate_global_indices[:20]}, "
            f"error_files={error_files[:20]}"
        )

    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped_results[result["job"]["env_name"]].append(result)

    task_stats = {}
    output_dir = output_dir_from_config(config)
    for env_name in manifest["selected_tasks"]:
        task_results = sorted(
            grouped_results[env_name], key=lambda result: result["job"]["episode"]
        )
        episodes = [result["stats"]["episodes"][0] for result in task_results]
        horizons = {result["stats"]["horizon"] for result in task_results}
        if len(horizons) != 1:
            raise RuntimeError(f"Inconsistent horizons for {env_name}: {sorted(horizons)}")
        successes = sum(int(episode["success"]) for episode in episodes)
        stats = {
            "env_name": env_name,
            "split": args.split,
            "num_episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes),
            "horizon": horizons.pop(),
            "episodes": episodes,
        }
        task_stats[env_name] = stats
        write_json_atomic(output_dir / env_name / "stats.json", stats)

    summary = entry.build_summary(args, task_stats)
    summary_path = output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    logging.info("Wrote merged summary to %s", summary_path)
    logging.info(
        "Success rate: %.2f%% (%d/%d)",
        100.0 * summary["episode_success_rate"],
        summary["successes"],
        summary["num_episodes"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the pending rollout queue.")
    init_parser.add_argument("--queue-dir", type=Path, required=True)
    init_parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)

    worker_parser = subparsers.add_parser("worker", help="Claim and evaluate pending rollouts.")
    worker_parser.add_argument("--queue-dir", type=Path, required=True)
    worker_parser.add_argument("--worker-id", required=True)
    worker_parser.add_argument("--server-addr", default="127.0.0.1")
    worker_parser.add_argument("--server-port", type=int, required=True)

    merge_parser = subparsers.add_parser("merge", help="Validate and merge rollout results.")
    merge_parser.add_argument("--queue-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.command == "init":
        initialize_queue(args.queue_dir, args.evaluation_args)
    elif args.command == "worker":
        run_worker(args.queue_dir, args.worker_id, args.server_addr, args.server_port)
    else:
        merge_results(args.queue_dir)


if __name__ == "__main__":
    main()
