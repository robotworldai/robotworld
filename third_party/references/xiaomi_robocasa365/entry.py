# Copyright (C) 2026 Xiaomi Corporation.
"""Evaluate Xiaomi-Robotics-1 on RoboCasa365 through deploy/client.py."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from client import Client  # noqa: E402


DEFAULT_MODEL_PATH = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60
ACTION_DIM = 12
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)

    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def observation_to_state(observation: dict[str, Any]) -> np.ndarray:
    """Build the EE-first 14D state used to train the RoboCasa365 policy."""
    state = np.concatenate(
        [
            np.asarray(observation["state.end_effector_position_relative"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(observation["state.end_effector_rotation_relative"]),
            np.asarray(observation["state.gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(observation["state.base_position"], dtype=np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(observation["state.base_rotation"]),
        ]
    ).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected a 14D RoboCasa365 state, got {state.shape}")
    return state


def collect_images(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.ascontiguousarray(observation[key], dtype=np.uint8)
        for key in CAMERA_KEYS
    }


def sample_history(history: collections.deque[np.ndarray], length: int, interval: int) -> np.ndarray:
    items = list(history)
    if not items:
        raise ValueError("Observation history cannot be empty")
    indices = [
        max(0, len(items) - 1 - (length - 1 - index) * interval)
        for index in range(length)
    ]
    return np.ascontiguousarray(np.stack([items[index] for index in indices], axis=0))


def center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image

    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def make_video_frame(observation: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([observation[key] for key in CAMERA_KEYS], axis=1))


class EvalClient:
    def __init__(
        self,
        model_path: str,
        host: str,
        port: int,
        robot_type: str,
        crop_ratio: float,
    ) -> None:
        self.client = Client(host=host, port=port, model_path=model_path)
        self.processor = self.client.processor
        self.robot_type = robot_type
        self.crop_ratio = crop_ratio

        robot_types = self.processor.list_robot_types()
        if robot_type not in robot_types:
            self.client.close()
            raise ValueError(
                f"Robot type {robot_type!r} is missing from the checkpoint. "
                f"Available robot types: {robot_types}."
            )
        logging.info("Loaded processor from %s; robot types: %s", model_path, robot_types)

    def _build_messages(self, image_history: dict[str, np.ndarray], instruction: str) -> list[dict[str, Any]]:
        videos = {
            key: [center_crop(frame, self.crop_ratio) for frame in image_history[key]]
            for key in CAMERA_KEYS
        }
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                    {
                        "type": "text",
                        "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

    def infer(
        self,
        state_history: np.ndarray,
        image_history: dict[str, np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        state_history = np.asarray(state_history, dtype=np.float32)
        state = np.zeros((1, state_history.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, : state_history.shape[-1]] = state_history

        inputs = self.processor.apply_chat_template(
            self._build_messages(image_history, instruction),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=state,
            robot_type=self.robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self.robot_type

        actions = self.client(**request)
        actions = actions[0, :, :ACTION_DIM].float().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)

    def close(self) -> None:
        self.client.close()


def reset_env(env: Any, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return env.reset(seed=seed)
    except TypeError as error:
        raise RuntimeError(
            "This evaluator requires a RoboCasa version whose reset() accepts a seed."
        ) from error


def evaluate_task(
    env_name: str,
    task_index: int,
    args: argparse.Namespace,
    client: EvalClient,
    gym: Any,
    get_task_horizon: Any,
    convert_action: Any,
    output_dir: Path,
    episode_indices: list[int] | None = None,
    show_progress: bool = True,
    write_task_stats: bool = True,
) -> dict[str, Any]:
    task_dir = output_dir / env_name
    task_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(f"robocasa/{env_name}", split=args.split, seed=args.seed)
    horizon = args.horizon if args.horizon is not None else get_task_horizon(env_name)
    episode_results = []

    episodes = range(args.num_trials) if episode_indices is None else episode_indices
    try:
        for episode in tqdm(episodes, desc=env_name, disable=not show_progress):
            global_episode_index = task_index * args.num_trials + episode
            episode_seed = args.seed + global_episode_index
            observation, _ = reset_env(env, episode_seed)
            instruction = observation["annotation.human.task_description"]

            queue_length = (args.obs_history - 1) * args.obs_interval + 1
            image_queues = {
                key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS
            }
            state_queue: collections.deque[np.ndarray] = collections.deque(maxlen=queue_length)
            for key, image in collect_images(observation).items():
                image_queues[key].append(image)
            state_queue.append(observation_to_state(observation))

            action_plan: collections.deque[np.ndarray] = collections.deque()
            video_frames = []
            capture_video = args.save_videos or args.save_failure_videos
            if capture_video:
                video_frames.append(make_video_frame(observation))

            success = False
            steps = 0
            while steps < horizon:
                if not action_plan:
                    states = sample_history(state_queue, args.obs_history, args.obs_interval)
                    images = {
                        key: sample_history(queue, args.obs_history, args.obs_interval)
                        for key, queue in image_queues.items()
                    }
                    action_chunk = client.infer(states, images, instruction)
                    if len(action_chunk) < args.replan_steps:
                        raise RuntimeError(
                            f"Model returned {len(action_chunk)} actions, but "
                            f"--replan-steps is {args.replan_steps}."
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                policy_action = np.asarray(action_plan.popleft(), dtype=np.float32)
                observation, _, done, truncated, info = env.step(convert_action(policy_action))
                steps += 1

                for key, image in collect_images(observation).items():
                    image_queues[key].append(image)
                state_queue.append(observation_to_state(observation))

                success = bool(info.get("success", False))
                if capture_video and (steps % args.video_stride == 0 or success or done or truncated):
                    video_frames.append(make_video_frame(observation))
                if success or done or truncated:
                    break

            status = "success" if success else "failure"
            if video_frames and (args.save_videos or (args.save_failure_videos and not success)):
                imageio.mimsave(
                    task_dir / f"episode_{episode:03d}_seed_{episode_seed}_{status}.mp4",
                    video_frames,
                    fps=args.video_fps,
                )

            episode_results.append(
                {
                    "episode": episode,
                    "global_episode_index": global_episode_index,
                    "seed": episode_seed,
                    "success": success,
                    "steps": steps,
                }
            )
            successes = sum(int(item["success"]) for item in episode_results)
            stats = {
                "env_name": env_name,
                "split": args.split,
                "num_episodes": len(episode_results),
                "successes": successes,
                "success_rate": successes / len(episode_results),
                "horizon": horizon,
                "episodes": episode_results,
            }
            if write_task_stats:
                with (task_dir / "stats.json").open("w", encoding="utf-8") as file:
                    json.dump(stats, file, indent=2)

        return stats
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Xiaomi-Robotics-1 on RoboCasa365.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--robot-type", default=ROBOT_TYPE)
    parser.add_argument("--server-addr", default="localhost")
    parser.add_argument("--server-port", type=int, default=10086)
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--task-set", default="target50")
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help="Evaluate one task. Repeat this option to select multiple tasks.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit selected tasks for a smoke test.")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--obs-history", type=int, default=4)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop-ratio", type=float, default=0.95)
    parser.add_argument("--save-root-dir", default="eval_results/robocasa365")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-failure-videos", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--max-tasks must be at least 1")
    if args.num_trials < 1:
        raise ValueError("--num-trials must be at least 1")
    if args.replan_steps < 1:
        raise ValueError("--replan-steps must be at least 1")
    if args.obs_history < 1 or args.obs_interval < 1:
        raise ValueError("--obs-history and --obs-interval must be at least 1")
    if not 0 < args.crop_ratio <= 1:
        raise ValueError("--crop-ratio must be in (0, 1]")
    if args.video_stride < 1:
        raise ValueError("--video-stride must be at least 1")


def select_tasks(
    args: argparse.Namespace,
    task_set_registry: dict[str, Any],
) -> tuple[dict[str, int], list[str]]:
    if args.task_set not in task_set_registry:
        choices = ", ".join(sorted(task_set_registry))
        raise KeyError(f"Unknown task set {args.task_set!r}. Available task sets: {choices}")

    all_tasks = list(task_set_registry[args.task_set])
    task_to_index = {task: index for index, task in enumerate(all_tasks)}
    selected_tasks = list(args.task_name) if args.task_name else all_tasks
    unknown_tasks = [task for task in selected_tasks if task not in task_to_index]
    if unknown_tasks:
        raise KeyError(f"Tasks are not in {args.task_set!r}: {unknown_tasks}")
    if len(selected_tasks) != len(set(selected_tasks)):
        raise ValueError("--task-name must not select the same task more than once")
    if args.max_tasks is not None:
        selected_tasks = selected_tasks[: args.max_tasks]
    return task_to_index, selected_tasks


def build_summary(
    args: argparse.Namespace,
    task_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    num_episodes = sum(stats["num_episodes"] for stats in task_stats.values())
    successes = sum(stats["successes"] for stats in task_stats.values())
    success_rates = [stats["success_rate"] for stats in task_stats.values()]
    return {
        "model_path": args.model_path,
        "robot_type": args.robot_type,
        "split": args.split,
        "task_set": args.task_set,
        "num_tasks": len(task_stats),
        "num_episodes": num_episodes,
        "successes": successes,
        "episode_success_rate": successes / num_episodes if num_episodes else None,
        "mean_task_success_rate": float(np.mean(success_rates)) if success_rates else None,
        "replan_steps": args.replan_steps,
        "obs_history": args.obs_history,
        "obs_interval": args.obs_interval,
        "tasks": task_stats,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    validate_args(args)

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    task_to_index, selected_tasks = select_tasks(args, TASK_SET_REGISTRY)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.save_root_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing evaluation outputs to %s", output_dir)

    client = EvalClient(
        model_path=args.model_path,
        host=args.server_addr,
        port=args.server_port,
        robot_type=args.robot_type,
        crop_ratio=args.crop_ratio,
    )
    task_stats = {}
    try:
        for env_name in selected_tasks:
            task_stats[env_name] = evaluate_task(
                env_name=env_name,
                task_index=task_to_index[env_name],
                args=args,
                client=client,
                gym=gym,
                get_task_horizon=get_task_horizon,
                convert_action=convert_action,
                output_dir=output_dir,
            )
    finally:
        client.close()

    summary = build_summary(args, task_stats)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    logging.info("Wrote summary to %s", summary_path)
    logging.info(
        "Success rate: %.2f%% (%d/%d)",
        100.0 * summary["episode_success_rate"],
        summary["successes"],
        summary["num_episodes"],
    )


if __name__ == "__main__":
    main()
