# RoboCasa365 agent adapter

This integration uses the official `gym.make('robocasa/<Task>', split=..., seed=...)`, `reset(seed=...)`, `convert_action`, and `env.step` interfaces. The task horizon comes from `get_task_horizon`; each step reads native `info['success']` and stops on success, termination, or truncation. Upstream scenes, objects, controllers, and scoring are retained.

Environment sources, containers, and assets are under `third_party/benchmarks/robocasa/`; robosuite is under `third_party/dependencies/robosuite/checkout`. This directory holds the agent policy and control protocol. Evaluation starts in `environment/integrations/robocasa_eval.py`.

## Native controls

PandaOmron uses a 12-dimensional action: end-effector translation (3), rotation (3), gripper (1), base (3), torso (1), and mode (1). Control follows the official OSC_POSE delta semantics.

- `move_eef`: six normalised increments. Values in `[-1, 1]` map to ±0.05 m translation and ±0.5 rad rotation. The increment repeats at each requested step.
- `move_base`: three normalised base inputs with automatic base-following mode; these are not velocities in m/s.
- `move_torso`: a normalised torso-joint increment, rather than an absolute height.
- `set_gripper`: 1 closes and 0 opens; the setting persists until changed.
- `move_robot`: coordinates the full native action with explicit mode 0 or 1. Mode 1 follows the base using the target pose; mode 0 updates arm targets using the measured pose.

Calls execute 1–30 steps. Invalid bounded requests do not advance the environment. Unspecified motion components are zero, although zero increments do not guarantee that every joint remains physically stationary. The adapter adds no IK solver or collision planner and exposes no hidden simulator state. Observations use three official RGB views and proprioception; depth is disabled by default.

## Evaluation protocol

The pinned reference under `third_party/references/xiaomi_robocasa365` supplies the Gym wrapper, horizon, split, seed schedule, and success aggregation conventions. Evaluation does not run Xiaomi model weights, processors, or services.

Defaults are the `pretrain` split, seed 7, and 50 trials per task. Tasks are selected from `all_tasks`. Episode seeds follow `seed + task_index * num_trials + episode`, where `task_index` comes from the fixed task set. Selecting `target50` uses the reference set and order. RobotWorld's selected subset is not the full target50 set, so its aggregate score is not directly comparable with the reference 2,500-episode result. Changing `num_trials` also changes subsequent task seeds.

Task mappings are in `environment/scenarios/robocasa/robotworld.json`. IDs 10 and 28 both map to `CoffeeSetupMug` and are counted once. The pinned version has a `CountertopCleanup` class but no official horizon; it supports only explicitly labelled diagnostics with `--horizon`. Requested item 7 remains unspecified. Original language instructions are retained; display aliases do not introduce additional objectives.

## Isolation and records

The integration reuses BEHAVIOR's source-binary verification, host bubblewrap sandbox, and trusted relay. The simulator runs in a separate Docker container. The agent sees `/workspace`, read-only `/observations`, and basic system programs, without simulator assets or evaluator internals. The network is shared with the host and has no allowlist. Authentication comes from a dedicated local login directory, outside the image.

Logs record each native action, proprioceptive feedback, and official result. Complete Codex/tool events and image-free copies are generated automatically. Video records one frame per control step at the native `control_freq`. Model input includes the current observation and up to three historical observations at a two-step interval, independently of video sampling.

See [the integration README](../../../third_party/benchmarks/robocasa/README.md) for build and run instructions and `STATUS.md` for recorded validation. Probe runs and horizon overrides are explicitly labelled; a working connection alone does not establish task success.
