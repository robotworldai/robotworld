# ReflexBench

Official Isaac Lab implementation of **ReflexBench**, the reaction-critical manipulation benchmark introduced in [*Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation*](https://arxiv.org/abs/2608.14379).

ReflexBench contains six dynamic manipulation tasks and a latency-aware evaluation framework. Simulator stepping is decoupled from robot control, allowing controlled evaluation of synchronous and asynchronous policy inference at configurable control frequencies and inference latencies.

- Paper: https://arxiv.org/abs/2608.14379
- Project page: https://reflexvla.github.io
- Dataset: https://huggingface.co/datasets/cyx337/ReflexBench_dataset
- Repository: https://github.com/LxRoboticsLab/ReflexBench

## Tasks

Each task provides joint-position, absolute-IK, relative-IK, play, and camera-enabled data-collection variants.

| Task | Gym ID prefix | Demonstration policy | Source |
| --- | --- | --- | --- |
| Conveyor Belt Pick-and-Place | `ConveyorBeltPickAndPlace-Franka` | RL checkpoint | [task](source/reflexbench/reflexbench/tasks/manager_based/conveyor_belt_pick_and_place/README.md) |
| Ball Catching | `BallCatching-Franka` | Planning | [task](source/reflexbench/reflexbench/tasks/manager_based/ball_catching/README.md) |
| Whack-a-Mole | `WhackAMole-Franka` | Planning | [task](source/reflexbench/reflexbench/tasks/manager_based/whack_a_mole/README.md) |
| Rolling Ball Interception | `RollingBallInterception-Franka` | Planning | [task](source/reflexbench/reflexbench/tasks/manager_based/rolling_ball_interception/README.md) |
| Ball Throwing | `BallThrowing-Franka` | Planning | [task](source/reflexbench/reflexbench/tasks/manager_based/ball_throwing/README.md) |
| Rotating Peg Insertion | `RotatingPegInsertion-Franka` | Planning | [task](source/reflexbench/reflexbench/tasks/manager_based/rotating_peg_insertion/README.md) |

Task suffixes follow this convention:

| Suffix | Control / purpose |
| --- | --- |
| `-v0` | Joint-position control |
| `-Play-v0` | Lightweight play/evaluation configuration |
| `-IK-Abs-v0` | Absolute end-effector pose control |
| `-IK-Rel-v0` | Relative end-effector pose control |
| `-DataCollection-v0` | Camera-enabled VLA data collection and evaluation |

Run `python scripts/list_envs.py` after installation to list every registered ID.

### Control and observation representations

Data collection and evaluation use the same `--control` setting so the controller, recorded actions, and policy outputs remain consistent.

| `--control` | Action format | Dimension | Default proprioception with `--state_format auto` |
| --- | --- | ---: | --- |
| `joint_pos` | Seven absolute arm-joint targets and gripper | 8 | Joint positions and gripper state |
| `ik_abs` | Absolute end-effector position and roll-pitch-yaw, plus gripper | 7 | End-effector position, quaternion, and gripper state |
| `ik_rel` | Relative end-effector position and roll-pitch-yaw delta, plus gripper | 7 | End-effector position, quaternion, and gripper state |

End-effector orientations in observations use quaternion order `(w, x, y, z)` by default. IK action orientations use roll-pitch-yaw. Evaluation can override proprioception with `--state_format joint`, `eef_pose`, or `both`, and orientation with `--orientation_rep quat` or `euler`.

## Installation

1. Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) and activate its Python environment.
2. Clone ReflexBench outside the Isaac Lab source directory.
3. Install the extension:

   ```bash
   git clone https://github.com/LxRoboticsLab/ReflexBench.git
   cd ReflexBench
   python -m pip install -e source/reflexbench
   ```

   Install the optional LeRobot conversion tools with:

   ```bash
   python -m pip install -e "source/reflexbench[data]"
   ```

   The planning-based collectors for five tasks additionally require [cuRobo](https://curobo.org) in the Isaac Lab environment. Conveyor Belt Pick-and-Place uses an RL checkpoint instead.

4. Verify task registration:

   ```bash
   python scripts/list_envs.py
   python scripts/zero_agent.py --task BallCatching-Franka-Play-v0
   ```

## Data collection

`collect_vla_data.py` records synchronized camera observations, robot state, actions, task metadata, and success labels to HDF5.

**Conveyor Belt Pick-and-Place is currently the only task whose demonstration collection depends on a trained RL policy. It does not support `--policy planning`.** The other five tasks use their task-specific planning policies. All examples use `DataCollection-v0` variants because those variants enable the fixed and wrist cameras.

### Released LeRobot v3.0 dataset

A curated version of the ReflexBench demonstrations is available on Hugging Face at [`cyx337/ReflexBench_dataset`](https://huggingface.co/datasets/cyx337/ReflexBench_dataset). The dataset uses the **LeRobot v3.0** format and contains approximately **200 episodes per task** for all six ReflexBench tasks.

Download the complete dataset with the Hugging Face CLI:

```bash
hf download cyx337/ReflexBench_dataset \
  --type dataset \
  --local-dir datasets/ReflexBench_dataset
```

### Conveyor Belt Pick-and-Place

#### Train the RL policy

The data collector currently loads skrl checkpoints. Train the PPO policy with:

```bash
python scripts/skrl/train.py \
  --task ConveyorBeltPickAndPlace-Franka-v0 \
  --headless
```

Training outputs are written below `logs/skrl/logs/conveyor_belt_pick_and_place/`. To inspect a checkpoint before collecting data:

```bash
python scripts/skrl/play.py \
  --task ConveyorBeltPickAndPlace-Franka-Play-v0 \
  --checkpoint /path/to/conveyor_policy.pt
```

An RSL-RL training configuration is also registered for RL experiments:

```bash
python scripts/rsl_rl/train.py \
  --task ConveyorBeltPickAndPlace-Franka-v0 \
  --headless
```

RSL-RL checkpoints are not currently accepted by `collect_vla_data.py`; use a skrl checkpoint for demonstration collection.

#### Collect demonstrations with the RL checkpoint

```bash
python scripts/data_collection/collect_vla_data.py \
  --task ConveyorBeltPickAndPlace-Franka-DataCollection-v0 \
  --policy checkpoint \
  --checkpoint /path/to/conveyor_policy.pt \
  --control joint_pos \
  --num_envs 5 \
  --num_episodes 200 \
  --headless
```

The checkpoint and `--control` mode must match the environment used during RL training. Passing `--policy planning` for this task is rejected explicitly.

### Ball Catching

```bash
python scripts/data_collection/collect_vla_data.py \
  --task BallCatching-Franka-DataCollection-v0 \
  --policy planning \
  --control joint_pos \
  --num_envs 5 \
  --num_episodes 200 \
  --planning_speed 4.0 \
  --headless
```

### Whack-a-Mole

```bash
python scripts/data_collection/collect_vla_data.py \
  --task WhackAMole-Franka-DataCollection-v0 \
  --policy planning \
  --control joint_pos \
  --num_envs 5 \
  --num_episodes 10 \
  --planning_speed 4.0 \
  --headless
```

### Rolling Ball Interception

```bash
python scripts/data_collection/collect_vla_data.py \
  --task RollingBallInterception-Franka-DataCollection-v0 \
  --policy planning \
  --control joint_pos \
  --num_envs 5 \
  --num_episodes 10 \
  --planning_speed 4.0 \
  --headless
```

### Ball Throwing

```bash
python scripts/data_collection/collect_vla_data.py \
  --task BallThrowing-Franka-DataCollection-v0 \
  --policy planning \
  --control joint_pos \
  --num_envs 5 \
  --num_episodes 200 \
  --headless
```

### Rotating Peg Insertion

```bash
python scripts/data_collection/collect_vla_data.py \
  --task RotatingPegInsertion-Franka-DataCollection-v0 \
  --policy planning \
  --control joint_pos \
  --num_envs 5 \
  --num_episodes 200 \
  --planning_speed 4.0 \
  --headless
```

Important collection options:

- `--num_envs` is the number of parallel simulation environments.
- `--num_episodes` is the total number of saved episodes, not the number per environment.
- `--planning_speed 4.0` shortens planned trajectories for the five planning-supported tasks; it does not apply to Conveyor Belt Pick-and-Place.
- `--save_failed` retains failed episodes; without it, only successful episodes are saved.
- `--fps` overrides both the robot control rate and dataset FPS while leaving the physics timestep unchanged.
- `--prompt` overrides the task-specific language instruction stored in the dataset.
- `--output` selects an explicit HDF5 path. If omitted, the collector writes under `scripts/data_collection/collected_data/<task_slug>/`.
- `--control` must match the policy action representation. `joint_pos` uses 8 values (7 arm joints and gripper); either IK mode uses 7 values (6-DoF end-effector command and gripper).

### Convert HDF5 to LeRobot

For example, the default Ball Catching collection path can be converted with:

```bash
python scripts/data_collection/convert_to_lerobot.py \
  --input scripts/data_collection/collected_data/ballcatching/ballcatching_franka_datacollection_v0.hdf5 \
  --output datasets/ballcatching
```

Useful conversion options:

- `--target_fps x` downsamples the source trajectories to x FPS.
- `--success_only` skips episodes not marked successful.
- `--output` may be an absolute path when the LeRobot dataset should live outside this repository.

The converter reads the task, cameras, FPS, state representation, action representation, and language prompt from the HDF5 metadata.

## Latency-aware evaluation

`scripts/evaluation/eval.py` supports a remote policy server and a local ACT checkpoint. The remote server must implement the [policy-server protocol](scripts/evaluation/POLICY_SERVER.md).

### Evaluate your own policy

The recommended way to evaluate a custom policy is to wrap its inference code in an HTTP server. This keeps model-specific dependencies separate from the Isaac Lab environment and supports policies implemented with any framework. Follow [`scripts/evaluation/POLICY_SERVER.md`](scripts/evaluation/POLICY_SERVER.md) for the complete request and response schemas and a minimal FastAPI example.

The evaluation client uses these endpoints:

| Endpoint | Required | Purpose |
| --- | --- | --- |
| `GET /info` | Yes | Report `action_dim`, `action_horizon`, model name, and control mode. |
| `POST /predict` | Yes | Receive a batched observation and return one action chunk per environment. |
| `POST /reset` | No | Clear recurrent state or observation history for environments that have reset. |

A typical integration workflow is:

1. Load your policy and preprocessing pipeline in a server process.
2. Implement `/info` and `/predict` according to the policy-server protocol. The `actions` response should normally have shape `(N, H, D)`, where `N` is the requested number of environments, `H` is the action horizon, and `D` is the action dimension.
3. If the policy keeps temporal state, implement `/reset` and clear only the environment IDs included in the request.
4. Start the server and verify its metadata endpoint:

   ```bash
   python path/to/your_policy_server.py
   curl http://localhost:8000/info
   ```

5. Run a short, single-environment smoke test before launching the full benchmark:

   ```bash
   python scripts/evaluation/eval.py --headless \
     --task BallCatching-Franka-DataCollection-v0 \
     --task_profile ball_catching \
     --backend server \
     --server_url http://localhost:8000 \
     --obs_mode vla \
     --control joint_pos \
     --task_description "hold the container with the gripper and catch the thrown ball in the container" \
     --inference_mode async \
     --infer_freq 10 \
     --execution_horizon 8 \
     --num_envs 1 \
     --num_episodes 5 \
     --output outputs/evaluation/smoke_test.json
   ```

Before a full evaluation, check the following:

- `--control` must match the policy action representation and the `control_mode` returned by `/info`. See the [action representation table](scripts/evaluation/POLICY_SERVER.md#action-representations) for dimensions and gripper conventions.
- For a vision-language policy, use `--obs_mode vla` and a `DataCollection-v0` task so fixed and wrist cameras are available. The request contains base64-encoded RGB images, proprioception, and `task_description`.
- For a state-only policy, use `--obs_mode state`; `/predict` then receives a flat state batch instead of images.
- The server must preserve the batch dimension and return actions for every requested environment. Begin with `--num_envs 1`, then test the intended parallel batch size.
- Set `--image_history` and `--image_history_stride` to the temporal context expected by the model. Multi-frame camera inputs are ordered from oldest to newest.
- Use `--execution_horizon` to cap how many actions from each returned chunk are executed. It may be shorter than the server's advertised `action_horizon`.
- Add `--use_real_latency` when measuring the effect of actual preprocessing, network transfer, and model inference time. Without it, `--infer_freq` defines a fixed simulated inference period.

The `--output` JSON contains the full configuration, aggregate success rate, returns, episode lengths, latency/real-time-factor statistics, and per-episode results. Use `--video_dir` when qualitative videos are also needed.

### Timing and temporal observations

The simulator, controller, and policy have separate time scales:

- `--ctrl_freq 25` makes the robot consume one action every 40 ms.
- `--infer_freq 10` defines a fixed policy period of 100 ms.
- `--use_real_latency` replaces the fixed inference period on every request with the measured wall-clock latency converted using the simulator real-time factor. When this flag is present, `--infer_freq` is not the per-cycle delay used for evaluation.
- `--inference_mode sync` applies a freeze-arm latency penalty before executing the newly predicted action chunk.
- `--inference_mode async` executes the previous chunk while a newly predicted chunk is pending.
- `--execution_horizon 8` caps each returned action chunk at eight executable actions.

For temporal observations, `--image_history T` sends `T` frames per camera, oldest first. `--image_history_stride S` selects every `S`-th captured frame. Thus history 5 with stride 4 spans 17 captured frames including the current frame; at 25 Hz this is nominally 0.64 seconds. History 2 with stride 1 spans two consecutive frames.

### Reference evaluation commands

The following commands reproduce the common five-environment, server-based evaluation setup. Each command saves task-specific videos and a JSON result file.

#### Conveyor Belt Pick-and-Place

```bash
python scripts/evaluation/eval.py --headless \
  --task ConveyorBeltPickAndPlace-Franka-DataCollection-v0 \
  --backend server \
  --server_url http://localhost:8000 \
  --task_profile conveyor_belt_pick_and_place \
  --inference_mode async --infer_freq 10 \
  --task_description "pick up the cube from the moving conveyor and place it into the bin" \
  --execution_horizon 8 \
  --image_history 5 \
  --image_history_stride 4 \
  --ctrl_freq 25 \
  --num_envs 5 \
  --video_dir eval_videos/conveyor_belt_pick_and_place \
  --num_episodes 150 \
  --use_real_latency \
  --output outputs/evaluation/conveyor_belt_pick_and_place.json
```

#### Whack-a-Mole

```bash
python scripts/evaluation/eval.py --headless \
  --task WhackAMole-Franka-DataCollection-v0 \
  --backend server \
  --server_url http://localhost:8000 \
  --task_profile whack_a_mole \
  --inference_mode async --infer_freq 10 \
  --task_description "hit the mole that pops up on the board with the closed gripper" \
  --execution_horizon 8 \
  --image_history 5 \
  --image_history_stride 4 \
  --ctrl_freq 25 \
  --num_envs 5 \
  --video_dir eval_videos/whack_a_mole \
  --num_episodes 150 \
  --use_real_latency \
  --output outputs/evaluation/whack_a_mole.json
```

#### Ball Catching

```bash
python scripts/evaluation/eval.py --headless \
  --task BallCatching-Franka-DataCollection-v0 \
  --backend server \
  --server_url http://localhost:8000 \
  --task_profile ball_catching \
  --inference_mode async --infer_freq 10 \
  --task_description "hold the container with the gripper and catch the thrown ball in the container" \
  --execution_horizon 8 \
  --image_history 5 \
  --image_history_stride 4 \
  --ctrl_freq 25 \
  --num_envs 5 \
  --video_dir eval_videos/ball_catching \
  --num_episodes 150 \
  --use_real_latency \
  --output outputs/evaluation/ball_catching.json
```

#### Rolling Ball Interception

```bash
python scripts/evaluation/eval.py --headless \
  --task RollingBallInterception-Franka-DataCollection-v0 \
  --backend server \
  --server_url http://localhost:8000 \
  --task_profile rolling_ball_interception \
  --inference_mode async --infer_freq 10 \
  --task_description "catch the ball rolling down the ramp with the container" \
  --execution_horizon 8 \
  --image_history 2 \
  --image_history_stride 1 \
  --ctrl_freq 25 \
  --num_envs 5 \
  --video_dir eval_videos/rolling_ball_interception \
  --num_episodes 150 \
  --use_real_latency \
  --output outputs/evaluation/rolling_ball_interception.json
```

#### Ball Throwing

```bash
python scripts/evaluation/eval.py --headless \
  --task BallThrowing-Franka-DataCollection-v0 \
  --backend server \
  --server_url http://localhost:8000 \
  --task_profile ball_throwing \
  --inference_mode async --infer_freq 10 \
  --task_description "throw the ball into the bin" \
  --execution_horizon 8 \
  --image_history 5 \
  --image_history_stride 4 \
  --ctrl_freq 25 \
  --num_envs 5 \
  --video_dir eval_videos/ball_throwing \
  --num_episodes 150 \
  --use_real_latency \
  --output outputs/evaluation/ball_throwing.json
```

#### Rotating Peg Insertion

```bash
python scripts/evaluation/eval.py --headless \
  --task RotatingPegInsertion-Franka-DataCollection-v0 \
  --backend server \
  --server_url http://localhost:8000 \
  --task_profile rotating_peg_insertion \
  --inference_mode async --infer_freq 10 \
  --task_description "insert the peg into the hole on the rotating disc" \
  --execution_horizon 8 \
  --image_history 5 \
  --image_history_stride 4 \
  --ctrl_freq 25 \
  --num_envs 5 \
  --video_dir eval_videos/rotating_peg_insertion \
  --num_episodes 150 \
  --use_real_latency \
  --output outputs/evaluation/rotating_peg_insertion.json
```


## Citation

If ReflexBench is useful in your research, please cite:

```bibtex
@article{chen2026reflex,
  title   = {Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation},
  author  = {Chen, Yuxuan and Zhang, Wanruo and Li, Xiao},
  journal = {arXiv preprint arXiv:2608.14379},
  year    = {2026}
}
```

## License

This project is released under the [BSD 3-Clause License](LICENSE). Individual files retain their original copyright notices.
