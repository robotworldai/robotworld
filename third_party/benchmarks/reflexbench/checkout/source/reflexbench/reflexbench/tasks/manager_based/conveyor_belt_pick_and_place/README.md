# Conveyor Belt Pick-and-Place (Franka)

Pick object from a moving conveyor belt and place it into a box. Supports joint position control and IK (absolute/relative) control.

## Registered tasks

| Task | Description |
|------|-------------|
| `ConveyorBeltPickAndPlace-Franka-v0` | Joint position control (7-DOF + gripper) |
| `ConveyorBeltPickAndPlace-Franka-Play-v0` | Same, play/eval variant (fewer envs, no obs noise) |
| `ConveyorBeltPickAndPlace-Franka-IK-Abs-v0` | IK absolute pose control |
| `ConveyorBeltPickAndPlace-Franka-IK-Rel-v0` | IK relative pose control |
| `ConveyorBeltPickAndPlace-Franka-DataCollection-v0` | Same task with fixed + wrist cameras for VLA data collection |


## Task phases and success

The task uses a phased state machine (`task_phase` in `mdp/events.py`):

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Approach | Move EE toward the object on the conveyor |
| 1 | Grasp | EE close to object; close gripper and lift |
| 2 | Transport | Object above lift height (0.25 m); IK moves arm toward box target |
| 3 | Release | Arm at box; delayed gripper open to drop object |
| 4 | Done | Object detected inside box |

**Success (episode termination):**
Defined by **object in box** (env-local coords: x ∈ [−0.2, 0.2], y ∈ [−0.6, −0.3], z ∈ [0.0, 0.08]). `task_completed` uses `object_in_box()` — the same thresholds in events.py, rewards.py, and terminations.py.

## RL training and data collection

Conveyor Belt Pick-and-Place currently requires a trained RL policy for demonstration collection. Unlike the other five ReflexBench tasks, it does not support the built-in planning policy.

Train an skrl checkpoint from the repository root:

```bash
python scripts/skrl/train.py \
  --task ConveyorBeltPickAndPlace-Franka-v0 \
  --headless
```

Collect demonstrations using the resulting checkpoint:

```bash
python scripts/data_collection/collect_vla_data.py \
  --task ConveyorBeltPickAndPlace-Franka-DataCollection-v0 \
  --policy checkpoint \
  --checkpoint /path/to/conveyor_policy.pt \
  --control joint_pos \
  --num_envs 5 \
  --num_episodes 200
```

- **`--checkpoint`**: Must point to a trained skrl `.pt` checkpoint.
- **`--control`**: Must match the policy (`joint_pos`, `ik_abs`, `ik_rel`). Sets action and state format.
- **`--fps`**: Robot control frequency and dataset FPS (Hz). Default: task config (25 Hz).
- **`--policy planning`**: Not supported for this task.

Output: HDF5 under `scripts/data_collection/collected_data/` (or `--output`).

## Evaluation

Policy served by a remote server (see `scripts/evaluation/POLICY_SERVER.md`):

```bash
python scripts/evaluation/eval.py \
  --task ConveyorBeltPickAndPlace-Franka-DataCollection-v0 \
  --server_url http://localhost:8000 \
  --control joint_pos \
  --inference_mode async --infer_freq 10 \
  --num_episodes 50 --output results.json
```

- **`--control`**: Must match the policy (use `--state_format auto` so state follows control).
- **`--infer_freq`**: Inference frequency in Hz (e.g. 10 = one chunk every 100 ms); internally converted to physics sub-steps (see main README Time model).
- **`--ctrl_freq`**: Override robot control frequency (Hz).
