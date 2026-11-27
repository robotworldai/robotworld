# Ball Catching (Franka)

Catch a launched ball with the Franka end-effector acting as a virtual bucket. Supports joint position control and IK (absolute/relative) control.

## Registered tasks

| Task | Description |
|------|-------------|
| `BallCatching-Franka-v0` | Joint position control (7-DOF + gripper) |
| `BallCatching-Franka-Play-v0` | Same, play/eval variant (fewer envs, no obs noise) |
| `BallCatching-Franka-IK-Abs-v0` | IK absolute pose control |
| `BallCatching-Franka-IK-Rel-v0` | IK relative pose control |
| `BallCatching-Franka-DataCollection-v0` | Same task with fixed + wrist cameras for VLA data collection |

## Task phases and success

The task uses a phased state machine (`task_phase` in `mdp/events.py`):

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Wait | Ball is reset at the launcher and waiting for launch |
| 1 | Launch detected | Ball has been launched and motion tracking begins |
| 2 | Intercept | A ballistic intercept point has been predicted |
| 3 | Catch | EE is near the intercept point and waiting to contain the ball |
| 4 | Done | Ball stays in the catch zone for the hold threshold |

**Success (episode termination):**
Defined by **phase 4**. The ball must remain inside the virtual catch zone below the end-effector for consecutive steps.

## Data collection

From the repo root:

```bash
python scripts/data_collection/collect_vla_data.py \
  --task BallCatching-Franka-DataCollection-v0 \
  --policy checkpoint \
  --control joint_pos \
  --num_episodes 200
```

- **`--control`**: Must match the policy (`joint_pos`, `ik_abs`, `ik_rel`).
- **`--fps`**: Robot control frequency and dataset FPS (Hz). Default: task config (25 Hz).

## Evaluation

```bash
python scripts/evaluation/eval.py \
  --task BallCatching-Franka-DataCollection-v0 \
  --server_url http://localhost:8000 \
  --control joint_pos \
  --inference_mode async --infer_freq 10 \
  --num_episodes 50 --output results.json
```

- **`--control`**: Must match the policy.
- **`--infer_freq`**: Inference frequency in Hz (e.g. 10 = one chunk every 100 ms); internally converted to physics sub-steps.
