# Whack-a-Mole (Franka)

Reactive striking task: moles pop up on a board on a fixed schedule; the robot must press the active mole with a **closed** gripper within each time window. Supports joint position control and IK (absolute/relative) control.

## Registered tasks

| Task | Description |
|------|-------------|
| `WhackAMole-Franka-v0` | Joint position control (7-DOF + gripper) |
| `WhackAMole-Franka-Play-v0` | Same, play/eval variant (fewer envs, no obs noise) |
| `WhackAMole-Franka-IK-Abs-v0` | IK absolute pose control |
| `WhackAMole-Franka-IK-Rel-v0` | IK relative pose control |
| `WhackAMole-Franka-DataCollection-v0` | Same task with fixed + wrist cameras for VLA data collection |

## Mechanics and success

- **Board**: Five moles in a fixed XY layout (`mdp/events.py`). Each episode samples a schedule of **8** popup windows; each window activates one mole for a duration (defaults in env cfg; eval profile `whack_a_mole` can override timing via `scripts/evaluation/task_profiles.py`).
- **Press detection**: End-effector must be within XY/Z thresholds over the active mole, gripper sufficiently closed, and hold for a short dwell (`PRESS_DWELL_STEPS` in `mdp/events.py`). Valid strikes increment `valid_hits`.
- **Episode termination**: Internal `task_phase` moves from **0** (windows in progress) to **4** after all windows have been processed (hit or missed). Training rewards use hit rate; unified eval profile success is **`valid_hits > 0`** (see `check_success` in `task_profiles.py`).

Diagnostics: run `scripts/evaluation/eval.py` with `--debug_whack` for per-step press-detection prints.

## Data collection

From the repo root:

```bash
python scripts/data_collection/collect_vla_data.py \
  --task WhackAMole-Franka-DataCollection-v0 \
  --policy checkpoint \
  --control joint_pos \
  --num_envs 1 \
  --num_episodes 100
```

- **`--control`**: Must match the policy (`joint_pos`, `ik_abs`, `ik_rel`).
- **`--fps`**: Robot control frequency and dataset FPS (Hz). Default follows task config.

## Evaluation

```bash
python scripts/evaluation/eval.py \
  --task WhackAMole-Franka-DataCollection-v0 \
  --task_profile whack_a_mole \
  --server_url http://localhost:8000 \
  --control joint_pos \
  --inference_mode async --infer_freq 10 \
  --num_episodes 50 --output results.json
```

- **`--task_profile whack_a_mole`**: Applies eval-specific popup timing and success logic when needed.
- **`--infer_freq`**: Inference frequency in Hz; internally converted to physics sub-steps.
