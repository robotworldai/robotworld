# Ball Throwing (Franka)

Throw a small ball, **already grasped by the gripper at episode start**, into a
target box placed directly in front of the robot. The ball is initialised
between the two fingers; the policy only needs to swing the arm and open the
gripper at the right moment.

## Registered tasks

| Task | Description |
|------|-------------|
| `BallThrowing-Franka-v0` | Joint position control (7-DOF + gripper) |
| `BallThrowing-Franka-Play-v0` | Same, play/eval variant (50 envs, no obs noise) |
| `BallThrowing-Franka-IK-Abs-v0` | IK absolute pose control |
| `BallThrowing-Franka-IK-Rel-v0` | IK relative pose control |
| `BallThrowing-Franka-DataCollection-v0` | Same task with fixed + wrist cameras |

## Scene

- **Robot**: Franka Panda (`FRANKA_PANDA_HIGH_PD_CFG`), default arm joint
  pose. Finger joints are initialised closed (≈0.026 m half-aperture) so the
  gripper is already clamped on the ball.
- **Ball**: blue solid sphere, radius 0.025 m, mass 0.04 kg. Spawned at the
  EE frame target minus 2.2 cm along its -Z (between the two fingers) by
  `reset_ball_in_gripper`. Gravity is **disabled at spawn** so the ball
  cannot drop before the gripper command stabilises; an interval event
  (`apply_post_release_gravity`) re-applies `mass * g` as an external force
  once the task transitions to phase 1.
- **Box**: Isaac KLT bin (`Props/KLT_Bin/small_KLT_visual_collision.usd`) at
  `[0.8, 0.0, 0.075]` (0.8 m in front of the robot along +X, well past the
  Franka workspace edge -- only reachable via a real throw). The bin is
  oriented so its long edge (~0.30 m) lies along +X (the throw direction),
  giving a more forgiving catch window for X-axis landing error. A startup
  event applies convex decomposition so the bin's opening is not sealed by
  a single convex hull.

## Task phases and success

Phase machine (`task_phase` in `mdp/events.py::check_phase_transitions`):

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Hold | Ball in gripper, robot is swinging the arm |
| 1 | Released | Gripper opened (finger width ≥ 0.06 m) or ball drifted > 0.08 m from EE |
| 2 | Done | Ball inside target box volume (success) |

**Success:** `task_phase == 2`. Object-in-box uses env-local bounds
`|x − cx| ≤ 0.13, |y − cy| ≤ 0.09, z ∈ [0.02, 0.18]` (rectangular catch
volume aligned with the bin's long edge) where `(cx, cy, cz)` is the
randomised env-local box centre (jittered ±5 cm in XY at every reset).
See `object_in_box()` in `mdp/terminations.py`.

**Failure:** time-out (`episode_length_s = 4 s`) or ball drops below
z = -0.05 m (missed the box and rolled off the floor).

## Quick smoke test

```bash
python3 -c "
import gymnasium as gym
import reflexbench.tasks.manager_based.ball_throwing  # registers Gym IDs
print(gym.spec('BallThrowing-Franka-Play-v0'))
"
```

## Data collection

```bash
python3 scripts/data_collection/collect_vla_data.py \
  --task BallThrowing-Franka-DataCollection-v0 \
  --policy planning \
  --control joint_pos \
  --num_episodes 200
```

## Evaluation

Use the matching task ID and control representation for evaluation.

```bash
python3 scripts/evaluation/eval.py \
  --task BallThrowing-Franka-DataCollection-v0 \
  --server_url http://localhost:8000 \
  --control joint_pos \
  --inference_mode async --ctrl_freq 10 \
  --num_episodes 50 --output results.json
```
