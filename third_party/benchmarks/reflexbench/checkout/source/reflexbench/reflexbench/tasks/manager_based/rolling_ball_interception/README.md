# Rolling Ball Interception (Franka)

Intercept a ball rolling down a ramp using a virtual catcher attached to the Franka end-effector. Supports joint position control and IK (absolute/relative) control.

## Registered tasks

| Task | Description |
|------|-------------|
| `RollingBallInterception-Franka-v0` | Joint position control (7-DOF + gripper) |
| `RollingBallInterception-Franka-Play-v0` | Same, play/eval variant (fewer envs, no obs noise) |
| `RollingBallInterception-Franka-IK-Abs-v0` | IK absolute pose control |
| `RollingBallInterception-Franka-IK-Rel-v0` | IK relative pose control |
| `RollingBallInterception-Franka-DataCollection-v0` | Same task with fixed + wrist cameras for VLA data collection |

## Task phases and success

The task uses a phased state machine (`task_phase` in `mdp/events.py`):

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Wait | Ball on ramp, gate closed, random delay counting down |
| 1 | Rolling | Gate opened, ball rolling down the ramp |
| 2 | Intercept | (placeholder) Move catcher to ramp exit point |
| 3 | Capture | (placeholder) Hold catcher at exit, absorb ball |
| 4 | Done | Ball retained in catcher for hold threshold |

**Success (episode termination):**
Defined by **phase 4**. The ball must remain inside the virtual catch zone below the end-effector for consecutive steps.

## Scene layout

- **Ramp**: 0.30 m long, 0.10 m wide, tilted 20°, with guide rails on both sides
- **Ball**: Sphere (r=0.038 m, m=0.10 kg) placed at the top of the ramp
- **Catcher**: Cylinder (r=0.06 m, h=0.10 m) teleported to follow end-effector
- **Gate**: Ball is held stationary for a random delay (0.3–0.8 s), then released to roll under gravity
