# Rotating Peg Insertion (Franka)

Insert a pre-grasped peg into a hole on a continuously rotating disc. The robot must synchronize with periodic alignment windows to perform timed insertion.

## Registered tasks

| Task | Description |
|------|-------------|
| `RotatingPegInsertion-Franka-v0` | Joint position control (7-DOF + gripper) |
| `RotatingPegInsertion-Franka-Play-v0` | Same, play/eval variant (fewer envs, no obs noise) |
| `RotatingPegInsertion-Franka-IK-Abs-v0` | IK absolute pose control |
| `RotatingPegInsertion-Franka-IK-Rel-v0` | IK relative pose control |
| `RotatingPegInsertion-Franka-DataCollection-v0` | Same task with fixed + wrist cameras for VLA data collection |

## Task phases

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Pre-align | Robot moves toward hover position above disc |
| 1 | Track disc | Monitoring disc rotation and predicting alignment windows |
| 2 | Wait for window | Alignment window predicted, waiting for lead time |
| 3 | Descend / Insert | Timed descent into hole during alignment window |
| 4 | Done | Peg held in hole for required duration |

## Scene layout

- Franka Panda at origin
- Rotating disc at (0.5, 0.0, 0.40) on a pedestal
- Disc: radius 0.14 m, 1 hole (diameter 0.10 m), angular velocity 20-60 deg/s
- Peg: radius 0.005 m, length 0.10 m, pre-grasped (follows EE)
- Green marker on disc surface indicates hole position

## Success condition

Peg tip enters hole within radius tolerance AND descends >= 0.02 m below disc surface AND holds for >= 5 control steps.
