# Built-in Robots

This is the canonical list of robot embodiments that ship with RoboLab. For how to *use* a robot
(registration wiring, defining your own robot, contact grippers, wrist cameras), see
[`docs/robots.md`](../../docs/robots.md).

| | Robot | Embodiment | Action spaces | Cameras |
|---|-------|------------|---------------|---------|
| <img src="../../docs/images/robots/droid.png" width="480"> | **DROID**<br>(Franka + Robotiq 2F-85)<br>`droid.py` | `single-arm` `fixed-base` `parallel-jaw` | joint position, absolute EE IK, relative EE IK | wrist |
| <img src="../../docs/images/robots/franka.png" width="480"> | **Franka Panda**<br>`franka.py`, `franka_high_pd.py` | `single-arm` `fixed-base` `parallel-jaw` | joint position, absolute EE IK, relative EE IK | — |
| <img src="../../docs/images/robots/kinova_gen3.png" width="480"> | **Kinova Gen3**<br>(Gen3 7-DoF + Robotiq 2F-85)<br>`kinova_gen3.py` | `single-arm` `fixed-base` `parallel-jaw` | joint position | wrist |

Gripper convention for all binary gripper actions: a scalar per gripper, `> 0.5` closes, `≤ 0.5` opens.
Quaternions are `(w, x, y, z)`; absolute IK targets are expressed in the robot root frame, translations in meters.

---

## DROID (Franka + Robotiq 2F-85)

`tags: single-arm · fixed-base · parallel-jaw · wrist-cam · gravity-disabled · high-PD · benchmark-default`

The default benchmark embodiment: a Franka Panda arm with a Robotiq 2F-85 gripper, matching the
[DROID](https://droid-dataset.github.io/) platform. High PD gains (400/80) with gravity disabled on the
arm, plus a 720p wrist camera whose intrinsics are calibrated to match pi05 / DreamZero training data.

| Action config | Layout | Dim |
|---------------|--------|-----|
| `DroidJointPositionActionCfg` | 7 arm joint targets + binary gripper | 8 |
| `DroidIKActionCfg` | absolute EE pose `(x, y, z, qw, qx, qy, qz)` + binary gripper | 8 |
| `DroidRelIKActionCfg` | relative EE pose `(dx, dy, dz, droll, dpitch, dyaw)` + binary gripper | 7 |

- **Config classes:** `DroidCfg` (robot + wrist camera + EE frame transformers)
- **Proprioception:** `ProprioceptionObservationCfg` — arm joint positions, gripper open fraction,
  EE pose (both the gripper mount flange `ee_*` and the rotated control frame `eef_*`)
- **Contact gripper:** `{"gripper": ...left_inner_finger}`
- **Registrations:** `robolab/registrations/droid/` (jointpos, abs-IK, rel-IK, lighting/background variations)

```python
from robolab.robots.droid import DroidCfg, DroidJointPositionActionCfg, contact_gripper
```

## Franka Panda

`tags: single-arm · fixed-base · parallel-jaw`

A stock Franka Panda with its factory finger gripper. Two articulation variants share the same action
configs: `franka.py` with standard PD gains (80/4), and `franka_high_pd.py` with high gains (400/80)
and gravity disabled (better target tracking for policy control).

| Action config | Layout | Dim |
|---------------|--------|-----|
| `FrankaJointPositionActionCfg` | 7 arm joint targets + binary gripper | 8 |
| `FrankaIKActionCfg` | absolute EE pose `(x, y, z, qw, qx, qy, qz)` + binary gripper | 8 |
| `FrankaRelIKActionCfg` | relative EE pose `(dx, dy, dz, droll, dpitch, dyaw)` + binary gripper | 7 |

- **Config classes:** `FrankaCfg` (one per variant file; action configs in `franka_definitions.py`)
- **Proprioception:** EE frame pose and finger joint positions (`franka_definitions.py`)
- **Contact gripper:** `{"gripper": ...panda_leftfinger}`

```python
from robolab.robots.franka import FrankaCfg                      # standard PD
from robolab.robots.franka_high_pd import FrankaCfg              # high PD, gravity disabled
from robolab.robots.franka_definitions import FrankaJointPositionActionCfg, contact_gripper
```

## Kinova Gen3 (Gen3 7-DoF + Robotiq 2F-85)

`tags: single-arm · fixed-base · parallel-jaw · wrist-cam · gravity-disabled`

A Kinova Gen3 7-DoF arm with a Robotiq 2F-85 gripper, welded to the world at the base
(`fix_root_link=True`, gravity disabled on the arm). The USD is vendored under
`assets/robots/kinova_gen3_robotiq_2f85/` and derived from Kinova `ros2_kortex` and PickNik
`ros2_robotiq_gripper` at pinned revisions — see the README and license files in that folder.

The gripper's six joints are driven together from one binary action rather than through PhysX mimic
constraints; the signed per-joint targets live in `GRIPPER_JOINT_COMMANDS`.

| Action config | Layout | Dim |
|---------------|--------|-----|
| `KinovaJointPositionActionCfg` | 7 arm joint targets + binary gripper | 8 |

- **Config classes:** `KinovaGen3Cfg` (robot + wrist camera + EE frame transformer),
  `KinovaWristCameraCfg` (exposes the wrist camera to image observations)
- **Proprioception:** `KinovaProprioceptionObservationCfg` — arm joint positions, gripper open
  fraction, EE pose at `robotiq_85_base_link`
- **Contact gripper:** `{"gripper": ...robotiq_85_.*_finger_tip_link}`
- **Registrations:** `robolab/registrations/kinova/` (jointpos)

```python
from robolab.robots.kinova_gen3 import (
    KinovaGen3Cfg,
    KinovaJointPositionActionCfg,
    KinovaProprioceptionObservationCfg,
    contact_gripper,
)
```

Actuator gains are simulation defaults, not measured from hardware — this is a functional
simulation model rather than a calibrated digital twin.

---

Also in this folder: `delta_actions.py`, a helper that converts a target EE pose into a relative
(delta) pose action — used by trajectory replay, not an action space itself.

The robot stills above are rendered in an empty scene at each robot's reset posture.
