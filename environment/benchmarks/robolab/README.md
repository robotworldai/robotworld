# RoboLab agent adapter

Source restoration, containers, assets, and task mappings are documented in [the RoboLab integration](../../../third_party/benchmarks/robolab/README.md). This directory implements the model interface and action conversion.

- `control.py` exposes the three native DROID control modes through `move_joints` or `move_eef`, plus `set_gripper` and `move_robot`. Every mode supports simultaneous arm and gripper control. The robot has a fixed base.
- `lifecycle.py` flushes native records and closes HDF5 files when Python execution is interrupted. Interrupted episodes are recorded as incomplete without inventing a terminal task state.
- `policy.py` implements the official `InferenceClient` interface through the locally built Codex relay. Local computation is permitted; hidden environment state is not exposed. The configured observation history uses interval 2, and both complete and image-free events are saved.
- `../../integrations/robolab_eval.py` registers native profiles and invokes the official evaluator, retaining upstream success criteria.

## Control conventions

Poses use the robot root frame and the `base_link` flange pose, with WXYZ quaternions.

| Mode | Command components |
| --- | --- |
| `joint_position` | Seven joint positions in radians and one gripper command. |
| `absolute_ik` | Three position values, four quaternion values, and one gripper command. |
| `relative_ik` | Six pose increments and one gripper command, with upstream scale 0.5. |

Control runs at 15 Hz. A call executes 1–30 steps; gripper 0 means open and 1 means closed. These conventions differ from RoboCasa's normalised increments and BEHAVIOR's XYZW quaternions.

Observation history does not alter video sampling. The policy exposes no ground-truth state, external solver, reset, or score-query tool.
