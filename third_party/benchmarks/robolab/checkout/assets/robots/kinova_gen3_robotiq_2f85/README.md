# Kinova Gen3 7-DoF + Robotiq 2F-85

This directory contains a self-contained, fixed-base USD asset for a Kinova
Gen3 7-DoF arm with a Robotiq 2F-85 gripper.

The arm uses Kinova's official vision-equipped bracelet geometry. The
policy-facing `wrist_cam` uses the corresponding Kortex simulation-camera link
transform and the packaged 1280x720 color-camera calibration.

## Provenance

The robot description and geometry were derived from:

- [Kinova `ros2_kortex`](https://github.com/Kinovarobotics/ros2_kortex) at
  revision `78e6dee598865f0f9b352f1a57b929dae14ce61a`.
- [PickNik `ros2_robotiq_gripper`](https://github.com/PickNikRobotics/ros2_robotiq_gripper)
  at revision `12e623212e6891a5fcc9af94d67b07e640916394`.

The source descriptions are BSD-licensed. See `LICENSE-KINOVA` and
`LICENSE-ROBOTIQ` in this directory.

## Simulation configuration

The USD was converted with Isaac Sim 5.0 and Isaac Lab 2.2. It contains seven
arm joints and six independently actuated gripper joints. RoboLab maps one
binary gripper action to the six signed joint targets in
`robolab/robots/kinova_gen3.py`; it does not rely on PhysX mimic constraints.

The actuator gains are initial simulation defaults, not measured gains from a
physical robot. The asset is a functional simulation model, not yet a
calibrated digital twin.

## RoboLab modifications

Changes authored by RoboLab on top of the converted asset, in
`configuration/kinova_gen3_7dof_robotiq_2f85_physics.usd`:

- `GripperPhysicsMaterial` (static 2.0, dynamic 2.0, restitution 0.0), bound to
  the four `robotiq_85_*_finger*_link` prims. The conversion shipped no physics
  material, so the gripper pads fell back to the PhysX default of 0.5.
