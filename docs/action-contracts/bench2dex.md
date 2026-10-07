# bench2dex: English action contract

Example regenerated from public action metadata in `outputs/bench2dex/41/run-smoke-native-one-per-bench-20260929-01-0001/artifacts/prompt.json`; not a new rollout.

ACTION CONTRACT — 41
Use exactly 52 finite scalars in action[0] through action[51], in the order below. All channels act simultaneously.
One control step = 0.05 s. Holding steps=10 lasts 0.5 simulated seconds; steps=50 lasts 2.5 s, unless terminated earlier.
apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.
Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.
Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.
Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.
coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.
For each channel u=action[i], target=offset+scale*u, followed by any listed native processed-target clip.
| i | Exact runtime joint | Unit | Input limits | offset | scale | Target at u=0 | Target at u=+0.1 | Target at u=+1 | Processed-target clip |
|---|---|---|---|---|---|---|---|---|---|
| 0 | shoulder_pan_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 1 | shoulder_lift_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 2 | L_arm_shoulder_pan_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 3 | elbow_joint | rad | [-3.1415927, 3.1415927] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 4 | L_arm_shoulder_lift_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 5 | wrist_1_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 6 | L_arm_elbow_joint | rad | [-3.1415927, 3.1415927] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 7 | wrist_2_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 8 | L_arm_wrist_1_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 9 | wrist_3_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 10 | L_arm_wrist_2_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 11 | L_arm_wrist_3_joint | rad | [-6.2831855, 6.2831855] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 12 | right_finger1_joint1 | rad | [0.036800027, 1.6124998] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 13 | right_finger2_joint1 | rad | [-0.16110003, 1.5587999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 14 | right_finger3_joint1 | rad | [-0.17190003, 1.5496] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 15 | right_finger4_joint1 | rad | [-0.16009998, 1.5533999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 16 | right_finger5_joint1 | rad | [-0.1674, 1.5538999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 17 | right_finger1_joint2 | rad | [-0.15759999, 0.93119997] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 18 | right_finger2_joint2 | rad | [-0.40439999, 0.30540001] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 19 | right_finger3_joint2 | rad | [-0.40139997, 0.29960001] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 20 | right_finger4_joint2 | rad | [-0.41339996, 0.31609997] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 21 | right_finger5_joint2 | rad | [-0.42029998, 0.29309997] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 22 | left_finger1_joint1 | rad | [0.058300018, 1.5940999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 23 | left_finger2_joint1 | rad | [-0.15600002, 1.5618999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 24 | left_finger3_joint1 | rad | [-0.15690005, 1.5535998] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 25 | left_finger4_joint1 | rad | [-0.15069997, 1.5635998] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 26 | left_finger5_joint1 | rad | [-0.15779996, 1.5631999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 27 | right_finger1_joint3 | rad | [-0.46380001, 1.5607002] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 28 | right_finger2_joint3 | rad | [-0.47140002, 1.5504] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 29 | right_finger3_joint3 | rad | [-0.46319997, 1.5612999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 30 | right_finger4_joint3 | rad | [-0.47819996, 1.5448] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 31 | right_finger5_joint3 | rad | [-0.48039997, 1.5419999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 32 | left_finger1_joint2 | rad | [-0.11989999, 0.93349993] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 33 | left_finger2_joint2 | rad | [-0.40949994, 0.29210001] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 34 | left_finger3_joint2 | rad | [-0.39769998, 0.31149998] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 35 | left_finger4_joint2 | rad | [-0.41009998, 0.30140001] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 36 | left_finger5_joint2 | rad | [-0.41029999, 0.29489997] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 37 | right_finger1_joint4 | rad | [-0.48290002, 1.5450999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 38 | right_finger2_joint4 | rad | [-0.46439993, 1.5752999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 39 | right_finger3_joint4 | rad | [-0.46969998, 1.5701998] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 40 | right_finger4_joint4 | rad | [-0.48249996, 1.5549999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 41 | right_finger5_joint4 | rad | [-0.47049999, 1.5709] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 42 | left_finger1_joint3 | rad | [-0.46459997, 1.5638999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 43 | left_finger2_joint3 | rad | [-0.48390001, 1.5465999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 44 | left_finger3_joint3 | rad | [-0.4846999, 1.5410998] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 45 | left_finger4_joint3 | rad | [-0.47469997, 1.5525999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 46 | left_finger5_joint3 | rad | [-0.47329992, 1.5559998] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 47 | left_finger1_joint4 | rad | [-0.45700002, 1.5683999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 48 | left_finger2_joint4 | rad | [-0.47229999, 1.5753999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 49 | left_finger3_joint4 | rad | [-0.46709991, 1.5788] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 50 | left_finger4_joint4 | rad | [-0.47290003, 1.5717999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
| 51 | left_finger5_joint4 | rad | [-0.46619999, 1.5759999] | 0 | 1 | 0 | 0.1 | 1 | none configured |
For a negative change du, the unclipped target changes by scale*du. A zero position input requests the listed offset, not the current measured joint angle. A zero-offset velocity input requests zero speed, not instantaneous braking.
To request an absolute joint target q with nonzero scale, use u=(q-offset)/scale within allowed bounds, not u=q unless offset=0 and scale=1. Example offset=0.2,scale=0.5,u=0.1 requests q=0.25rad on every repeated step, not measured_q+0.05 each time. To hold the measured pose, first undo any observation normalization/default subtraction; do not copy relative/scaled observations directly into the action vector.
For position channels, radians convert to degrees by *57.2957795: 0.025 rad is about 1.43 degrees; 0.05 rad about 2.86 degrees.
