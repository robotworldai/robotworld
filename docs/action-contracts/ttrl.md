# ttrl: English action contract

Example regenerated from public action metadata in `outputs/ttrl/humanoid_table_tennis_return/run-smoke-native-one-per-bench-20260929-01-0001/artifacts/prompt.json`; not a new rollout.

ACTION CONTRACT — T02
Use exactly 21 finite scalars in action[0] through action[20], in the order below. All channels act simultaneously.
One control step = 0.02 s. Holding steps=10 lasts 0.2 simulated seconds; steps=50 lasts 1 s, unless terminated earlier.
apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.
Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.
Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.
Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.
coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.
T02 clips raw u to [-100,100] BEFORE scaling; valid tool inputs must already lie within these bounds. This wide input range is not a recommended motion.
For each channel u=action[i], target=offset+scale*u, followed by any listed native processed-target clip.
| i | Exact runtime joint | Unit | Input limits | offset | scale | Target at u=0 | Target at u=+0.1 | Target at u=+1 | Processed-target clip |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Left_Shoulder_Pitch | rad | [-100, 100] | 0.2 | 0.25 | 0.2 | 0.225 | 0.45 | none configured |
| 1 | Left_Shoulder_Roll | rad | [-100, 100] | -1.35 | 0.25 | -1.35 | -1.325 | -1.1 | none configured |
| 2 | Left_Elbow_Pitch | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 3 | Left_Elbow_Yaw | rad | [-100, 100] | -0.5 | 0.25 | -0.5 | -0.475 | -0.25 | none configured |
| 4 | Right_Shoulder_Pitch | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 5 | Right_Shoulder_Roll | rad | [-100, 100] | -0.1 | 0.25 | -0.1 | -0.075000001 | 0.15 | none configured |
| 6 | Right_Elbow_Pitch | rad | [-100, 100] | 0.2 | 0.25 | 0.2 | 0.225 | 0.45 | none configured |
| 7 | Right_Elbow_Yaw | rad | [-100, 100] | 0.5 | 0.25 | 0.5 | 0.525 | 0.75 | none configured |
| 8 | Waist | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 9 | Left_Hip_Pitch | rad | [-100, 100] | -0.2 | 0.25 | -0.2 | -0.175 | 0.049999997 | none configured |
| 10 | Left_Hip_Roll | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 11 | Left_Hip_Yaw | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 12 | Left_Knee_Pitch | rad | [-100, 100] | 0.41999999 | 0.25 | 0.41999999 | 0.44499999 | 0.66999999 | none configured |
| 13 | Left_Ankle_Pitch | rad | [-100, 100] | -0.23 | 0.25 | -0.23 | -0.205 | 0.019999996 | none configured |
| 14 | Left_Ankle_Roll | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 15 | Right_Hip_Pitch | rad | [-100, 100] | -0.2 | 0.25 | -0.2 | -0.175 | 0.049999997 | none configured |
| 16 | Right_Hip_Roll | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 17 | Right_Hip_Yaw | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
| 18 | Right_Knee_Pitch | rad | [-100, 100] | 0.41999999 | 0.25 | 0.41999999 | 0.44499999 | 0.66999999 | none configured |
| 19 | Right_Ankle_Pitch | rad | [-100, 100] | -0.23 | 0.25 | -0.23 | -0.205 | 0.019999996 | none configured |
| 20 | Right_Ankle_Roll | rad | [-100, 100] | 0 | 0.25 | 0 | 0.025 | 0.25 | none configured |
For a negative change du, the unclipped target changes by scale*du. A zero position input requests the listed offset, not the current measured joint angle. A zero-offset velocity input requests zero speed, not instantaneous braking.
To request an absolute joint target q with nonzero scale, use u=(q-offset)/scale within allowed bounds, not u=q unless offset=0 and scale=1. Example offset=0.2,scale=0.5,u=0.1 requests q=0.25rad on every repeated step, not measured_q+0.05 each time. To hold the measured pose, first undo any observation normalization/default subtraction; do not copy relative/scaled observations directly into the action vector.
For position channels, radians convert to degrees by *57.2957795: 0.025 rad is about 1.43 degrees; 0.05 rad about 2.86 degrees.
