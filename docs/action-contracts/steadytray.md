# steadytray: English action contract

Example regenerated from public action metadata in `outputs/steadytray/tray_balancing_walk/run-smoke-native-one-per-bench-20260929-01-0001/artifacts/prompt.json`; not a new rollout.

ACTION CONTRACT — T01
Use exactly 29 finite scalars in action[0] through action[28], in the order below. All channels act simultaneously.
One control step = 0.02 s. Holding steps=10 lasts 0.2 simulated seconds; steps=50 lasts 1 s, unless terminated earlier.
apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.
Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.
Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.
Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.
coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.
For each channel u=action[i], target=offset+scale*u, followed by any listed native processed-target clip.
| i | Exact runtime joint | Unit | Input limits | offset | scale | Target at u=0 | Target at u=+0.1 | Target at u=+1 | Processed-target clip |
|---|---|---|---|---|---|---|---|---|---|
| 0 | left_hip_pitch_joint | rad | [-inf, +inf] | -0.1 | 0.54754645 | -0.1 | -0.045245357 | 0.44754644 | [-100,100] rad |
| 1 | right_hip_pitch_joint | rad | [-inf, +inf] | -0.1 | 0.54754645 | -0.1 | -0.045245357 | 0.44754644 | [-100,100] rad |
| 2 | waist_yaw_joint | rad | [-inf, +inf] | 0 | 0.54754645 | 0 | 0.054754645 | 0.54754645 | [-100,100] rad |
| 3 | left_hip_roll_joint | rad | [-inf, +inf] | 0 | 0.35066146 | 0 | 0.035066146 | 0.35066146 | [-100,100] rad |
| 4 | right_hip_roll_joint | rad | [-inf, +inf] | 0 | 0.35066146 | 0 | 0.035066146 | 0.35066146 | [-100,100] rad |
| 5 | waist_roll_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 6 | left_hip_yaw_joint | rad | [-inf, +inf] | 0 | 0.54754645 | 0 | 0.054754645 | 0.54754645 | [-100,100] rad |
| 7 | right_hip_yaw_joint | rad | [-inf, +inf] | 0 | 0.54754645 | 0 | 0.054754645 | 0.54754645 | [-100,100] rad |
| 8 | waist_pitch_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 9 | left_knee_joint | rad | [-inf, +inf] | 0.30000001 | 0.35066146 | 0.30000001 | 0.33506616 | 0.65066147 | [-100,100] rad |
| 10 | right_knee_joint | rad | [-inf, +inf] | 0.30000001 | 0.35066146 | 0.30000001 | 0.33506616 | 0.65066147 | [-100,100] rad |
| 11 | left_shoulder_pitch_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 12 | right_shoulder_pitch_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 13 | left_ankle_pitch_joint | rad | [-inf, +inf] | -0.2 | 0.43857732 | -0.2 | -0.15614227 | 0.23857732 | [-100,100] rad |
| 14 | right_ankle_pitch_joint | rad | [-inf, +inf] | -0.2 | 0.43857732 | -0.2 | -0.15614227 | 0.23857732 | [-100,100] rad |
| 15 | left_shoulder_roll_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 16 | right_shoulder_roll_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 17 | left_ankle_roll_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 18 | right_ankle_roll_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 19 | left_shoulder_yaw_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 20 | right_shoulder_yaw_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 21 | left_elbow_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 22 | right_elbow_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 23 | left_wrist_roll_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 24 | right_wrist_roll_joint | rad | [-inf, +inf] | 0 | 0.43857732 | 0 | 0.043857732 | 0.43857732 | [-100,100] rad |
| 25 | left_wrist_pitch_joint | rad | [-inf, +inf] | 0 | 0.074500874 | 0 | 0.0074500874 | 0.074500874 | [-100,100] rad |
| 26 | right_wrist_pitch_joint | rad | [-inf, +inf] | 0 | 0.074500874 | 0 | 0.0074500874 | 0.074500874 | [-100,100] rad |
| 27 | left_wrist_yaw_joint | rad | [-inf, +inf] | 0 | 0.074500874 | 0 | 0.0074500874 | 0.074500874 | [-100,100] rad |
| 28 | right_wrist_yaw_joint | rad | [-inf, +inf] | 0 | 0.074500874 | 0 | 0.0074500874 | 0.074500874 | [-100,100] rad |
For a negative change du, the unclipped target changes by scale*du. A zero position input requests the listed offset, not the current measured joint angle. A zero-offset velocity input requests zero speed, not instantaneous braking.
To request an absolute joint target q with nonzero scale, use u=(q-offset)/scale within allowed bounds, not u=q unless offset=0 and scale=1. Example offset=0.2,scale=0.5,u=0.1 requests q=0.25rad on every repeated step, not measured_q+0.05 each time. To hold the measured pose, first undo any observation normalization/default subtraction; do not copy relative/scaled observations directly into the action vector.
For position channels, radians convert to degrees by *57.2957795: 0.025 rad is about 1.43 degrees; 0.05 rad about 2.86 degrees.
