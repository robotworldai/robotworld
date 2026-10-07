# wheeled_quadruped: English action contract

Example regenerated from public action metadata in `outputs/wheeled_quadruped/rear_wheel_upright_balance/run-smoke-native-one-per-bench-20260929-01-0001/artifacts/prompt.json`; not a new rollout.

ACTION CONTRACT — T09
Use exactly 4 finite scalars in action[0] through action[3], in the order below. All channels act simultaneously.
One control step = 0.02 s. Holding steps=10 lasts 0.2 simulated seconds; steps=50 lasts 1 s, unless terminated earlier.
apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.
Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.
Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.
Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.
coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.
For each channel u=action[i], target=offset+scale*u, followed by any listed native processed-target clip.
| i | Exact runtime joint | Unit | Input limits | offset | scale | Target at u=0 | Target at u=+0.1 | Target at u=+1 | Processed-target clip |
|---|---|---|---|---|---|---|---|---|---|
| 0 | robot1_front_left_thigh_joint | rad | [-inf, +inf] | 0 | 0.5 | 0 | 0.05 | 0.5 | none configured |
| 1 | robot1_front_right_thigh_joint | rad | [-inf, +inf] | 0 | 0.5 | 0 | 0.05 | 0.5 | none configured |
| 2 | robot1_rl_wheel_joint | rad/s | [-inf, +inf] | 0 | 5 | 0 | 0.5 | 5 | none configured |
| 3 | robot1_rr_wheel_joint | rad/s | [-inf, +inf] | 0 | 5 | 0 | 0.5 | 5 | none configured |
For a negative change du, the unclipped target changes by scale*du. A zero position input requests the listed offset, not the current measured joint angle. A zero-offset velocity input requests zero speed, not instantaneous braking.
To request an absolute joint target q with nonzero scale, use u=(q-offset)/scale within allowed bounds, not u=q unless offset=0 and scale=1. Example offset=0.2,scale=0.5,u=0.1 requests q=0.25rad on every repeated step, not measured_q+0.05 each time. To hold the measured pose, first undo any observation normalization/default subtraction; do not copy relative/scaled observations directly into the action vector.
For position channels, radians convert to degrees by *57.2957795: 0.025 rad is about 1.43 degrees; 0.05 rad about 2.86 degrees.
