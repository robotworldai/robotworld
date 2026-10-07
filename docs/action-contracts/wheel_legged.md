# wheel_legged: English action contract

Example regenerated from public action metadata in `outputs/wheel_legged/wheel_legged_upright_recovery/run-smoke-native-one-per-bench-20260929-01-0001/artifacts/prompt.json`; not a new rollout.

ACTION CONTRACT — T07
Use exactly 6 finite scalars in action[0] through action[5], in the order below. All channels act simultaneously.
One control step = 0.01 s. Holding steps=10 lasts 0.1 simulated seconds; steps=50 lasts 0.5 s, unless terminated earlier.
apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.
Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.
Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.
Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.
coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.
All six inputs are in [-1,1]. Positive input increases the corresponding virtual target; these are NOT six raw joint angles.
| Index | Native control | Target formula | u=-1 / 0 / +1 |
|---|---|---|---|
| 0 | left_virtual_leg_angle | angle=0.35*u rad | -0.35 / 0 / +0.35 rad |
| 1 | left_virtual_leg_length | length=clip(0.237+0.06*u,0.18,0.30) m | 0.18 / 0.237 / 0.297 m |
| 2 | left_wheel_velocity | wheel_velocity=24*u rad/s | -24 / 0 / +24 rad/s |
| 3 | right_virtual_leg_angle | angle=0.35*u rad | -0.35 / 0 / +0.35 rad |
| 4 | right_virtual_leg_length | length=clip(0.237+0.06*u,0.18,0.30) m | 0.18 / 0.237 / 0.297 m |
| 5 | right_wheel_velocity | wheel_velocity=24*u rad/s | -24 / 0 / +24 rad/s |
A change of +0.1 changes the unclipped angle target by +0.035 rad, leg length by +0.006 m, or wheel speed by +2.4 rad/s, respectively.
Virtual leg angle is measured from the native vertical reference, not world yaw: theta0=atan2(end_y,end_x)-pi/2 in the native leg plane. Increasing length requests extension along that virtual leg, not a guaranteed base-height change. Zero requests virtual angle 0, length 0.237 m and wheel speed 0. It does not supply automatic balance. Original VMC converts these targets into motor torques.
