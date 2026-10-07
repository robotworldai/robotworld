# reflexbench: English action contract

Example regenerated from public action metadata in `outputs/reflexbench/cup_ball_catching/run-smoke-native-one-per-bench-20260929-01-0001/artifacts/prompt.json`; not a new rollout.

ACTION CONTRACT — T03
Use exactly 8 finite scalars in action[0] through action[7], in the order below. All channels act simultaneously.
One control step = 0.04 s. Holding steps=10 lasts 0.4 simulated seconds; steps=50 lasts 2 s, unless terminated earlier.
apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.
Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.
Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.
Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.
coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.
Robot-root coordinates, metres; quaternion order WXYZ. The seven pose entries form ONE absolute EEF pose, not seven joints or increments.
| Index | Meaning | Scale / zero / sign |
|---|---|---|
| 0,1,2 | x,y,z | Absolute robot-root position; +0.01 changes a target coordinate by +1 cm. Repeating it does not add another centimetre. |
| 3,4,5,6 | qw,qx,qy,qz | Unit quaternion; identity is [1,0,0,0]. [0,0,0,0] is invalid. Quaternion magnitude is not rotation strength. |
| 7 | gripper | >0 opens to 0.015 m per finger; <=0 closes to 0. No proportional strength or speed is encoded here. |
For example a +10 degree root-Z rotation has quaternion [0.9961947,0,0,0.08715574]; this is an orientation example, not a command to execute.
Do not use an all-zero action to hold still. Preserve a valid desired pose and an explicit gripper choice. Native damped IK resolves arm joints.
