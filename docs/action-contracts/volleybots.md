# volleybots: English action contract

Example regenerated from public action metadata in `outputs/volleybots/drone_volleyball_solo_juggle/run-smoke-native-one-per-bench-20260929-01-0001/artifacts/prompt.json`; not a new rollout.

ACTION CONTRACT — T05-single
Use exactly 4 finite scalars in action[0] through action[3], in the order below. All channels act simultaneously.
One control step = 0.02 s. Holding steps=10 lasts 0.2 simulated seconds; steps=50 lasts 1 s, unless terminated earlier.
apply_action repeats the SAME command for its steps; it does not multiply its amplitude or interpolate a trajectory. A repeated position target stays absolute relative to its stated reference; an increment interface explicitly accumulates.
Magnitude examples below are unit conversions, NOT recommended motions, safe ranges or measured gains. Input 1 is not universally maximum strength. Input limits, processed-target clipping and physical actuator limits are different.
Positive joint commands increase that named joint coordinate along its authored axis. Left/right mirrored joints can have different geometric effects; do not assume positive means forward/up for every joint.
Commanded targets are not guaranteed achieved positions/velocities or direct torque/force commands. Native servos, inertia, contacts, saturation and configured delays remain active. Use fresh allowed observations to assess the actual response.
coding_control uses this SAME action contract, with fresh observation on every control tick; max_steps is a cap, not an amplitude.
| Index | Control | Input limits |
|---|---|---|
| 0 | rotor_0 | [-1, 1] |
| 1 | rotor_1 | [-1, 1] |
| 2 | rotor_2 | [-1, 1] |
| 3 | rotor_3 | [-1, 1] |
u=-1 requests zero steady-state rotor thrust; u=0 requests 50% of that rotor's maximum steady-state thrust; u=+1 requests 100%. u=-0.5/+0.5 requests 25%/75%. Zero is NOT hover or stop.
Original motor mapping: desired internal throttle s*=sqrt(clip((u+1)/2,0,1)); actual thrust=KF*s^2. Motor lag means a command is not achieved instantly.
The normalized rotor-throttle observation is v=2*s-1; its current thrust fraction is ((v+1)/2)^2, NOT (v+1)/2.
Each rotor thrust acts along its own local +Z. Tilting the aircraft changes its world force direction. Differential rotor inputs create moments; collective input alone cannot correct tilt. No hover/balance controller is supplied.
