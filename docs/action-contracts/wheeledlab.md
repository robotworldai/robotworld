# wheeledlab: English action contract

Illustrative 50Hz profiles. Actual prompts substitute the task control timestep.

ACTION CONTRACT — WheeledLab
action[0]=normalized speed; action[1]=normalized central steering; both finite in[-1,1].
Speed target=3*u m/s: u=0/0.1/0.5/1 requests0/0.3/1.5/3m/s. Any negative u is clamped to zero target speed; this task cannot reverse.
Steering parameter=0.488*u radians: u=0/0.1/-0.1/1 gives0/0.0488/-0.0488/0.488rad before the original RC-car tan conversion: both front steering-joint targets=tan(0.488*u) rad, e.g. u=0.1 gives about0.048839rad and u=1 about0.53082rad. Zero requests straight steering. In the published upright MuSHR/F1Tenth mounting, positive steering rotates about vehicle +Z (left turn when driving forward); negative steers right. Reversing reverses the nominal yaw response for the same steering input; tire slip can change the achieved yaw. Vehicle +X is forward, +Y left, +Z up, not camera image axes.
Example action=[0.1,0] requests0.3m/s wheel-ground speed and straight steering; at dt=0.02 and steps=10 the ideal no-slip travel is6cm, not a guaranteed displacement. With wheel radius0.05m, the rear-wheel target is6rad/s in MuSHR; F1Tenth distributes wheel speeds through its native geometry. Both entries are targets, not acceleration/torque percentages or guaranteed motion. A zero speed target does not teleport velocity to zero. Tire slip and native drives remain active.
One control step=0.02s; steps=10 holds ONE command for0.2s, not ten increasing commands. coding_control, if enabled, may update both entries every tick.
These numerical examples explain units and gain, not safe speeds or a supplied driving controller.

ACTION CONTRACT — WheeledLab
action[0]=normalized speed; action[1]=normalized central steering; both finite in[-1,1].
Speed target=3*u m/s: u=0/0.1/0.5/1 requests0/0.3/1.5/3m/s. u=-1/-0.1 requests -3/-0.3 m/s (reverse).
Steering parameter=0.488*u radians: u=0/0.1/-0.1/1 gives0/0.0488/-0.0488/0.488rad before the original RC-car tan conversion: both front steering-joint targets=tan(0.488*u) rad, e.g. u=0.1 gives about0.048839rad and u=1 about0.53082rad. Zero requests straight steering. In the published upright MuSHR/F1Tenth mounting, positive steering rotates about vehicle +Z (left turn when driving forward); negative steers right. Reversing reverses the nominal yaw response for the same steering input; tire slip can change the achieved yaw. Vehicle +X is forward, +Y left, +Z up, not camera image axes.
Example action=[0.1,0] requests0.3m/s wheel-ground speed and straight steering; at dt=0.02 and steps=10 the ideal no-slip travel is6cm, not a guaranteed displacement. With wheel radius0.05m, the rear-wheel target is6rad/s in MuSHR; F1Tenth distributes wheel speeds through its native geometry. Both entries are targets, not acceleration/torque percentages or guaranteed motion. A zero speed target does not teleport velocity to zero. Tire slip and native drives remain active.
One control step=0.02s; steps=10 holds ONE command for0.2s, not ten increasing commands. coding_control, if enabled, may update both entries every tick.
These numerical examples explain units and gain, not safe speeds or a supplied driving controller.
