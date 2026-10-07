# SteadyTray Recovery V2 (Task 79)

Revision requested on 2026-10-01. Selector remains world-state-v1; results identify
steadytray-recovery-v2. Historical v1 proposals, traces and scores remain archived.

- Maximum 20 simulated seconds / 1000 control ticks at 50 Hz.
- Episode-seeded uniform trigger ticks: object in 1..250, robot in 251..500.
  Disable both recurring interval terms. Invoke each original native callback
  once with unchanged parameters. Object x/y and roll/pitch ranges remain
  +/-0.3 m/s and +/-0.3 rad/s; robot x/y remain +/-0.5 m/s. Actual behavior is
  defined by the native callback, not a replacement force implementation.
- Record scheduled and applied events. Isolate/reseed push RNG and restore its
  state so action-dependent noise does not change the sampled disturbance.
- Remove the World always-supported and <=20-degree object-tilt predicates.
  Native failure/termination and its author-defined delays are NOT changed.
- Only after both callbacks have completed, require goal distance <=0.25 m,
  horizontal speed <=0.10 m/s, and object_supported continuously for 2 seconds.
  object_supported retains the existing filtered Object-to-Tray upward force
  >1e-4 N, tray-local center bounds |x|<=0.127 m, |y|<=0.176 m and z>0.
- Interruptions reset the two-second timer. Successful completion ends early;
  the 1000-tick maximum without success is failure. Native failure dominates
  success on the same tick. Missing measurements remain invalid/unscored.

Initial contact establishment before scored time zero is unchanged. The goal
remains 3 m ahead of reset heading; actor observations and actions are unchanged.
This revision does not rescore or resume an old rollout. A fresh physical run is
needed to validate the revised timing and early-success behavior end-to-end.
