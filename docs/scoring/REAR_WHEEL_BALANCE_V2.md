# Rear-Wheel Balance V2

Human-approved revision on 2026-10-01 for task 85 only. ANYmal is unchanged.

- Total duration remains 20 simulated seconds, 1000 control ticks at 50 Hz.
- Sample one trigger tick uniformly from 1..500 using the episode seed (default
  rollout seed remains unchanged). Thus the push occurs in (0,10] seconds.
- Disable the native recurring interval event and call its original function
  exactly once with its original parameters. Seed its Torch RNG separately and
  restore RNG state afterwards; action-dependent random draws cannot change the
  push sample. No push after 10 seconds. Native mode is unchanged.
- At the final tick, require three elapsed seconds of uninterrupted valid rear
  balance, including both endpoints 17 and 20 seconds (ticks 850..1000).
- Keep rear-wheel support, front-wheel clearance, no other body support, height
  error <=0.08 m from 0.828 m, and upright tilt <=15 degrees unchanged.
- Remove only the extra 0.75 m reset-centred position failure. Preserve native
  fall checks and termination; a fall before the push still ends the episode.

The command-line profile selector remains world-state-v1 for compatibility;
world_evaluation.version and success_definition identify rear-wheel-balance-v2.
The older proposal's recorded screenshots/comparisons/verification evidence are
historical v1 material; this amendment supersedes its 10-second hold, recurring
pushes and practice-circle restriction. No old score is overwritten. This
revision requires a fresh physical rollout, not a rescore of the 64-tick run.

Scheduled/applied push times and seeds are recorded in world_scene_events.
Unit tests do not constitute a successful physical demonstration.
