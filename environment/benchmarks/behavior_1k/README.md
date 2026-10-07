# BEHAVIOR-1K / World adapter

Agent access and execution: [AGENT_BOUNDARY.md](AGENT_BOUNDARY.md). The current
launcher isolates source Codex outside the simulator container, enables shell
and code mode, and exports only allowed observations to its workspace boundary.

External integration targeting upstream **v3.9.3**, commit `6cbf70b075816096e9be53958780769f3264d25d`. Source and evaluator are not edited. Repeated Isaac 6.0.1 probes now pass with the upstream RealTimePathTracing renderer and upstream R1Pro JointController configuration (23D). A source-Codex episode using the external IK tool profile (21D) also completed 181 steps with that renderer, exit 0, success=false/Q=0. Darkness is not yet explained conclusively. See [RENDERING_STATUS.md](RENDERING_STATUS.md) for current evidence and [STATUS.md](STATUS.md) for historical experiments.

Preserve the original scene, initial state, physics, rendering and task rules. The historical alternative-renderer option is disabled. Actual upstream source is included locally at [World/third_party/benchmarks/behavior_1k/checkout](../../../third_party/benchmarks/behavior_1k/checkout/).

## Execution path

`local source-built Codex app-server → benchmark-specific tools → WorldPolicy → native R1Pro controller action → official BatchedEvaluator → sensors / metrics / video`

The official evaluator's supported Hydra local-policy injection point loads `WorldPolicy`; this first integration uses in-process policy callbacks rather than the optional websocket transport. Simulator operations stay on the evaluator's main thread. Codex owns the model conversation; there is no direct model HTTP client. A tool response is sent only after the evaluator has executed its requested steps and supplied fresh observations. At terminal steps, the launcher flushes final observation and pending tool feedback without advancing the simulator again.

Files:

- `control.py`: validated robot-base EEF pose to absolute-IK action mapping, bounded local base velocities, gripper target persistence.
- `policy.py`: Codex session and tool dispatch across evaluator callbacks, camera history, per-step events.
- `../../integrations/behavior_eval.py`: official evaluator configuration, runtime action-index profile, official result finalization.
- `../../robots/r1pro/behavior_ik.yaml`: explicit custom robot configuration, based on the upstream R1Pro file, changing arm controllers to absolute IK and base inputs to physical units.
- `../../containers/behavior_1k/`: isolated Docker launcher and official asset preparation.

## Observation and action contract

Only the configured robot's RGB, depth and proprioception are consumed. Unknown fields, task object states, global robot poses and camera extrinsics are dropped before prompting. Proprioception layout is verified against runtime dimensions. RGB-D wrapper is upstream `RGBDFullResWrapper`: head 720x720, wrists 480x480. Original depth arrays are saved as NPY; the model sees a labeled 0–3m grayscale depth visualization. Histories retain four observations sampled every two decision rounds. Official videos record every evaluator step.

`move_arms`, `move_base`, `set_grippers`, and `move_torso` have separate allowed target fields and matching descriptions. `move_robot` exposes their union for simultaneous control: both EEF positions/orientations, both grippers, base velocities, and trunk targets when the runtime profile provides a trunk controller. All components share one 1..30-step segment and consume that many environment steps, not a sum per component. Separate tool calls remain sequential. Gripper commands begin alongside motion, not after arrival; separate approach and closure when required. Arm targets remain robot-root relative during base motion. Trunk targets are a complete absolute joint-position vector in proprioception order, within runtime joint limits; omitted trunk holds its measured target. Unsupported camera/head or raw arm-joint controls are not exposed. They share `note`/`targets`/`steps` envelopes. Coordinate semantics are explicitly **robot articulation-root relative**, unlike RoboDojo world coordinates. Position is metres; orientation is XYZW quaternion (converted to controller rotation vector). Gripper 0=closed, 1=open. Optional `base_vx/base_vy/base_wz` are local m/s and rad/s, bounded to 0.3/0.3/0.5; zero by default per call. `steps` is 1..30 at the official 30 Hz action rate. Omitted arms hold their current observed pose; gripper targets persist. IK is not collision-free path planning. No ground-truth scene planner or object teleportation.

This initial profile supports only single-environment R1Pro with the supplied IK configuration. Do not infer support for all robots or all household tasks.

## Metrics and protocol

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for allowed observations, global-map restrictions, tool coordinate conventions and the 5000-step development run.

The versioned evaluator computes success, Q-score and efficiency; World does not replace these with model self-reports. Default timeout is upstream 1.5x task mean human-demo length. Official report indices are public 0–9, once each; public 10–19 are reserved here for development. One invocation runs one instance; run official instances sequentially for a report and retain every result, not the best run. Do not aggregate a development subset as the full challenge score.

The policy keeps trying while the episode is active and hard budgets remain. `give_up` is not offered; ordinary model turn completion continues in the same thread with current observations, without resetting the environment. Failed grasps and IK residuals should trigger revised attempts. Hard tool/time budget exhaustion, explicit interruption or a failed model turn can still stop decisions; the evaluator then holds until its own termination/timeout so metrics come from the official path. Gripper targets persist during holding. Infrastructure errors are not assigned fabricated benchmark scores.

Outputs: upstream `json/`, `videos/`, `results.json`, exact `evaluation-config.json`, `robot-profile.json`, and per-episode `rollout-000/{episode.json,prompt.json,frames/,events/}`. `events/` and `events/no-images/` are generated automatically, including Codex messages, requested/completed tools and native actions with post-step proprioception.

## Validation limits

Controller conversion, state dimensions, terminal action accounting, observation filtering, events and image-free logs are tested without Isaac. A fake Codex session in tests is explicitly not an end-to-end runtime result. No BEHAVIOR success has been claimed. Actual GPU rendering and a bounded source-Codex IK smoke passed on the external 6.0.1 runtime; the official 5.1 image and full task completion remain unvalidated.

Official reference: https://behavior.stanford.edu/challenge/evaluation.html (checked 2026-09-26). Local source documents and evaluator are pinned to the commit above; future changes require rechecking these contracts.


## Step-by-step control `coding_control`

### BEHAVIOR special sports tips

[motion_prompt.py](motion_prompt.py) will be added to the system/baseInstructions actually sent to the local Codex, and the normal action mode and program mode will apply. Make it clear that `steps` only repeats the final target and does not automatically insert values or speed limits. (a) Normal actions require a move from the latest physical position: each end-point shift does not exceed 3 cm and attitude changes do not exceed 5°; 5 mm / 2 ° for close exposure to check for tracking errors before advancing. The maximum 0.02 rad is changed at each joint of the torso and the lower speed limit is used for the chassis.

In program mode, generate continuous reference targets using dt: translation reference speed should stay within 0.06 m/s, or 0.015 m/s near contact; The attitude reference speed does not exceed 15 °/ s, 6 °/ s, respectively, while checking actual tracking errors and prohibiting the continuation of cumulative targets when robots fail to catch up. Complete binding is based on the prompting file.

These values are for initial conservative hints of significant target jumps that have not yet been validated through a new set; Not a new enforcement end hard limit or physical speed guarantee. Action conversions, upstream controllers, scoring and video recording remain the same. Only the following new Codex sessions, old video and old prompt will not be rewrited.

`run_behavior_1k.sh --code-control on` (Default) opens the tool, `off` removes it from schema and the execution side refuses to call. The container entry corresponds to `--disable-coding-control`. Local source code Codex, external isolation and official BatchedEvaluator maintain the same path and the upstream source code remains unchanged.

Parameters: `note` (for this paragraph purpose), `max_steps` (1 – 600), `code` (limited Python for defining `control(obs, memory)`, maximum 16000 characters). Each official `forward(obs)` received new observations, the program was executed once and returned to `{'targets': {...}}`, with the target name, unit, and range fully consistent with `move_robot`; The arms, chassis, torso and claws together form a primitive action. Returns `{'done': True}` to end this segment and return the model, without declaring the mission successful and without extra execution to keep moving.

`obs` contains the permitted current proprio, the thin 12×16 depth grid for each machine-mounted camera (mi, with original map sampling coordinates, with invalid values null), dt=1/30, number of completed steps, number of in-procedure steps, nominal remaining budget, previous target and clawing command. No simulator handle, object real position, global map or rating. The RGB scenario is still understood through images normally received by the model. The original action verifyr is responsible for border checks and the official evaluator is responsible for advancing physics, recording video, success/failure and ending the round.

(a) The end/dry objectives of the program ' s initial lock measurement, which were subsequently omitted to retain the last program target; The chassis speed is default zero at each turn, and the claw opening is maintained continuously. Full observation schema, examples and interpreter limited to [code_control.py](code_control.py), which injects system prompt. The program only supports restricted AST calculations and math, and cannot access files, networks or any Python module. Up to 2 seconds, 100000 interpretation operation per recall; A single-part maximum of 600 moves (20 simulation) can be terminated earlier. The environment was suspended during the calculation of model reasoning and procedures and no real-time hardware control was claimed.

`programs/` save codes and parameters; `events/environment.jsonl` records permitted observations, return values, memory, and validated actions for each callback; Complete with `no-images` track is still automatically saved. An error in the program only terminates the paragraph and returns the error to the model and the error does not execute the action; The action executed will not roll back. Infrastructure failure no longer fills the round with hidden action.

Tests of the non-model control interface have been completed (real limited interpreter + simulation official callback input), and the full GPT scene assessment of the new tool is not yet operational; The current smoke batch remains suspended.

