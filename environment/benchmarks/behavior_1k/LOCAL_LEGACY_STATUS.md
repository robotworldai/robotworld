# Historical local status before main merge

These records describe the local legacy runtime, not the upstream deployment.

# Local Main Integration Status

## 2026-09-26: Shared Visual-Memory Window Implemented

New source entry points for BEHAVIOR and RoboDojo share current observation plus
four prior observations at stride two. The per-run request audit keeps images
only from the latest complete packet and leaves text, actions, call IDs, original
image files and videos intact. BEHAVIOR sends at most five mosaics. Request
selection verifies both ordered image hashes and temporal labels. All 101
environment tests pass, including 65-round one-/three-camera histories, retries,
identical pixels, missing/reordered packets and native-worker parity.

Real Codex/Astra probe `image-window-probe-20260926-01` completed 42 image-reading
calls, all 42 correct. Raw request history reached 195 images; outgoing requests
never exceeded five and no 50-image rejection occurred. The 52-call target did
NOT complete: the 900-second probe budget expired during slow API responses.
Its report is `passed=false`, not a full acceptance. The temporal-label guard
was added while the probe was running; it was then tested separately against
43 archived real frames, not live in that process. No robot episode, image
rebuild or existing-result rewrite was performed. A fresh source-matched gateway
and completed request probe are needed before claiming new simulator acceptance.
Future four-task batches retain the requested 5000-control-step default.

## 2026-09-26 18:15 CST: Four GPU Lanes Started

Batch `first-four-parallel-20260926-02` runs tasks 1..4 on GPUs 0..3.
Four live containers were checked independently: each sees only its assigned
GPU UUID. All four loaded OmniGibson / Isaac 5.1 and entered scene loading.
At this checkpoint native reset and physical/RGB acceptance are still pending;
no model episode or new score has been obtained. Each lane automatically starts
its own fresh Codex + EEF episode after its recorded gateway passes.
The 10000-control-step budget, request retries and video recording are retained.
All 86 environment tests passed with the designated ffmpeg on PATH.

Attempt `first-four-parallel-20260926-01` failed during Docker container creation:
the original `/dockerdata` volume had only six free inodes despite 8.9 TB free.
No physics or model evaluation occurred; these are infrastructure failures,
not zero scores. An offline 4.6 GiB copy of the idle VFS store now serves parallel
Docker under the private local directory. Old Docker data, native worker code,
licensed assets, host driver and Codex authentication were not changed.

## Correction: Confirmed Proxy Image Loss And Direct Route

The local evaluation proxy classified all tool output images as
historical and removed them when their count exceeded two. It retained
the initial user image. Logs at 10:59:00..10:59:39 and 12:05:20 / 12:05:29 confirm
this happened in agent03 and video-agent01. Their original accepted/score_valid
fields are superseded by private `image-routing-incident.json` files. Original
checker Q=0 remains factual, but those runs are NOT valid capability scores.

An independent `--aihub-direct` evaluation route now sends to AI Hub's native
Azure Responses endpoint, resolving only the model alias and authentication.
No image/history trimming or protocol conversion occurs. Shared proxy, global
Codex configuration, authentication and simulator are untouched.

`aihub-direct-image-probe-20260926-01` passed FOUR sequential randomized tool
image checks. Five outgoing requests preserved all images (counts 1..5), reached
AI Hub directly, and completed successfully. The model read all four codes,
including the third/fourth images that previously triggered deletion. This is
transport validation, not a new task score. All 51 software tests pass. Real
simulation must get a fresh source-matched gateway receipt before next scoring.
Independent repeat `aihub-direct-image-probe-20260926-02` also passed all four
randomized image checks after enforcing successful terminal events. Both tests
used main's source-built Codex and the same configured GPT-6 Astra Azure model,
without starting any simulator or changing global provider/authentication files.

## 2026-09-26: Native Docker Gateway Passed

Private evidence:
`/root/robotworld-behavior-private/native-task2-scaffold-gateway-20260926-01`.
No model was called. This is physical/IPC acceptance, not a scored episode.

- Existing BEHAVIOR Docker reused without changes: OmniGibson 3.9.3,
  Isaac Sim 5.1.0.0, Python 3.11, Torch 2.7.0+cu128.
- Original task2 reset and original checker ran. Scene is house_single_floor,
  clean_up_your_desk, official train instance 0 / seed 0 / R1Pro.
- Left/right forward displacement: 0.01000276 m / 0.00999871 m.
- Left/right target error: 0.00008731 m / 0.00008763 m.
- Left gripper close/reopen joint-vector motion: 0.07070954 / 0.07070912.
- Four commands, 60 native steps, 240 physics callbacks; nonempty three-camera
  RGB mosaic changed. Read-only observe did not advance physics. A stale
  command was rejected without motion.
- Source hashes unchanged; original checker completed; container exited 0;
  no forced cleanup or uncertain command acknowledgement.
- All 42 environment tests pass, including the existing RoboDojo tests.

Source-built Codex 8ae55c863db26d417e83390c5854f1144114276b subsequently compiled
with its unchanged lockfile and Rust 1.95.0. Build time: 12m06s.
Binary SHA256: c6c5c3c7593ab88e900357ae34bf43580e14c053128cff8640957a6e1aad234e.
The original exported script modes and bubblewrap LICENSE symlink were restored
to the pinned upstream tree; no Rust content or dependency lock was changed.
The nested Codex Git status is clean.

Actual App Server tool registration passed. Agent attempt
`native-task2-scaffold-agent-20260926-01` stalled before reset in native
SimulationContext.play during OmniGibson scene import. Two read-only Python
stack samples and unchanged I/O confirmed several minutes without progress.
The owned container was stopped; model_turn_started=false, model_score=null,
score_valid=false. This is an environment initialization failure, not model 0.

Attempt `native-task2-scaffold-agent-20260926-02` completed with original
checker Q=0, success=false and clean Docker exit. However, its raw score_valid
field is superseded by qualification-review.json: a subsequent same-config
request audit exposed extra agent/goal/user-input tools. It is NOT a qualified
pure-EEF result. The model's reported loss of subsequent RGB was not independently
established: a synthetic tool-image round trip successfully identified a new
green K7 image. The initialization stall on attempt01 was not a proven deadlock;
attempt02 eventually completed initialization with the same configuration.

The follow-up uses agents.enabled=false (feature flags alone do not override
the model catalog), disables goals/user-input, checks thread isolation and
rejects requests before forwarding unless the exact tool allowlist and current
RGB hash are present. Request bytes are not rewritten. Raw RGB and credentials
are never included in the audit. Fresh physical acceptance is required after
these source changes before another scored attempt.

Follow-up synthetic probe `main-image-probe-20260926-02` passed. Both upstream
requests had only observe/give_up; the current image hash was present, and the
model identified the tool's green K7 image. Thread response confirmed the
requested gpt-6-astra-azure model, no instruction sources/workspace roots, and
read-only/no-network permissions. All 46 environment tests pass with the
project venv/bin on PATH (the host Conda ffmpeg lacks libx264/preset support).

Fresh gateway `native-task2-scaffold-gateway-20260926-02` passed with the
audited source revision: four physical commands, 60 native / 240 physics steps,
left/right displacement 0.01000276 / 0.00999871 m, finger close/reopen and changed
nonempty RGB. Source hashes remained unchanged; Docker exited cleanly.
Agent attempt03 uses that source-matched gateway and the request audit.

## Audited Task2 Episode Completed

`native-task2-scaffold-agent-20260926-03` completed with gpt-6-astra-azure:
original checker Q=0.0, success=false. Five motion requests, two observations,
one give_up; 64 native steps / 256 physics callbacks / 2.1333 simulation seconds.
Right-arm moves measured 0.02636, 0.03772, 0.03799 and 0.04158 m; the first two
arm targets were not reached, later two were. No goal progress was recorded.

All eight model requests had exactly observe/step/give_up and the current RGB
hash, returned HTTP200, and contained no detected image-omission notices.
Thread isolation passed. Sources were unchanged, acknowledgement was certain,
and the original checker and clean Docker exit completed. GPU0 was released.

The agent nevertheless gave up saying current RGB was unavailable. This is
not supported by the sending-side audit and does not establish a simulator
failure. Downstream model perception is not proven by an HTTP200/hash check;
the discrepancy remains open. Keep this as a diagnostic run, not a clean
capability comparison or full BEHAVIOR leaderboard score. See the private
run's qualification-review.json. No further task or batch was started.

Initialization took about 12 minutes. Read-only py-spy and gdb samples found
SimulationContext.play waiting in carb.tasking / gpu.foundation / RTX scene
synchronization; loading subsequently progressed. The inspection briefly
paused the owned process before reset, and any effect on scheduling is unknown.
No evidence establishes a deadlock or a dependency-installation failure.

## Continuous Video Acceptance

Opt-in recording gateway `native-task2-scaffold-video-gateway-20260926-01`
passed: original reset/checker, measured EEF motion and gripper actuation,
60 native steps / 240 physics callbacks. Four H.264 streams decoded to exactly
60 frames at 30 FPS (2 seconds); head recording is 640x480, original head/wrist
streams remain 128x128. MP4 frame counts were verified with ffprobe.

The separate recording launcher/overlay leaves the original worker unchanged,
adds a render product to the original head camera, and captures after each
original env.step. No extra physical step or interpolation is used. Camera
aspect ratio differs from model observations. The first high-resolution frames
show renderer convergence noise; later frames are clearer. This is not a new
camera viewpoint or a guarantee of photorealism.

All 48 environment tests pass. Source-matched video agent attempt01 was then
started with the original 600-second / 40-action model budget and request audit.

Recording agent `native-task2-scaffold-video-agent-20260926-01` completed:
gpt-6-astra-azure, two base-yaw commands, 30 control steps / 120 physics callbacks,
then give_up. Original checker Q=0.0, success=false. All four videos contain
30 decoded frames at 30 FPS (one second). Head video is native 640x480; no
upscaling or temporal interpolation. This is a new live run, not the old episode.
Sources unchanged, all five audited requests had the intended tools and latest
mosaic, Docker exited cleanly and GPU0 was released. The model again claimed
tool RGB unavailable; the audit covers delivery to the local API proxy, not
its downstream transformation or model-side visibility. Capability attribution
remains unresolved. The recording task is complete; no extra episode was run.

The worker reports mode=agent / eef_verified=false because it serves IPC
rather than executing its own built-in smoke. The outer main gateway carried
out and verified the physical smoke; see result.json / physical_smoke.
