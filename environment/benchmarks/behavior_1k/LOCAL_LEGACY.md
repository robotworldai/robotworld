# Historical local deployment (before 2026-09-30 merge)

Preserved for the local legacy launcher; not the default upstream protocol.

# BEHAVIOR-1K: Local Native First Four

## Current default: 5000 control steps

New four-task parallel and serial batches default to `steps5000`: at most 5000
native 30Hz control steps (166.67 simulated seconds), not wall-clock seconds.
The prompt, host protocol validation and native task termination all use this
same cap. API retries, no-give-up policy, recording and operational watchdogs
are unchanged. Use `--budget-profile steps5000` for individual gateway/agent
commands; their low-level default remains `legacy` for backward compatibility.
The explicit `steps10000` and `legacy` profiles and all old results are retained.
Changing the control budget alone does not fix an API image limit. The separate
visual-memory policy below bounds images on the actual outgoing request.

## Shared visual memory

BEHAVIOR and RoboDojo now share `current-plus-4-history-stride-2-v1`. An
observation packet contains current round t and available rounds t-2/t-4/t-6/t-8;
rounds count observations, not control steps. All cameras of a selected round
stay together. BEHAVIOR uses one native mosaic per round (at most five images).
RoboDojo preserves its camera set (three cameras yield at most fifteen images).

The request audit retains only the latest complete packet's images in the
outgoing Responses input, replacing older packet images with explicit notices.
All original text, action results, call IDs, image files and full video remain.
The exact latest packet hashes must match the adapter, including current RGB;
missing/reordered images or opaque server-side history fail closed. Packets
above fifty images are rejected, not silently cropped. Incoming/outgoing counts,
selected rounds, hashes and removed occurrences are recorded without image bytes.
API retries replay the same already-selected body. Original no-rewrite runs and
their source snapshots are unchanged and must not be pooled with this policy.

## Four GPU lanes

`python -m environment.benchmarks.behavior_1k.run_four_parallel --output
/root/robotworld-behavior-private/first-four-parallel-NEW --worker-dir
/absolute/robotworld/runtime/behavior1k/docker --manifest /absolute/build.json`

This additive supervisor maps tasks 1..4 to distinct GPUs (default 0..3), resolving
DRM nodes by PCI address, not assuming matching numbering. Each lane holds its
own existing GPU lock, receives up to 64 distinct CPU cores, and has independent
Kit apps, appdata, caches, temp files, recordings and results. The software venv
is read-only; only the per-lane Isaac apps mount accepts native startup writes.
Licensed assets remain shared read-only. No global driver or glibc changes.

One batch owner holds the Docker-daemon lock across all eight gateway/agent
containers. Lane cleanup cannot stop that daemon or another lane. The owner
refuses an existing daemon and only cleans its labelled containers. Each lane
starts its model episode immediately after its own 630-step recorded gateway
passes on the same GPU with matching source hashes and allocation. An individual
lane failure is recorded separately while the other lanes continue. Parallel
wall-clock/API load differs from the serial profile and is recorded explicitly.

Parallel Docker uses a separate offline copy of the existing VFS image store at
`/root/robotworld-behavior-private/docker-parallel-runtime/data`. Its socket and
exec state also live there. The original `/dockerdata` image store is untouched:
that volume exhausted its inode quota despite having free bytes. Startup checks
both daemon locations and requires 250000 available inodes and 16 GiB free.

## 10000-control-step profile

Use `--budget-profile steps10000` on `run_first_four` (or on both gateway and
agent invocations). The default `legacy` profile remains reproducible. This is
a new run profile; existing results and report snapshots are not rewritten.

- Exactly 10000 control steps at 30Hz, at most 333.33 simulated seconds. The
  original task termination_config and host/worker validators all use this cap.
- No 600-second wall-clock task deadline. At most 10000 action requests and
  20000 tool calls protect against invalid-call loops without reducing a run
  that takes one or more valid control steps per action.
- Per-command duration stays 1..15 steps. Pose/velocity limits, camera inputs,
  fixed trunk, physical grasping and original success/Q checker are unchanged.
- No give_up tool. Premature successful text completion continues in the same
  Codex thread and physical episode with fresh RGB and a generic persistence
  reminder, recorded in episode.json. No scene reset or hidden goal hint.
- Three consecutive turns with no executed control steps, 64 nonstepping tool
  calls, 480 seconds without an app-server event, upstream/IPC errors, or the
  approximately 24.5-hour launcher fail-safe stop as operational failures. They
  do not receive a full-budget model score. The worker's 720-second watchdog is
  an idle timer refreshed after requests, not a total episode timer.

The long gateway includes real 42-command / 630-control-step acceptance, crossing
both obsolete limits. A clean new simulator reset precedes the model episode.
Agent score_valid additionally requires native success OR exactly 10000 executed
control steps, plus the existing environment, image boundary and video checks.
API requests use the explicit shared visual-memory window described above, with
a larger local byte/request guard. Unexpected provider/context limits remain
infrastructure errors, never silently fixed by further trimming. A 10000-step episode requires at least 667 model
actions, so wall-clock cost can be hours and is not 333 seconds.

The long profile retries opening a model request up to 3 times (4 attempts total),
with identical selected body, images and headers. Each attempt has a 90-second socket
timeout and exponential backoff of 2/4/8 seconds. Retryable failures are transient
transport errors and HTTP 408/429/500/502/503/504 before ANY downstream headers
or body are sent. Retry-After up to 30 seconds is respected; longer server delays
are returned instead of retried early. Authentication, validation and certificate
errors are not retried. Once a response starts streaming, it is never replayed.
All attempts, errors, delays and retry counts are saved without credentials.
Recovered transient attempts do not invalidate an otherwise complete audited
request. Exhaustion, partial-stream errors and missing images still invalidate it.
The 480-second model-idle and 720-second worker-idle guards accommodate four
90-second attempts and bounded backoff. No robot command is replayed or reset.
This is request-level recovery, not automatic episode restart or checkpoint resume.
Per-process provider overrides disable Codex's nested request/stream retries and
set its stream-idle timeout to 480 seconds, so the relay owns the retry bound.
Concurrent audit writers serialize atomic saves; no host authentication is changed.

This local addition to main connects its CodexSession / episode runner to the
existing RobotWorld BEHAVIOR Docker worker. It is not part of upstream commit
0b7ce83, and is not the RoboDojo move_eef action schema.

Scope: clean_up_your_desk, Feishu task 2, official 2026 train instance 0,
house_single_floor, R1Pro, seed 0. The exact instance is a diagnostic selection,
not verified as a Feishu-specified instance. An explicit `--task 1|2|3|4`
now selects the corresponding pinned native train instance via the additive
`task_worker.py`; omitting it preserves the original task-2 worker path.

The first four are carrying_in_groceries (house_double_floor_lower),
clean_up_your_desk, slicing_vegetables, and sorting_vegetables (house_single_floor).
Template hashes are checked against the local partial asset receipt before
startup. Task 3 enables GPU dynamics because native dicing creates particle
systems. No skill or scripted solution is added. EEF execution, observations
and checker functions are AST-compared with the frozen original worker in tests.
The template selection is diagnostic, not a verified exact Feishu instance.

## Boundary

Host: source-built Codex pinned by main, existing private AI Hub credential, and
main's episode runner. Docker: the existing OmniGibson 3.9.3 / Isaac Sim 5.1.0.0 /
Python 3.11 runtime, original reset, native IK, EEF execution and original checker.
The worker, protocol, upstream sources and existing launcher are not modified.

The model sees observe / step / give_up. step retains the native local
behavior-r1pro-eef-v1 protocol: base-relative position deltas <=0.04 m,
axis-angle deltas <=0.15 rad, XYZW observations, gripper 0=closed/1=open,
1..15 native 30Hz steps, base commands <=0.2 and fixed trunk. No world-jaw
targets, CuRobo planner or high-level manipulation skills are substituted.

Only the worker's sensor-only state and camera mosaic are forwarded. Object
truth, goal predicates and Q-score are evaluator-only. Main receives a labelled
current-plus-history packet, bounded at the outgoing request as described above.
This never discards the current tool RGB, unlike the old proxy incident. Results
from different visual-memory profiles must not be silently pooled.

A local request audit selects the declared image window, preserves tool/text
inputs, and resolves gpt-6-astra-azure to gpt-6-astra for AI Hub Azure. Before
forwarding each request it requires the profile's exact tool list and the hash
of the current RGB mosaic. Missing RGB or extra tools reject the request and
invalidate the score. Only tool names, image hashes/counts and status are logged,
not image bytes or credentials. Thread startup also requires no instruction
sources, no workspace roots, read-only/no-network permissions and the requested
model. Per-process overrides disable agents, goals and user-input tools; the
host's authentication/config files are unchanged.

Budgets: 600 seconds, 96 tool calls, 40 motion requests, 500 native steps.
observe does not consume the motion-request budget. Uncertain execution stops
the run without replay; infrastructure errors do not receive a model score.
Original worker limits also apply.

On controller-requested timeout/budget/stop, the audit may qualify only the final
in-flight HTTP200 request as expected cancellation. It must have no image/tool/API
violations and must end with a client disconnect. Earlier disconnects, incomplete
API responses, and unacknowledged physical execution remain invalid. The marker
is recorded at the actual runner interrupt, not inferred retrospectively.

## First-Four Attempt

`run_first_four` runs serially on GPU0, preserving the existing locks. Each task
gets a source-matched recorded gateway followed by a fresh recorded agent run;
failed gateways skip inference rather than emit a model zero. The existing host
venv's ffmpeg must precede the host PATH (which also supplies ffprobe).

```bash
python -m environment.benchmarks.behavior_1k.run_first_four \
  --output /root/robotworld-behavior-private/first-four-NEW \
  --worker-dir /absolute/robotworld/runtime/behavior1k/docker \
  --manifest /absolute/verified/build.json
```

Private `status.json` links every gateway/agent run, the raw checker, score
validity, termination and video verification. Budget is 600 wall seconds after
reset per agent, 40 motion requests, 96 tools, 500 control steps. This is a bounded
fixed-trunk sensor-only diagnostic, not full-budget official challenge scoring.

## Run

Use new output paths under the private local BEHAVIOR directory, never the
shared source checkout. Pass the existing worker directory explicitly.

```bash
python -m environment.benchmarks.behavior_1k.run gateway \
  --worker-dir /absolute/robotworld/runtime/behavior1k/docker \
  --run /root/robotworld-behavior-private/native-task2-scaffold-gateway-NEW

python -m environment.benchmarks.behavior_1k.run agent --aihub-direct \
  --worker-dir /absolute/robotworld/runtime/behavior1k/docker \
  --run /root/robotworld-behavior-private/native-task2-scaffold-agent-NEW \
  --accepted-smoke /root/robotworld-behavior-private/native-task2-scaffold-gateway-ACCEPTED \
  --manifest /absolute/verified/build.json
```

gateway uses no model: observe non-stepping, stale request rejection, each
arm forward 1 cm, physical gripper close/reopen, 240 physics steps, changing
RGB, original checker and clean Docker exit. Agent mode requires matching
source hashes from a passing gateway and the source-built Codex manifest.

Source snapshots, exact command, budget, model result, native report and
observations remain in each private run. Existing worker images are sparse
observation captures, not continuous video. This adapter does not claim
continuous recording. Software tests do not establish simulator acceptance
or a model score.

## Continuous Recording

Add `--record-video` to both gateway and agent commands. The agent requires a
source-matched, video-accepted gateway. This opt-in uses a separate
`run_video.sh` launcher and `record_worker.py` overlay; the original native
launcher, worker and checker remain unchanged.

The overlay invokes the original env.step once per control step, then saves
its three native RGB streams and a separate 640x480 render product of the
existing head-camera prim. Model observations remain 128x128 per camera.
There is no interpolation, added physics step or model-wait padding. Videos
start after the first control step and run at the native 30 FPS, so duration
is `native_steps / 30`, not episode wall-clock duration. The 4:3 recording
aspect ratio differs from the square model sensor output.

Files live under the private run's `simulator/video/`: `head_640x480.mp4`,
`head.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, capture and encoding manifests.
Acceptance checks decoded frame counts, resolution, frame rate, consecutive
step IDs and exactly four native physics callbacks per recorded step.
This is a new recording-enabled run, never retroactive video of an old run.

## AI Hub Azure Route

Agent mode now requires `--aihub-direct`. The legacy port-8789 proxy was found
to drop ALL tool images once their count exceeded two, including the latest
observation. Old Q=0 runs affected by this bug are invalid capability evidence;
see their `image-routing-incident.json`, which supersedes raw score_valid.

The isolated route sends Responses to
`http://api.aihub.woa.com/openai/v1/responses` with model `gpt-6-astra` and Azure
provider authentication, following the user-supplied AI Hub example. It reads
only the existing private `~/.codex/robotworld-api.env` (mode 0600); credentials
are never written to source, logs, argv or simulator mounts. The shared proxy,
Codex auth/config and model catalog are not edited or restarted.

Only the model alias and authentication headers change. Tool results, image
blocks, history, reasoning and store settings are forwarded unchanged. There
is no image trimming, summary substitution, Chat Completions conversion or
silent image-error recovery. Outgoing hashes are audited at the AI Hub request
boundary. Terminal API errors invalidate the run even after an HTTP200 header.

Before a real episode, test FOUR sequential tool images, not a single image:

```bash
python -m environment.scripts.check_behavior_images --aihub-direct \
  --manifest /absolute/verified/build.json \
  --output /root/robotworld-behavior-private/aihub-image-check-NEW
```

Random codes exist only in image pixels; the model must read each code, including
the third and fourth tool images. This verifies transport/visibility, not task
success. A fresh source-matched physical gateway is still required after code
changes. Full BEHAVIOR scoring must not be claimed from this synthetic probe.
