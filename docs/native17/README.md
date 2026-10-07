# 17 access: source, running and verifying boundary

The task is from [Original Handover Page](handoff/index.html), the source package is stored as `handoff/` and is not a validated environment per se.
User has confirmed the current access to ** T06 and T12, not T02**. This time add the remaining 15 questions and merge them into 13 stand-alone projects; Do not change Codex and upstream source code, do not update upload-github.

** name distinction: ** already existing `robolab` is ** NVlabs/RoboLab** (desktop operation, classification, spatial placement); `robot_lab` for this batch of T11 is ** agilexrobotics/robot_lab** (prelegation support for A1). Two stand-alone warehouses, adapters and scripts are kept separately, and T11 blockages do not affect RoboLab which is accessed.

| Node. | Project Directory | Interface/task |
|---|---|---|
| T01 | steadytray | G1 Full body joint, body/carrying and author IsaacLab fork |
| T02 | ttrl | T1, Ping-Pong, whole body joint, touch ball and effective return ball. |
| T03 | reflexbench | Franka Absolute EEF position + claw, original catcher event and prediction observation |
| T04 | aerial_balance | Vertical velocity increment, original 15 step delay and OU disturbance |
| T05 | volleybots | Original flight movements, fixed rival conditions |
| T06 (available) | wheeledlab | Original RSS drift; Play or custom driving scenes |
| T07/T08 | wheel_legged | Original 6 V VMC target, Recovery/Terrain-Reactive |
| T09 | wheeled_quadruped | The forward leg position and the rear wheel. |
| T10 | go2_push | Go2 original joint target and random thrust course |
| T11 | robot_lab | A1 Post-leg task, original fixed version 10 second round |
| T12 (available) | humanoid_soccer | Currently MuJoCo sim2sim; Different from hand-over package Isaac |
| T13 | digit | Digit original hands target/walk joint joint joint joint joint joint control |
| T14 | omniisaacgymenvs | Original AnymalTerrain and Propulsion, Independent old Sim environment |
| T15 | flamingo | Original TrackJump, rev01_5_2 robot |
| T16/T17 | omnidrones | Hummingbird Original 4 rotor, action_transform=null |

Each project consists of:

- `third_party/benchmarks/<project>/checkout/`: Full fixed upstream source code and licence.
- Peer `project.json`, asset inspection/description, `docker/Dockerfile`, `docker/run.py`: Environment and dependency.
- `environment/benchmarks/<project>/project.py`: Native environmental construction, action description, prompt, observation white list and rating reading.
- `scripts/eval/<project>.sh`: Single item entry. The public transfer layer only recycles tools and does not change physics or tasks.

## Current Verifiable Status

The results of the actual operation of this round and the reasons for the jamming are found in [Authentication records](VALIDATION.md), which is based on the records and `exit.json`, `result.json` of each run.
The machine is readable at `var/runs/docker/native17/final-validation.json` and lists 12 active model rounds and 3 blocking tasks. The external `a1-feet` asset compatible configuration for T11 has been modeled, as detailed in [Restoration of records](../../third_party/benchmarks/robot_lab/ASSET_COMPATIBILITY.md).

The static preparation progress of the projects is shown in their respective README; ** cannot connect directory existence, Dockerfile existence or CPU test to ** as GPU.
Docker permissions have been restored to 2026-09-28; The default mirror for the 13 project and the corresponding 3 stand-alone mirrors for T02/T14/T16/T17 have been constructed. Of the 15 newly integrated tasks, T01, T02, T03, T07, T08, T09, T10, T11, T13, T14, T16, and T17 completed valid local source-built Codex/GPT episodes. See the validation records for their outcomes; “Fulfiling” does not represent a successful mission. T07/T08 Early Lock Wheel Round, T16/T17 Early Externalally Lost Round has been marked invalid and only recalculated. T04 does not have original payload assets, T05 does not have target 1 v1 and is not compatible with the original CUDA, T15 has an original configuration conflict and has not yet completed model measurements. T11 has been unblocked through an independent a1-feet asset compatible configuration and the original USD block still keeps records. Could not close temporary folder: %s The initial Docker permission error was retained at `var/runs/docker/native17/initial-preflight/T07/`.

Static checks covered the pinned sources for 13 projects and launch dry-runs for 15 tasks. The selected interface, observation-boundary, metric, and termination tests reported 61 passed and 1 skipped (the host lacked Torch, so the TTRL batch test was skipped; Corresponds to container Torch otherwise tested. dry-run does not activate Docker or model, and mock interface tests do not represent physical feasibility. Source list in root directory `sources.lock.json`, including project-specific IsaacLab sources; Upstream codes were not rewritten for testing.

Also confirmed is the T04 master USD quote author 's 4's payload, which was not released with the repository. `aerial_balance/asset-manifest.json` lists the original path; Do not impersonate the environment with alternative models. The compatibility of the other items with the old Sim/RTX5090 must also be confirmed through probe.

T05 tracked 34 weights of PT belong to 3 v3 or hierarchy skills and cannot be directly identified as complete rivals to the target 1 v1; Competing condition `third_party/benchmarks/volleybots/opponents/README.md`. The OmniDrones geometry asset is local and has completed the GPU model round; Start still accesses the original remote material and extension service and cannot be declared completely offline.

The existing T12 integration uses MuJoCo sim2sim at the same commit, with direct/hybrid modes; The handing over package specifies the Isaac task `Tracking-Flat-G1-SoccerMoving-RNN-v0`, and therefore does not declare that the handing over package settings have been reproduced. Existing access is maintained in accordance with this directive and its achievements are not replicated as those of the new agreement.

## Run Mode

Runs at World root, and only this new 15 question is covered by default:

The host starts the script using Python 3.11 or updates (model configuration needs `tomllib` for reading); Simulate packagings with original item Python versions. Host needs Docker socket access; This machine has been processed. If other machines encounter permission denied, the machine administrator should configure privileges at the terminal and not pass passwords in chats or logs.

```bash
bash scripts/eval/native17.sh list
bash scripts/eval/native17.sh check
# Ready./Validation of project assets; missing author resources will clearly fail and cannot be claimed to be ready
bash scripts/eval/native17.sh assets
# All project build definitions; first added--dry-runView
bash scripts/eval/native17.sh build
# Single project environment probe: real loading of assets and execution4A primary control step.
bash scripts/eval/native17.sh probe --project reflexbench --output var/runs/docker/native17/probe01
# SingleGPTwhole-grown budget; output directory must be created every time
bash scripts/eval/native17.sh run --task T03 --model gpt-6-astra \
  --codex-home var/auth/robodojo-codex --output var/runs/docker/native17/gpt01
# All new themes run sequentially, avoiding full-filling simultaneouslyGPU
bash scripts/eval/native17.sh run --model gpt-6-astra \
  --codex-home var/auth/robodojo-codex --output var/runs/docker/native17/all-gpt01
```

`--steps` only shortens the operating budget and does not change the duration of the original task, random disturbance or success conditions. A shorter round that does not reach its original state cannot serve as a complete official assessment.
First probe by running GPT; `summary.json` records infrastructure return codes, primary indicators and success fields. Keep `null` when the official binary SR is missing.

### Independent Isaac6 Experiment Configuration

The user has agreed to provide an additional Isaac Sim6.0.1 compatible configuration for T02, T14, T16/T17. Default `--runtime-profile default` to retain the original configuration of the items; Visible `--runtime-profile isaac6` uses the new configuration and its results cannot be used as the performance of the original run. Item `runtime-profiles/isaac6.json` covers run-time fields only, and the starter refuses to overwrite tasks, steps, source repository or commit.

```bash
bash scripts/eval/native17.sh build --project ttrl --runtime-profile isaac6
bash scripts/eval/native17.sh probe --task T02 --runtime-profile isaac6 \
  --output var/runs/docker/native17/ttrl-isaac6-probe
bash scripts/eval/native17.sh run --task T02 --runtime-profile isaac6 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex \
  --output var/runs/docker/native17/ttrl-isaac6-gpt
```

T14 uses `--project omniisaacgymenvs`, T16/T17 uses `--project omnidrones`. Build separate records of files, mirrors, caches and running outputs; Original mirror and failed evidence retention. Current construction/probation/model completion is based on validation records. The source code, 16 mirrors and 15 + 4 start-up commands are stored in `var/runs/docker/native17/source-check-final03/`.

11, which has been validated on RTX5090, runs in two groups and maintains the active running selection; 4 questions, which are not yet available, or which have conflicting configurations, are not commingled in this group:

```bash
bash scripts/eval/native17.sh run \
  --task T01 --task T03 --task T07 --task T08 --task T09 --task T10 --task T13 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex \
  --output var/runs/docker/native17/retest01/base
bash scripts/eval/native17.sh run --runtime-profile isaac6 \
  --task T02 --task T14 --task T16 --task T17 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex \
  --output var/runs/docker/native17/retest01/explicit-isaac6
```

Here, the default configuration of group 1 itself discloses the Isaac6 test compatible boundary; “default” is not an official physical equivalent statement. Change `--model` and corresponding `--codex-home` to replace the model, imitation, tools and original calibration to maintain the original configuration.

## Model & Process Boundary

`--model` replace model names; `--codex-home` specifies independent authentication/provider configuration. Use [Model service description](../MODELS.md) for configuration.
The starter reads commit of `World/codex`, using only local source products validated by Hash in `var/build/codex/<commit>/build.json`. codex, installed in PATH, will not be called, and there is no alternative path to the Python straight-link model API.
Codex app-server is in host bubblewrap, imitated in Docker. Exchange observations and actions only through controlled dynamic tools; agent does not see environmental source code, private rating and third person review lens.

Common tools are `observe`, `apply_action`, `coding_control`. `apply_action` also enters all the freedom of the original action vector, and each item retains the actual dimension, zoom, original PD/IK/VMC. Models can write step-by-step feedback on control procedures, but do not automatically give authors a trained mission strategy. The model required for the evaluation of T05 rival strategy etc. is described separately.

## Recording and calibration

- Original `events/{codex,tools,environment}.jsonl` is saved with automatically generated `events/no-images/`.
- `prompt.json`, source code/build Hash, complete configuration, seed, action metadata, observation layout and reason for stopping in each run.
- Current +4 historical observations, interval2 instrument observations; Videos are stored in the original control steps without following the prompt frame.
- Numerical actor observations may contain simulators ' state, target commands, projections, disclosed on a case-by-case basis and do not claim that all missions are purely visual. critic and rating state do not give the model.
- LLM pauses simulation during reflection, code controls each primary control step back; Retain the original task delay/disturb. This does not constitute real time reasoning delays or robotic deployment experiments.
- Original reward/failure/timeout is separated from derivative indicator. No primary SR can be replaced with "survival to end" The failure environment does not initiate a forgery or mock GPT test.
