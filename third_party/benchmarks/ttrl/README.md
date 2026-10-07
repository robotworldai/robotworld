# TTRL / T02. Man-shaped robots continue to pick up random ping-pong.

Full Upstream `purdue-tracelab/TTRL-ICRA2026` Fixed
`fdb192f8fff9f5ca54c4ec4a8441fda5174ca078` placed `checkout/`, unmodified.
Use the original `t1_tt_eval` customized `TTEnv` instead of the generic IsaacLab environment.
Docker reserves ** Isaac Sim 4.5.0/ IsaacLab 2.1.0 ** as requested by the author. Author clearly reported new version simulation
(a) Decline performance and therefore move to 6.0.1 without delay; Availability under current hardware must be verified separately.

The image builds, but the original Torch 2.5.1+cu118 CUDA distribution does not support this host's RTX 5090 sm_120 architecture
kernel support range, minimum `no kernel image` reported. This machine can't be executed as it was running.
Full T02 or GPT assessment; This is an environmental barrier and cannot be counted as a failure of a model. See `runtime-status.json` for evidence.

## Install and Run

World root directory execution:

```bash
bash scripts/eval/ttrl.sh list
bash scripts/eval/ttrl.sh assets
bash scripts/eval/ttrl.sh build
bash scripts/eval/ttrl.sh probe --output var/runs/docker/ttrl/probe01
bash scripts/eval/ttrl.sh run --output var/runs/docker/ttrl/gpt6-01 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

app-server built with local `World/codex`. The model configuration is `docs/MODELS.md`.
The environment executes the original action at local Docker; Do not request the model API directly and do not load the author 's balancing/bouncing policy.
`prepare_assets.py --check` requires Python for pxr. Default World asset venv.

## Action and observation of boundaries

- Original ** 21-Dividation **, target = default joint position + 0.25 x clip(action, -100, 100).
  Dynamic tools to expose the original joint order, default position and scale; Bottom is a joint service, not walking controller.
- Physical 500 Hz, decimation 10, controls 50 Hz. Keep action delay and perception delay.
- Original actor ** 5 frame history, 81 dimension/ frame, total 405 dimension **, from old to new. Original state, delay ball/body
  Positions, projection slots, relative targets and directional reservations. Field slices are recorded in observation-metadata.json.
- Optional learned predictor ** Disables ** and three projection slots maintain original zero values; Its derived relative objective is not effective.
  Sphere trajectory prediction. This setting must be marked as an authorless predictor condition, which does not match the author predictor-enabled scores.
- The real future ball state is used only internally in the original critic/reward and does not enter actor or coding_control.
  coding_control produces the same actions only with permitted observations; The third person claimed that the review video was not shared.
- Following the original `eval.py` visible cover: noise.add_noise=True, push_robot=None, height drift=0;
  Physical disturbance during training cannot be claimed. The scenes, air resistance, contact, balling and reward keep them alive.

## Assessment and boundary

Following the original `legged_lab/scripts/eval.py`, latch each ball's hit and valid-return events after `step` returns:
`ball_contact_rew > 0` is hit;`has_touch_opponent_table_just_now & has_touch_paddle`
It's an effective backball event. Archive only when the original `ball_reset_ids` indicates that one ball has been completed; The first **two complete serves**
Skipping is not seen as a success or failure in the hemisphere where the budget ends. `success=null`.
No full round of SR is fabricated. Follow the post-step sample calibration of the original script without changing the sequence of events.

Single ball timeout 1.8 seconds; The ball fell from the original environment. counter has been added from 0 to 1 at the time of construction;
The original count ends with `ball_reset_counter > 5`, so a maximum of five balls is completed without falling, of which three are used to score.
Original robot falls/border terminates retention. Only the robot episode automatically reset is disabled and the original ball continues.
The 540 step is the worst 450 maximum of the harness step than the serve protocol and does not change the timeout or allow crossing done.
The huge time limit of the original eval scenario does not amount to a requirement to keep running for that period.

Simulations are suspended during model thinking. 50 Hz is the simulation period for executing the action, not proof that GPT can respond per 20 ms in real time.

## Assets, source code and build

- `checkout/`: Full author code and embedded robot/desktop USD; The ball is the geometry of the original program.
- `asset-manifest.json`: really go look for USD sublayer, reference, payload dependencies and SHA256.
  The current two entry points consist of ** 8, original USD**, all of which are locally dependent on resolution and no author's absolute path is missing.
- `assets/`: original floor, materials, texture, HDR with ** 9 official resources **; Could not close temporary folder: %s
- `prepare_assets.py`: Re-opening dependent locking; Mistakes are reported when there is a lack of reliance, and not only the top file is checked.
- `docker/`: Fixed official 4.5 mirror and Lab 2.1 commit; Do not contain model evidence.
- `environment/benchmarks/ttrl/project.py`: Independent adapter, original serve statistics and final protection.

The resource path points to the same local file by external code; Upstream USD, configuration source code and physical parameters are not modified.
For verification status see `VALIDATION.md`.

## Optional Isaac Sim 6.0.1 Experimental Files

The default remains the original Isaac4.5. Show selection `--runtime-profile isaac6` only
`runtime-profiles/isaac6.json`, `docker/Dockerfile.isaac6` and Independent `project_isaac6.py`.
This profile reuses the existing 6.0.1 image software, but loads the complete pinned source from `isaaclab21_checkout/`,
Lab 2.1.0 at `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`, rather than the base image's different Lab version.

```bash
python scripts/eval/native_projects.py build --project ttrl --runtime-profile isaac6
python -m environment.runtime.native_project_launch --project ttrl --runtime-profile isaac6 \
  --task T02 --mode probe --output var/runs/docker/ttrl/isaac6-probe
python -m environment.runtime.native_project_launch --project ttrl --runtime-profile isaac6 \
  --task T02 --mode codex --output var/runs/docker/ttrl/isaac6-gpt6 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

The experiment file shares with the default file the original 21-D actions, 405-D actor history, projector shutdown, original ball and serve count,
Two complete warmup scores, final processing. External bridges are only compatible with API and record rendering; Do not change mission objectives and scores.
New Torch executes the original eager version of the function using the official TORCHDYNAMO_DISABLE=1 entry; World Coordinates of Original Aerodynamics
PhysX is_global=True is kept in every physical writing by force/recent, which is called locally to continue the same path.
The engine version itself may change the physical result, so the scores must be individually marked and cannot be declared equivalent to the author 4.5 running time.
The construction, detection and GPT field run are recorded separately and the incomplete stages are not written through.

A source-built Codex/GPT-6 episode completed on the experimental profile: a 540-step budget ended natively at step 93 (1.86 seconds)
The non-temporal robot is terminated. Only one warmup ball completed; no scored ball was reached. Hit rate and valid-return rate are both null,
and must not be reported as 0% or as success. Record: `var/runs/docker/native17/profiles-isaac6/gpt01/ttrl/T02`;
The video is `video/camera.mp4`, 94 frame/ 50 FPS, which reviews the lenses covering the entire table and robot.
The original 4.5 and experimental 6.0.1 states are found in `runtime-status.json` and `runtime-status.isaac6.json`, respectively.
