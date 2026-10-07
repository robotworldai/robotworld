# ReflexBench / T03

Environmental source code, original official asset, Docker construction is in this directory. `checkout/` is complete and unmodified
LxRoboticsLab/ReflexBench, fixed `8bb931485093c6d98f8729774ad01bf824964e16`.
World adapter in `environment/benchmarks/reflexbench/project.py` without modifying Codex.

## Directory and Run

- `checkout/`: upstream source code and licence; Author policy is not available to detected models.
- `project.json`: T03 entrance, original 100 step cap and operating conditions.
- `assets/`: Franka, catch cup, ground and dependent files, local cache, no Git.
- `prepare_assets.py` / `asset-manifest.json`: Only download from NVIDIA official and verify Hashi.
- `docker/`: Reuse the software layer of the existing Isaac6.0.1 mirror; Runs the read-only source code and assets.

In World root directory:

```bash
bash scripts/eval/reflexbench.sh list
bash scripts/eval/reflexbench.sh assets
bash scripts/eval/reflexbench.sh build
bash scripts/eval/reflexbench.sh run --output var/runs/docker/reflexbench/t03-gpt6-01 --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

The public starter connects the local Docker environment with a dynamic action tool using local source code constructed Codex app-server.
See `docs/MODELS.md` for model configuration. Full track and no image version saved in events for run.
Asset commands require Python for `pxr`; The existing asset tool venv can be used by default by specifying `WORLD_ASSET_PYTHON`.

## Native mechanism and rating

** BallCatchingEnv** with `BallCatching-Franka-IK-Abs-v0` to retain its custom step
Order (spacing events prior to the termination decision), random pitching, cup follow, virtual capture area, obstruction and original termination conditions.
Physical 100 Hz, action 25 Hz (decimation=4), 4 seconds / 100 control step; The budget cannot be exceeded by the original assessment.
Original rewards is empty. Only the original `task_phase == 4` is successful; The ball that has been launched below 0.03 m failed.
Automatic reset is prohibited after the first reset, and the last and original done signals are kept.

Action 8 dimension: The robotic root coordinates are absolute EEF `[x,y,z,qw,qx,qy,qz]` and binary claws.
The original policy status contains the ball position, velocity, projected interception position/time, and ** is not purely visual **; Future launch angles/timers are not disclosed.
Following mechanisms and virtual capture with glasses installed on wrists are not natural, and this limitation is retained in the report.
The body action server/ IK is maintained without a pre-trained ball receiver for the model. code control can only generate the same action from permitted observations.

## Compatibility and authentication

Isaac is not fixed upstream; This run-off is ** Isaac Sim 6.0.1 + IsaacLab 2.2 compatible with **
No physical equivalence with the author's original platform. API alias/empty recorder method only for external compatibility; Do not change upstream documents.
The cups strictly use the upstream Isaac 5.1 USD, Franka assets corresponding to Isaac 4.5 for ground-based use of experimental run-off Lab2.2.
Actual assets SHA256 see manifest; Static code completion does not amount to simulation/ GPT pass and is currently `VALIDATION.md`.

## Fixed-source review

Full source code has been placed in local checkout. To reconstruct the source, check out the repository and commit in project.json,
Do not switch to main; SteadyTray also press runtime_source to check out the author frame and keep the Git LFS entity of source/assets.
Unmounted author model, policy action from World local Codex.

## 2026-09-28 Actual Authentication

- Mirror built; 4 step GPU probe through: `var/runs/docker/native17/gpu-probe03/reflexbench/T03`.
- Local source code Codex + gpt-6-astra complete single round: `var/runs/docker/native17/gpt03/reflexbench/T03`. With a 100-step budget, native ball_on_ground termination occurred at step 36, with success=false; It's not an infrastructure error.
- Video 25 FPS, 37 frame (initial frame + 36 control step), complete events and no-images are kept.
- Externally compatible global texture for Kit110 hangs, using non-promoting physical review rendering; Completing old Lab aliases for num_rerenders_on_reset. Upstream source code and task criteria have not been modified.
- This result is derived from the Isaac6 Experiment Compatible Environment and does not claim a numerically equivalent to the official version of Isaac.
