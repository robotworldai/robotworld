# Each benchmark one entrance

Bench2Dex Add 41 – 49: `bash scripts/eval/bench2dex.sh list|assets|build|check`，and `probe/run --cases 41,42 --output ...`。 See [Note by Bench2Dex](../../third_party/benchmarks/bench2dex/README.md) for each original budget, active joint tool and experimental running boundary.

Add a single inverted ball: `bash scripts/eval/volleybots_single.sh list`/ `build`/ `probe --output ...`/ `run --codex-home ... --output ...`. It selects a clearly defined `T05-single` compatible configuration with the `isaac6` experiment, the original 800 step and fails to stop early; These are not results for the original T05 1v1 task. There is no need to download rival weights.

Use `bash scripts/eval/native17.sh list` for the new 15 for the 17 interlocking package; Supports the selection of `check/assets/build/probe/run`, `--project` or `--task`, with output separated by item and title. Full usage and true unverified items are found in [17 notes](../../docs/native17/README.md). T06/T12 follows the existing entrance and does not automatically rerun or swap them.

Use the root directory in a repository, which is only listed by default and is not connected, built or run. Each .sh top is also listed directly.

```bash
bash scripts/eval/all.sh list
bash scripts/eval/robodojo.sh list
bash scripts/eval/behavior_1k.sh list
bash scripts/eval/robocasa.sh list
bash scripts/eval/robolab.sh list
bash scripts/eval/humanoid_soccer.sh list
```

Unified sub-command: `sources` restores fixed upstream checkout; `build` build mirrors; `assets` prepares official assets; `run` Each selected task runs a round; `all` executes the first four steps sequentially. The main entrance all.sh is executed in the order of benchmark and does not rob GPU. `--dry-run` only print plans and still check parameters; Does not mean that the model, GPU or asset has been validated.

```bash
# Only two short missions.
bash scripts/eval/robolab.sh run --cases cube-left,cube-front --runtime isaac601 \
  --model YOUR_MODEL --codex-home "$PWD/var/auth/my-codex" --dry-run

# Build Independence RoboCasa mirror;fixed Git checkout You can skip. sources
bash scripts/eval/robocasa.sh sources
bash scripts/eval/robocasa.sh build
bash scripts/eval/robocasa.sh assets --robocasa-assets /path/to/official/robocasa/models/assets

# All environments ready to run the whole set at once (default official running time)
bash scripts/eval/all.sh run --model YOUR_MODEL --codex-home "$PWD/var/auth/my-codex" \
  --output "$PWD/var/runs/suites/model-a"
```

The modeling method is described in detail in [MODELS.md](../../docs/MODELS.md), the environment depends on [DOCKER.md](../../docs/DOCKER.md) and the official asset source [ASSETS.md](../../docs/ASSETS.md). The snapshot in the public package is not an independent Git repository, and the sources sub-command will keep the snapshot backup and restore checkout.

## Title, data and assets

Unified machine readable list: `environment/evaluation/suites.json`. The current four cores of benchmark are now 15 RoboDojo + 10 BEHAVIOR + 10 RoboCasa + 10 RoboLab by default. RoboCasa CountertopCleanup does not have verified official horizon, no --steps visible not_run; ID28 is an alias of ID10, which is defaulted to avoid double counting. To reproduce two analytical perspectives, two distinct options -- cases 10 or cases 28 -- cannot be called different official tasks.

- RoboDojo: 15 selected tasks, see `bash scripts/eval/robodojo.sh list`, official arx_x5 layout; `assets --robodojo-assets /path/to/official/Assets` only copys the selected scene. Python is required with pxr.
- BEHAVIOR: 10 selected tasks (retention of groceries, desk aliases), fixed public_test index 10/ Example 311; `assets --accept-behavior-license` Downloads Selected Official Closed. Each user must accept the upstream agreement on its own; Keys/assets may no longer be distributed.
- RoboCasa: 10 to reload; 28 is a historical distinguishing alias for 10; Official asset downloaders are downloaded by large asset groups and do not claim to support the exact minimum closure of each item. Priority - robocasa-assets reuse of official assets; It is true that you need to download the Visible Fax for the first time - download-large-assets (approximately 10 GB, the official tool is interactively confirmed).
- RoboLab: 10 selected tasks, 450 – 3000 steps, converted to the original `episode_length_s`; cube-front, the old hammers is only a visible entrance to history; `assets --cases cube-left,cube-front` retrieves only the official LFS closure of the Cube scene. Need git-lfs and usd-core.

Python can be selected by `WORLD_PYTHON=/path/to/python` to execute the entry. `python -m pip install usd-core`, for example, for USD asset preparation; Not replace Docker internal emulator Python.

## Budget and running time

`--runtime official` is default; `--runtime isaac601` to BEHAVIOR/RoboLab selects the active experiment to run. RoboDojo is currently using the verified 6.0.1 snapshot, and RoboCasa is always MuJoCo. The 6.0.1 derivative mirror relies on an existing RoboDojo homeshot and is not fully independently installed.

`--steps N` against RoboDojo/BEHAVIOR/RoboCasa is covered by the diagnostic budget and will be charged to suite.json; The default does not cover the official budget. RoboLab refuses to overwrite this by using the selected task budget. `--timeout` is the budget per round of model wall clocks, with Docker plus 1800 seconds for the total budget; It's different from a simulation.

RoboCasa `--seed` Default 7, `--episode-index` Default 0, '--split pretrain|target ` Default pretrain. The rest of the benchmark uses the list to specify the example or original single-round feed, and there is no fictional unified seed syntax.

suite.json save models, tasks, budgets, commands and starter states; The starter exits 0 by itself does not amount to a successful mission, and the final result depends on the official output. Failed or unrunable entries will be retained for reasons that the total process returns non-0 but continues to process the remaining entries. The output directory must be created; This script does not automatically restore the interrupted batch. Do not launch multiple full campaigns on the same GPU concurrently.

## HumanoidSoccer

`humanoid_soccer.sh` by default only fixed ** play-soccer** and `all.sh` by default added this topic: seed=2, geostationary, training-pitch Background, 1000 control step/ 20 seconds, direct + coding_control, ankle-com balance support. The environment, assets and scores are upstream; 1000 step and support to World customised control diagnostics without impersonating an official 300 step protocol.

```bash
bash scripts/eval/humanoid_soccer.sh assets
bash scripts/eval/humanoid_soccer.sh build
bash scripts/eval/humanoid_soccer.sh run \
  --model gpt-6-astra --codex-home "$PWD/var/auth/my-codex" \
  --output "$PWD/var/runs/suites/soccer-gpt6-01"
```

Change model only to `--model` and corresponding `--codex-home`. The actual monograph output is in `OUTPUT/humanoid_soccer/play-soccer/` and contains video, complete/ungraphed events, model control code, actual prompt and official results. Fixed profile is `play-soccer-v1`, and all models use the same previous failure lessons file `third_party/benchmarks/humanoid_soccer/profiles/play-soccer-v1.txt`, which clearly marks informed diagnostic; This file is saved as notes from the measured second round and is not mixed with action.

This question does not accept different `--seed` or `--steps`; seed is still fixed to 2 when it is not passed, not affected by other benchmark default 7. The original 300-step baseline/hybrid/direct modes remain optional comparisons, selected explicitly with `--cases baseline,hybrid,direct --seed 7`; they are not mixed into the default run. To change the physical condition, control mode or duration, a new diagnosis is made using an independent Docker launcher.

## AI-CPS

`ai_cps.sh` default ID22 catch, 23 tray balance, 24 insertion, original 300 step cap, respecting early termination. `all.sh` automatically contains three questions (ID34 removed as requested by user). `--steps` only diagnoses when shortened, and full scoring cannot be replaced by short tracks.

```bash
bash scripts/eval/ai_cps.sh sources
bash scripts/eval/ai_cps.sh build
WORLD_PYTHON=var/venvs/robolab-assets/bin/python bash scripts/eval/ai_cps.sh assets
bash scripts/eval/ai_cps.sh run --model gpt-6-astra --codex-home "$PWD/var/auth/my-codex" --output "$PWD/var/runs/suites/ai-cps-01"
```

Only model/auth can compare different models; Unified source code Codex bridge, complete/ungraphed events, code and video. The current re-use of the Isaac6.0.1 experiment mirror, in particular the problem of the peg collision convection retreat, can be seen in `third_party/benchmarks/ai_cps/README.md` and cannot claim the official physical equivalent of the old version.

## WheeledLab

`bash scripts/eval/wheeledlab.sh list` displays four configurations: mushr-drift, f1tenth-drift (each 250 control step/ 5 second), elevation (200 step/ 20 second), visual (50 step/ 10 second). `assets` check the original USD in checkout and download about 2 MB official ground dependency; `build` returns the existing RoboLab 6.0.1 mirror. The current runtime is an experimental Isaac6.0.1/IsaacLab2.2; suite.json is recorded and is not misled by the generic-runtime default value.

```bash
bash scripts/eval/wheeledlab.sh run --model YOUR_MODEL --codex-home /path/to/model-codex-home
bash scripts/eval/wheeledlab.sh run --cases elevation --steps 100 --model YOUR_MODEL --codex-home /path/to/model-codex-home
```

Use local source code Codex; Model access is the same as other bench. The original configuration does not have a uniform binary simulation success rate. See `third_party/benchmarks/wheeledlab/TASKS.md` for details. The video is from an additional appraisal camera and is not fed to policy. Complete primary observations, actions, coding_control records and logs for images are automatically saved. all.sh sequence runs this bench.

### WheeledLab RobotWorld Custom Four Questions (2000 Step)

`bash scripts/eval/wheeledlab_custom.sh list` view four; `run --model MODEL --codex-home HOME` sequence assessment. The default set of original WheeledLab remains unchanged; The new issue has an independent agreement with success, which cannot be confused with official success rates. See `third_party/benchmarks/wheeledlab/robotworld/README.md` more.
# WheeledLab Precision Driving Supplement

`bash scripts/eval/wheeledlab_precision.sh list` lists the two narrow beam bridges, the reverse garage and the side parking; `run --model YOUR_MODEL --codex-home /path/to/model-home` runs. Both 2000 steps, enabling the reverse vehicle, with the model only looking at the front/back view and its own sensors. Solid-line contact is a failure, and the specific size/scoring/ solvency boundary is `third_party/benchmarks/wheeledlab/robotworld/PRECISION.md`. The default selection for the original set remains unchanged.

For a description of the non-model verification and code/asset aggregation for this round, see `environment/validation/README.md` and `environment/datasets/ASSET_LAYOUT.md`. The original termination condition is not disabled because the video is too short.
