# RoboCasa Environment

- `checkout/`: Same fixed submission `456174f62b89b8fca99eaaf33949c29fec9cfc2a` for local `/opt/robotworld/robocasa`, clean Git checkout, without replay_actions.py or output.
- `assets/`: Recyclers have downloaded independent copies of official assets without downloading a second time; Separate copies can be mounted for upstream generation/ deletion of temporary XML without Docker mirrors or GitHub.
- `docker/`: Dockerfile, source-code archive build scripts, starters and build records.
- robosuite: `third_party/dependencies/robosuite/checkout`, local fixation of `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`.

This is MuJoCo 3.3.1 / Python3.11 / RoboCasa1.0.1 environment, not dependent on Isaac Sim. Mirrors only require the installation of evaluation without training library, model weights, account numbers or scene assets. All sources remain unchanged.

## Build

Execute in World root directory with GPU Docker:

```bash
python third_party/benchmarks/robocasa/docker/build.py
```

Mirror tag `world/robocasa:1.0.1`. Dockerfile installs and packs the original source code directly from the Python base mirror instead of relying on the existing RoboDojo/BEHAVIOR mirror. `docker/sources.json` save source version, `World/var/build/docker/robocasa/` save build context, build.log and image-inspect.json; No binary mirror uploaded registry.

The new machine acquires the source code from the upstream version of the sources.local.json record, runs the `python -m robocasa.scripts.download_kitchen_assets` under the official installation description, obtains the official asset and mounts a separate copy by the path here. The official asset source is defined by `checkout/robocasa/models/assets/box_links/box_links_assets.json` and does not use World HF mirrors. Gym reset does not need a training data set to start the evaluation.

## Do not call the real GPU probe

```bash
python third_party/benchmarks/robocasa/docker/run.py \
  --probe-only --task-name CloseDrawer --num-trials 1 --probe-steps 8 \
  --output "$PWD/var/runs/docker/robocasa/close-drawer-probe"
```

The output directory must be created, the original horizon remains unchanged, but probe stops early at the specified step and marks informal assessments.

## Source code Codex scene assessment

Build with your login directory using the existing source code for World/codex:

```bash
python third_party/benchmarks/robocasa/docker/run.py \
  --codex-home "$PWD/var/auth/robodojo-codex" \
  --task-name CloseDrawer --num-trials 1 \
  --timeout 7200 --wall-timeout 9000 \
  --output "$PWD/var/runs/docker/robocasa/close-drawer-codex"
```

Complete selected registered task: Repeat `--task-name SortingCleanup --task-name CoffeeSetupMug --task-name NavigateKitchen`, default 50 times/task; One round first. `--split target` toggle the official target scenario distribution, default pretrain. Short-link tests only are visible `--horizon 8`, and records mark non-standard budgets.

`CountertopCleanup` lacks the official registry horizon and therefore does not quietly add a Standard Step; `--horizon` is passed and can be diagnosed, but cannot report standard performance assessments. The remaining question 7 is not specified.

Outputs include launcher.log, command.json, exit.json, summary.json, per-task stats.json, episode-*/events (including no-images), prompts, frames, and continuous video.mp4. The running script will await the end of the entire container and leave no limitless budgetary tasks behind.
