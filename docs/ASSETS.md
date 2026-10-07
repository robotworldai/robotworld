# Assets

## Pinned release

The default source is [visity/RobotWorld-Assets](https://huggingface.co/datasets/visity/RobotWorld-Assets). `environment/datasets/asset-source.json` pins the revision and manifest hash.

```bash
bash scripts/setup_handoff.sh assets
bash scripts/setup_handoff.sh assets --bench robocasa
bash scripts/setup_handoff.sh assets --local-source /path/to/upload-hf
```

The downloader verifies each file's SHA-256 and restores runtime paths. Explicit `--repo-id` and `--revision` overrides are available; the asset manifest must still match the code. Restricted BEHAVIOR assets are excluded. Every upstream asset retains its own terms.

## Layout and restoration

Assets are stored under `Assets/<benchmark>/`. See [asset layout](../environment/datasets/ASSET_LAYOUT.md). Restore existing matching assets with:

```bash
python scripts/restore_assets.py --apply --bench robocasa
```

This restores local paths; it does not download missing files. For official downloaders that populate legacy paths, use `python -m environment.datasets.centralize apply --bench BENCH --report reports/assets/BENCH-migration.json` after downloading. Centralisation verifies hashes, moves resources into `Assets/`, and creates compatibility links. `python -m environment.datasets.embedded_assets --apply` collects embedded upstream assets without changing their contents.

Source exports omit registered assets, nested Git repositories, authentication, caches, builds, and run outputs. Keep code and asset releases separate.

## RoboDojo

Follow the [official asset documentation](https://robodojo-benchmark.com/doc/) and the bundled project README. Existing official assets can be reused. With a Python interpreter that provides `pxr`, create a scene subset:

```bash
python environment/containers/robodojo/copy_scene_assets.py \
  --assets /path/to/official/Assets \
  --layout Eval_Layout/RoboDojo/arx_x5/1/match_and_pick_from_conveyor_0.json \
  --output "$PWD/var/datasets/robodojo/match_and_pick_from_conveyor_0"
```

The coin layout is `Eval_Layout/RoboDojo/arx_x5/0/deposit_coin_0.json`. The script preserves the original assets and emits a subset manifest. Historical single-scene HF utilities are not the default complete-suite downloader.

## BEHAVIOR-1K

Read the asset terms for the pinned OmniGibson revision. Every user must accept the terms independently before downloading:

```bash
python scripts/prepare_behavior_assets.py --accept-license --dry-run
python scripts/prepare_behavior_assets.py --accept-license
```

The selected-task preparation targets all ten registered tasks and stores data under `Assets/behavior_1k/data`. Inspect the resulting manifests for actual coverage. Assets and decryption keys must not be redistributed in the source or shared asset pack. See the [container guide](../environment/containers/behavior_1k/README.md).

## RoboCasa

Use the official downloader in an environment with the pinned RoboCasa package installed:

```bash
python -m robocasa.scripts.download_kitchen_assets
```

Official URLs are recorded in `checkout/robocasa/models/assets/box_links/box_links_assets.json`. Follow the integration README to prepare the separate `third_party/benchmarks/robocasa/assets/` copy: the upstream loader requires temporary XML write access. Evaluation does not require training trajectories.

## RoboLab

Restore sources, install the USD reader in a dedicated asset environment, and request selected scenes:

```bash
python scripts/fetch_sources.py robolab
python3 -m venv var/venvs/robolab-assets
var/venvs/robolab-assets/bin/pip install usd-core
var/venvs/robolab-assets/bin/python third_party/benchmarks/robolab/prepare_assets.py \
  --scene rubiks_cube_banana_bowl.usda
```

Other scene names include `tools_container.usda`, `foodpacking_1bin_2box_2can.usda`, and `wire_shelf_mugs_plate_spatula.usda`; repeat `--scene` as needed. Only the selected official Git LFS dependency closures are fetched. External NVIDIA material references remain recorded in `assets.local.json`, so complete offline rendering is not guaranteed.

## HumanoidSoccer

```bash
python scripts/fetch_sources.py humanoid_soccer
bash scripts/eval/humanoid_soccer.sh assets
```

This checks or retrieves official G1 mesh/MJCF resources, soccer-standard motions, and `policy_30000.onnx` from the pinned project. The upstream licence is CC BY-NC 4.0. The source snapshot does not include the excluded ONNX/mesh/motion asset binaries.

## Bench2Dex, AI-CPS, and driving

- **Bench2Dex:** `bash scripts/eval/bench2dex.sh assets` prepares selected official USD closures and first-origin scene anchors. Resources are organised under `Assets/bench2dex/{dex2bench_dataset,anchors,shared-assets,upstream}`. Scene anchors are not demonstrations of successful actions by the evaluated robot.
- **AI-CPS:** `bash scripts/eval/ai_cps.sh assets` prepares the pinned Franka and ground dependencies. See its project README for the exact sources and compatibility limits.
- **WheeledLab:** `bash scripts/eval/wheeledlab.sh assets` checks native MuSHR/F1TENTH and ground resources. The seven custom RobotWorld courses are generated separately; scoring records `scenario.json` and source hashes.

## Other native-project tasks

Each project exposes `bash scripts/eval/PROJECT.sh assets`. This prepares or checks published resources; it cannot retrieve files that the upstream author did not release.

| Project | Tasks | Notes |
| --- | --- | --- |
| `steadytray` | T01 | Project-specific robot and IsaacLab resources. |
| `ttrl` | T02 | Native table-tennis environment dependencies. |
| `reflexbench` | T03 | Original catching scene and robot resources. |
| `aerial_balance` | T04 | Historical entry with missing original payload USD resources; excluded from the distributed asset pack. |
| `volleybots` | T05, T05-single | Single-drone juggling needs no opponent; 1v1 requires the specified complete opponent. |
| `wheel_legged` | T07, T08 | Recovery and terrain-reactive tasks. |
| `wheeled_quadruped` | T09 | Native wheeled-quadruped resources. |
| `go2_push` | T10 | Go2 and the pinned runtime resources. |
| `robot_lab` | T11 | A1 project, distinct from manipulation `robolab`. |
| `digit` | T13 | Digit resources. |
| `omniisaacgymenvs` | T14 | Legacy-runtime dependencies. |
| `flamingo` | T15 | Historical original configuration has a critic/sensor inconsistency. |
| `omnidrones` | T16, T17 | Hummingbird resources; remote material/extension dependencies may remain. |

T06 uses WheeledLab and T12 uses the existing HumanoidSoccer integration. See [native-project documentation](native17/README.md) and individual project READMEs for exact source revisions and validation status.

Shared embedded robot and motion resources live under `Assets/_shared/upstream/dependencies/`. Project-specific fork assets belong under that benchmark's `upstream/` directory. Asset restoration supports including shared resources where needed. Successful local preparation does not grant additional redistribution rights.
