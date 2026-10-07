# GalaxeaVLA: G0.5 Visual-Language-Action Model

[[item home page]](https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=github)](https://opengalaxea.github.io/G05/)
[[Thesis]](https://img.shields.io/badge/Paper-8A2BE2?style=for-the-badge&logo=arxiv)](https://opengalaxea.github.io/G05/Galaxea_G0_5.pdf)
[[Video]](https://img.shields.io/badge/Videos-FF0000?style=for-the-badge&logo=youtube)](https://opengalaxea.github.io/G05/videos/introduction_g05.mp4)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF6B35?style=for-the-badge&logo=huggingface)](https://huggingface.co/OpenGalaxea/G05)

[English](README.md) | Chinese

<div align="left">
  <img src="assets/r1_mascot.jpeg" alt="mascot" style="height: 160px; margin-right: 0px;">
  <img src="assets/g05_logo.png" alt="logo" style="height: 160px;">
</div>


## Recent developments

[2026 year 6 month 29] We add ** G0.5** RoboTwin 2.0 assessment support, including assessment entrance and `g05-robotwin20` [Weights](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-robotwin20). More simulation benchmark evaluation features will be updated soon.

[17 June 2026] Released `g05-so101` [weights](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-so101) for zero-shot G0.5 deployment on so-100/101 robots.

[16 June 2026] Added zero-shot G0.5 deployment on R1 Lite and DROID, LIBERO simulation evaluation, and post-training support for R1 Lite/R1 Pro.

[2026 year 6 month 1] We released ** G0.5**, the latest self-return VLA model with leading performance. Please see [Project home page](https://opengalaxea.github.io/G05/).

[2026 year 2 month 12] Updates ** G0Plus** pre-training weights trained on more large-scale remote operations and web-page data. Release ** G0Tiny** (250 M, SmolVLM2 backbone network) for deployment on the edge of R1 Pro Orin. New open box for demonstration tasks: ** Fold Towels** and ** Handover Gift** (run G0Tiny reasoning at the end of the device through TensorRT, maximum 10 Hz). Add ** pi0/pi0fast** fine tune support based on [openpi](https://github.com/Physical-Intelligence/openpi).

[2026 year 1 month 4] We released ** G0Plus**, the latest pre-trained VLA model for multi-mission robotics.

[7 October 2025] The Galaxea Open World Dataset is available in LeRobot format on [Hugging Face](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset).

[2025 year 9 month 17] Release G0-VLA fine-tuned and real robotic reasoning codes.

[9 September 2025] G0-VLA pretrained weights released on [Hugging Face](https://huggingface.co/OpenGalaxea/G0-VLA) and [ModelScope](https://www.modelscope.cn/models/Galaxea/G0-VLA).

[2025 Year 9 Day 9] Launch Galaxea Open World Data Sets at [Hugging Face](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) and [ModelScope](https://www.modelscope.cn/datasets/Galaxea/Galaxea-Open-World-Dataset)!


## G0.5 Overview

** G0.5** is the Galaxea pre-training for universal robotic control self-return visual-linguistic-action model. G0.5 does not use VLM only as a visual language encoder for independent action experts, but rather maintains VLM as an implementer: a single transformer decoder generates reasoning token and action token in a unified self-regression stream and uses the same next-token prediction target.

G0.5 Technical Report's Core Approach:

1. ** Unified self-return VLA**
   - G0.5 enters it under multi-perspective RGB observations, physical identification, natural language mission commands and robotic body state.
   - Models can be output-based and then continue output-structured actions token in the same generation stream.
   - Action token is decoded as continuous robotic control, implemented in the form of chunk, and closed-ring re-engineering based on new observations.

2. ** Crossset ActionCodec**
   - The isomer robot action is mapped to the shared 27 dimension action space:
     `left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1) | lower_body(7)`。
   - Learning residual vector-quantized action tokenizer indicates semantic alignment of sports groups such as left arm, right arm and lower body.
   - Only active sports groups are produced when actions are generated, avoiding creating unnecessary padding for idle freedom.

3. ** Controlled Primary chain-of-thought**
   - G0.5 trains reasoning and action as the same token sequence, rather than treating reasoning as an external module or as a supplementary objective for training only.
   - chain-of-thought Snippets can contain `Subtask`, `BBox`, `Trace` and `ActionHint` fields for task decomposition, target positioning, 2-D clamp track and frame level motion tips.
   - This design enhances the ability of grounding and long-range execution because the resulting reasoning is directly visible to subsequent action token.

4. ** Visual memory **
   - G0.5 Injected multisecond visual history through visual encoders and decomposition time and space attention.
   - Pre-trained to use the 6 frame sampled in the 5 second window, and to reduce alignment by random historical frame dropout.
   - History token is discarded on the last floor to keep the reasoning delayed.

5. ** Pre-training and Evaluation **
   - G0.5 Initializes from Qwen3.5 2 B VLM and pre-trained in robotic demonstration data from 14 prototypes as well as large web pages and body VQA data.
   - The robotic action sample and the VQA sample are optimized using the same cross-breath target and are mixed by VQA-to-action ratio of 1:4.
   - Reported results include 82.5% zero-shot success on DROID, 87.3% on Bridge-SimplerEnv, 93.3% on RoboTwin 2.0, 98.9% on LIBERO, a BEHAVIOR-1K task-success score of 0.3136, and 76.7% average success after real-world fine-tuning on R1-Lite/R1-Pro.

<p align="center">
  <img src="assets/tokenizer.png" alt="G0.5 Tokenizer" width="700"/>
  <br>
  <em>G0.5 Tokenizer</em>
</p>

<p align="center">
  <img src="assets/token_template.png" alt="G0.5 Token Sequence Template" width="700"/>
  <br>
  <em>G0.5 Token Sequence Template</em>
</p>

## G0 / G0Plus Overview

This section retains an earlier overview of G0Plus in GalaxeaVLA to facilitate the tracing of project history. G0Plus and G0Tiny belong to the old G0/G0Plus distribution line; The main elements of the current branch have been switched to G0.5 and no longer contain the old version G0Plus Dockerfile, Fold Towels/Handover Gift presentation guide or pi0/pi0fast training configuration. For the previous README and file layout, see commit [13a16a9](https://github.com/OpenGalaxea/GalaxeaVLA/tree/13a16a9049aee8f1d799b56fccc0c5832a75fc2f). Historical weights remain in [G0-VLA Hugging Face repository](https://huggingface.co/OpenGalaxea/G0-VLA), including:

   - ** G0Plus_3B-base**: Training in real-world robotic data using ** 2 k+ hours ** for custom task fine-tuning.
   - **G0Tiny_250M-base:** a lightweight pretrained model trained on **1k hours** of R1 Pro VR teleoperation data for deployment on R1 Pro Orin.
   - ** G0Plus_3B-pick_and_place**: Post-training weights for pick-and-place deployment.
   - Demos of Pick Up Anything, Fold Towels, and Handover Gift.
   - ** pi0/pi0fast** fine tune based on openpi.

## Galaxea Open World Data Set

### ** Primary feature **

- ** 500 + hour ** Real World Mobile Operating Data.
- All data are collected using the ** unified robotic body ** to ensure consistency.
- ** sub-tasks of fine particle size are marked with **.
- Covers residential, kitchen, retail, and office settings.
- The data set provides ** RLDS** and ** LeRobot** formats.

For more data set formats and examples see [Here.](docs/data/schema.md) for details.


## G0.5 Quick start

### GPU request

To run the G0.5 pre-training model in this repository, NVIDIA GPU is required to meet at least the following specifications. These estimates are based on single GPU.

Multi-GPU fine-tuning launched through [scripts/run/finetune.sh](scripts/run/finetune.sh) uses `torchrun` distributed data parallelism by default. It lifts throughput, but each GPU keeps a complete copy of the model, so it should not be described as a parallel model, nor should it lower the single-card model display. Multi-machine operation is controlled by environmental variables such as `WORLD_SIZE`, `RANK`, `MASTER_ADDR` and `MASTER_PORT`.

| Mode               | Visibility Requirements | Example GPU                 |
| ------------------ | -------- | ------------------------ |
| Arguments               | > 8 GB   | RTX 3090 / ** RTX 4090 (recommended) ** |
| Full fine           | > 70 GB  | A100 (80GB) / H20 (96GB) |

### Install

Tested environment:

- Linux
- Python `>=3.10.16,<3.11`
- CUDA 12.8 and PyTorch 2.7.1 `cu128` wheel
- Primary CUDA extensions such as `flash-attn-4`, `flash-linear-attention`; Make sure CUDA runtime, compile the tool chain and PyTorch match
- Video data sets and assessment requirements `ffmpeg`

```bash
git clone https://github.com/OpenGalaxea/GalaxeaVLA
cd GalaxeaVLA

# Operating environment
uv sync --index-strategy unsafe-best-match

# If dependent on development and testing, change
uv sync --extra dev --index-strategy unsafe-best-match

source .venv/bin/activate
```

Before installation, note:

1. We suggest [Install uv](https://docs.astral.sh/uv/getting-started/installation/) without using conda environment.
2. If a network problem is encountered, it is suggested that the following uv environment variable be tried:
   ```bash
   export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
   export UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download
   ```


### Model Checkpoint

Please download the Hugging Face repository to `checkpoints/` to keep the local path consistent with the current configuration:

```bash
huggingface-cli download OpenGalaxea/G05 \
    --repo-type model \
    --local-dir checkpoints \
    --local-dir-use-symlinks False
```

Synchronising folder

```text
checkpoints/
├── action_tokenizer.pt
├── qwen3_5_2b_base_processor/
├── g05-base/
│   ├── .hydra/config.yaml
│   ├── checkpoints/model_state_dict.pt
│   └── dataset_stats.json
├── g05-so101/
│   ├── .hydra/config.yaml
│   ├── checkpoints/model_state_dict.pt
│   └── dataset_stats.json
├── g05-droid/
│   ├── .hydra/config.yaml
│   ├── checkpoints/model_state_dict.pt
│   └── dataset_stats.json
├── g05-libero/
│   ├── .hydra/config.yaml
│   ├── model.pt
│   └── dataset_stats.json
└── g05-robotwin20/
    ├── .hydra/config.yaml
    ├── checkpoints/model_state_dict.pt
    └── dataset_stats.json
```

When containing RoboTwin checkpoint, the full weight is approximately 55 GB. Each G0.5 model check point is about 11 GB, shared `action_tokenizer.pt` is about 484 MB, `qwen3_5_2b_base_processor/` is about 22 MB.

| Model | Use | Local `--ckpt_path` |
| ---- | ---- | ------------------ |
| [G05-base](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-base) | fine-tuning of R1 Lite sample deployment | `checkpoints/g05-base/checkpoints/model_state_dict.pt` |
| [G05-so101](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-so101) | SO-100/101 Zero sample deployment | `checkpoints/g05-so101/checkpoints/model_state_dict.pt` |
| [G05-droid](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-droid) | DROID Zero sample deployment | `checkpoints/g05-droid/checkpoints/model_state_dict.pt` |
| [G05-libero](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-libero) | LIBERO assessment | `checkpoints/g05-libero/model.pt` |
| [G05-robotwin20](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-robotwin20) | RoboTwin 2.0 Assessment | `checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt` |

Please keep `.hydra/config.yaml` and `dataset_stats.json` next to each checkpoint. The reasoning and evaluation portal will resolve `dataset_stats.json` up from the checkpoint parent directory, configure the default use of `checkpoints/qwen3_5_2b_base_processor` as share processor and use `checkpoints/action_tokenizer.pt` as share action tokenizer.

The LIBERO export configuration will refer to bundle internal sidecar. The `eval_libero.sh` command below relies on these paths. If the download tool does not generate these files, create a soft link:

```bash
ln -sf ../action_tokenizer.pt checkpoints/g05-libero/action_tokenizer.pt
ln -sfn ../qwen3_5_2b_base_processor checkpoints/g05-libero/hf_processor
```


### Real robotic reasoning.

Real robot deployment uses a service/client structure. The GPU machine runs the G0.5 policy service of this repository, collects original observations from the robotic client, sends observations through WebSocket + msgpack, receives action chunk and executes them.

#### R1 Lite

Start policy service on GPU machine:

```bash
python scripts/serve_policy.py \
    --ckpt_path checkpoints/g05-base/checkpoints/model_state_dict.pt \
    --host 0.0.0.0 \
    --port 8080 \
    --device cuda \
    --action_steps 16 \
    eval_embodiment=galaxea_r1lite
```

After setting the service address in `experiments/r1lite/config.toml`, start the R1 Lite client on the robot machine:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
cd experiments/r1lite
python run.py --config config.toml
```

ROS2 topics, client/service-end protocols, command files, recording and testing details can be found in [experiments/r1lite/README.md](experiments/r1lite/README.md).

#### SO-100 / SO-101

SO-100/101 deploys with the same G0.5 policy service and with a lightweight LeRobot client. Start the strategy service on GPU:

```bash
bash experiments/so100/start_server.sh checkpoints/g05-so101/checkpoints/model_state_dict.pt
```

Then configure and activate the client on another terminal or robotic machine:

```bash
conda env create -f experiments/so100/environment.yml
conda activate lerobot
bash experiments/so100/start_client.sh
```

Before running the client, update the camera index and camera-slot map in `experiments/so100/start_client.sh`. See [experiments/so100/README.md](experiments/so100/README.md) for details.

#### DROID / Franka

DROID deployed to provide G0.5 tactical services from this warehouse and using independent
[OpenGalaxea/droid-franka-client](https://github.com/OpenGalaxea/droid-franka-client)
Warehouse control robots. Start the strategy service on GPU:

```bash
CHECKPOINT_DIR=checkpoints/g05-droid \
POLICY_PORT=8000 \
POLICY_DEVICE=cuda:0 \
bash experiments/droid/start_server.sh \
    model.model_arch.discrete_action=true model.model_arch.continuous_action=false
```

Then clone and configure robotic client:

```bash
git clone git@github.com:OpenGalaxea/droid-franka-client.git
```

Full settings and protocols are agreed in [experiments/droid/README.md](experiments/droid/README.md) and [experiments/droid/PROTOCOL.md](experiments/droid/PROTOCOL.md).

### LIBERO assessment

LIBERO evaluates the same service/client settings. This command activates the batch processing strategy service in the backstage, starts a parallel client for each LIBERO suite and stops the service after the evaluation.

```bash
bash scripts/run/eval_libero.sh checkpoints/g05-libero/model.pt \
    --num_trials 50 \
    --num_parallel 10 \
    --save_videos
```

By default, scripts evaluate `libero_goal`, `libero_spatial`, `libero_object` and `libero_10`, and then write `outputs/libero_eval_<checkpoint_name>/` logs and `summary.json` to each suite. Use `--suites "libero_goal libero_10"` to run only a subset; Hydra-style overrides can be added, for example `model.model_arch.discrete_action=false`.

The LIBERO installation description, generated path configuration, output layout and single suite debug command can be found in [experiments/libero/README.md](experiments/libero/README.md).

### RoboTwin assessment

The RoboTwin assessment is run in the RoboTwin emulator and requires additional simulations and assets. It uses a separate `.venv-robotwin` environment to avoid the `sapien` 3. x required for the emulator to cover the main warehouse environment. Follow [the RoboTwin guide](experiments/robotwin/README_zh.md) to clone RoboTwin, download assets, create `.venv-robotwin`, and verify rendering.

Full RoboTwin job evaluation command:

```bash
.venv-robotwin/bin/python -u experiments/robotwin/run_robotwin_manager.py \
    task=robotwin \
    ckpt=checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt \
    EVALUATION.robotwin_root=third_party/RoboTwin \
    MULTIRUN.num_gpus=8 \
    MULTIRUN.max_tasks_per_gpu=1
```

This command evaluates all tasks in RoboTwin `_eval_step_limit.yml` using the episode number in the configuration. Under the standard `g05-robotwin20` directory structure, `dataset_stats.json` automatically resolves up from checkpoint parent directory; Only stats files need `EVALUATION.dataset_stats_path=/path/to/dataset_stats.json` if they are in another location. The sum result is written to `evaluate_results/robotwin/<ckpt_tag>/<timestamp>/`, RoboTwin rendering video is kept on `third_party/RoboTwin/eval_result/`.

### Zulu, fine-tune the base model on a Galaxea robot.

If you need to use your own data fine-tuning model, do the following four steps. The configuration level and common change position are [configs/QUICK_START.md](configs/QUICK_START.md).

1. Create or adjust task config under `configs/task/` and update the corresponding data set path under `configs/data/`. Galaxea robot fine tunes can start with the following configuration:
   - R1 Lite: [configs/task/r1lite.yaml](configs/task/r1lite.yaml), the data set path is updated in [configs/data/r1lite.yaml](configs/data/r1lite.yaml).
   - R1 Pro joint space training: [configs/task/r1pro.yaml](configs/task/r1pro.yaml), data set path updated in [configs/data/r1pro.yaml](configs/data/r1pro.yaml).
   - R1 Pro training for WBC with body state/action: [configs/task/r1pro_wbc.yaml](configs/task/r1pro_wbc.yaml), data set path updated in [configs/data/r1pro_wbc.yaml](configs/data/r1pro_wbc.yaml).

   All three types of Galaxea robotic data configuration include examples of local data paths that must be replaced before training:
   - R1 Lite: Replace `data/r1lite/task*_lerobot` in `configs/data/r1lite.yaml`.
   - R1 Pro: Replace `data/r1pro/fold_carton_lerobot` in `configs/data/r1pro.yaml`.
   - R1 Pro WBC: Replace `data/r1pro_wbc/stack_box_lerobot` in `configs/data/r1pro_wbc.yaml`.

   These paths should point to your local LeRobot data set catalogue.

2. Installation of required software packages.

   ```bash
   sudo apt install ffmpeg
   ```

3. Sets the environment variable.
    - `G05_OUTPUT_DIR`: `configs/train.yaml` is required; Checkpoint and Hydra log will be written to this directory.
    - `HF_HOME`/ `HF_HUB_CACHE`: Recommended cache position for the Hugging Face model and tokenizer snapshot.
    - `HF_DATASETS_CACHE`: Recommended cache position for Hugging Face dataset and LeRobot metadata.
    - `LIBERO_CONFIG_PATH`: Only LIBERO simulation needs. It points to a directory where LIBERO reads or writes `config.yaml`, which contains `bddl_files`, `init_states`, `assets` and data set paths.
    - `SWANLAB_API_KEY`: Only `logger.type=swanlab` is required.
    - `WANDB_API_KEY`: Only `logger.type=wandb` and `logger.mode=online` are required.

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HOME=<YOUR_HF_CACHE_ROOT>
    export HF_HUB_CACHE=$HF_HOME/hub
    export HF_DATASETS_CACHE=$HF_HOME/datasets
    export G05_OUTPUT_DIR=<YOUR_OUTPUT_DIR>
    export LIBERO_CONFIG_PATH=$(pwd)/experiments/libero
    export SWANLAB_API_KEY=<YOUR_SWANLAB_API_KEY>
    ```

    Copy repository local environment templates and fill out machine-related paths and optional API key:

    ```bash
    cp .env.example .env
    source .env
    ```

4. Run fine-tuning.

   ```bash
   bash scripts/run/finetune.sh <num_of_gpu> <task_path>

   # dry-run：Parsing configuration and exiting without training
   bash scripts/run/finetune.sh 1 r1pro --dry-run --max_datasets 1

   # Single card smoke test
   bash scripts/run/finetune.sh 1 r1pro --test --max_datasets 1 model.max_steps=<num_steps>

   # More GPU Example:
   bash scripts/run/finetune.sh 8 r1lite
   bash scripts/run/finetune.sh 8 r1pro
   bash scripts/run/finetune.sh 8 r1pro_wbc
   ```

   R1 Pro configuration using grouped 27 D ActionCodec layout:
   `left_control(9) | left_gripper(1) | right_control(9) | right_gripper(1) | lower_body(7)`。
   WBC version will merge `torso` with [configs/data/parts_meta/r1pro.yaml](configs/data/parts_meta/r1pro.yaml) to the `lower_body` group.

#### Align FAQ

1. How does Q: convert my data to [LeRobot](https://github.com/huggingface/lerobot) data sets?

   A: We provided [Presentation Data Set](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_Finetune_LeRobot_Datasets_Demo) on Hugging Face for rapid testing.

2. Q: Why can't you see the training log in SwanLab?

   A: Set your own SwanLab `workspace` in [train.yaml](configs/train.yaml).

3. Q:. Why can't you find a pre-training model?

   The A: G0.5 model configuration is [g05.yaml](configs/model/g05.yaml). Update `model.model_arch.pretrained_model_path` or task level `model.pretrained_ckpt` if the checkpoint is in a custom path.

4. Why does Q: have out-of-memory (OOM) error)?

   A: please confirm that you have sufficient GPU visibility as mentioned above. Or reduce `model.batch_size` in the corresponding task config under [configs/task](configs/task).

### Minimum Authentication

After installing and downloading checkpoint, it is recommended to run a light check before starting a high-cost task:

```bash
python tools/resolve_config.py r1pro --key model.model_arch
python tools/resolve_config.py r1pro --key data.embodiment_datasets.galaxea_r1pro.dataset_groups
python -c "import torch, g05; print(torch.__version__); print(g05.__file__)"
```

Tests are differentiated by operational dependency. No real robotic data are required for synthetic data/unit testing; serving, pretrained-policy and dataset tests require CUDA, downloaded checkpoint or local LeRobot data sets. The LIBERO volume assessment also relies on the Linux `ss` command because [scripts/run/eval_libero.sh](scripts/run/eval_libero.sh) will be used to wait for the policy server port to be ready.

## Acknowledgement

The project builds on existing work in open-source communities. This was inspired by [open-pi-zero](https://github.com/allenzren/open-pi-zero), [OpenVLA](https://github.com/openvla/openvla), [Octo](https://github.com/octo-models/octo), [OpenPI](https://github.com/Physical-Intelligence/openpi) and [LeRobot](https://github.com/huggingface/lerobot), using data sets including [OXE](https://github.com/google-deepmind/open_x_embodiment), [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [BridgeV2](https://github.com/rail-berkeley/bridge_data_v2) and [DROID](https://github.com/droid-dataset/droid). We sincerely thank the authors of these projects for their public codes and data.


## Reference

If using our data sets or models, please refer to:

```bibtex
@article{galaxea2026g05,
  title={Galaxea G0.5 Technical Report},
  author={Galaxea Team},
  year={2026},
  url={https://opengalaxea.github.io/G05/Galaxea_G0_5.pdf}
}
```

## Licence

This warehouse contains materials subject to different licences according to the date of submission:
- Apache-2.0 (historical version): All the elements submitted before 2026-01-04 are based on Apache License 2.0 authorization.
- G0 PLUS Community License Agreement: content committed on or after 2026-01-04 and before 2026-06-16 is licensed under the G0 PLUS Community License (Non-Commercial + Limited Patent License). See [G0 Plus Community License Agreement](licenses/LICENSE-G0Plus) for details.
- G0.5 Community License Agreement: All submissions of 2026-06-16 made on or after that day are authorized on the basis of G0.5 Community License (Non-Commercial + Limited Patent License). See [G0.5 Community License Agreement](licenses/LICENSE-G0.5) for details.

In order to avoid doubt, there are two licence boundaries, each of which is determined by the introduction of the corresponding commit for the first time:
- Boundary 1 (Apache-2.0 / G0 PLUS): commit under G0 PLUS licence (include G0 PLUS licence switching): [38b31e4](https://github.com/OpenGalaxea/GalaxeaVLA/tree/38b31e4f732ef28719a5458a18e2836dd52f9d12)
- Border 2 (G0 PLUS / G0.5): First commit under G0.5 license (include G0.5 licence switch): [dc0a1ef](https://github.com/OpenGalaxea/GalaxeaVLA/tree/dc0a1ef4531256adea4ee9f3d7d2fa44613cb866)

If there is any inconsistency between the date description and commit hash, then commit hash.

### Use allowed under G0.5 Community License

You can use, reproduce, modify and distribute G0.5 materials for non-commercial purposes only, such as academic research, personal use, education and evaluation. Commercial use (including production deployment, provision of services to third parties or productization) requires a separate commercial licence from us.

G0.5 materials include model codes, weights, configurations, training/adjection scripts, documents and supporting materials. See [LICENSE-G0.5](./LICENSE-G0.5) and [licenses/LICENSE-G0.5](licenses/LICENSE-G0.5) for further details.

### Declaration and signature

If any part of the G0.5 material is redistributed, it must include:
- Copies of or links to G0.5 Community License Agreement;
- NOTICE files in this warehouse;
- Visible changes to the revised document.

### Third-party licences

G0.5 uses Qwen3.5 as a pre-trained VLM backbone model and contains Qwen3.5 derived realization components. Qwen3.5 model weights, configuration files and related upstream materials are authorized by Qwen based on Apache License 2.0. See [LICENSE_QWEN3_5.txt](./LICENSE_QWEN3_5.txt) for details.

### Legal and security compliance

The publication of this warehouse does not in itself constitute a generating artificial intelligence service for the public. If you deploy, fine-tune, re-distribut or expose to external exposure any model, service, output or robotic control process, you need to ensure compliance with applicable laws and regulations, including data rights, personal information protection, security assessment or filing, content security, synthetic content identification and robotic safety.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenGalaxea/GalaxeaVLA&type=date&legend=top-left)](https://www.star-history.com/#OpenGalaxea/GalaxeaVLA&type=date&legend=top-left)
