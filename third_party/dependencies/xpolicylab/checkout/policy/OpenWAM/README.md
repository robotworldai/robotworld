# OpenWAM

**Contributor:** OpenWAM Contributors | **Paper:** An Open, Modular Exploration Towards Systematic World–Action Model Pretraining | **arXiv:** [2609.07398](https://arxiv.org/abs/2609.07398) | **Original code:** https://github.com/OpenWAM-Official/OpenWAM

`OpenWAM` adapts the OpenWAM world-action model to XPolicyLab/RoboDojo (`arx_x5`, absolute EE control, batched inference). Integration scripts live at this directory level; the vendored upstream implementation lives in `OpenWAM/`. Official OpenWAM does not expose batch inference; the vendored tree adds `generate_batch` for `eval_batch: true`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Create a Python >= 3.10 environment, then install the official PyTorch CUDA wheel before the adapter packages:

```bash
conda create -n openwam python=3.10
conda activate openwam
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

cd XPolicyLab/policy/OpenWAM
bash install.sh
```

`install.sh` editable-installs XPolicyLab and the vendored `OpenWAM/` package. Cosmos-Predict2.5 extras are not required for the RoboDojo Wan checkpoint.

## Data Processing

OpenWAM consumes native RoboDojo HDF5 (`<dataset_dir>/<task>/<embodiment>/data/episode_*.hdf5`). There is no LeRobot conversion.

```bash
cd XPolicyLab/policy/OpenWAM
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example
export OPENWAM_DATASET_DIR=/path/to/robodojo_data
bash process_data.sh RoboDojo cotrain arx_x5 ee
```

If `OPENWAM_DATASET_DIR` (or `policy/OpenWAM/data/<bench>-<ckpt>-<env>-<action>/`) already exists, the script exits. Otherwise it launches the official interactive downloader `OpenWAM/scripts/download_assets/download_benchmark_data.py` — select RoboDojo.

## Model Assets

Released RoboDojo SFT checkpoints are self-contained (`config.yaml`, `checkpoint_step_*.safetensors`, `normalization_stats.npy`). Download from [Hugging Face OpenWAM](https://huggingface.co/OpenWAM) or with the official helper:

```bash
cd XPolicyLab/policy/OpenWAM/OpenWAM
python scripts/download_assets/download_openwam_checkpoints.py
```

Place or symlink the checkpoint directory where eval can resolve it:

```bash
cd XPolicyLab/policy/OpenWAM
mkdir -p checkpoints
ln -sfn /path/to/New_OpenWAM_RoboDojo_SFT_60k checkpoints/New_OpenWAM_RoboDojo_SFT_60k
```

Training from scratch also needs the Wan video backbone (`python scripts/download_assets/download_video_backbone.py` inside `OpenWAM/`). Fine-tuning a released checkpoint uses `OPENWAM_FINETUNE_CKPT_PATH`.

## Training

```bash
cd XPolicyLab/policy/OpenWAM
export OPENWAM_DATASET_DIR=/path/to/robodojo_data

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: RoboDojo SFT on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 ee 0 0
```

This wraps official `OpenWAM/scripts/train.sh` with `dataloader=robodojo`. Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. Extra Hydra overrides go in `OPENWAM_TRAIN_OVERRIDES`.

## Evaluation

```bash
cd XPolicyLab/policy/OpenWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: offline wiring check (no simulator, no checkpoint load)
EVAL_ENV_TYPE=debug OPENWAM_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo stack_bowls dummy arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a released checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls New_OpenWAM_RoboDojo_SFT_60k arx_x5 ee 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. `ckpt_name` may be the short directory name, the full 5-tuple run directory, or an absolute path. Override with `OPENWAM_CKPT_DIR`. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `eval_batch`, `action_type`, `device`, `ckpt_dir`, `openwam_root`, `openwam_deploy_config`, `replan_steps`, `allow_dummy_policy`.

| Variable | Notes |
|---|---|
| `OPENWAM_DATASET_DIR` | Native RoboDojo HDF5 root for `process_data.sh` / `train.sh`. |
| `OPENWAM_CKPT_DIR` | Explicit eval checkpoint directory (`config.yaml` + safetensors). |
| `OPENWAM_ROOT` | Optional OpenWAM source override; defaults to the vendored `OpenWAM/`. |
| `OPENWAM_DEPLOY_CONFIG` | Optional deploy yaml override; defaults to `OpenWAM/configs/deploy.yaml`. |
| `OPENWAM_TRAIN_OVERRIDES` | Extra Hydra overrides forwarded to official `scripts/train.sh`. |
| `OPENWAM_FINETUNE_CKPT_PATH` | Warm-start directory for official `training.finetune_ckpt_path`. |
| `OPENWAM_RESUME_CKPT_PATH` | Resume directory for official `training.resume_ckpt_path`. |
| `OPENWAM_ALLOW_DUMMY_POLICY` | Debug-only: skip checkpoint load and return hold-position chunks. |

`eval_batch: true` stacks every running env into one `engine.generate_batch` forward. `model.py` re-forces `dit_cache` / `compile` / `decode_video` off and `inference_mode=sync`. `device: cuda` loads Wan, UMT5-XXL, and ActionDiT on GPU, matching official OpenWAM deploy.
