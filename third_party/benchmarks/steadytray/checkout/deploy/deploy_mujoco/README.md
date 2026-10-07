# Deploy MuJoCo

Lightweight MuJoCo deployment for visualizing trained policies without requiring Isaac Sim or Isaac Lab.

## Setup

Install the required environment (skip if Isaac Lab is already installed):

```bash
conda env create -f environment.yml
conda activate g1_deploy
```

## Usage

Run the deployment script:

```bash
python deploy_mujoco.py --policy <POLICY_PATH> --config <CONFIG_PATH>
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--policy` | `exported/policy_9999.pt` | Path to the exported JIT policy file |
| `--config` | `deploy/configs/g1_29dof_walk.yaml` | Path to the robot configuration file |
| `--encoder_seq_len` | `32` | Encoder sequence length (for distillation policies) |

### Example

A pretrained exported policy is provided at `exported/policy_9999.pt`. Run it with:

```bash
python deploy_mujoco.py --policy exported/policy_9999.pt --config deploy/configs/g1_29dof_walk.yaml
```