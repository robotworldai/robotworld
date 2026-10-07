#!/bin/bash
set -euo pipefail

# XPolicyLab-standard training wrapper for OpenWAM.
# Launches the official OpenWAM train.sh with the RoboDojo dataloader and
# writes checkpoints to the 5-tuple directory eval resolves by default:
#   <POLICY_DIR>/checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/
#
# Usage:
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
#
# Required:
#   OPENWAM_DATASET_DIR   native RoboDojo HDF5 root
# Optional:
#   OPENWAM_TRAIN_OVERRIDES   extra Hydra overrides (word-split)
#   OPENWAM_FINETUNE_CKPT_PATH / OPENWAM_RESUME_CKPT_PATH

bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_id=${6:?}

if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi

if [[ "${action_type}" != "ee" ]]; then
    echo "[OpenWAM] ERROR: OpenWAM RoboDojo is an EE-space policy; action_type must be 'ee', got '${action_type}'." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENWAM_ROOT="${OPENWAM_ROOT:-${POLICY_DIR}/OpenWAM}"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
dataset_dir="${OPENWAM_DATASET_DIR:-${POLICY_DIR}/data/${data_setting}}"

if [[ ! -d "${OPENWAM_ROOT}" ]]; then
    echo "[OpenWAM] ERROR: missing vendored OpenWAM at ${OPENWAM_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${dataset_dir}" ]]; then
    echo "[OpenWAM] ERROR: native RoboDojo dataset not found at ${dataset_dir}." >&2
    echo "[OpenWAM] Set OPENWAM_DATASET_DIR or run process_data.sh first." >&2
    exit 1
fi

mkdir -p "${ckpt_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NPROC_PER_NODE="${num_gpus}"

echo "[OpenWAM train] bench=${bench_name} ckpt=${ckpt_name} env=${env_cfg_type} action=${action_type} seed=${seed}"
echo "[OpenWAM train] dataset_dir=${dataset_dir}"
echo "[OpenWAM train] output_path=${ckpt_dir}"
echo "[OpenWAM train] gpus=${gpu_id} nproc_per_node=${num_gpus}"

train_args=(
    "dataloader=robodojo"
    "dataloader.dataset_dir=${dataset_dir}"
    "training.output_path=${ckpt_dir}"
    "project.seed=${seed}"
)

if [[ -n "${OPENWAM_FINETUNE_CKPT_PATH:-}" ]]; then
    train_args+=("training.finetune_ckpt_path=${OPENWAM_FINETUNE_CKPT_PATH}")
fi
if [[ -n "${OPENWAM_RESUME_CKPT_PATH:-}" ]]; then
    train_args+=("training.resume_ckpt_path=${OPENWAM_RESUME_CKPT_PATH}")
fi
if [[ -n "${OPENWAM_TRAIN_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206
    train_args+=(${OPENWAM_TRAIN_OVERRIDES})
fi

cd "${OPENWAM_ROOT}"
exec bash scripts/train.sh "${train_args[@]}"
