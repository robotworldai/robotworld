#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/g1_flat"

RUN_NAME="${RUN_NAME:-${1:-test}}"
NUM_ENVS="${NUM_ENVS:-8192}"
MIMIC_MAX_ITERATIONS="${MIMIC_MAX_ITERATIONS:-4000}"

cd "${REPO_ROOT}"

python scripts/rsl_rl/train_multi.py --task Tracking-Terrain-G1-RNN-v0 \
    --motion_path motions/soccer-standard \
    --run_name "${RUN_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MIMIC_MAX_ITERATIONS}" \
    --headless

LOAD_RUN="${LOAD_RUN:-$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)}"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve load_run from ${EXPERIMENT_DIR}"
    exit 1
fi

echo "Resolved load_run=${LOAD_RUN}"
echo "Run name=${RUN_NAME}"

python scripts/rsl_rl/train_multi.py --task Tracking-Flat-G1-SoccerMoving-RNN-v0 \
    --motion_path motions/soccer-standard \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --resume True \
    --headless
