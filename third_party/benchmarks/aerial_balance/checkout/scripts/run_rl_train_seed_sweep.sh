#!/usr/bin/env bash
set -uo pipefail

# Sequentially train RL policies with different random seeds.
#
# Common usage:
#   bash scripts/run_rl_train_seed_sweep.sh
#
# Useful overrides:
#   NUM_SEEDS=5 bash scripts/run_rl_train_seed_sweep.sh
#   SEEDS="0 1 2" bash scripts/run_rl_train_seed_sweep.sh
#   NUM_ENVS=256 MAX_ITERATIONS=300 DEVICE=cuda:0 bash scripts/run_rl_train_seed_sweep.sh
#   EXTRA_ARGS="--headless" bash scripts/run_rl_train_seed_sweep.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-baselines/configs/rl_target_position_rpo_train.yaml}"
LOG_ROOT="${LOG_ROOT:-logs/rl_train/rl_train_seed_sweeps}"
RUN_PREFIX="${RUN_PREFIX:-rpo_target_position_legacy8}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEED_MIN="${SEED_MIN:-0}"
SEED_MAX="${SEED_MAX:-1000000}"
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
DEVICE="${DEVICE:-}"
HEADLESS="${HEADLESS:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${LOG_ROOT}"

sample_unique_seeds() {
    if [[ -n "${SEEDS:-}" ]]; then
        printf "%s\n" ${SEEDS}
        return
    fi

    if command -v shuf >/dev/null 2>&1; then
        shuf -i "${SEED_MIN}-${SEED_MAX}" -n "${NUM_SEEDS}"
        return
    fi

    python3 - <<PY
import random
seed_min = int("${SEED_MIN}")
seed_max = int("${SEED_MAX}")
num_seeds = int("${NUM_SEEDS}")
if seed_max - seed_min + 1 < num_seeds:
    raise SystemExit("Seed range is smaller than NUM_SEEDS.")
for seed in random.sample(range(seed_min, seed_max + 1), num_seeds):
    print(seed)
PY
}

mapfile -t RAW_SEEDS < <(sample_unique_seeds)

declare -A SEEN_SEEDS=()
SEED_LIST=()
for seed in "${RAW_SEEDS[@]}"; do
    [[ -z "${seed}" ]] && continue
    if [[ -z "${SEEN_SEEDS[${seed}]+x}" ]]; then
        SEED_LIST+=("${seed}")
        SEEN_SEEDS["${seed}"]=1
    fi
done

if [[ "${#SEED_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] No seeds to run. Set SEEDS or NUM_SEEDS." >&2
    exit 1
fi

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Log root: ${LOG_ROOT}"
echo "[INFO] Seeds: ${SEED_LIST[*]}"

for seed in "${SEED_LIST[@]}"; do
    timestamp="$(date +%Y%m%d_%H%M%S)"
    run_name="${RUN_PREFIX}_seed_${seed}"
    log_file="${LOG_ROOT}/${run_name}_${timestamp}.log"

    cmd=(python3 scripts/rl_train.py --config "${CONFIG}" --seed "${seed}" --run_name "${run_name}")
    if [[ -n "${NUM_ENVS}" ]]; then
        cmd+=(--num_envs "${NUM_ENVS}")
    fi
    if [[ -n "${MAX_ITERATIONS}" ]]; then
        cmd+=(--max_iterations "${MAX_ITERATIONS}")
    fi
    if [[ -n "${DEVICE}" ]]; then
        cmd+=(--device "${DEVICE}")
    fi
    if [[ "${HEADLESS}" == "1" || "${HEADLESS}" == "true" || "${HEADLESS}" == "True" ]]; then
        cmd+=(--headless)
    fi
    if [[ -n "${EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        extra_args_array=(${EXTRA_ARGS})
        cmd+=("${extra_args_array[@]}")
    fi

    echo "[INFO] Starting seed ${seed}. Log: ${log_file}"
    echo "[INFO] Command: ${cmd[*]}" > "${log_file}"
    echo "[INFO] Started at: $(date --iso-8601=seconds)" >> "${log_file}"

    "${cmd[@]}" >> "${log_file}" 2>&1
    status=$?

    echo "[INFO] Finished at: $(date --iso-8601=seconds)" >> "${log_file}"
    echo "[INFO] Exit status: ${status}" >> "${log_file}"

    if [[ "${status}" -ne 0 ]]; then
        echo "[ERROR] Seed ${seed} failed with status ${status}. See ${log_file}" >&2
        if [[ "${STOP_ON_FAILURE}" == "1" || "${STOP_ON_FAILURE}" == "true" || "${STOP_ON_FAILURE}" == "True" ]]; then
            exit "${status}"
        fi
    else
        echo "[INFO] Seed ${seed} completed successfully."
    fi
done

echo "[INFO] All requested RL training runs finished."
