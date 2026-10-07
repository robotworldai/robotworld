#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/logs/rl/rpo_models}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/logs/rl/rpo_results}"
SUMMARY_FILE="${RESULTS_ROOT}/eval_status.tsv"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-5}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-3m}"
TIMEOUT_KILL_AFTER="${TIMEOUT_KILL_AFTER:-5s}"
CLEANUP_AFTER_RUN="${CLEANUP_AFTER_RUN:-true}"
CLEANUP_GRACE_PERIOD="${CLEANUP_GRACE_PERIOD:-2s}"
TARGET_EPISODES_VALUE="${TARGET_EPISODES:-config_default}"
NUM_ENVS_VALUE="${NUM_ENVS:-config_default}"

if [[ -n "${CONFIG:-}" ]]; then
    CONFIGS=("${CONFIG}")
else
    CONFIGS=(
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_delay_free.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_delay_wo_comp.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_delay_w_comp.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_mass.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_gain.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_disturbance.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_delay_30.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_delay_45.yaml"
        "${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_delay_60.yaml"
    )
fi

mkdir -p "${RESULTS_ROOT}"

cleanup_process_group() {
    local pgid="$1"
    if [[ "${CLEANUP_AFTER_RUN}" != "true" ]]; then
        return
    fi
    if [[ -z "${pgid}" ]]; then
        return
    fi
    if kill -0 "-${pgid}" 2>/dev/null; then
        echo "[INFO] Cleaning remaining processes in process group ${pgid}"
        kill -TERM "-${pgid}" 2>/dev/null || true
        sleep "${CLEANUP_GRACE_PERIOD}"
        kill -KILL "-${pgid}" 2>/dev/null || true
    fi
}

for arg in "$@"; do
    if [[ "${arg}" == "--episodes" || "${arg}" == --episodes=* ]]; then
        echo "[ERROR] Do not pass --episodes directly to this sweep script. Use TARGET_EPISODES=<positive-int> instead." >&2
        exit 1
    fi
    if [[ "${arg}" == "--num_envs" || "${arg}" == --num_envs=* ]]; then
        echo "[ERROR] Do not pass --num_envs directly to this sweep script. Use NUM_ENVS=<positive-int> instead." >&2
        exit 1
    fi
done

EVAL_OVERRIDES=()
if [[ -n "${TARGET_EPISODES:-}" ]]; then
    if ! [[ "${TARGET_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] TARGET_EPISODES must be a positive integer, got '${TARGET_EPISODES}'." >&2
        exit 1
    fi
    EVAL_OVERRIDES+=(--episodes "${TARGET_EPISODES}")
fi
if [[ -n "${NUM_ENVS:-}" ]]; then
    if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] NUM_ENVS must be a positive integer, got '${NUM_ENVS}'." >&2
        exit 1
    fi
    EVAL_OVERRIDES+=(--num_envs "${NUM_ENVS}")
fi

mapfile -t CHECKPOINTS < <(find "${MODEL_ROOT}" -mindepth 3 -maxdepth 3 -type f -path "*/checkpoints/best_agent.pt" | sort)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "[ERROR] No best_agent.pt checkpoints found under: ${MODEL_ROOT}" >&2
    exit 1
fi

for idx in "${!CONFIGS[@]}"; do
    if [[ "${CONFIGS[$idx]}" != /* ]]; then
        CONFIGS[$idx]="${PROJECT_ROOT}/${CONFIGS[$idx]}"
    fi
    if [[ ! -f "${CONFIGS[$idx]}" ]]; then
        echo "[ERROR] Config file does not exist: ${CONFIGS[$idx]}" >&2
        exit 1
    fi
done

printf "config_name\tconfig_path\trun_name\tcheckpoint\ttarget_episodes\tnum_envs\tstatus\tlog_file\n" > "${SUMMARY_FILE}"

overall_status=0
echo "[INFO] Found ${#CHECKPOINTS[@]} checkpoint(s). Results will be written to: ${RESULTS_ROOT}"
echo "[INFO] Found ${#CONFIGS[@]} config(s)."
echo "[INFO] TARGET_EPISODES=${TARGET_EPISODES_VALUE}; NUM_ENVS=${NUM_ENVS_VALUE}"

for config in "${CONFIGS[@]}"; do
    config_name="$(basename "${config}" .yaml)"
    config_results_root="${RESULTS_ROOT}/${config_name}"
    config_summary_file="${config_results_root}/eval_status.tsv"
    mkdir -p "${config_results_root}"
    printf "config_name\tconfig_path\trun_name\tcheckpoint\ttarget_episodes\tnum_envs\tstatus\tlog_file\n" > "${config_summary_file}"

    echo "[INFO] Starting config: ${config_name}"
    echo "[INFO] Config path: ${config}"

    for checkpoint in "${CHECKPOINTS[@]}"; do
        model_dir="$(dirname "$(dirname "${checkpoint}")")"
        run_name="$(basename "${model_dir}")"
        log_file="${config_results_root}/${run_name}.log"

        echo "[INFO] Evaluating ${run_name}"
        echo "[INFO] Config: ${config_name}"
        echo "[INFO] Checkpoint: ${checkpoint}"
        echo "[INFO] Log file: ${log_file}"
        echo "[INFO] TARGET_EPISODES=${TARGET_EPISODES_VALUE}; NUM_ENVS=${NUM_ENVS_VALUE}"
        echo "[INFO] Eval timeout: ${EVAL_TIMEOUT} (kill after ${TIMEOUT_KILL_AFTER})"

        PYTHONUNBUFFERED=1 setsid timeout --kill-after="${TIMEOUT_KILL_AFTER}" "${EVAL_TIMEOUT}" \
            "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/rl_policy_eval.py" \
            --config "${config}" \
            --checkpoint "${checkpoint}" \
            --run_name "${run_name}" \
            "${EVAL_OVERRIDES[@]}" \
            "$@" > "${log_file}" 2>&1 &
        eval_pid=$!
        wait "${eval_pid}"
        status=$?
        cleanup_process_group "${eval_pid}"

        if [[ ${status} -eq 0 ]]; then
            echo "[INFO] Finished ${run_name} (${config_name})"
            status_text="ok"
        elif [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
            echo "[ERROR] Evaluation timed out or was killed for ${run_name} (${config_name}) after ${EVAL_TIMEOUT}" >&2
            status_text="timeout_or_killed:${status}"
            overall_status=${status}
        else
            echo "[ERROR] Evaluation failed for ${run_name} (${config_name}) with exit code ${status}" >&2
            status_text="failed:${status}"
            overall_status=${status}
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${config_name}" "${config}" "${run_name}" "${checkpoint}" \
            "${TARGET_EPISODES_VALUE}" "${NUM_ENVS_VALUE}" "${status_text}" "${log_file}" >> "${SUMMARY_FILE}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${config_name}" "${config}" "${run_name}" "${checkpoint}" \
            "${TARGET_EPISODES_VALUE}" "${NUM_ENVS_VALUE}" "${status_text}" "${log_file}" >> "${config_summary_file}"

        echo "[INFO] Sleeping ${SLEEP_BETWEEN_RUNS}s before the next evaluation to let Isaac/driver resources settle."
        sleep "${SLEEP_BETWEEN_RUNS}"
    done
done

echo "[INFO] Evaluation sweep complete. Status summary: ${SUMMARY_FILE}"
exit "${overall_status}"
