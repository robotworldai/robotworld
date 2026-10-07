#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/logs/nmpc/nmpc_eval_sweeps}"
SUMMARY_FILE="${RESULTS_ROOT}/eval_status.tsv"
TARGET_EPISODES="${TARGET_EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-12}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-60m}"
TIMEOUT_KILL_AFTER="${TIMEOUT_KILL_AFTER:-10s}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-10}"
CLEANUP_AFTER_RUN="${CLEANUP_AFTER_RUN:-true}"
CLEANUP_GRACE_PERIOD="${CLEANUP_GRACE_PERIOD:-3s}"

if [[ -n "${HORIZONS:-}" ]]; then
    read -r -a HORIZON_VALUES <<< "${HORIZONS}"
else
    HORIZON_VALUES=(20 30 40)
fi

if [[ -n "${CONFIG:-}" ]]; then
    CONFIGS=("${CONFIG}")
else
    CONFIGS=(
        "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_delay_free.yaml"
        "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_delay_wo_comp.yaml"
        "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_delay_w_comp.yaml"
        # "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_mass.yaml"
        # "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_gain.yaml"
        # "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_disturbance.yaml"
        # "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_delay_30.yaml"
        # "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_delay_45.yaml"
        # "${PROJECT_ROOT}/baselines/configs/nmpc_target_position_eval_delay_60.yaml"
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
    if [[ "${arg}" == "--n_horizon" || "${arg}" == --n_horizon=* ]]; then
        echo "[ERROR] Do not pass --n_horizon directly to this sweep script. Use HORIZONS=\"15 25 35\" instead." >&2
        exit 1
    fi
done

for idx in "${!CONFIGS[@]}"; do
    if [[ "${CONFIGS[$idx]}" != /* ]]; then
        CONFIGS[$idx]="${PROJECT_ROOT}/${CONFIGS[$idx]}"
    fi
    if [[ ! -f "${CONFIGS[$idx]}" ]]; then
        echo "[ERROR] Config file does not exist: ${CONFIGS[$idx]}" >&2
        exit 1
    fi
done

for horizon in "${HORIZON_VALUES[@]}"; do
    if ! [[ "${horizon}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] Invalid horizon '${horizon}'. Horizons must be positive integers." >&2
        exit 1
    fi
done

printf "horizon\tconfig_name\tconfig_path\tnum_envs\ttarget_episodes\tstatus\tlog_file\n" > "${SUMMARY_FILE}"

overall_status=0
echo "[INFO] Found ${#HORIZON_VALUES[@]} horizon(s) and ${#CONFIGS[@]} config(s). Results will be written to: ${RESULTS_ROOT}"
echo "[INFO] Default NUM_ENVS=${NUM_ENVS}; each NMPC env starts one do-mpc worker process."

for horizon in "${HORIZON_VALUES[@]}"; do
    horizon_dir="${RESULTS_ROOT}/horizon_${horizon}"
    mkdir -p "${horizon_dir}"
    echo "[INFO] Starting NMPC horizon sweep for n_horizon=${horizon}"

    for config in "${CONFIGS[@]}"; do
        config_name="$(basename "${config}" .yaml)"
        log_file="${horizon_dir}/${config_name}.log"

        echo "[INFO] Evaluating NMPC config: ${config_name}"
        echo "[INFO] Horizon: ${horizon}"
        echo "[INFO] Config path: ${config}"
        echo "[INFO] Log file: ${log_file}"
        echo "[INFO] Episodes: ${TARGET_EPISODES}; num_envs: ${NUM_ENVS}"
        echo "[INFO] Eval timeout: ${EVAL_TIMEOUT} (kill after ${TIMEOUT_KILL_AFTER})"

        PYTHONUNBUFFERED=1 setsid timeout --kill-after="${TIMEOUT_KILL_AFTER}" "${EVAL_TIMEOUT}" \
            "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/nmpc_policy_eval.py" \
            --config "${config}" \
            --n_horizon "${horizon}" \
            --episodes "${TARGET_EPISODES}" \
            --num_envs "${NUM_ENVS}" \
            "$@" > "${log_file}" 2>&1 &
        eval_pid=$!
        wait "${eval_pid}"
        status=$?
        cleanup_process_group "${eval_pid}"

        if [[ ${status} -eq 0 ]]; then
            echo "[INFO] Finished ${config_name} with n_horizon=${horizon}"
            status_text="ok"
        elif [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
            echo "[ERROR] Evaluation timed out or was killed for ${config_name} with n_horizon=${horizon} after ${EVAL_TIMEOUT}" >&2
            status_text="timeout_or_killed:${status}"
            overall_status=${status}
        else
            echo "[ERROR] Evaluation failed for ${config_name} with n_horizon=${horizon}, exit code ${status}" >&2
            status_text="failed:${status}"
            overall_status=${status}
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${horizon}" \
            "${config_name}" \
            "${config}" \
            "${NUM_ENVS}" \
            "${TARGET_EPISODES}" \
            "${status_text}" \
            "${log_file}" >> "${SUMMARY_FILE}"

        echo "[INFO] Sleeping ${SLEEP_BETWEEN_RUNS}s before the next evaluation to let Isaac/driver resources settle."
        sleep "${SLEEP_BETWEEN_RUNS}"
    done
done

echo "[INFO] NMPC evaluation sweep complete. Status summary: ${SUMMARY_FILE}"
exit "${overall_status}"
