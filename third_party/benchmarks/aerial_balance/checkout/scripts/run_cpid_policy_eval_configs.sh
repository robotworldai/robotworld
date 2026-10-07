#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/logs/cpid_500/cpid_eval_logs}"
SUMMARY_FILE="${RESULTS_ROOT}/eval_status.tsv"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-5}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-2m}"
TIMEOUT_KILL_AFTER="${TIMEOUT_KILL_AFTER:-5s}"
CLEANUP_AFTER_RUN="${CLEANUP_AFTER_RUN:-true}"
CLEANUP_GRACE_PERIOD="${CLEANUP_GRACE_PERIOD:-3s}"

if [[ -n "${CONFIG:-}" ]]; then
    CONFIGS=("${CONFIG}")
else
    CONFIGS=(
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_delay_free.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_delay_wo_comp.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_delay_w_comp.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_mass.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_gain.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_disturbance.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_delay_30.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_delay_45.yaml"
        "${PROJECT_ROOT}/baselines/configs/cpid_target_position_eval_delay_60.yaml"
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

for idx in "${!CONFIGS[@]}"; do
    if [[ "${CONFIGS[$idx]}" != /* ]]; then
        CONFIGS[$idx]="${PROJECT_ROOT}/${CONFIGS[$idx]}"
    fi
    if [[ ! -f "${CONFIGS[$idx]}" ]]; then
        echo "[ERROR] Config file does not exist: ${CONFIGS[$idx]}" >&2
        exit 1
    fi
done

printf "config_name\tconfig_path\tstatus\tlog_file\n" > "${SUMMARY_FILE}"

overall_status=0
echo "[INFO] Found ${#CONFIGS[@]} config(s). Results will be written to: ${RESULTS_ROOT}"

for config in "${CONFIGS[@]}"; do
    config_name="$(basename "${config}" .yaml)"
    log_file="${RESULTS_ROOT}/${config_name}.log"

    echo "[INFO] Evaluating CPID config: ${config_name}"
    echo "[INFO] Config path: ${config}"
    echo "[INFO] Log file: ${log_file}"
    echo "[INFO] Eval timeout: ${EVAL_TIMEOUT} (kill after ${TIMEOUT_KILL_AFTER})"

    PYTHONUNBUFFERED=1 setsid timeout --kill-after="${TIMEOUT_KILL_AFTER}" "${EVAL_TIMEOUT}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/cpid_policy_eval.py" \
        --config "${config}" \
        "$@" > "${log_file}" 2>&1 &
    eval_pid=$!
    wait "${eval_pid}"
    status=$?
    cleanup_process_group "${eval_pid}"

    if [[ ${status} -eq 0 ]]; then
        echo "[INFO] Finished ${config_name}"
        status_text="ok"
    elif [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
        echo "[ERROR] Evaluation timed out or was killed for ${config_name} after ${EVAL_TIMEOUT}" >&2
        status_text="timeout_or_killed:${status}"
        overall_status=${status}
    else
        echo "[ERROR] Evaluation failed for ${config_name} with exit code ${status}" >&2
        status_text="failed:${status}"
        overall_status=${status}
    fi

    printf "%s\t%s\t%s\t%s\n" "${config_name}" "${config}" "${status_text}" "${log_file}" >> "${SUMMARY_FILE}"

    echo "[INFO] Sleeping ${SLEEP_BETWEEN_RUNS}s before the next evaluation to let Isaac/driver resources settle."
    sleep "${SLEEP_BETWEEN_RUNS}"
done

echo "[INFO] CPID evaluation sweep complete. Status summary: ${SUMMARY_FILE}"
exit "${overall_status}"
