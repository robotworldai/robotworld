#!/usr/bin/env bash
# End-to-end Docker procedure runner + live monitor for RoboDojo.
#
# It drives (and asserts) the whole containerized eval loop so you can see the
# whole procedure is working:
#   [1] GPU is visible inside a container   (docker run --gpus all ... nvidia-smi)
#   [2] build the robodojo image            (docker build)
#   [3] start the demo_policy server (host, WebSocket / protocol: ws)
#   [4] run a real container eval           (docker run ... robodojo.sh client)
#   [5] assert eval_result/.../_result.json (eval_time >= 1)
#
# Usage:
#   bash docker/smoke_docker.sh run        # run + monitor the whole procedure
#   bash docker/smoke_docker.sh monitor    # attach a live view to the latest run
#   bash docker/smoke_docker.sh clean      # stop leftover server, prune smoke logs
#
# Everything is overridable via env (defaults chosen to match the verified run):
#   ROBODOJO_IMAGE=robodojo:cuda12.8  ROBODOJO_SMOKE_TASK=stack_bowls
#   ROBODOJO_SMOKE_POLICY=demo_policy ROBODOJO_SMOKE_PORT=6060
#   ROBODOJO_SMOKE_ENV_CFG=arx_x5     ROBODOJO_SMOKE_ACTION_TYPE=joint
#   ROBODOJO_SMOKE_EVAL_NUM=1         ROBODOJO_CONDA_ENV=RoboDojo

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/docker/smoke_logs"
mkdir -p "${LOG_DIR}"

IMAGE="${ROBODOJO_IMAGE:-robodojo:cuda12.8}"
TASK="${ROBODOJO_SMOKE_TASK:-stack_bowls}"
POLICY="${ROBODOJO_SMOKE_POLICY:-demo_policy}"
PORT="${ROBODOJO_SMOKE_PORT:-6060}"
ENV_CFG="${ROBODOJO_SMOKE_ENV_CFG:-arx_x5}"
ACTION_TYPE="${ROBODOJO_SMOKE_ACTION_TYPE:-joint}"
EVAL_NUM="${ROBODOJO_SMOKE_EVAL_NUM:-1}"
CONDA_ENV="${ROBODOJO_CONDA_ENV:-RoboDojo}"
CUDA_BASE_IMAGE="${ROBODOJO_CUDA_BASE_IMAGE:-nvidia/cuda:12.8.1-base-ubuntu22.04}"

# ── China mirror build args (resolved at run time; ROBODOJO_CN_MIRRORS=1/0) ───
CN_MIRRORS=""
BUILD_ARGS=()
resolve_build_args() {
    CN_MIRRORS="${ROBODOJO_CN_MIRRORS:-auto}"
    if [[ "${CN_MIRRORS}" == "auto" ]]; then
        if curl -fsS --connect-timeout 8 https://download.docker.com/linux/ubuntu/gpg -o /dev/null 2>/dev/null; then
            CN_MIRRORS=0
        else
            CN_MIRRORS=1
        fi
    fi
    if [[ "${CN_MIRRORS}" == "1" ]]; then
        BUILD_ARGS=(
            --build-arg "UBUNTU_MIRROR=${ROBODOJO_UBUNTU_MIRROR:-mirrors.tuna.tsinghua.edu.cn}"
            --build-arg "PIP_INDEX_URL=${ROBODOJO_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
            --build-arg "MINICONDA_URL=${ROBODOJO_MINICONDA_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
            --build-arg "CONDA_CHANNEL_MIRROR=${ROBODOJO_CONDA_CHANNEL_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
        )
    fi
}

# ── tiny helpers ──────────────────────────────────────────────────────────────
c_grn=$'\e[1;32m'; c_red=$'\e[1;31m'; c_ylw=$'\e[1;33m'; c_rst=$'\e[0m'
now()  { date +'%Y-%m-%d %H:%M:%S'; }
stamp(){ date +'%Y-%m-%d_%H-%M-%S'; }

# ── docker invocation (fall back to `sg docker` if the shell lacks the group) ──
detect_docker() {
    if docker info >/dev/null 2>&1; then
        _DOCKER_MODE="direct"
    elif ! command -v docker >/dev/null 2>&1; then
        echo "${c_red}[smoke] docker is not installed. Run: sudo bash docker/install_docker_nvidia.sh${c_rst}" >&2
        exit 3
    elif id -nG "$(id -un)" 2>/dev/null | tr ' ' '\n' | grep -qx docker \
         || getent group docker 2>/dev/null | grep -qw "$(id -un)"; then
        _DOCKER_MODE="sg"
    else
        echo "${c_red}[smoke] cannot access docker daemon and user not in 'docker' group.${c_rst}" >&2
        echo "        Re-run installer or add yourself: sudo usermod -aG docker $(id -un); then re-login." >&2
        exit 3
    fi
}
dk() {
    if [[ "${_DOCKER_MODE}" == "direct" ]]; then
        command docker "$@"
    else
        local q; printf -v q '%q ' "$@"
        sg docker -c "docker ${q}"
    fi
}

# ── status file (append-only key=value; monitor reads the last of each key) ────
RUN_ID=""; RUN_LOG=""; BUILD_LOG=""; SERVER_LOG=""; EVAL_LOG=""; STATUS_FILE=""
SERVER_PID=""

log()   { echo "[$(now)] $*" | tee -a "${RUN_LOG}"; }
put()   { echo "$1=$2" >> "${STATUS_FILE}"; }
phase() { put phase "$1"; put active_log "${2:-}"; log "${c_ylw}== phase: $1 ==${c_rst}"; }

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        log "stopped policy server (pid ${SERVER_PID})"
    fi
}
trap cleanup_server EXIT

fail() { put "$1" FAIL; put overall FAILED; log "${c_red}[FAIL] ${2:-$1}${c_rst}"; final_report; exit 1; }
pass() { put "$1" PASS; log "${c_grn}[ok] $1${c_rst}"; }

# ── phases ────────────────────────────────────────────────────────────────────
phase_gpu() {
    phase gpu_check "${RUN_LOG}"
    if dk run --rm --gpus all "${CUDA_BASE_IMAGE}" nvidia-smi >>"${RUN_LOG}" 2>&1; then
        pass gpu_check
    else
        fail gpu_check "GPU not visible in container (nvidia-smi failed). Check NVIDIA Container Toolkit."
    fi
}

phase_build() {
    phase build "${BUILD_LOG}"
    # Iterate on the runtime path without paying for the ~1h image build: reuse an
    # existing image when ROBODOJO_SKIP_BUILD=1. Never skips silently if missing.
    if [[ "${ROBODOJO_SKIP_BUILD:-0}" == "1" ]]; then
        if dk image inspect "${IMAGE}" >/dev/null 2>&1; then
            log "skipping build (ROBODOJO_SKIP_BUILD=1; reusing existing ${IMAGE})"
            pass build; return
        fi
        log "ROBODOJO_SKIP_BUILD=1 but ${IMAGE} not found — building anyway"
    fi
    log "building ${IMAGE} (cn_mirrors=${CN_MIRRORS}; this is the slow one; tail ${BUILD_LOG})"
    dk build "${BUILD_ARGS[@]}" -t "${IMAGE}" -f "${ROOT_DIR}/Dockerfile" "${ROOT_DIR}" >"${BUILD_LOG}" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        log "$(tail -n 30 "${BUILD_LOG}")"
        fail build "docker build failed (rc=${rc}); see ${BUILD_LOG}"
    fi
    dk image inspect "${IMAGE}" >/dev/null 2>&1 || fail build "image ${IMAGE} missing after build"
    pass build
}

phase_server() {
    phase server "${SERVER_LOG}"
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh" 2>/dev/null \
        || fail server "cannot source conda from ${conda_base}"
    conda activate "${CONDA_ENV}" 2>/dev/null || fail server "cannot activate conda env '${CONDA_ENV}'"

    PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/XPolicyLab" \
    nohup python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${ROOT_DIR}/XPolicyLab/policy/${POLICY}/deploy.yml" \
        --protocol ws \
        --overrides port="${PORT}" host=127.0.0.1 dataset_name=RoboDojo \
            task_name="${TASK}" ckpt_name=demo env_cfg_type="${ENV_CFG}" \
            seed=0 policy_name="${POLICY}" action_type="${ACTION_TYPE}" \
        >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    put server_pid "${SERVER_PID}"
    log "policy server pid=${SERVER_PID}, waiting for 127.0.0.1:${PORT} ..."

    local i
    for i in $(seq 1 30); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "$(tail -n 20 "${SERVER_LOG}")"
            fail server "policy server process died during startup; see ${SERVER_LOG}"
        fi
        if timeout 2 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
            pass server; return
        fi
        sleep 1
    done
    fail server "policy server not listening on ${PORT} after 30s"
}

phase_eval() {
    phase eval "${EVAL_LOG}"
    log "running container eval (task=${TASK}, policy=${POLICY})"

    # Mount the host's Isaac Sim / Omniverse caches so Kit resolves extensions and
    # shaders locally instead of downloading them from the Omniverse registry /
    # CloudFront at runtime (which stalls on restricted networks). A fresh
    # container has an empty extension cache and would otherwise hang pulling
    # e.g. omni.kit.pip_archive. Only host dirs that exist are mounted.
    # NOTE: ~/.cache/warp holds curobo's JIT-compiled warp/NVRTC kernels. Without
    # it the container recompiles them on every curobo warmup (very slow, minutes),
    # so mounting the host cache is essential for the planner to start promptly.
    local cache_mounts=() pair host
    for pair in \
        "${HOME}/.local/share/ov:/root/.local/share/ov" \
        "${HOME}/.cache/ov:/root/.cache/ov" \
        "${HOME}/.cache/warp:/root/.cache/warp" \
        "${HOME}/.cache/nvidia:/root/.cache/nvidia" \
        "${HOME}/.cache/pip:/root/.cache/pip" \
        "${HOME}/.nv:/root/.nv" \
        "${HOME}/.nvidia-omniverse:/root/.nvidia-omniverse"; do
        host="${pair%%:*}"
        [[ -d "${host}" ]] && cache_mounts+=( -v "${pair}" )
    done
    log "isaac cache mounts: ${#cache_mounts[@]} host dir(s)"

    # curobo robot configs (Assets/Robots/**/curobo.yml, generated from *_tmp.yml
    # by utils/update_embodiment_config_path.py) bake in ABSOLUTE paths using the
    # host repo root (os.getcwd()), e.g. ${ROOT_DIR}/Assets/Robots/x5/X5A.urdf.
    # Mount Assets both at the in-container repo path (where the code looks) AND at
    # the host path (so those baked absolute paths resolve). Both read-only, so the
    # host's configs are never modified.
    local assets_mounts=( -v "${ROOT_DIR}/Assets:/workspace/RoboDojo/Assets:ro" )
    [[ "${ROOT_DIR}/Assets" != "/workspace/RoboDojo/Assets" ]] \
        && assets_mounts+=( -v "${ROOT_DIR}/Assets:${ROOT_DIR}/Assets:ro" )

    # Headless RTX rendering (single Vulkan ICD + libXt.so.6) is baked into the
    # image (Dockerfile), so nothing extra is needed here at run time.
    dk run --rm --gpus all --network host --ipc host \
        -e ROBODOJO_MAX_BASH_RETRIES=2 \
        "${cache_mounts[@]}" \
        "${assets_mounts[@]}" \
        -v "${ROOT_DIR}/eval_result:/workspace/RoboDojo/eval_result" \
        "${IMAGE}" \
        bash scripts/robodojo.sh client \
            --task "${TASK}" --policy-name "${POLICY}" \
            --policy-host 127.0.0.1 --policy-port "${PORT}" \
            --ckpt demo --action-type "${ACTION_TYPE}" --eval-num "${EVAL_NUM}" \
        >"${EVAL_LOG}" 2>&1
    local rc=$?
    [[ ${rc} -eq 0 ]] || { log "$(tail -n 30 "${EVAL_LOG}")"; fail eval "container eval exited rc=${rc}; see ${EVAL_LOG}"; }
    pass eval
}

phase_verify() {
    phase verify "${RUN_LOG}"
    # fix ownership of any root-created result files (container runs as root)
    dk run --rm -v "${ROOT_DIR}/eval_result:/workspace/RoboDojo/eval_result" \
        "${IMAGE}" chown -R "$(id -u):$(id -g)" /workspace/RoboDojo/eval_result >/dev/null 2>&1 || true

    local base="${ROOT_DIR}/eval_result/RoboDojo/${TASK}/${POLICY}"
    local rj
    rj="$(find "${base}" -name _result.json -newermt "@${RUN_START}" 2>/dev/null | sort | tail -1)"
    [[ -n "${rj}" ]] || fail verify "no _result.json produced under ${base}"
    put result_json "${rj}"
    log "result: ${rj}"
    log "$(cat "${rj}")"
    if python3 - "$rj" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
et=d.get("eval_time",0)
print(f"[verify] success_rate={d.get('success_rate')} eval_time={et} score={d.get('score')}")
sys.exit(0 if (isinstance(et,(int,float)) and et>=1) else 1)
PY
    then pass verify; else fail verify "eval_time < 1 in ${rj}"; fi
}

final_report() {
    echo
    echo "================ RoboDojo Docker procedure ================"
    for k in gpu_check build server eval verify; do
        printf "  %-10s : %s\n" "$k" "$(grep "^${k}=" "${STATUS_FILE}" | tail -1 | cut -d= -f2- || echo '-')"
    done
    echo "  overall    : $(grep '^overall=' "${STATUS_FILE}" | tail -1 | cut -d= -f2-)"
    echo "  logs       : ${LOG_DIR}  (run_id=${RUN_ID})"
    echo "=========================================================="
}

cmd_run() {
    detect_docker
    resolve_build_args
    RUN_ID="$(stamp)"
    RUN_LOG="${LOG_DIR}/run_${RUN_ID}.log"
    BUILD_LOG="${LOG_DIR}/build_${RUN_ID}.log"
    SERVER_LOG="${LOG_DIR}/server_${RUN_ID}.log"
    EVAL_LOG="${LOG_DIR}/eval_${RUN_ID}.log"
    STATUS_FILE="${LOG_DIR}/status_${RUN_ID}.txt"
    : >"${RUN_LOG}"; : >"${STATUS_FILE}"
    ln -sfn "$(basename "${STATUS_FILE}")" "${LOG_DIR}/status_latest.txt"
    RUN_START=$(date +%s)

    put run_id "${RUN_ID}"; put overall RUNNING
    log "docker mode: ${_DOCKER_MODE} | image=${IMAGE} task=${TASK} policy=${POLICY} port=${PORT} | cn_mirrors=${CN_MIRRORS}"

    phase_gpu
    phase_build
    phase_server
    phase_eval
    phase_verify
    cleanup_server
    put overall PASSED
    log "${c_grn}ALL PHASES PASSED${c_rst}"
    final_report
}

cmd_monitor() {
    local status="${LOG_DIR}/status_latest.txt"
    [[ -e "${status}" ]] || { echo "no run yet — start one with: bash docker/smoke_docker.sh run"; exit 1; }
    echo "monitoring $(readlink -f "${status}") (Ctrl-C to stop)"
    while true; do
        local overall phase active
        overall="$(grep '^overall=' "${status}" | tail -1 | cut -d= -f2-)"
        phase="$(grep '^phase=' "${status}" | tail -1 | cut -d= -f2-)"
        active="$(grep '^active_log=' "${status}" | tail -1 | cut -d= -f2-)"
        echo "----- [$(now)] overall=${overall:-?} phase=${phase:-?} -----"
        for k in gpu_check build server eval verify; do
            printf "  %-10s %s\n" "$k" "$(grep "^${k}=" "${status}" | tail -1 | cut -d= -f2- || echo '-')"
        done
        if [[ -n "${active}" && -f "${active}" ]]; then
            echo "  ...tail $(basename "${active}"):"
            tail -n 6 "${active}" | sed 's/^/    /'
        fi
        [[ "${overall}" == "PASSED" || "${overall}" == "FAILED" ]] && { echo "run finished: ${overall}"; break; }
        sleep 4
    done
}

cmd_clean() {
    pkill -f "setup_policy_server.py.*policy_name=${POLICY}" 2>/dev/null && echo "stopped leftover server(s)" || true
    echo "smoke logs in ${LOG_DIR}:"; ls -1 "${LOG_DIR}" 2>/dev/null || true
}

case "${1:-run}" in
    run)     cmd_run ;;
    monitor) cmd_monitor ;;
    clean)   cmd_clean ;;
    -h|--help) sed -n '2,30p' "$0" ;;
    *) echo "unknown subcommand: $1 (use: run | monitor | clean)"; exit 2 ;;
esac
