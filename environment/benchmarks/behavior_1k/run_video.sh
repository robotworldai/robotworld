#!/usr/bin/env bash
# Additive recording launcher; the original native launcher and worker remain unchanged.
set -Eeuo pipefail
BASE=/dockerdata/robotworld-r580
PRIVATE=/root/robotworld-behavior-private
TRIAL=$BASE/docker-trial-20260925
RECORD=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HERE=${BEHAVIOR_NATIVE_WORKER:?Pass the existing worker directory}
TASK=${BEHAVIOR_FEISHU_ID:-2}
[[ $TASK =~ ^[1-4]$ ]] || exit 2
PROFILE=${BEHAVIOR_BUDGET_PROFILE:-legacy}
[[ $PROFILE == legacy || $PROFILE == steps5000 || $PROFILE == steps10000 ]] || exit 2
[[ $PROFILE == legacy || -n ${BEHAVIOR_FEISHU_ID:-} ]] || exit 2
FFMPEG=$(readlink -f -- "${BEHAVIOR_VIDEO_FFMPEG:?Pass the verified host ffmpeg}")
VIDEO=$(realpath -- "$RECORD/../robodojo/video.py")
SOURCE=$HERE/../source/BEHAVIOR-1K-6cbf70b075816096e9be53958780769f3264d25d
DOCKER=$TRIAL/bin/docker
export DOCKER_HOST=unix://$TRIAL/run/docker.sock
NAME=rw-behavior-native-video
IMAGE=robotworld-behavior-python-compiler:ubuntu24.04-py311
if [[ ${1:-} != --locked ]]; then
  exec flock --nonblock "$BASE/unified-eef-gpu0.lock" \
    flock --nonblock "$BASE/robodojo51.lock" bash "$0" --locked "$@"
fi
shift
[[ $# == 1 || ( $# == 2 && ( $2 == --app-only || $2 == --partial || $2 == --eef-smoke || $2 == --eef-agent ) ) ]] || exit 2
RUN=$(realpath -m -- "$1")
[[ $RUN == "$PRIVATE"/native-* && ! -e $RUN ]] || exit 2
APP_ONLY=0
PARTIAL=0
EEF_MODE=none
SCRIPT=native_probe.py
TIMEOUT=1200
[[ ${2:-} != --app-only ]] || APP_ONLY=1
[[ ${2:-} != --partial ]] || PARTIAL=1
if [[ ${2:-} == --eef-smoke || ${2:-} == --eef-agent ]]; then
  PARTIAL=1
  EEF_MODE=${2#--eef-}
  SCRIPT=eef_worker.py
  TIMEOUT=2100
  [[ $PROFILE == legacy ]] || TIMEOUT=88200
fi
RECEIPT=$PRIVATE/asset-result.json
if [[ $PARTIAL == 1 ]]; then
  RECEIPT=$PRIVATE/partial/receipt.json
  python3 "$HERE/partial_assets.py" --verify
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert int(sys.argv[2]) in d["tasks"] and d["all_systems_included"]' "$RECEIPT" "$TASK"
elif [[ $APP_ONLY == 0 ]]; then
  [[ -s $PRIVATE/asset-result.json ]] || {
    printf 'Scene assets have no validated receipt; native task not started.\n' >&2; exit 2;
  }
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="prepared" and d["key_installed"]' "$PRIVATE/asset-result.json"
fi
[[ $(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader) == 580.105.08 ]]
[[ $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits) == 0 ]]
[[ $(readlink -f /sys/class/drm/renderD128/device) == */0000:03:00.0 ]]
umask 077
mkdir "$RUN"
mkdir -p "$PRIVATE/appdata" "$RUN/tmp" "$RUN/cache"
sha256sum "$HERE/$SCRIPT" "$HERE/run_native.sh" "$HERE/eef_protocol.py" > "$RUN/inputs.sha256"
cp -p "$HERE/$SCRIPT" "$HERE/run_native.sh" "$HERE/eef_protocol.py" "$RUN/"
[[ ! -f $RECEIPT ]] || cp -p "$RECEIPT" "$RUN/asset-receipt.json"
daemon_pid=
cleanup() {
  trap - EXIT INT TERM
  if [[ -S $TRIAL/run/docker.sock ]]; then
    "$DOCKER" stop --timeout 12 "$NAME" >/dev/null 2>&1 || true
    "$DOCKER" rm "$NAME" >/dev/null 2>&1 || true
  fi
  if [[ -n $daemon_pid ]]; then
    if [[ -f $TRIAL/run/dockerd.pid ]]; then
      pid=$(<"$TRIAL/run/dockerd.pid")
      [[ $pid =~ ^[0-9]+$ ]] && kill -INT "$pid" 2>/dev/null || true
    fi
    wait "$daemon_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -S $TRIAL/run/docker.sock ]]; then
  [[ $("$DOCKER" inspect -f '{{.State.Running}}' rw-behavior-assets) == true ]] || {
    printf 'Unrecognized existing daemon left untouched\n' >&2; exit 2;
  }
else
  exec 9>"$BASE/docker-trial.lock"
  flock --nonblock 9 || exit 2
  [[ ! -e $TRIAL/run/dockerd.pid ]] || exit 2
  env PATH="$TRIAL/bin:$TRIAL/toolkit/root/usr/bin:/usr/bin:/bin" \
    LD_LIBRARY_PATH="$TRIAL/toolkit/root/usr/lib64:/lib64" \
    NVIDIA_CONTAINER_RUNTIME_CONFIG="$TRIAL/nvidia-container-runtime/config.toml" \
    timeout --signal=INT --kill-after=15s "$((TIMEOUT+60))s" "$TRIAL/bin/dockerd" \
    --host "$DOCKER_HOST" --pidfile "$TRIAL/run/dockerd.pid" \
    --data-root "$TRIAL/data" --exec-root "$TRIAL/exec" --config-file "$TRIAL/daemon.json" \
    --storage-driver vfs --bridge none --iptables=false --ip6tables=false \
    --ip-forward=false --ip-masq=false --log-level warn > "$RUN/daemon.log" 2>&1 &
  daemon_pid=$!
  for _ in {1..100}; do
    if "$DOCKER" info >/dev/null 2>&1; then break; fi
    sleep .1
  done
fi
"$DOCKER" image inspect "$IMAGE" > "$RUN/image.json"
[[ -x $FFMPEG && -f $VIDEO ]]
sha256sum "$RECORD/record_worker.py" "$RECORD/run_video.sh" "$VIDEO" "$FFMPEG" > "$RUN/recording-inputs.sha256"
cp -p "$RECORD/record_worker.py" "$RECORD/run_video.sh" "$VIDEO" "$RUN/"
command=("$DOCKER" run --name "$NAME" --network none --gpus device=0
  --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card1:/dev/dri/card1
  --shm-size 8g --cap-drop MKNOD --security-opt no-new-privileges
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1
  -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e LD_LIBRARY_PATH=/opt/r580:/opt/ubuntu-libs
  -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
  -e VK_ICD_FILENAMES=/opt/r580/nvidia_icd.json -e VK_DRIVER_FILES=/opt/r580/nvidia_icd.json
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/opt/r580/10_nvidia.json
  -e XDG_RUNTIME_DIR=/run/user/0 -e OMNIGIBSON_HEADLESS=1 -e OMNIGIBSON_GPU_ID=0
  -e TMPDIR=/output/tmp -e XDG_CACHE_HOME=/output/cache
  -e OMNIGIBSON_DATA_PATH=/data -e OMNIGIBSON_APPDATA_PATH=/appdata
  -e BEHAVIOR_APP_ONLY="$APP_ONLY"
  -e BEHAVIOR_PARTIAL_ASSETS="$PARTIAL"
  -e BEHAVIOR_EEF_MODE="$EEF_MODE"
  --mount "type=bind,src=$BASE/behavior51-venv,dst=/opt/behavior"
  --mount "type=bind,src=$SOURCE/docs,dst=/opt/behavior/lib/python3.11/docs,readonly"
  --mount "type=bind,src=$BASE/driver-libs,dst=/opt/r580,readonly"
  --mount "type=bind,src=$BASE/robodojo51-ubuntu/usr/lib/x86_64-linux-gnu,dst=/opt/ubuntu-libs,readonly"
  --mount "type=bind,src=$PRIVATE/appdata,dst=/appdata"
  --mount "type=bind,src=$HERE,dst=/work,readonly"
  --mount "type=bind,src=$RECORD,dst=/recording,readonly"
  --mount "type=bind,src=$VIDEO,dst=/usr/local/lib/continuous_video.py,readonly"
  -e PYTHONPATH=/usr/local/lib
  --mount "type=bind,src=$FFMPEG,dst=/usr/local/bin/ffmpeg,readonly"
  --mount "type=bind,src=$RUN,dst=/output"
  --mount type=tmpfs,dst=/run/user/0)
if [[ -n ${BEHAVIOR_FEISHU_ID:-} ]]; then
  command+=(-e BEHAVIOR_FEISHU_ID="$TASK")
  command+=(-e BEHAVIOR_BUDGET_PROFILE="$PROFILE")
  sha256sum "$RECORD/task_worker.py" "$RECORD/tasks.py" "$RECORD/budgets.py" >> "$RUN/recording-inputs.sha256"
  cp -p "$RECORD/task_worker.py" "$RECORD/tasks.py" "$RECORD/budgets.py" "$RUN/"
fi
if [[ $APP_ONLY == 0 ]]; then
  command+=(--mount "type=bind,src=$RECEIPT,dst=/data-receipt.json,readonly")
fi
if [[ $PARTIAL == 1 ]]; then
  command+=(--mount type=tmpfs,dst=/data
    --mount "type=bind,src=$PRIVATE/data/omnigibson-robot-assets,dst=/data/omnigibson-robot-assets,readonly"
    --mount "type=bind,src=$PRIVATE/data/2026-challenge-task-instances,dst=/data/2026-challenge-task-instances,readonly"
    --mount "type=bind,src=$PRIVATE/data/omnigibson.key,dst=/data/omnigibson.key,readonly")
  command+=(--mount "type=bind,src=$PRIVATE/subset-data/behavior-1k-assets,dst=/data/behavior-1k-assets,readonly")
else
  command+=(--mount "type=bind,src=$PRIVATE/data,dst=/data,readonly")
fi
[[ $SCRIPT == eef_worker.py ]]
command+=("$IMAGE" /opt/behavior/bin/python -u /recording/record_worker.py)
printf '%q ' "${command[@]}" > "$RUN/command.sh"
printf '\n' >> "$RUN/command.sh"
set +e
timeout --signal=INT --kill-after=20s "${TIMEOUT}s" "${command[@]}" > "$RUN/process.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$RUN/exit-code.txt"
"$DOCKER" inspect "$NAME" > "$RUN/container-final.json" 2>/dev/null || true
[[ $code == 0 && -s $RUN/native-result.json ]] || exit 1
python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads((p/"native-result.json").read_text()); s=json.loads((p/"container-final.json").read_text())[0]["State"]; ok=s["Status"]=="exited" and s["ExitCode"]==0 and not s["OOMKilled"] and not s["Error"]; (p/"exit-summary.json").write_text(json.dumps({"process_exit_clean":ok,"probe_passed":d["status"]=="passed","shutdown_requested":d.get("shutdown_requested",False)},indent=2)+"\n"); assert ok and d["status"]=="passed" and d.get("shutdown_requested")' "$RUN"
printf 'Native probe passed (app_only=%s): %s\n' "$APP_ONLY" "$RUN"
