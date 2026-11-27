#!/usr/bin/env bash
# One lane only: daemon lifecycle belongs exclusively to the batch supervisor.
set -Eeuo pipefail
BASE=/dockerdata/robotworld-r580
PRIVATE=/root/robotworld-behavior-private
TRIAL=$BASE/docker-trial-20260925
RECORD=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HERE=${BEHAVIOR_NATIVE_WORKER:?}
BATCH=${BEHAVIOR_MANAGED_BATCH:?}
GPU=${BEHAVIOR_HOST_GPU:?}
TASK=${BEHAVIOR_FEISHU_ID:?}
PROFILE=${BEHAVIOR_BUDGET_PROFILE:?}
FFMPEG=$(readlink -f -- "${BEHAVIOR_VIDEO_FFMPEG:?}")
VIDEO=$(realpath -- "$RECORD/../robodojo/video.py")
SOURCE=$HERE/../source/BEHAVIOR-1K-6cbf70b075816096e9be53958780769f3264d25d
DOCKER=$TRIAL/bin/docker
export DOCKER_HOST=unix://$PRIVATE/docker-parallel-runtime/run/docker.sock
IMAGE=robotworld-behavior-python-compiler:ubuntu24.04-py311
[[ $GPU =~ ^[0-7]$ && $TASK =~ ^[1-4]$ && ( $PROFILE == steps5000 || $PROFILE == steps10000 ) ]]
if [[ ${1:-} != --locked ]]; then
  exec flock --nonblock "$BASE/unified-eef-gpu$GPU.lock" bash "$0" --locked "$@"
fi
shift
[[ $# == 2 && $2 == --eef-agent ]]
RUN=$(realpath -m -- "$1")
[[ $RUN == "$PRIVATE"/native-task"$TASK"-scaffold-first-four-parallel-*/simulator && ! -e $RUN ]]
mapfile -t DEVICES < <(python3 "$RECORD/parallel_runtime.py" --batch "$BATCH" --gpu "$GPU")
[[ ${#DEVICES[@]} == 5 ]]
GPU_UUID=${DEVICES[0]}
RENDER=${DEVICES[1]}
CARD=${DEVICES[2]}
PCI=${DEVICES[3]}
CPUS=${DEVICES[4]}
[[ $(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader) == "$GPU_UUID" ]]
[[ $(nvidia-smi -i "$GPU" --query-gpu=driver_version --format=csv,noheader) == 580.105.08 ]]
[[ $(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) == 0 ]]
[[ $(readlink -f "/sys/class/drm/${RENDER##*/}/device") == */"$PCI" ]]
[[ $(readlink -f "/sys/class/drm/${CARD##*/}/device") == */"$PCI" ]]
NAME=rw-behavior-p-$(basename "$BATCH")-t$TASK-g$GPU
APPDATA=$BATCH/lane$TASK/appdata
APPS=$BATCH/lane$TASK/apps
[[ -d $APPDATA && -f $APPS/omnigibson_5_1_0.kit && -x $FFMPEG ]]
umask 077
mkdir -p "$RUN/tmp" "$RUN/cache"
RECEIPT=$PRIVATE/partial/receipt.json
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert int(sys.argv[2]) in d["tasks"] and d["status"]=="materialized"' "$RECEIPT" "$TASK"
cp -p "$RECEIPT" "$RUN/asset-receipt.json"
sha256sum "$RECORD/run_parallel.sh" "$RECORD/parallel_runtime.py" "$RECORD/task_worker.py" "$RECORD/budgets.py" "$RECORD/tasks.py" "$RECORD/record_worker.py" "$HERE/eef_protocol.py" > "$RUN/inputs.sha256"
created=0
cleanup() {
  trap - EXIT INT TERM
  if [[ $created == 1 && -S $PRIVATE/docker-parallel-runtime/run/docker.sock ]]; then
    "$DOCKER" stop --timeout 15 "$NAME" >/dev/null 2>&1 || true
    "$DOCKER" rm "$NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$DOCKER" image inspect "$IMAGE" > "$RUN/image.json"
command=("$DOCKER" run --name "$NAME" --label "robotworld.batch=$BATCH" --label "robotworld.task=$TASK"
  --network none --gpus "device=$GPU_UUID" --cpuset-cpus "$CPUS"
  --device "$RENDER:$RENDER" --device "$CARD:$CARD"
  --shm-size 8g --cap-drop MKNOD --security-opt no-new-privileges
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1
  -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8
  -e LD_LIBRARY_PATH=/opt/r580:/opt/ubuntu-libs
  -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1
  -e VK_ICD_FILENAMES=/opt/r580/nvidia_icd.json -e VK_DRIVER_FILES=/opt/r580/nvidia_icd.json
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/opt/r580/10_nvidia.json
  -e XDG_RUNTIME_DIR=/run/user/0 -e OMNIGIBSON_HEADLESS=1 -e OMNIGIBSON_GPU_ID=0
  -e TMPDIR=/output/tmp -e XDG_CACHE_HOME=/output/cache
  -e OMNIGIBSON_DATA_PATH=/data -e OMNIGIBSON_APPDATA_PATH=/appdata
  -e BEHAVIOR_PARTIAL_ASSETS=1 -e BEHAVIOR_EEF_MODE=agent
  -e BEHAVIOR_FEISHU_ID="$TASK" -e BEHAVIOR_BUDGET_PROFILE="$PROFILE"
  -e PYTHONPATH=/usr/local/lib
  --mount "type=bind,src=$BASE/behavior51-venv,dst=/opt/behavior,readonly"
  --mount "type=bind,src=$APPS,dst=/opt/behavior/lib/python3.11/site-packages/isaacsim/apps"
  --mount "type=bind,src=$SOURCE/docs,dst=/opt/behavior/lib/python3.11/docs,readonly"
  --mount "type=bind,src=$BASE/driver-libs,dst=/opt/r580,readonly"
  --mount "type=bind,src=$BASE/robodojo51-ubuntu/usr/lib/x86_64-linux-gnu,dst=/opt/ubuntu-libs,readonly"
  --mount "type=bind,src=$APPDATA,dst=/appdata"
  --mount "type=bind,src=$HERE,dst=/work,readonly"
  --mount "type=bind,src=$RECORD,dst=/recording,readonly"
  --mount "type=bind,src=$VIDEO,dst=/usr/local/lib/continuous_video.py,readonly"
  --mount "type=bind,src=$FFMPEG,dst=/usr/local/bin/ffmpeg,readonly"
  --mount "type=bind,src=$RUN,dst=/output"
  --mount type=tmpfs,dst=/run/user/0 --mount type=tmpfs,dst=/data
  --mount "type=bind,src=$RECEIPT,dst=/data-receipt.json,readonly"
  --mount "type=bind,src=$PRIVATE/data/omnigibson-robot-assets,dst=/data/omnigibson-robot-assets,readonly"
  --mount "type=bind,src=$PRIVATE/data/2026-challenge-task-instances,dst=/data/2026-challenge-task-instances,readonly"
  --mount "type=bind,src=$PRIVATE/data/omnigibson.key,dst=/data/omnigibson.key,readonly"
  --mount "type=bind,src=$PRIVATE/subset-data/behavior-1k-assets,dst=/data/behavior-1k-assets,readonly"
  "$IMAGE" /opt/behavior/bin/python -u /recording/record_worker.py)
printf '%q ' "${command[@]}" > "$RUN/command.sh"
printf '\n' >> "$RUN/command.sh"
created=1
set +e
timeout --signal=INT --kill-after=20s 88200s "${command[@]}" > "$RUN/process.log" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$RUN/exit-code.txt"
"$DOCKER" inspect "$NAME" > "$RUN/container-final.json" 2>/dev/null || true
[[ $code == 0 && -s $RUN/native-result.json ]]
python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads((p/"native-result.json").read_text()); s=json.loads((p/"container-final.json").read_text())[0]["State"]; ok=s["Status"]=="exited" and s["ExitCode"]==0 and not s["OOMKilled"] and not s["Error"]; (p/"exit-summary.json").write_text(json.dumps({"process_exit_clean":ok,"probe_passed":d["status"]=="passed","shutdown_requested":d.get("shutdown_requested",False)},indent=2)+"\n"); assert ok and d["status"]=="passed" and d.get("shutdown_requested")' "$RUN"
