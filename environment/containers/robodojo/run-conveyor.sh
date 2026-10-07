#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORLD_ROOT:?Set the World checkout path}"
mode="${1:-probe}"
if [ "$#" -gt 0 ]; then shift; fi
image="${WORLD_ROBODOJO_IMAGE:-world/robodojo:isaac6.0.1-local}"
if [ "$mode" = doctor ]; then
  exec docker run --rm --gpus all --memory 2g "$image" python -c \
    'import importlib.metadata as m; print({p:m.version(p) for p in ("isaacsim", "isaaclab", "torch")})'
fi
extra=()
for arg in "$@"; do
  if [ "$arg" = --run-task ]; then mode=task; else extra+=("$arg"); fi
done
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="${WORLD_RUNS:-$WORLD_ROOT/var/runs/docker/isaac601-conveyor}/$run_id"
args=("$mode" --image "$image" --output "$output")
if [ "$mode" = smoke ] || [ "$mode" = task ]; then
  : "${WORLD_CODEX_HOME:?Set the dedicated Codex config/auth directory}"
  args+=(--codex-home "$WORLD_CODEX_HOME")
fi
printf 'Run artifacts: %s\n' "$output"
exec python3 "$here/run_isaac601_local.py" "${args[@]}" "${extra[@]}"
