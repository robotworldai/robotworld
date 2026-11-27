#!/usr/bin/env bash
set -euo pipefail
export WORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_ASSETS="${WORLD_ASSETS:-$WORLD_ROOT/Assets}"
WORLD_ASSETS="$(python3 "$WORLD_ROOT/scripts/check_harbor_assets.py" "$WORLD_ASSETS" --prepare-mount "$WORLD_ROOT")"
WORLD_NVIDIA_RUNTIME="$(python3 "$WORLD_ROOT/scripts/harbor_gpu_runtime.py")"
export WORLD_NVIDIA_RUNTIME
export WORLD_EPISODE_TOKEN="${WORLD_EPISODE_TOKEN:-$(python -c 'import secrets;print(secrets.token_hex(32))')}"
HARBOR="${HARBOR:-harbor}"
# Default full-budget transport probe uses no model and is not a task solution.
exec "$HARBOR" run --path "$WORLD_ROOT/tasks/close-drawer" \
  --agent "${HARBOR_AGENT:-agents.robotworld:RobotWorldProbeAgent}" --n-concurrent 1 --n-attempts "${ROLLOUTS:-1}" \
  --jobs-dir "$WORLD_ROOT/outputs/harbor" "$@"
