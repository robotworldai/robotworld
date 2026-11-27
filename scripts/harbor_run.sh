#!/usr/bin/env bash
# Environment preparation only. All agent execution, scheduling and grading belong to Harbor.
set -euo pipefail
ROBOTWORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROBOTWORLD_ROOT"
export WORLD_ROOT="$ROBOTWORLD_ROOT"
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
WORLD_ASSETS="$(python3 scripts/check_harbor_assets.py "${WORLD_ASSETS:-$WORLD_ROOT/Assets}" --prepare-mount "$WORLD_ROOT")"
export WORLD_ASSETS
WORLD_NVIDIA_RUNTIME="$(python3 scripts/harbor_gpu_runtime.py)"
export WORLD_NVIDIA_RUNTIME
if [[ -z "${WORLD_EPISODE_TOKEN:-}" ]]; then
    WORLD_EPISODE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    export WORLD_EPISODE_TOKEN
fi
export WORLD_CODE_CONTROL="${WORLD_CODE_CONTROL:-0}"
exec "${HARBOR_BIN:-harbor}" run "$@"
