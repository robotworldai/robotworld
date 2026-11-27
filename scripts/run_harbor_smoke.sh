#!/usr/bin/env bash
set -euo pipefail
export WORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORLD_ROOT"
export PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_ASSETS="${WORLD_ASSETS:-$WORLD_ROOT/Assets}"
WORLD_ASSETS="$(python3 "$WORLD_ROOT/scripts/check_harbor_assets.py" "$WORLD_ASSETS" --prepare-mount "$WORLD_ROOT")"
WORLD_NVIDIA_RUNTIME="$(python3 "$WORLD_ROOT/scripts/harbor_gpu_runtime.py")"
export WORLD_NVIDIA_RUNTIME
export WORLD_EPISODE_TOKEN="${WORLD_EPISODE_TOKEN:-$(python3 -c 'import secrets;print(secrets.token_hex(32))')}"
exec "${HARBOR:-harbor}" run --config configs/jobs/robotworld-smoke100.yaml --jobs-dir "$WORLD_ROOT/outputs/harbor" "$@"
