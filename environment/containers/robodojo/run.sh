#!/usr/bin/env bash
set -euo pipefail
: "${WORLD_ROOT:?Set the absolute path to the World checkout}"
: "${ROBODOJO_ASSETS:?Set the real absolute path to RoboDojo Assets}"
world="$(realpath "$WORLD_ROOT")"
assets="$(realpath "$ROBODOJO_ASSETS")"
test -d "$world/codex/.git"
test -d "$assets"
mode="${1:-probe}"
if [ "$#" -gt 0 ]; then shift; fi
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="${WORLD_RUNS:-$world/var/runs/docker/robodojo}/$run_id"
cache="${WORLD_CACHE:-$world/var/cache/docker/robodojo}"
mkdir -p "$output" "$cache"
args=(run --rm --gpus "${WORLD_GPUS:-all}" --shm-size 8g
  --env "WORLD_ROOT=$world" --env WORLD_OUTPUT=/runs
  --mount "type=bind,src=$world,dst=$world,readonly"
  --mount "type=bind,src=$assets,dst=/workspace/RoboDojo/Assets,readonly"
  --mount "type=bind,src=$assets,dst=$assets,readonly"
  --mount "type=bind,src=$output,dst=/runs"
  --mount "type=bind,src=$cache,dst=/root/.cache")
# Preserve the path used in existing CuRobo asset YAMLs without rewriting them.
if [ -n "${ROBODOJO_ASSET_ALIAS:-}" ] && [ "$ROBODOJO_ASSET_ALIAS" != "$assets" ] && [ "$ROBODOJO_ASSET_ALIAS" != /workspace/RoboDojo/Assets ]; then
  args+=(--mount "type=bind,src=$assets,dst=$ROBODOJO_ASSET_ALIAS,readonly")
fi
if [ "$mode" = smoke ]; then
  : "${WORLD_CODEX_MANIFEST:?Set the source-built Codex manifest path}"
  : "${WORLD_MODEL_CATALOG:?Set the direct tool catalog path within World}"
  : "${WORLD_CODEX_HOME:?Set a dedicated Codex runtime config/auth directory}"
  test -f "$WORLD_CODEX_MANIFEST"
  test -f "$WORLD_MODEL_CATALOG"
  test -d "$WORLD_CODEX_HOME"
  args+=(--env "WORLD_CODEX_MANIFEST=$WORLD_CODEX_MANIFEST"
    --env "WORLD_MODEL_CATALOG=$WORLD_MODEL_CATALOG"
    --env CODEX_HOME=/codex-home
    --mount "type=bind,src=$(realpath "$WORLD_CODEX_HOME"),dst=/codex-home")
fi
printf 'Run artifacts: %s\n' "$output"
docker "${args[@]}" "${WORLD_ROBODOJO_IMAGE:-world/robodojo:local}" "$mode" "$@"
