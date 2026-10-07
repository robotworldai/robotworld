#!/usr/bin/env bash
set -euo pipefail
: "${WORLD_BEHAVIOR_DATA:?Set the absolute destination for official BEHAVIOR datasets}"
: "${WORLD_ROOT:?Set the World checkout path}"
if [[ "${1:-}" != --accept-license ]]; then
  echo 'Read the official BEHAVIOR asset license, then pass --accept-license.' >&2
  exit 2
fi
PYTHONPATH="$WORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m environment.datasets.behavior_1k.prepare_scene \
  --destination "$WORLD_BEHAVIOR_DATA" --cache "$WORLD_ROOT/var/cache/behavior-zip-index" --accept-license
# The subset directory already exists, so the official function only installs the key.
docker run --rm --network bridge \
  --mount "type=bind,src=$WORLD_BEHAVIOR_DATA,dst=/data" \
  -e OMNIGIBSON_DATA_PATH=/data -e OMNIGIBSON_NO_OMNIVERSE=1 \
  "${WORLD_BEHAVIOR_IMAGE:-stanfordvl/behavior:3.9.3}" python -c \
  'from pathlib import Path; assert Path("/data/behavior-1k-assets/VERSION").is_file(); from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)'
