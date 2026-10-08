#!/usr/bin/env bash
set -euo pipefail
# No GPT/Codex inference: loads scenes, advances native actions, records original outcomes and MP4.
# Catalog: environment/validation/selected-tasks.json (84 entries; 45 selected core tasks).
WORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORLD_ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.validation.campaign "$@"
