#!/usr/bin/env bash
set -euo pipefail
WORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$WORLD_ROOT/scripts/eval/native_projects.py" "$@"
