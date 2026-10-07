#!/usr/bin/env bash
set -euo pipefail
WORLD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$WORLD/scripts/eval/native_projects.py" "${1:-list}" --project volleybots "${@:2}"
