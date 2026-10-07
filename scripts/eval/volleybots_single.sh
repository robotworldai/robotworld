#!/usr/bin/env bash
set -euo pipefail
WORLD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORLD"
case "${1:-list}" in
  list|check|build|probe|run)
    command="${1:-list}"
    if [[ $# -gt 0 ]]; then shift; fi
    exec python scripts/eval/native_projects.py "$command" --project volleybots --task T05-single --runtime-profile isaac6 "$@"
    ;;
esac
exec python -m environment.runtime.native_project_launch --project volleybots --task T05-single --runtime-profile isaac6 "$@"
