#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command="${1:-list}"
shift || true
case "$command" in
  list) cat third_party/benchmarks/steadytray/project.json ;;
  assets)
    PY="${WORLD_ASSET_PYTHON:-$ROOT/var/venvs/robolab-assets/bin/python}"
    exec "$PY" third_party/benchmarks/steadytray/prepare_assets.py "$@" ;;
  build) exec docker build -t world/steadytray:isaac6.0.1-experimental third_party/benchmarks/steadytray/docker "$@" ;;
  run) exec "${WORLD_PYTHON:-python3}" -m environment.runtime.native_project_launch --project steadytray --task T01 "$@" ;;
  *) echo "Usage: $0 {list|assets|build|run --output PATH [--mode probe|zero|codex] ...}" >&2; exit 2 ;;
esac
