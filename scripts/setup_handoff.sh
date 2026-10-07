#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${WORLD_PYTHON:-python3}"
case "${1:-help}" in
  sources) exec "$PY" scripts/fetch_sources.py --all ;;
  assets) shift; exec "$PY" scripts/download_assets.py "$@" ;;
  docker) shift; exec "$PY" scripts/prepare_docker.py "$@" ;;
  codex) exec "$PY" -m environment.scripts.build_codex --with-cli --with-code-mode-host ;;
  *) echo 'Usage: bash scripts/setup_handoff.sh sources|assets|docker|codex [options]'; exit 0 ;;
esac
