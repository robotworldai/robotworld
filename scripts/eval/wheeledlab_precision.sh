#!/usr/bin/env bash
set -euo pipefail
# Sensor-only custom precision driving; solid-line contact is a failure.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench wheeledlab \
  --cases rw-twin-beam,rw-reverse-bay,rw-parallel-park "$@"
