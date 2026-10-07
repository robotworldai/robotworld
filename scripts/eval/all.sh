#!/usr/bin/env bash
set -euo pipefail
# Run all benchmark suites registered in environment/evaluation/suites.json, sequentially.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench all "$@"
