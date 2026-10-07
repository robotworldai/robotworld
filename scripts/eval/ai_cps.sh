#!/usr/bin/env bash
set -euo pipefail
# 22 catch, 23 balance, 24 peg insertion. ID34 removed from the active suite.
# Native horizon300; two physics substeps per action. Isaac6 is experimental.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench ai_cps "$@"
