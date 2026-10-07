#!/usr/bin/env bash
set -euo pipefail
# Default: play-soccer, fixed seed=2, 1000 steps / 20s, training-pitch background.
# Direct joint + coding_control tools, ankle-com assistance, fixed prior-attempt notes.
# Native 300-step baseline/hybrid/direct remain optional via --cases.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench humanoid_soccer "$@"
