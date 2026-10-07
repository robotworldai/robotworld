#!/usr/bin/env bash
set -euo pipefail
# RobotWorld-authored: one baseline-target + three hard-target cases, 2000 steps each.
# Selection is explicit; upstream WheeledLab suite is kept unchanged.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench wheeledlab \
  --cases rw-courtyard,rw-hairpins,rw-gate-dock,rw-drift-switch "$@"
