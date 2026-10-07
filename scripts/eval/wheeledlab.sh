#!/usr/bin/env bash
set -euo pipefail
# MuSHR drift 250; F1TENTH drift 250; elevation 200; visual 50 native action steps.
# Current runtime: Isaac6.0.1 + IsaacLab2.2, experimental compatibility.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.runner --bench wheeledlab "$@"
