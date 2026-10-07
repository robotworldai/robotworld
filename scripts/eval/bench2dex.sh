#!/usr/bin/env bash
set -euo pipefail
# RobotWorld 41=piano79,42=soup76,43=tools12,44=wine34,45=faucet67,
# 46=microwave44,47=desk80,48=puzzle73,49=glasses03.
WORLD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORLD"
exec "${WORLD_PYTHON:-python3}" -m environment.benchmarks.bench2dex.suite "$@"
