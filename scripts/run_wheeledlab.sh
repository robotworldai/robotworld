#!/usr/bin/env bash
set -euo pipefail
# wheeledlab: Start a fresh environment and source-built agent for every task rollout.
ROLLOUTS="${ROLLOUTS:-3}"
MODEL="${MODEL:-}"
CODE_CONTROL="${CODE_CONTROL:-off}"  # Benchmark default; per-task on/off overrides it; default inherits it.
SCORING_PROFILE="${SCORING_PROFILE:-auto}"  # Use World v1 for adapted scenarios and native scoring elsewhere.
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-}"
# native=original budget; profile=scenario budget; integer=explicit steps. BEHAVIOR defaults to min(native,2000).
TASK_STEPS=(
  "mushr-drift=profile"  # Native budget=250
  "f1tenth-drift=profile"  # Native budget=250
  "elevation=native"  # Native budget=200
  "visual=profile"  # Native budget=50
  "rw-courtyard=native"  # Native budget=2000
  "rw-hairpins=native"  # Native budget=2000
  "rw-gate-dock=native"  # Native budget=2000
  "rw-drift-switch=native"  # Native budget=2000
  "rw-twin-beam=native"  # Native budget=2000
  "rw-reverse-bay=native"  # Native budget=2000
  "rw-parallel-park=native"  # Native budget=2000
)
# Set each task to on/off; default inherits CODE_CONTROL (initially off).
TASK_CODE_CONTROL=(
  "mushr-drift=default"
  "f1tenth-drift=default"
  "elevation=default"
  "visual=default"
  "rw-courtyard=default"
  "rw-hairpins=default"
  "rw-gate-dock=default"
  "rw-drift-switch=default"
  "rw-twin-beam=default"
  "rw-reverse-bay=default"
  "rw-parallel-park=default"
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
args=(--bench wheeledlab --scoring-profile "$SCORING_PROFILE" --rollouts "$ROLLOUTS" --code-control "$CODE_CONTROL")
for setting in "${TASK_STEPS[@]}"; do args+=(--task-steps "$setting"); done
for setting in "${TASK_CODE_CONTROL[@]}"; do args+=(--task-code-control "$setting"); done
if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
if [[ -n "$CODEX_AUTH_HOME" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.rollouts "${args[@]}" "$@"
