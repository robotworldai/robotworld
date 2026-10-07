#!/usr/bin/env bash
set -euo pipefail
# behavior_1k: Start a fresh environment and source-built agent for every task rollout.
ROLLOUTS="${ROLLOUTS:-3}"
MODEL="${MODEL:-}"
CODE_CONTROL="${CODE_CONTROL:-off}"  # Benchmark default; per-task on/off overrides it; default inherits it.
SCORING_PROFILE="${SCORING_PROFILE:-auto}"  # Use World v1 for adapted scenarios and native scoring elsewhere.
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-}"
# native=original budget; profile=scenario budget; integer=explicit steps. BEHAVIOR defaults to min(native,2000).
TASK_STEPS=(
  "carrying_in_groceries=2000"  # Native budget=21412
  "clean_up_your_desk=2000"  # Native budget=32126
  "slicing_vegetables=2000"  # Native budget=22267
  "sorting_vegetables=2000"  # Native budget=17855
  "clean_boxing_gloves=2000"  # Native budget=12353
  "putting_up_Christmas_decorations_inside=2000"  # Native budget=20578
  "setting_the_table=2000"  # Native budget=26712
  "putting_dishes_away_after_cleaning=2000"  # Native budget=16430
  "can_meat=2000"  # Native budget=17770
  "freeze_pies=2000"  # Native budget=18682
)
# Set each task to on/off; default inherits CODE_CONTROL (initially off).
TASK_CODE_CONTROL=(
  "carrying_in_groceries=default"
  "clean_up_your_desk=default"
  "slicing_vegetables=default"
  "sorting_vegetables=default"
  "clean_boxing_gloves=default"
  "putting_up_Christmas_decorations_inside=default"
  "setting_the_table=default"
  "putting_dishes_away_after_cleaning=default"
  "can_meat=default"
  "freeze_pies=default"
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
args=(--bench behavior_1k --scoring-profile "$SCORING_PROFILE" --rollouts "$ROLLOUTS" --code-control "$CODE_CONTROL")
for setting in "${TASK_STEPS[@]}"; do args+=(--task-steps "$setting"); done
for setting in "${TASK_CODE_CONTROL[@]}"; do args+=(--task-code-control "$setting"); done
if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
if [[ -n "$CODEX_AUTH_HOME" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.rollouts "${args[@]}" "$@"
