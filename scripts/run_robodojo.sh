#!/usr/bin/env bash
set -euo pipefail
# robodojo: Start a fresh environment and source-built agent for every task rollout.
ROLLOUTS="${ROLLOUTS:-3}"
MODEL="${MODEL:-}"
CODE_CONTROL="${CODE_CONTROL:-off}"  # Benchmark default; per-task on/off overrides it; default inherits it.
SCORING_PROFILE="${SCORING_PROFILE:-auto}"  # Use World v1 for adapted scenarios and native scoring elsewhere.
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-}"
# native=original budget; profile=scenario budget; integer=explicit steps. BEHAVIOR defaults to min(native,2000).
TASK_STEPS=(
  "hang_mugs=native"  # Native budget=800
  "sweep_blocks=native"  # Native budget=1000
  "pour_liquid_into_cup=native"  # Native budget=400
  "make_toast=native"  # Native budget=1400
  "store_laptop_and_headphones=native"  # Native budget=800
  "insert_tubes=native"  # Native budget=500
  "plug_in_charger=native"  # Native budget=400
  "pour_balls_into_vase=native"  # Native budget=600
  "play_Xylophone=native"  # Native budget=500
  "fill_pen_holder=native"  # Native budget=1100
  "fill_egg_holder=native"  # Native budget=700
  "make_kong=native"  # Native budget=600
  "pour_by_language=native"  # Native budget=800
  "match_and_pick_from_conveyor=native"  # Native budget=700
  "deposit_coin=native"  # Native budget=300
)
# Set each task to on/off; default inherits CODE_CONTROL (initially off).
TASK_CODE_CONTROL=(
  "hang_mugs=default"
  "sweep_blocks=default"
  "pour_liquid_into_cup=default"
  "make_toast=default"
  "store_laptop_and_headphones=default"
  "insert_tubes=default"
  "plug_in_charger=default"
  "pour_balls_into_vase=default"
  "play_Xylophone=default"
  "fill_pen_holder=default"
  "fill_egg_holder=default"
  "make_kong=default"
  "pour_by_language=default"
  "match_and_pick_from_conveyor=default"
  "deposit_coin=default"
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
args=(--bench robodojo --scoring-profile "$SCORING_PROFILE" --rollouts "$ROLLOUTS" --code-control "$CODE_CONTROL")
for setting in "${TASK_STEPS[@]}"; do args+=(--task-steps "$setting"); done
for setting in "${TASK_CODE_CONTROL[@]}"; do args+=(--task-code-control "$setting"); done
if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
if [[ -n "$CODEX_AUTH_HOME" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.rollouts "${args[@]}" "$@"
