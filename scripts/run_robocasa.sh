#!/usr/bin/env bash
set -euo pipefail
# robocasa: Start a fresh environment and source-built agent for every task rollout.
ROLLOUTS="${ROLLOUTS:-3}"
MODEL="${MODEL:-}"
CODE_CONTROL="${CODE_CONTROL:-off}"  # Benchmark default; per-task on/off overrides it; default inherits it.
SCORING_PROFILE="${SCORING_PROFILE:-auto}"  # Use World v1 for adapted scenarios and native scoring elsewhere.
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-}"
# native=original budget; profile=scenario budget; integer=explicit steps. BEHAVIOR defaults to min(native,2000).
TASK_STEPS=(
  "CountertopCleanup=600"  # No verified official horizon; explicit 600-step World budget.
  "SortingCleanup=native"  # Native budget=3000
  "CoffeeSetupMug=native"  # Native budget=600
  "CloseDrawer=native"  # Native budget=450
  "NavigateKitchen=native"  # Native budget=450
  "PackIdenticalLunches=native"  # Native budget=3900
  "OrganizeMugsByHandle=native"  # Native budget=1350
  "LoadDishwasher=native"  # Native budget=1800
  "MicrowaveCorrectMeal=native"  # Native budget=1500
  "ResetCabinetDoors=native"  # Native budget=3300
)
# Set each task to on/off; default inherits CODE_CONTROL (initially off).
TASK_CODE_CONTROL=(
  "CountertopCleanup=default"
  "SortingCleanup=default"
  "CoffeeSetupMug=default"
  "CloseDrawer=default"
  "NavigateKitchen=default"
  "PackIdenticalLunches=default"
  "OrganizeMugsByHandle=default"
  "LoadDishwasher=default"
  "MicrowaveCorrectMeal=default"
  "ResetCabinetDoors=default"
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
args=(--bench robocasa --scoring-profile "$SCORING_PROFILE" --rollouts "$ROLLOUTS" --code-control "$CODE_CONTROL")
for setting in "${TASK_STEPS[@]}"; do args+=(--task-steps "$setting"); done
for setting in "${TASK_CODE_CONTROL[@]}"; do args+=(--task-code-control "$setting"); done
if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
if [[ -n "$CODEX_AUTH_HOME" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.rollouts "${args[@]}" "$@"
