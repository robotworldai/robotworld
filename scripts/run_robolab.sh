#!/usr/bin/env bash
set -euo pipefail
# robolab: Start a fresh environment and source-built agent for every task rollout.
ROLLOUTS="${ROLLOUTS:-3}"
MODEL="${MODEL:-}"
CODE_CONTROL="${CODE_CONTROL:-off}"  # Benchmark default; per-task on/off overrides it; default inherits it.
SCORING_PROFILE="${SCORING_PROFILE:-auto}"  # Use World v1 for adapted scenarios and native scoring elsewhere.
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-}"
# native=original budget; profile=scenario budget; integer=explicit steps. BEHAVIOR defaults to min(native,2000).
TASK_STEPS=(
  "ToolOrganizationTask=native"  # Native budget=2700
  "NonHammerToolsInRightBinTask=native"  # Native budget=2700
  "FoodPacking2CansTask=native"  # Native budget=2700
  "RubiksCubeLeftOfBowlTask=native"  # Native budget=450
  "FruitsOnPlate3Task=native"  # Native budget=3000
  "PutTwoMugsOnShelfTask=native"  # Native budget=2700
  "BlockStackingSpecifiedOrderTask=native"  # Native budget=1350
  "ClutterPlasticTask=native"  # Native budget=2700
  "ReorientWhiteMugsTask=native"  # Native budget=900
  "WhiteMugInCenterOfTableTask=native"  # Native budget=450
)
# Set each task to on/off; default inherits CODE_CONTROL (initially off).
TASK_CODE_CONTROL=(
  "ToolOrganizationTask=default"
  "NonHammerToolsInRightBinTask=default"
  "FoodPacking2CansTask=default"
  "RubiksCubeLeftOfBowlTask=default"
  "FruitsOnPlate3Task=default"
  "PutTwoMugsOnShelfTask=default"
  "BlockStackingSpecifiedOrderTask=default"
  "ClutterPlasticTask=default"
  "ReorientWhiteMugsTask=default"
  "WhiteMugInCenterOfTableTask=default"
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
args=(--bench robolab --scoring-profile "$SCORING_PROFILE" --rollouts "$ROLLOUTS" --code-control "$CODE_CONTROL")
for setting in "${TASK_STEPS[@]}"; do args+=(--task-steps "$setting"); done
for setting in "${TASK_CODE_CONTROL[@]}"; do args+=(--task-code-control "$setting"); done
if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
if [[ -n "$CODEX_AUTH_HOME" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.rollouts "${args[@]}" "$@"
