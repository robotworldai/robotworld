#!/usr/bin/env bash
set -euo pipefail
trap 'exit 130' INT
trap 'exit 143' TERM
# Run sequentially to avoid GPU contention; wrappers retain their own default budgets.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# BENCHMARKS selects a space-separated subset. All registered entries are selected by default; blocked entries remain errors.
BENCHMARKS="${BENCHMARKS:-robodojo behavior_1k robocasa robolab humanoid_soccer ai_cps wheeledlab bench2dex digit flamingo go2_push omnidrones omniisaacgymenvs reflexbench robot_lab steadytray ttrl volleybots wheel_legged wheeled_quadruped}"
BATCH="${BATCH:-$(date +%Y%m%dT%H%M%S)-$$}"
export ROLLOUTS="${ROLLOUTS:-3}"
export CODE_CONTROL="${CODE_CONTROL:-off}"
status=0
for bench in $BENCHMARKS; do
  case "$bench" in
    robodojo|behavior_1k|robocasa|robolab|humanoid_soccer|ai_cps|wheeledlab|bench2dex|digit|flamingo|go2_push|omnidrones|omniisaacgymenvs|reflexbench|robot_lab|steadytray|ttrl|volleybots|wheel_legged|wheeled_quadruped) ;;
    *) echo "Unknown benchmark: $bench" >&2; exit 2 ;;
  esac
  bash "$ROOT/scripts/run_${bench}.sh" --batch "$BATCH" "$@" || status=1
done
exit "$status"
