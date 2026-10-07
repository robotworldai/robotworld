#!/usr/bin/env bash
set -euo pipefail
# Native 1v1 rules, experimental Isaac6 runtime, frozen World-scripted opponent.
# MODE=zero/probe: unscored diagnostic; no model API calls.
# MODE=codex: local source Codex player1 receives from scripted player0.
MODE="${MODE:-codex}"
CODE_CONTROL="${CODE_CONTROL:-off}"  # on enables per-tick model-written feedback; report separately
MODEL="${MODEL:-}"
SEED="${SEED:-7}"
export WORLD_VOLLEY_PLAYER="${PLAYER:-1}"  # Which side is replaced by Codex in codex mode.
STEPS="${STEPS:-1000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-$ROOT/var/auth/robodojo-codex}"
OUTPUT="${OUTPUT:-$ROOT/outputs/volleybots/drone_volleyball_1v1/run-$(date +%Y%m%d-%H%M%S)}"
cd "$ROOT"
args=()
case "$CODE_CONTROL" in
  off) args+=(--disable-coding-control);;
  on) ;;
  *) echo "CODE_CONTROL must be on or off" >&2; exit 2;;
esac
if [[ -n "${CODEX_AUTH_HOME:-}" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.runtime.native_project_launch \
 --project volleybots --task T05 --runtime-profile isaac6-scripted-1v1 \
 --mode "$MODE" --model "$MODEL" --seed "$SEED" --steps "$STEPS" --output "$OUTPUT" "${args[@]}" "$@"
