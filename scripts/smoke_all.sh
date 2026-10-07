#!/usr/bin/env bash
set -euo pipefail
# One real source-built agent rollout per benchmark with native step limits. Default: list; run calls the model.
CODE_CONTROL="${CODE_CONTROL:-on}"
MODEL="${MODEL:-gpt-6-astra}"
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
args=(--code-control "$CODE_CONTROL" --model "$MODEL")
if [[ -n "$CODEX_AUTH_HOME" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.smoke "${args[@]}" "$@"
