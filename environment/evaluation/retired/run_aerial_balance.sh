#!/usr/bin/env bash
set -euo pipefail
# aerial_balance: 每个 run 是一次独立 rollout；默认遵循上游 horizon。
# 修改这里，或用同名环境变量 / 后面的 CLI 参数覆盖。
CODE_CONTROL="${CODE_CONTROL:-on}"  # on/off；不提供此工具的 bench 会记录 effective=false
ROLLOUTS="${ROLLOUTS:-3}"
MODEL="${MODEL:-gpt-6-astra}"
# CODEX_AUTH_HOME 指向已登录或配置 provider 的目录；不使用系统安装的 Codex。
CODEX_AUTH_HOME="${CODEX_AUTH_HOME:-}"
# 各题可将 native 改为正整数；上限不能超过原生 horizon。
TASK_STEPS=(
  "T04=native"  # 600; native
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
args=(--bench aerial_balance --rollouts "$ROLLOUTS" --code-control "$CODE_CONTROL" --model "$MODEL")
for setting in "${TASK_STEPS[@]}"; do args+=(--task-steps "$setting"); done
if [[ -n "$CODEX_AUTH_HOME" ]]; then args+=(--codex-home "$CODEX_AUTH_HOME"); fi
exec "${WORLD_PYTHON:-python3}" -m environment.evaluation.rollouts "${args[@]}" "$@"
