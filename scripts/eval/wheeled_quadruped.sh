#!/usr/bin/env bash
set -euo pipefail
WORLD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="$WORLD/third_party/benchmarks/wheeled_quadruped"
PYTHON_BIN="${WORLD_PYTHON:-python3}"
mode="${1:-list}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  list)
    "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["repository"],p["commit"]); [print(k,t["id"],str(t["steps"])+" steps",t["title"]) for k,t in p["tasks"].items()]' "$PROJECT/project.json"
    ;;
  assets)
    if [[ ! -d "$PROJECT/checkout/.git" ]]; then "$PYTHON_BIN" "$PROJECT/fetch_sources.py"; fi
    "$PYTHON_BIN" "$PROJECT/prepare_assets.py" --write-manifest
    ;;
  build)
    docker build -t world/wheeled-quadruped:isaac6.0.1-experimental -f "$PROJECT/docker/Dockerfile" "$PROJECT/docker"
    ;;
  run|probe)
    task="${1:?Supply T09, or all}"
    shift
    output_root="${WORLD_RUN_ROOT:-$WORLD/var/runs/docker/wheeled_quadruped}"
    tasks=("$task")
    if [[ "$task" == all ]]; then tasks=(T09); fi
    for task in "${tasks[@]}"; do
      run_id="$(date -u +%Y%m%dT%H%M%SZ)-$task-$mode"
      if [[ "$mode" == probe ]]; then
        "$PYTHON_BIN" "$PROJECT/docker/run.py" --task "$task" --mode probe --steps 4 --output "$output_root/$run_id" "$@"
      else
        # --codex-home may also be supplied explicitly after the task selector.
        args=(--task "$task" --mode codex --model "${WORLD_MODEL:-gpt-6-astra}" --output "$output_root/$run_id")
        if [[ -n "${WORLD_CODEX_HOME:-}" ]]; then args+=(--codex-home "$WORLD_CODEX_HOME"); fi
        "$PYTHON_BIN" "$PROJECT/docker/run.py" "${args[@]}" "$@"
      fi
    done
    ;;
  *) echo 'Usage: wheeled_quadruped.sh list|assets|build|probe T09|all [options]|run T09|all --codex-home PATH [options]' >&2; exit 2;;
esac
