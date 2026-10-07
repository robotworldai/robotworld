#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo
: "${WORLD_ROOT:?Mount the World checkout read-only at its original absolute path}"
: "${WORLD_OUTPUT:=/runs/episode}"
export PYTHONPATH="${WORLD_ROOT}:/opt/world-control-libs:/workspace/RoboDojo:/workspace/RoboDojo/XPolicyLab"
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${WORLD_ROOT}/codex"
mkdir -p "$WORLD_OUTPUT"
cd "$WORLD_OUTPUT"
mode="${1:-probe}"
if [ "$#" -gt 0 ]; then shift; fi
case "$mode" in
  probe)
    exec python -m environment.integrations.robodojo_smoke \
      --root /workspace/RoboDojo --output "$WORLD_OUTPUT" \
      --probe-only --headless --enable_cameras "$@"
    ;;
  smoke)
    : "${WORLD_CODEX_MANIFEST:?Specify the source-build manifest}"
    : "${WORLD_MODEL_CATALOG:?Specify a container-accessible direct-tool catalog}"
    exec python -m environment.integrations.robodojo_smoke \
      --root /workspace/RoboDojo --output "$WORLD_OUTPUT" \
      --manifest "$WORLD_CODEX_MANIFEST" --model-catalog "$WORLD_MODEL_CATALOG" \
      --headless --enable_cameras "$@"
    ;;
  doctor)
    nvidia-smi
    python -c 'import sys, importlib.metadata as m; print(sys.version); print({p:m.version(p) for p in ("isaacsim","isaaclab","torch","nvidia-curobo","websockets")})'
    cat /opt/world-source-manifest.json
    ;;
  *) exec "$mode" "$@" ;;
esac
