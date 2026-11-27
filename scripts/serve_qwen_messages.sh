#!/usr/bin/env bash
# Standard LiteLLM protocol translation; no Codex process or account login.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DASHSCOPE_API_KEY:?Load the third-party API key from a private environment file}"
: "${WORLD_QWEN_BASE_URL:?Set the third-party compatible-mode/v1 URL}"
: "${WORLD_LITELLM_KEY:?Set a private key for the local LiteLLM listener}"
exec "${WORLD_PYTHON:-python}" -c 'from litellm import run_server; run_server()' --config "${WORLD_LITELLM_CONFIG:-$ROOT/configs/providers/qwen-litellm.yaml}" --host 127.0.0.1 --port "${WORLD_LITELLM_PORT:-18881}"
