#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
for arg in "$@"; do
  if [ "$arg" = --verify-only ]; then exec python3 "$here/download_assets.py" "$@"; fi
done
# Fully verified local assets need neither network nor the HF Python dependency.
if python3 "$here/download_assets.py" "$@" --verify-only >/dev/null 2>&1; then
  exec python3 "$here/download_assets.py" "$@" --verify-only
fi
venv="$root/var/venvs/asset-downloader"
if [ ! -x "$venv/bin/python" ]; then python3 -m venv "$venv"; fi
if ! "$venv/bin/python" -c 'import huggingface_hub' >/dev/null 2>&1; then
  "$venv/bin/python" -m pip install 'huggingface_hub>=1,<3'
fi
exec "$venv/bin/python" "$here/download_assets.py" "$@"
