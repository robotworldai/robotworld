#!/usr/bin/env bash
set -euo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v docker >/dev/null || { echo 'Docker Engine is required; no image was built.' >&2; exit 2; }
docker info >/dev/null
cd "$bundle"
sha256sum --check SHA256SUMS
image="${WORLD_ROBODOJO_IMAGE:-world/robodojo:local}"
base="${image}-upstream"
build_env="${image}-build-env"
curobo_version="$(python3 -c 'import json; print("0.0.0+git." + json.load(open("source-manifest.json"))["sources"]["third_party/curobo"]["commit"][:12])')"
docker build --build-arg "CUROBO_VERSION=$curobo_version" \
  -f Dockerfile.build-env -t "$build_env" .
# Use the exported upstream Dockerfile without changing any line.
docker build --build-arg "CUDA_IMAGE=$build_env" -t "$base" upstream
docker build --build-arg "ROBODOJO_BASE=$base" -t "$image" .
docker image inspect "$image" > image-inspect.json
echo "Built $image. Run ./run.sh doctor, then ./run.sh probe."
