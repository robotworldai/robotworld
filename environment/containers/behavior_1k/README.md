# BEHAVIOR-1K container

Use the upstream `stanfordvl/behavior:3.9.3` image, not the RoboDojo Isaac Sim 6.0.1 snapshot. The image has been pulled and its digest recorded in `environment/registry/behavior_1k.json`. The first scene probe failed during Isaac Sim startup, before scene loading or Codex execution; see `environment/benchmarks/behavior_1k/STATUS.md`.

The local source is pinned to v3.9.3 (`6cbf70b075816096e9be53958780769f3264d25d`) and mounted read-only at `/behavior-src`. World and Codex sources are also read-only; caches, output and the user's dedicated Codex auth directory are writable. No upstream patch and no global Codex binary. Runtime verifies the source-built Codex manifest/hash. Dataset mount is read-only for evaluation.

## Preparation

User authorized isolated Docker setup and accepted the official dataset license on 2026-09-26. Use the standalone official image, not a host conda install.

```bash
docker pull stanfordvl/behavior:3.9.3
export WORLD_BEHAVIOR_DATA=/opt/robotworld/World/var/datasets/behavior_1k
export WORLD_ROOT="$PWD"
bash environment/containers/behavior_1k/prepare-assets.sh --accept-license
```

Read the upstream asset agreement before passing --accept-license. The downloader uses verified HTTP ranges to extract only carrying_in_groceries dependencies, R1Pro and public instance 311. It preserves original asset bytes and obtains the key with the official function; it does not download the complete scene archive. Unlike the RoboDojo subset distribution, do not mirror BEHAVIOR assets or the decryption key to GitHub/HF. Existing licensed datasets can instead be passed with `--assets`.

## Run a development instance

From World:

```bash
python3 environment/containers/behavior_1k/run.py \
  --source "$PWD/third_party/benchmarks/behavior_1k/checkout" \
  --assets "$WORLD_BEHAVIOR_DATA" \
  --codex-home "$PWD/var/auth/robodojo-codex" \
  --task-name carrying_in_groceries --instance-index 10 \
  --output "$PWD/var/runs/docker/behavior_1k/groceries-dev10-001"
```

Use your own Codex auth directory. A shared login directory should not be used by concurrent runs. `--dry-run` prints the exact command without starting Docker. Output directories must be new. Default memory 48 GB, no extra swap, 8 GB shared memory, 3600s outer wall limit. Override via CLI if needed; wall-clock timeout is an infrastructure limit, distinct from official simulator-step budget. No privileged container or host-network permission is required.

No default `--max-steps` override is supplied: the official evaluator uses its task-specific timeout. `--max-steps` is only for diagnostic runs and recorded as nonstandard. Videos use the official per-step writer, independently of model history sampling.

## Current status

Code/controller/protocol tests pass without simulator. The selected assets and official key are prepared. Isaac Sim 5.1 startup currently fails on this host; asset closure, IK tracking, source Codex execution inside this image and end-to-end evaluation remain unverified.

## Isaac 6.0.1 external compatibility, upstream scene renderer

`Dockerfile.isaac601` derives a local image from the already built RoboDojo 6.0.1 snapshot (shared Docker layers). This is a local runtime reuse recipe, not an independent official BEHAVIOR image. Additional Python packages are installed in the derived image only; sources and assets are mounted read-only. Dependency versions are recorded at `/opt/world-behavior/packages.txt`.

```bash
docker build --build-context behavior_source=third_party/benchmarks/behavior_1k/checkout -t world/behavior:isaac6.0.1-experimental \
  -f environment/containers/behavior_1k/Dockerfile.isaac601 environment/containers/behavior_1k
python3 environment/containers/behavior_1k/run.py \
  --image world/behavior:isaac6.0.1-experimental --isaac601-compat --render-diagnostics \
  --assets "$PWD/var/datasets/behavior_1k" \
  --codex-home "$PWD/var/auth/robodojo-codex" \
  --output "$PWD/var/runs/docker/behavior_1k/groceries-isaac601-probe" --probe-only
```

The probe command above uses the upstream R1Pro robot configuration and preserves the upstream RealTimePathTracing renderer. It captures 60 stationary frames per camera without physics steps. Model runs (without --probe-only) still use the external IK tool profile, and are not an unchanged official robot-control baseline.

The compatibility layer is opt-in and records its PR commit and hashes in each run. Isaac 5.1 and 6.0.1 use separate caches. See the benchmark STATUS.md for actual validation results.

The historical RayTracedLighting override is disabled. Repeated upstream-renderer probes now complete; initialization takes several minutes, including a measured ~12-second render call. Use render-events.jsonl and progress.json to distinguish initialization from a persistent hang. Darkness remains under upstream exposure and is being audited; no scene light or image enhancement is applied.

