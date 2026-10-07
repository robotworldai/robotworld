# Deployment guide

This guide follows the scripts in this repository. Run commands from the repository root unless stated otherwise. The supported simulation deployment uses Linux and NVIDIA GPUs; macOS can be used for source inspection and CPU-only checks, but is not a substitute for the simulator host.

## 1. Prepare the host

Install:

- An NVIDIA driver compatible with the simulator versions you intend to run.
- Docker Engine, the Compose and Buildx plugins, and NVIDIA Container Toolkit configured for Docker.
- Git and Git LFS.
- Python 3.11 or later with `venv` and pip.
- Rust/rustup; the runtime pins its toolchain in `codex/codex-rs/rust-toolchain.toml`.
- A C/C++ compiler, CMake, pkg-config, OpenSSL development headers, and Clang/libclang development libraries.
- bubblewrap (`bwrap`) with usable user namespaces, plus FFmpeg/ffprobe for video inspection.

For Debian/Ubuntu, the following installs common build tools after a suitable Python version is available. Docker, NVIDIA drivers/toolkit, and Rust must also be installed using their supported platform instructions.

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config \
  libssl-dev clang libclang-dev bubblewrap ffmpeg zstd python3-venv python3-dev
git lfs install
python3 --version
nvidia-smi
docker version
docker compose version
docker buildx version
rustup --version
bwrap --ro-bind / / --proc /proc --dev /dev /bin/true
```

The bubblewrap command checks basic namespace availability. The actual agent sandbox must still pass its own runtime checks. Docker must be usable by the account running the benchmark, and NVIDIA Container Toolkit must expose the GPU inside containers. Allocate disk space for source snapshots, Rust builds, Docker layers, assets, and rollout videos. This source snapshot does not establish a universal minimum VRAM or disk requirement for all integrations.

## 2. Clone and install the Python package

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]' huggingface_hub
export WORLD_PYTHON="$PWD/.venv/bin/python"
```

Repeat the virtual-environment activation and `WORLD_PYTHON` export in new shells. The outer Python environment runs orchestration; simulator dependencies are installed in their respective images.

## 3. Restore pinned sources

```bash
python scripts/fetch_sources.py --all --dry-run
bash scripts/setup_handoff.sh sources
```

`sources.lock.json` lists every repository and commit. The restoration script fetches each commit and its submodules, keeps bundled copies under `var/source-snapshots/`, and creates independent checkouts. LFS smudging is disabled during source retrieval because assets are restored separately. Existing checkouts with the wrong revision or local modifications are rejected.

**Restore sources before assets.** Replacing a source snapshot after installing embedded asset links can remove those links. A narrower source setup is available through `python scripts/fetch_sources.py SOURCE_ID ...`, but include every shared dependency required by the selected integration.

Source restoration requires access to every pinned repository and commit. If a locked revision cannot be fetched, installation is incomplete: retain the error and resolve that dependency rather than substituting an arbitrary newer commit.

## 4. Download assets

The default asset repository and immutable revision are recorded in `environment/datasets/asset-source.json`.

```bash
# One benchmark plus its shared resources:
bash scripts/setup_handoff.sh assets --bench robocasa

# All distributed assets:
bash scripts/setup_handoff.sh assets

# Offline installation from a previously exported asset directory:
bash scripts/setup_handoff.sh assets --local-source /path/to/upload-hf
```

The downloader verifies the manifest and file SHA-256 hashes, then restores compatibility paths. Assets are stored under `Assets/`; keep this directory and the source tree on storage with adequate capacity. For private or gated Hugging Face access, authenticate locally using your account; do not put credentials in tracked files.

BEHAVIOR assets are excluded from the distributed asset pack. Read and accept the upstream terms yourself, then use:

```bash
python scripts/prepare_behavior_assets.py --accept-license --dry-run
python scripts/prepare_behavior_assets.py --accept-license
```

This prepares the selected BEHAVIOR tasks; it is not permission to redistribute their assets or keys. See [Assets](ASSETS.md) for official download alternatives and unresolved dependencies.

## 5. Prepare container images

### Import the prebuilt bundle

[RobotWorld Images](https://huggingface.co/datasets/visity/RobotWorld-Images) provides 20 runtime images, including the Isaac base image. Follow the [prebuilt-image guide](PREBUILT_IMAGES.md) to download the pinned four-part archive, verify it, and import it. The repository currently requires maintainer-granted access. This is the simplest path for the packaged runtime profiles. Compare your dry-run image tags with `images.json`; legacy profiles may require separate builds.

The following sections describe building images when a suitable prebuilt image is unavailable.

### RoboCasa: independent MuJoCo recipe

```bash
bash scripts/setup_handoff.sh docker --bench robocasa --dry-run
bash scripts/setup_handoff.sh docker --bench robocasa
```

The planner uses the repository's RoboCasa build wrapper. The HumanoidSoccer MuJoCo recipe is also independent of the Isaac 6.0.1 base:

```bash
bash scripts/setup_handoff.sh docker --bench humanoid_soccer --dry-run
bash scripts/setup_handoff.sh docker --bench humanoid_soccer
```

Successful builds still require their upstream dependencies and assets. Run a short episode after provisioning.

### Isaac-based integrations

Most dependent images require the local tag `world/robodojo:isaac6.0.1-local`. It is included in the access-controlled [prebuilt bundle](PREBUILT_IMAGES.md). Alternatively, choose one of the following provisioning approaches.

**A. Authorised registry image.** Add a mapping to `environment/containers/image-sources.json`:

```json
{
  "images": {
    "world/robodojo:isaac6.0.1-local": "YOUR_REGISTRY/YOUR_IMAGE@sha256:YOUR_DIGEST"
  }
}
```

Replace the placeholders with an image you are authorised to use and authenticate to that registry locally. The preparation script pulls and retags it. Additional final images can be mapped in the same file.

**B. Image archive supplied by a maintainer.** Verify the archive against the maintainer's checksum, run `docker load --input /path/to/image.tar`, and inspect the required tag:

```bash
docker image inspect world/robodojo:isaac6.0.1-local
```

**C. Matching local runtime snapshot.** On a machine that already has the authorised, compatible runtime, use the parameterised snapshot recipe:

```bash
docker buildx create --name world-isaac601-bounded --driver docker-container \
  --driver-opt memory=8g --driver-opt memory-swap=8g
python environment/containers/robodojo/build_isaac601_snapshot.py \
  --runtime /path/to/python312-isaac601-runtime \
  --isaaclab /path/to/pinned-IsaacLab \
  --upstream /path/to/robodojo-runtime-source-with-dependencies \
  --system-libs /path/to/required-system-libraries \
  --controls /path/to/control-python-libraries \
  --ffmpeg /path/to/ffmpeg
```

Inputs must match `environment/containers/robodojo/Dockerfile.isaac601.snapshot`. An arbitrary Python virtual environment is not equivalent. This is a snapshot recipe, not a verified clean-machine Isaac installation procedure.

After the base image is available:

```bash
bash scripts/setup_handoff.sh docker --dry-run
bash scripts/setup_handoff.sh docker
```

Use `--bench BENCHMARK` to build a subset. The script resolves local image dependencies and skips existing images. RoboLab's experimental image additionally uses configuration from its 5.0 image during construction; this does not mean the experimental run uses the 5.0 simulator. Containers share the host kernel driver.

## 6. Build the agent runtime

```bash
bash scripts/setup_handoff.sh codex
RUNTIME_REV=$(git -C codex rev-parse HEAD)
ls "var/build/codex/$RUNTIME_REV/debug/codex" \
   "var/build/codex/$RUNTIME_REV/debug/codex-app-server" \
   "var/build/codex/$RUNTIME_REV/debug/codex-code-mode-host"
```

The builder uses `cargo build --locked`, writes outputs under `var/build/`, and requires a clean independent `codex/` checkout. Inspect `var/build/codex/<revision>/build.log` if it fails. A globally installed CLI is not used as the evaluation binary.

## 7. Configure the API

```bash
python scripts/configure_api.py --base-url https://YOUR_API_HOST/v1 --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"
# Set WORLD_MODEL_API_KEY in your shell or secret manager.
python scripts/check_api.py
```

Use a streaming Responses-compatible service with image and tool support. The connection check makes two small model requests. See [API configuration](MODELS.md) for options and protocol requirements. Credentials stay outside tracked files.

## 8. Verify before starting a campaign

```bash
python -m pytest
bash scripts/run_robocasa.sh list
ROLLOUTS=1 bash scripts/run_all.sh run --dry-run
bash scripts/setup_handoff.sh docker --dry-run

# This command starts a real simulator episode and calls the selected model.
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --batch install-check
```

The first four commands check code and plans; they do not establish simulator readiness. Inspect the last command's `run.json`, event logs, and video. A valid robot-task failure still demonstrates a completed execution path; an infrastructure error requires investigation before scaling the campaign.

## 9. Run, resume, and archive

```bash
ROLLOUTS=3 bash scripts/run_all.sh run --batch model-a-full
ROLLOUTS=3 bash scripts/run_all.sh run --batch model-a-full --resume
```

Use the same model and evaluation settings when resuming. Preserve the complete `outputs/` run directories, including `artifacts/`, because video and event indexes may use relative links. See the [evaluation guide](../scripts/ROLLOUTS.md) for metrics and per-task configuration.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Locked source cannot be fetched | Repository access, commit availability, and submodule access. Do not replace the lock silently. |
| Existing checkout differs or is dirty | Inspect local changes and the locked revision. The setup script deliberately refuses to overwrite them. |
| Missing Isaac base image | Provision an authorised image or the matching snapshot before building dependent images. |
| GPU unavailable inside Docker | Host driver, NVIDIA Container Toolkit, Docker runtime configuration, and GPU visibility. |
| bubblewrap operation denied | User-namespace availability and the host's security policy. |
| Asset hash mismatch | Correct asset revision, complete download, and matching source manifest. |
| Runtime build failure | Pinned Rust toolchain, development libraries, network access, and `build.log`. |
| Model endpoint or tool-call failure | Model permission and streaming Responses/image/tool support; check gateway routing. |
| Host-local endpoint fails in RoboDojo | Its app-server runs in a container, where `localhost` means that container. |
| `success: null` | Read the status and artifacts for infrastructure or incomplete-execution errors; do not turn it into a model failure. |

Full clean-machine deployment and all-task GPU evaluation are separate validation milestones. Document the image digests, source revisions, asset revision, model configuration, and tested tasks when reporting a reproduced installation.
