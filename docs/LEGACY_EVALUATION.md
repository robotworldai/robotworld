# Legacy suite workflow

These commands do not run Harbor. Retained for reproducing historical runs only. Use the [Harbor quickstart](../README.md#standard-harbor-workflow) for new evaluations.

### Observation history, tools and validation

The CloseDrawer observation packet contains the current frame plus up to four
previous frames at a two-control-step interval, from three cameras (at most
15 images). This does not automatically prune a harness's conversation.
The Codex pilot explicitly trims older packets on outgoing Responses requests;
other harnesses must implement and verify their own declared context policy.
Full archived observations and per-control-step video remain intact.

Offline shell analysis and environment-side `code_control` are separate
permissions. Neither permits bypassing the action API. Both current neutral
pilots disable shell tools and environment-side code control.

To accept a new harness: verify describe/observe, image delivery, one valid
action, stale-ID rejection, duplicate-ID idempotency, timeout receipt recovery,
native termination, complete video and native trace collection. Then run a
model-backed trial. A successful API text request alone is insufficient.

Standalone pilot (after provisioning its image and assets):

```bash
PYTHONPATH=. python scripts/run_harness_pilot.py \
  --harness probe --assets /absolute/path/to/robocasa/data \
  --output outputs/protocol/close-drawer-probe-001
# Model-backed modes: --harness claude-code or --harness codex.
# Configure the provider and verified runtime first using the guide below.
```

The native budget is 450 control steps; the default wall-clock allowance is
43200 seconds. They measure different quantities. Latest standalone DeepSeek
trials completed all 450 steps in both harnesses without task success.
Harbor migration is separate: its prior no-op validation is not a model trial.

See [the protocol and migration plan](../docs/harness-protocol.md),
[provider/runtime configuration](../docs/HARBOR_VALIDATION.md), and
[the Harbor pilot](../tasks/close-drawer/README.md).

> **Legacy suite workflow:** The sections below retain the source-built runtime and `scripts/run_*.sh` workflow. They are not additional steps for the native Harbor quickstart above. Use one profile consistently and record it with results.

## 🛠️ Installation

### Hardware and dependencies

- **Current host reference:** NVIDIA L40S 48GB (driver `580.105.08`); this describes the deployment host, not certification of all 84 tasks.
- **Other GPUs:** RTX 5090 has historical partial-task validation with experimental Isaac Sim 6.0.1 profiles; older Isaac/CUDA stacks can be incompatible. RTX 4090, RTX A6000, RTX 6000 Ada and L40 are RT-capable candidates, not a verified support list.
- **Isaac rendering:** use a GPU with RT Cores. A100/H100 are [not supported by Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html), although they can host a separate model inference service. With a remote model API, local GPUs primarily run simulation and rendering.
- **Compatibility:** match the task's pinned image, driver and CUDA/Isaac versions, then run a smoke episode before scaling. No suite-wide minimum VRAM, RAM or disk requirement has been established.

Host dependencies: Linux, a compatible NVIDIA driver, Docker with Compose/Buildx and NVIDIA Container Toolkit, Git/Git LFS, Python 3.11+, the pinned Rust toolchain, C/C++ build tools (CMake, Clang and OpenSSL headers), bubblewrap with user namespaces, FFmpeg/ffprobe and zstd. Provision the pinned simulator images and assets, including any required access or licences; see the [deployment guide](../docs/DEPLOYMENT.md).

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' huggingface_hub
export WORLD_PYTHON="$PWD/.venv/bin/python"

# Restore independent source repositories at the locked revisions.
# Do this before restoring assets.
bash scripts/setup_handoff.sh sources

# Authenticate if the asset repository requires access.
hf auth login

# Download the pinned asset release, verify hashes, and restore paths.
bash scripts/setup_handoff.sh assets --bench robocasa

# Inspect the build plan, then build this benchmark's images.
bash scripts/setup_handoff.sh docker --bench robocasa --dry-run
bash scripts/setup_handoff.sh docker --bench robocasa

# Build the local app-server, login CLI, and code-mode helper.
bash scripts/setup_handoff.sh codex
```

For all integrations, omit `--bench` from the asset and image commands after provisioning the required base images. BEHAVIOR assets are downloaded separately after the user accepts the upstream licence. The [full deployment guide](../docs/DEPLOYMENT.md) covers these dependencies and verification steps.

Use the [prebuilt image bundle](../docs/PREBUILT_IMAGES.md) instead of the Docker build commands if the required tags are already available. Existing installations can be reused after checking the pinned source revisions, asset hashes, and image tags. Do not replace them with newer versions silently.

### Direct-tool runtime option

If `codex-code-mode-host` cannot be built (for example, an unavailable upstream V8 archive), use direct tool calling for the isolated agent runtime:

```bash
export WORLD_CODEX_DISABLE_CODE_MODE=1
bash scripts/setup_handoff.sh codex
```

Keep this variable set when running the evaluation. The relay selects direct tool exposure in a private model catalogue without changing the model ID or Codex source, and records `runtime-tool-mode.json`. This was used for the RoboCasa API smoke test. It does **not** enable environment-side `code_control`; action tools, observations, budgets and scoring remain unchanged. Record this interaction setting when comparing results. Other runtime entry points require separate verification.

## 🔌 Connect Your Model API

After deploying the simulator and building the agent runtime, configure your model API. The endpoint must accept streaming **Responses API** requests with images and function calls; a Chat Completions-only endpoint is not sufficient. The examples assume the endpoint is reachable from the runtime.

```bash
python scripts/configure_api.py \
  --base-url https://YOUR_API_HOST/v1 \
  --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"
# Enter the key without echoing it or putting it in shell history.
read -rsp "Model API key: " WORLD_MODEL_API_KEY; printf '\n'
export WORLD_MODEL_API_KEY
python scripts/check_api.py
```

The configuration stores the name of the credential environment variable, never its value. The API check makes two small model requests to verify image input and a tool-result round trip; normal provider charges apply. It does not start a simulator.

Evaluation inherits the configured model. Set `MODEL` or pass `--model` only when intentionally overriding it. Requests use the configured provider without model-name-based substitutions. See [API configuration](../docs/MODELS.md).

## 🚀 Run an Evaluation

Run commands from the repository root with `WORLD_PYTHON` and `CODEX_AUTH_HOME` set as above.

```bash
# Inspect tasks, effective budgets, scoring profiles, and control settings.
bash scripts/run_robocasa.sh list

# Inspect the complete campaign without calling a model or simulator.
ROLLOUTS=1 bash scripts/run_all.sh run --dry-run

# Start with one task and one rollout.
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --batch drawer-smoke

# Run all selected RoboCasa tasks, three times each.
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01

# Override a task budget explicitly.
bash scripts/run_robocasa.sh run --tasks CloseDrawer \
  --task-steps CloseDrawer=450 --rollouts 3 --batch drawer-01

# Enable environment-side code control for a supported BEHAVIOR task.
bash scripts/run_behavior_1k.sh run --tasks clean_up_your_desk \
  --task-code-control clean_up_your_desk=on --rollouts 2 --batch desk-code-01

# Run selected benchmarks sequentially.
BENCHMARKS='robocasa robolab' ROLLOUTS=2 bash scripts/run_all.sh run --batch subset-01

# Run the full registered suite after provisioning all dependencies.
ROLLOUTS=3 bash scripts/run_all.sh run --batch model-a-full

# Retry unfinished task/rollout slots with the same settings and batch ID.
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01 --resume
```

`--resume` restarts unfinished rollouts from their initial state and preserves previous attempts. It does not restore an interrupted physics state. Use a new batch ID when changing the model, budgets, assets, or tool conditions.

### Evaluation settings

| Setting | Meaning |
| --- | --- |
| `TASK_STEPS` / `--task-steps` | Per-task `native`, `profile`, or a positive step limit within the known scenario limit. |
| `CODE_CONTROL` | Benchmark default for environment-side generated control programs; initially `off`. |
| `TASK_CODE_CONTROL` / `--task-code-control` | Per-task `on`, `off`, or `default`; explicit values override the benchmark default. |
| `ROLLOUTS` / `--rollouts` | Trials **per task**; default: 3. |
| `MODEL` / `CODEX_AUTH_HOME` | Model ID and authentication/provider directory. |
| `SCORING_PROFILE` | `auto` uses the registered World rules for adapted scenarios and native rules elsewhere; `native` selects original objectives, budgets, and scoring. |
| `--batch` | Experiment identifier. |
| `--output-root` | Output location; defaults to `outputs/`. |
| `--timeout` | Wall-clock limit per run; default: 43,200 seconds, separate from simulation steps. |
| `--seed` / `--seed-stride` | Seed configuration where the integration supports randomisation. |

Later CLI arguments override the corresponding script settings. RoboDojo, RoboCasa, and RoboLab do not expose environment-side code control in this snapshot; requesting it does not add that capability, and the effective setting is recorded as false. Disabling code control still allows offline computation on permitted observations.

BEHAVIOR uses `min(native_budget, 2000)`. `CountertopCleanup` has an explicit 600-step World budget because its official horizon is unverified in the pinned source. Other task budgets are recorded in the catalogue. Native termination conditions remain active.

## 📊 Outputs and Metrics

```text
outputs/<benchmark>/<task>/run-<batch>-0001/
  run.json                 # Settings, effective budgets, status, success, and raw score
  videos/                  # Relative links to MP4 files
  events/                  # Agent, tool, and environment events; image-free copies
  programs/                # Index of generated control programs
  agent-workspace/         # Agent files, where supported by the integration
  artifacts/               # Original benchmark output; retain with the links above
outputs/<benchmark>/summaries/<batch>.json
outputs/<benchmark>/summaries/<batch>.csv
outputs/<benchmark>/summaries/<batch>-runs.csv
```

Per-task success rate is the mean of valid Boolean outcomes. Benchmark summaries average task success rates with equal task weights and also provide pooled rollout statistics. Population and sample variance are reported separately; sample variance is undefined for a single observation. Raw scores with different units are retained per task.

A valid failure is `false`. Infrastructure errors and incomplete or indeterminate outcomes are unscored (`null`), with coverage reported explicitly. Incomplete coverage does not constitute a complete benchmark score. Copy the whole run directory when archiving results so that relative artifact links remain usable.

