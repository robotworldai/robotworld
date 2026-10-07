# RobotWorld

**A benchmark for multimodal robot use across manipulation, locomotion, driving, and flight.**

RobotWorld connects a source-built multimodal agent runtime to robot simulators through explicit observation and action interfaces. Agents can analyse observations, issue bounded robot commands, and use execution feedback to revise their actions. Task evaluators assess the resulting trajectories independently of the agent's completion claims.

This source snapshot registers **84 tasks across 20 benchmark integrations**. Registration does not imply that every task has a verified successful physical trajectory. The default batch protocol runs three rollouts per task with environment-side code control disabled. Simulation pauses during agent reasoning and workspace analysis.

## Start here

- **[Deployment guide](docs/DEPLOYMENT.md)** — host requirements, source restoration, assets, Docker images, runtime build, authentication, verification, and evaluation.
- **[Assets](docs/ASSETS.md)** — pinned downloads, upstream asset sources, and restricted datasets.
- **[Model configuration](docs/MODELS.md)** — API credentials and compatible model gateways.
- **[Evaluation guide](scripts/ROLLOUTS.md)** — per-task budgets, control modes, repeated trials, resume, and summaries.
- **[Source and deployment limitations](docs/HANDOFF.md)** — what this snapshot can reproduce and which dependencies still need provisioning.

> **Prebuilt containers:** download the 20-image bundle from [RobotWorld Images](https://huggingface.co/datasets/visity/RobotWorld-Images). Access is currently restricted to authorised accounts. Follow the [download and import guide](docs/PREBUILT_IMAGES.md) before evaluation; some legacy task profiles need additional images.

## Installation

Run the simulator stack on Linux with an NVIDIA GPU, a compatible driver, Docker with Compose/Buildx and NVIDIA Container Toolkit, Git/Git LFS, Python 3.11 or later, Rust/rustup, and bubblewrap with user namespaces enabled. Simulator-specific hardware and software requirements also apply.

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

# Download the pinned asset release, verify hashes, and restore paths.
bash scripts/setup_handoff.sh assets --bench robocasa

# Inspect the build plan, then build this benchmark's images.
bash scripts/setup_handoff.sh docker --bench robocasa --dry-run
bash scripts/setup_handoff.sh docker --bench robocasa

# Build the local app-server, login CLI, and code-mode helper.
bash scripts/setup_handoff.sh codex
```

For all integrations, omit `--bench` from the asset and image commands after provisioning the required base images. BEHAVIOR assets are downloaded separately after the user accepts the upstream licence. The [full deployment guide](docs/DEPLOYMENT.md) covers these dependencies and verification steps.

## Connect your model API

After deploying the simulator and building the agent runtime, configure your model API:

```bash
python scripts/configure_api.py \
  --base-url https://YOUR_API_HOST/v1 \
  --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"
# Set WORLD_MODEL_API_KEY through your shell or secret manager.
python scripts/check_api.py
```

The configuration stores the name of the credential environment variable, never its value. The API check makes two small model requests to verify image input and a tool-result round trip; normal provider charges apply. It does not start a simulator.

Evaluation inherits the configured model. Set `MODEL` or pass `--model` only when intentionally overriding it. Requests use the configured provider without model-name-based substitutions. See [API configuration](docs/MODELS.md).

## Run an evaluation

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

## Outputs and metrics

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

## Repository layout

| Path | Contents |
| --- | --- |
| `environment/` | Adapters, observation/action interfaces, runtime, scoring, and validation. |
| `scripts/` | Setup, evaluation, asset preparation, and reporting commands. |
| `docs/` | Deployment and protocol documentation, task specifications, and historical audits. |
| `codex/` | Bundled agent-runtime source; the independent checkout is restored from `sources.lock.json`. |
| `third_party/` | Source snapshots, integration wrappers, Docker recipes, and upstream licences. |
| `sources.lock.json` | Pinned source repositories and revisions. |
| `Assets/` | Downloaded assets; created during installation. |
| `var/`, `outputs/` | Local builds, authentication, caches, and run artifacts; excluded from source export. |

## Validation and provenance

The source for this release is `visitworld123/RoboWorld_Scaffold`, branch `main`, commit `132972eec7ea4e9a5b59f8be7398f19d7681fc52`. Deployment instructions describe that snapshot. Historical reports and task notes remain in the repository; their recorded results are not new release-validation results.

Run the CPU tests with `python -m pytest`. Task-list and Docker-plan checks do not verify physical task feasibility. GPU simulation, model connectivity, and successful trajectories require separate checks. Experimental Isaac compatibility layers are identified in run metadata and should not be treated as proof of equivalence to the original simulator version.

Third-party code and assets retain their respective licences and notices. BEHAVIOR assets and decryption keys are not redistributed. This snapshot has no top-level licence for RobotWorld-owned code; the maintainers need to select one before presenting it as a fully licensed open-source release.
