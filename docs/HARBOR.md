# RobotWorld with Harbor

For the recommended built-in agent workflow, start with the [root quickstart](../README.md#standard-harbor-workflow). For changes and regression checks, use the [developer guide](HARBOR_DEVELOPMENT.md). The probe and source-built adapters below are separate diagnostic or legacy profiles, not prerequisites for the built-in workflow.

[Validation evidence and pending items / 验收证据与待办](HARBOR_VALIDATION.md).

[简体中文](HARBOR.zh-CN.md) · [Episode API](harness-protocol.md) · [Installation](DEPLOYMENT.md)

**Use standard Harbor tasks, agents, jobs and trials; RobotWorld provides the physical environment and action API.** A task contains no model-specific code. Changing the agent or model must not change physics, observations, initial-state sampling or scoring.

This integration targets **Harbor 0.24.0**. All 84 selected tasks have exported
standard packages; export/schema validation is not per-scene runtime acceptance.
Use the [acceptance record](HARBOR_VALIDATION.md) for actual execution evidence.

A **task** defines one environment and its goal. An **agent** runs the harness
that communicates with a model and the task. A **trial** is one agent attempt on
one task; a **job** schedules and collects trials. Change a task in `tasks/`, an
agent implementation in `agents/`, and the evaluation selection or repetitions
in `configs/jobs/`. `n_attempts` repeats a task configuration; it does not
implicitly randomise its seed.

## What runs where

| Harbor component | Contents | Authority |
|---|---|---|
| Host scheduler | Job configuration, trial creation and collection | Starts and stops containers |
| `main` service | Installed agent, model client, RobotWorld HTTP/MCP client | Reads public observations and submits actions |
| `simulator` service | Native benchmark, assets, episode service | Advances physics, records video and evaluates state |
| Separate verifier | Trusted grading code and collected simulator artifacts | Emits reward and outcome details |

A remote model API does not require a GPU in `main`. Simulation images remain benchmark-specific. The agent receives no Docker socket, simulator source, assets or writable simulator-results volume. Provider credentials belong only in the agent's runtime configuration. The episode bearer token is a different credential used solely between agent and simulator.

## Install and run a standard task

First follow the [deployment guide](DEPLOYMENT.md) to provision Docker GPU support, the benchmark image and its licensed assets. Install Harbor in its own Python environment:

```bash
python -m pip install 'harbor==0.24.0'
# From the repository root:
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_ROOT="$PWD"
# Validate the asset root and prepare the mountpoint before Docker mounts the checkout read-only.
WORLD_ASSETS="$(python scripts/check_harbor_assets.py "${WORLD_ASSETS:-$PWD/Assets}" --prepare-mount "$WORLD_ROOT")"
export WORLD_ASSETS
export WORLD_NVIDIA_RUNTIME="$(python scripts/harbor_gpu_runtime.py)"
export WORLD_EPISODE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
harbor run --path tasks/close-drawer --agent-import-path agents.robotworld:RobotWorldProbeAgent \
  --n-concurrent 1 --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

The scripted probe executes all 450 native control steps and produces a complete video and scored outcome. It tests infrastructure, not policy performance or task feasibility. By contrast, `--agent nop` finalizes without actions. The completion verifier returns 0, but the host audit excludes this probe from model scores. Run probes in separate jobs: raw Harbor reward aggregation does not distinguish them from model agents. Model runs require a configured installed agent and credentials; installing Harbor alone does not install every agent runtime or provision assets.

Harbor accepts a custom installed agent directly:

```bash
# Replace the import path and model with your installed implementation:
harbor run --path tasks/close-drawer \
  --agent-import-path your_package.agent:YourInstalledAgent --model your-model-id \
  --n-concurrent 1 --n-attempts 3 --jobs-dir "$PWD/outputs/harbor"
```

Use `harbor agent schema <agent>` to inspect supported arguments; `--ak key=value` supplies agent-specific configuration. A custom agent must implement Harbor's installed-agent interface and install or launch its runtime inside `main`. RobotWorld's supplied agents are documented in [agents/README.md](../agents/README.md). Their tool, continuation and image policies are explicit. Harbor's built-in agents are separate profiles until those behaviours have been compared.

### Optional legacy profile: supplied installed agents

Reuse a verified source build for Codex, and prepare a relocatable runtime bundle
(the provider configuration contains variable names, not secret values):

```bash
export WORLD_CODEX_SOURCE=/absolute/path/to/verified/codex/source
python scripts/prepare_harbor_codex.py \
  --provider-config /private/path/provider.json --output var/harbor-runtimes/codex-provider
export WORLD_HARBOR_CODEX_RUNTIME="$PWD/var/harbor-runtimes/codex-provider"
# Load the provider's credential environment privately before starting Harbor.
harbor run --path tasks/close-drawer \
  --agent-import-path agents.robotworld:RobotWorldCodexAgent --model your-model-id \
  --n-concurrent 1 --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

For Claude Code, provision the pinned official CLI (currently 2.1.292) and a
Messages-compatible endpoint reachable from the container:

```bash
export WORLD_HARBOR_CLAUDE_BINARY=/absolute/path/to/claude
# Privately load ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY.
harbor run --path tasks/close-drawer \
  --agent-import-path agents.robotworld:RobotWorldClaudeCodeAgent --model your-model-id \
  --n-concurrent 1 --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

These adapters inject selected credentials at process launch; do not put key
values into `--agent-env` or job files. The simulator does not receive them.
For a no-model full-budget CloseDrawer transport check, select
`agents.robotworld:RobotWorldProbeAgent`. This scripted probe is not a reference
solution and must not be included in model rankings.

A standard job configuration can replace CLI flags:

```yaml
# Example job.yaml, run from this repository's root.
jobs_dir: outputs/harbor
n_attempts: 3
n_concurrent_trials: 1
agents:
  - import_path: agents.robotworld:RobotWorldProbeAgent
tasks:
  - path: tasks/close-drawer
```

```bash
harbor run -c job.yaml --jobs-dir "$PWD/outputs/harbor"
```

Use an absolute `--jobs-dir` with Harbor 0.24.0: Compose can otherwise resolve artifact-copy paths relative to the task directory.

There is no separate RobotWorld task scheduler to learn. Existing Harbor tooling can consume the task directory and its verifier reward. Remote workers must have the images and assets available at their own configured paths; a local bind mount is not a remotely published asset package.

## 100-step scene smoke test

This is a no-model deployment diagnostic, separate from model evaluation. Prepare
pinned sources first, restore assets next, and provision every image requested
by the selected task packages:

```bash
# Follow DEPLOYMENT.md for host prerequisites and the Python installation.
bash scripts/setup_handoff.sh sources
bash scripts/setup_handoff.sh assets
# If assets already exist, restore their runtime links instead of downloading:
# python scripts/restore_assets.py --apply
# Import images following PREBUILT_IMAGES.md; build any missing task profile.
python -m pip install 'harbor==0.24.0'
export WORLD_ROOT="$PWD"
# Validate the asset root and prepare the mountpoint before Docker mounts the checkout read-only.
WORLD_ASSETS="$(python scripts/check_harbor_assets.py "${WORLD_ASSETS:-$PWD/Assets}" --prepare-mount "$WORLD_ROOT")"
export WORLD_ASSETS
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_NVIDIA_RUNTIME="$(python scripts/harbor_gpu_runtime.py)"
export WORLD_EPISODE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
harbor run --config configs/jobs/robotworld-smoke100.yaml \
  --jobs-dir "$PWD/outputs/harbor"
```

Asset download does not include restricted BEHAVIOR data: follow its individual
licence and official download procedure in [ASSETS.md](ASSETS.md). Image import
restores only the bundle's listed tags; it does not guarantee every legacy or
experimental runtime is present. The host still needs a compatible GPU driver
and Docker GPU support. Existing matching local installations may be reused.

The launch scripts select the registered NVIDIA runtime for RoboDojo on Docker
hosts using the default `runc` runtime. Set `WORLD_NVIDIA_RUNTIME` to select a
runtime explicitly, or set it to an empty string to use the daemon default.

The variables above are environment wiring: `WORLD_ROOT` and `WORLD_ASSETS`
select container mounts, `PYTHONPATH` exposes the installed-agent module, and
`WORLD_EPISODE_TOKEN` authenticates the episode API. Tasks, agents, jobs, trials
and verification follow Harbor's existing conventions. The 100-step stopping
condition belongs to `agents.smoke100:RobotWorldSmokeAgent`; it does not rewrite
`task.toml`, change the task's native budget, or replace its success checker.

For convenience only, `bash scripts/run_harbor_smoke.sh` wraps that same native
Harbor command. The wrapper locates this repository automatically, defaults `WORLD_ASSETS` to
`<repository>/Assets`, creates an episode token if needed, and uses an absolute
`<repository>/outputs/harbor` results directory. Override `HARBOR` to select an
installed CLI and `WORLD_ASSETS` to reuse an external asset tree. All script
arguments are forwarded to `harbor run`, for example:

```bash
HARBOR=/absolute/path/to/venv/bin/harbor \
WORLD_ASSETS=/absolute/path/to/Assets \
bash scripts/run_harbor_smoke.sh --job-name smoke100-local-01
```

Use a new job name for each fresh batch. The default configuration selects all
84 tasks, one attempt each, with one concurrent trial. Edit its `tasks` list for
a subset and `n_concurrent_trials` for suitable host capacity; no model API key
is needed. The imported agent is `agents.smoke100:RobotWorldSmokeAgent`.

The smoke job uses `configs/jobs/robotworld-smoke100.yaml`, with one trial per
task. Its scripted agent submits legal actions through the same public episode
API used by model agents. Its 100-step limit counts native **control steps**,
not model turns, tool calls, video frames or physics substeps. The original
budget, native success/failure and terminal state take precedence. A task that
ends before step 100 must record the actual count and reason; do not mask its
termination or reset it mid-trial to manufacture a longer run. Macro-action
interfaces report actual executed steps rather than claiming one command is one
step.

The check establishes scene loading/reset, accepted action execution, observation
transport, recording/finalisation, and verifier execution. Neutral or scripted
actions need not solve the task. A native failure is distinct from an
infrastructure error. A readable video alone is insufficient to establish a
correct success checker: positive and negative completion examples require
separate validation. Smoke trials must stay out of model success-rate reports.

Harbor stores each run as a job containing trials:

```text
outputs/harbor/<job>/
  config.json                 # actual job configuration
  result.json                 # job summary, including trial exceptions
  <trial>/
    result.json               # trial outcome and exception, if any
    agent/robotworld/          # summary.json and receipts.jsonl for smoke100
    artifacts/simulation/     # simulator videos and native/action records
    verifier/                 # reward and verification details
```

Video names and subdirectories differ by benchmark; inspect
`<trial>/artifacts/simulation/` for the actual `.mp4` files. An interrupted or
failed trial may have incomplete artifacts; preserve its exception and do not
count it as a completed 100-step check. Run `scripts/check_harbor_results.py` on
the resulting job for the independent evidence audit. It excludes probes from
model scores rather than interpreting their reward as model performance.

For formal evaluation, configure an installed model agent and its credentials,
then run `configs/jobs/robotworld-codex.yaml` (or your own standard job). Native
task budgets and the explicitly declared World profiles remain in the task
packages; this diagnostic does not replace them with 100 steps.

## Mixed-agent 400-step diagnostic

All task packages live directly at `tasks/<task-name>/`; `tasks/catalog.json`
records their source benchmark and native horizon. No benchmark-specific nesting
or separate pilot task is required.

To distribute the 84 tasks reproducibly between Codex and Claude Code, first
prepare two ordinary Harbor jobs:

```bash
python scripts/run_harbor_mixed.py --prepare-only \
  --model deepseek-v4.1-flash --steps 400 --seed 17 --name mixed400-01
```

This creates `outputs/harbor-plans/mixed400-01/assignment.json`, `codex.yaml`
and `claude-code.yaml`. A seeded shuffle assigns 42 tasks to each harness, with
one trial per task. The seed controls **assignment**, not simulator initial
states. The same model ID is used by both harnesses; it is not a paired comparison
because each task is assigned to only one harness.

Prepare both installed agents and privately load provider credentials as described
above. With the usual mount/API environment variables set, launch the jobs using
Harbor directly:

```bash
export WORLD_CONTROL_STEP_LIMIT=400
harbor run --config outputs/harbor-plans/mixed400-01/codex.yaml \
  --jobs-dir "$PWD/outputs/harbor" --job-name mixed400-01-codex
harbor run --config outputs/harbor-plans/mixed400-01/claude-code.yaml \
  --jobs-dir "$PWD/outputs/harbor" --job-name mixed400-01-claude-code
unset WORLD_CONTROL_STEP_LIMIT
```

`WORLD_CONTROL_STEP_LIMIT` is a diagnostic cap, not a new official task horizon.
It limits execution to the smaller of 400 and the declared task budget; native
termination and the existing checker remain authoritative. Record the cap with
the results and do not report these as original-budget benchmark scores. Removing
the variable restores the declared budget for subsequent jobs.

Alternatively, use the convenience wrapper:

```bash
bash scripts/run_harbor_mixed.sh --steps 400 --model deepseek-v4.1-flash \
  --seed 17 --name mixed400-02
```

It prepares the same configurations and runs both jobs sequentially, one active
trial at a time. `HARBOR` selects the CLI; `WORLD_ASSETS` may point to external assets.
The wrapper records job logs and `launch-results.json` alongside the plan. Actual
videos, agent traces and verifier results remain in each standard Harbor trial
under `outputs/harbor/<job>/<trial>/`. Preparing a plan does not start a trial or
establish successful execution; check the acceptance record for run evidence.

## Connect another agent

The task exposes the following MCP server in `task.toml`:

```toml
[[environment.mcp_servers]]
name = "robotworld"
transport = "stdio"
command = "python"
args = ["-m", "environment.bridges.mcp"]
```

An MCP-capable agent configures this server inside `main`, inheriting `WORLD_EPISODE_URL` and `WORLD_EPISODE_TOKEN`. A custom installed agent must explicitly forward task MCP configuration to its runtime; it is not enough to place TOML in a task if the agent ignores it. The five operations are `describe`, `observe`, `status`, `execute` and `receipt`.

An HTTP agent uses the same contract without MCP:

```python
from environment.clients.episode import EpisodeClient
client = EpisodeClient()  # URL and token inherited from the environment
spec = client.request("describe")
observation = client.request("observe")
status = client.request("status")
# Build commands from spec, never assume that another robot uses the same units.
```

`GET /v1/describe` supplies instructions, action schemas, frames, units and budgets. `GET /v1/observe` supplies public state and images. Submit an ordered chunk to `POST /v1/execute` with a unique `request_id` and the current `episode_id` and `observation_id`. See [the complete request example](harness-protocol.md#public-api). Query `GET /v1/receipts/{request_id}` after a transport timeout rather than sending a new action blindly.

The agent chooses chunk length, within the transport and native command limits. The environment validates the whole chunk before executing it, serialises commands, counts actual native control steps and interrupts on termination or budget exhaustion. Concurrent arm/base motion must be expressed through a combined native command, not competing HTTP writes. Observation reads do not advance physics.

The agent also owns model context. For the RoboCasa profile, one observation contains the current frame plus up to four previous frames at interval two control steps, for each camera. This does not automatically remove older packets from conversation history. Preserve original traces while applying a documented outbound image window. Agent text saying “finished” cannot set task success; use environment termination and bounded continuation consistently across models.

Offline shell code may process public images or calculate actions. Environment-side feedback code is a separately declared capability, disabled by default; it is not unrestricted access to the simulator. A different harness must not gain hidden state or reset authority simply because it supports shell execution.

## Model/provider configuration

The model and agent are independent. A DeepSeek model running inside Claude Code is still a Claude Code agent; a GPT model running inside another installed agent does not make it Codex.

Codex expects Responses; Claude Code expects Messages. An OpenAI Chat Completions endpoint is not interchangeable with either merely because its API key works. Configure a protocol converter only when needed, disclose it, and test text, images, tool calls and streaming. See [provider examples](HARBOR_VALIDATION.md). In containers, `127.0.0.1` means that container: use an address reachable from `main` for an external gateway. Never write actual keys into task instructions, YAML, published logs or command-line examples.

## Create or export tasks

```bash
python scripts/export_harbor_tasks.py --list
python scripts/export_harbor_tasks.py --bench robocasa \
  --tasks CloseDrawer,CoffeeSetupMug --seed 7 --output /tmp/robotworld-task-export
```

The exporter includes the catalog and only creates supported task packages. Unsupported legacy integrations must fail explicitly rather than receive placeholder grades. Consult its `--help` for the currently implemented adapters.

For a new benchmark, implement and test its episode adapter first. Preserve native action semantics, reset and termination; identify any World-authored criterion explicitly. Then create the ordinary Harbor layout:

```text
my-task/
  instruction.md                 # public task goal and interaction instructions
  task.toml                      # agent/verifier limits, MCP and artifact collection
  environment/Dockerfile         # agent-side task dependencies
  environment/docker-compose.yaml # main and simulator services
  tests/test.sh                  # invokes the trusted verifier
  tests/grade.py                 # checks collected simulator artifacts
  solution/solve.sh              # optional, only for a real validated solution
```

See [templates/harbor-task](../templates/harbor-task/README.md) for a deliberately non-runnable authoring scaffold. Keep model identities and secrets out of the task. Include native task ID, seed, control-step budget, checker version and asset provenance in metadata/artifacts. Do not include a fake `solution` that writes reward files.

## Budgets, repeats and results

- Native control steps, physics substeps, model calls and wall time are different quantities. `task.toml` agent timeout is wall time; the simulator enforces the control-step horizon.
- CloseDrawer uses 450 native control steps; the agent wall-clock allowance is 43200 seconds. Do not substitute one for the other.
- `--n-attempts 3` creates three trials; it does **not** invent three environment seeds. A fixed-seed package measures agent variability at that initialisation. Export and record separate seeds for environment variation.
- The verifier grades simulator-owned `episode.json` after recordings are finalised, never agent-authored claims. Valid success receives 1, valid non-success 0; native scores remain separate named fields.
- Simulator failures, missing/corrupt artifacts or incompatible checkers must produce an error, not an ordinary failure reward. Inspect agent errors as well: a verifier reward alone does not prove that a model run was valid.
- Keep the number of requested, valid and invalid trials. Report valid-trial success rate together with coverage/invalid rate; disclose any alternate all-attempt denominator. Sample variance needs at least two valid trials.

Harbor's job/trial directory is authoritative. Retain agent traces and files, simulator video and native-step journal, verifier reward/details, and configuration. An interrupted rerun starts a fresh episode unless an explicit verified state-restore implementation exists. Do not infer success from a watchable video alone or silently discard unsuccessful trials.

## Before accepting a new task

Check startup/reset, native action units and frames, idempotent request recovery, budget termination, complete video, evaluator positive/negative examples and artifact isolation. Run a no-model infrastructure test before a model-backed trial. Validate time-dependent criteria in physical seconds using actual control frequency. Keep “adapter implemented”, “smoke tested”, “model run completed” and “verified success” as distinct statuses.

Official references: [task format](https://docs.harborframework.com/tasks/overview), [multi-container tasks](https://docs.harborframework.com/tasks/multi-container), [custom agents](https://docs.harborframework.com/agents/custom-agents), [separate verification](https://docs.harborframework.com/tasks/separate-verifier). CLI and configuration examples here were checked against the installed Harbor 0.24.0 source.

## Observable ATIF trajectories

Custom installed agents retain native traces and support a limited ATIF export:

```bash
python -m agents.trajectory --logs-dir /absolute/job/trial/agent --harness codex
# For Claude Code: --harness claude-code
```

The exporter validates `trajectory.json` against Harbor's trajectory schema and
writes exact decoded image data under `trajectory-media/`. Messages, tool calls
and available replies come from recorded events. Native logs remain authoritative.
The export is not lossless: hidden reasoning, internal compaction and unrecorded
continuation prompts are not reconstructed; token, cost and reward metrics are
not inferred. Check `extra.limitations` and source references before training.
ATIF step numbers are event indices, not simulator control steps.

## Selected-suite job

After provisioning each required environment and editing `YOUR_MODEL_ID`, use
the supplied standard job to schedule all 84 tasks:

```bash
harbor run --config configs/jobs/robotworld-codex.yaml --jobs-dir "$PWD/outputs/harbor"
```

Edit `n_attempts` (default 3), `n_concurrent_trials` (default 1), `agents` and
the `tasks` list in that file. This schedules 252 trials; it is not a smoke test.
Seeds are fixed by exported task variants, not automatically changed by attempts.
Current physical validation and scoring limitations remain documented in
[the acceptance record](HARBOR_VALIDATION.md).

### Completion score and audit

A model that returns normally before reaching the goal receives completion reward
0. This does not fabricate an upstream native score: `native_success` stays null
when that scorer has not run. API/protocol/execution exceptions, including Harbor
`AgentTimeoutError`, remain invalid and must be reported with coverage.

```bash
python scripts/check_harbor_results.py "$PWD/outputs/harbor/JOB_NAME" --output audit.json
```

The audit uses host-owned Harbor agent configuration/context to exclude probes,
checks physical evidence, and refuses a complete task-macro score while evaluated
attempts are missing or invalid. Do not mix probes into model jobs or present raw
Harbor reward means as audited model results. A missing native score is distinct
from the completion reward assigned after a normal unsuccessful agent return.

### Environment-owned outcome

The simulator service is the sole authority for episode completion. After each action it receives the configured environment/checker outcome and determines whether execution continues. It saves the final outcome in `episode.json`; it never waits for a Harbor verifier response. Harbor runs the verifier after execution and artifact collection.

The verifier checks task identity and the final outcome format, then maps `success` to reward 1 or 0. It does not read auxiliary `result.json` / `native-result(s).json`, recompute physical predicates, or use video quality to change the score. These files remain available for optional diagnostics. Missing, nonterminal, malformed or infrastructure-error outcomes are invalid runs, not zero-score task failures. Early agent return and budget exhaustion without confirmed completion receive 0. Diagnostic regrading is saved separately and never silently rewrites historical trials.
