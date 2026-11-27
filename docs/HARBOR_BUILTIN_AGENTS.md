# Standard Harbor agents

[中文](HARBOR_BUILTIN_AGENTS.zh-CN.md)

RobotWorld tasks expose the same MCP interface to Harbor's built-in agents. Use `name: codex` or `name: claude-code`; these profiles do not import `agents.robotworld:RobotWorldCodexAgent` or `RobotWorldClaudeCodeAgent` and do not require our source-built harness bundle.

## Select tasks, agent and model independently

Generate an ordinary Harbor YAML; this does not introduce another agent loop:

```bash
python scripts/make_harbor_job.py --agent codex --model openai/gpt-6-astra \
  --agent-version 0.164.0 --task close-drawer --attempts 3 --concurrency 1 \
  --output configs/jobs/my-experiment.yaml
bash scripts/harbor_run.sh --config configs/jobs/my-experiment.yaml
```

Repeat `--task` to select tasks, or use `--task all` for all 84. Choose `--agent claude-code` and a compatible model/provider to change harnesses. A custom implementation uses `--agent-import-path my_package.agent:MyAgent` instead of `--agent`. The custom module must be installed in Harbor's Python environment. Agent-specific kwargs can be set in the generated YAML. Credentials and endpoint settings remain outside the task and generated YAML, in the native agent's supported environment configuration.

Selecting a model identifier does not convert its API protocol. Codex and Claude Code have different model interfaces; a provider connection must support the selected harness's required images, streaming and tool calls. An external protocol adapter, if needed, is independently versioned and tested. Never silently substitute the model or drop tool calls. This helper does not register a `robotworld` dataset with Harbor and does not claim every agent/model pairing is supported.


## ChatGPT subscription login (Harbor native Codex)

Harbor 0.24.0 already supports Codex login credentials. This is direct CLI subscription authentication, not a subscription-to-API proxy and not an OpenAI Platform API credit grant.

```bash
codex login
unset OPENAI_API_KEY OPENAI_BASE_URL
export CODEX_FORCE_AUTH_JSON=1
# For a non-default login file, use CODEX_AUTH_JSON_PATH=/private/path/auth.json instead.
bash scripts/harbor_run.sh --path tasks/close-drawer --agent codex \
  --model openai/gpt-6-astra --n-attempts 1 --n-concurrent 1 \
  --jobs-dir "$PWD/outputs/harbor" --job-name codex-subscription-pilot
```

The helper only prepares asset paths and an episode capability, then executes the installed `harbor run` with your arguments unchanged. Harbor owns container preparation, the built-in agent, artifact collection, the separate verifier, and result aggregation. It installs the CLI in the agent container; the host CLI is used for login. Credentials must stay outside the repository and result artifacts. Model availability depends on the logged-in account.

For the full selected suite, pass `--config configs/jobs/robotworld-builtin-codex.yaml` after replacing its model placeholder. This pins the CLI version. Use Harbor's attempt/concurrency settings; avoid GPU concurrency beyond available capacity. `WORLD_CODE_CONTROL` defaults to 0; Shell remains enabled. A custom `--path` / job YAML is currently the supported dataset entry; `harbor run -d robotworld` is **not yet registered**.

Local validation: subscription pilot queued on 2026-11-02; credential-file discovery is verified, model execution is not yet certified. This does not change simulator success criteria or legacy-wrapper experiment results.


## Prepare the environment

Install Harbor and the simulator images and assets using the main README. The current integration was developed with Harbor 0.24.0; other versions require validation. From the repository root:

```bash
export WORLD_ROOT="$PWD"
# Validate the asset root and prepare the mountpoint before Docker mounts the checkout read-only.
WORLD_ASSETS="$(python scripts/check_harbor_assets.py "${WORLD_ASSETS:-$PWD/Assets}" --prepare-mount "$WORLD_ROOT")"
export WORLD_ASSETS
export WORLD_NVIDIA_RUNTIME="$(python scripts/harbor_gpu_runtime.py)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_EPISODE_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export WORLD_CODE_CONTROL=0
unset WORLD_CONTROL_STEP_LIMIT
```

The task starts the simulator, then initializes the agent container. `/workspace/ROBOTWORLD.md` contains the full live public task description, action schemas, units, coordinate frames and controller information. The `robotworld` MCP server exposes `describe`, `observe`, native action tools, `status` and `receipt`. Shell remains available through the selected harness. `WORLD_CODE_CONTROL=0` disables simulator-side feedback programs, not Shell.

The task's MCP bootstrap reads a per-episode connection file in the agent container. This works even when a harness filters inherited environment variables. The file contains only the capability to use this episode's public API, never model credentials or verifier access; it is not an evaluation artifact. Do not manually collect or publish that connection file.

## Run an installed official CLI through Harbor

Provide credentials through environment variables or Harbor's documented authentication configuration. Never put keys into task YAML or instruction files.

```bash
# Configure OPENAI_API_KEY externally. OPENAI_BASE_URL is optional;
# a custom endpoint must be reachable from the container and support Responses.
harbor run --path tasks/close-drawer --agent codex \
  --model openai/YOUR_MODEL_ID --n-attempts 1 --n-concurrent 1 \
  --jobs-dir "$PWD/outputs/harbor"

# Configure ANTHROPIC_API_KEY externally (and ANTHROPIC_BASE_URL if needed).
# The endpoint must implement the protocol expected by Claude Code.
harbor run --path tasks/close-drawer --agent claude-code \
  --model anthropic/YOUR_MODEL_ID --n-attempts 1 --n-concurrent 1 \
  --jobs-dir "$PWD/outputs/harbor"
```

For all selected tasks, edit the model and attempt count in:

- `configs/jobs/robotworld-builtin-codex.yaml` (pins CLI 0.164.0).
- `configs/jobs/robotworld-builtin-claude-code.yaml` (pins CLI 2.1.292).

Then run `harbor run --config <file> --jobs-dir "$PWD/outputs/harbor"`. These configurations use native task horizons (including the selected BEHAVIOR cap) and sequential GPU trials. Harbor installs/provisions the CLI; the environment still needs network access for installation unless the executable is already available at the requested version.

Other Harbor agents can use the task if they support its MCP transport and image responses. Compatibility is not implied merely by appearing in Harbor's agent list. A custom source harness can implement Harbor's agent interface and consume this same MCP API without simulator-specific changes.

## Evaluation boundary

The simulator owns success, native failure, truncation and the control-step budget. Harbor collects its final `episode.json`; the shared verifier maps that outcome to reward. A textual completion claim cannot create success. No new physical criterion is added by this integration.

Built-in harness behavior is a separate experimental profile. Unlike the legacy RobotWorld model wrappers, it does **not** add our automatic continuation loop or request-image trimming proxy. Each observation packet retains the environment's declared history; retention of earlier packets in model requests belongs to the selected harness. Do not claim identical prompts, inference history or rollout results without measuring them. Record harness/CLI version, provider, model and configuration with every comparison. An early agent return remains observable as `agent_stopped`.

## No-model acceptance

```bash
harbor run --config configs/jobs/robotworld-native-budget-probe.yaml \
  --jobs-dir "$PWD/outputs/harbor" --job-name native-budget-probe
```

This exercises public observations, bounded neutral actions, receipt lookup, idempotency, native termination, artifact collection and verification. It is not a solution, oracle or model benchmark, even when its environment reward happens to be 1. Reference solutions must independently be packaged under `solution/solve.sh`; do not replace them with reward-writing scripts.

Use one Harbor job for the batch. For unattended runs, supervise the Harbor process with your job scheduler or a service manager; do not rely on an editor terminal remaining open. Interrupted attempts and infrastructure errors must remain separate from task failures.

## Validation status (2026-10-31)

All 84 packages contain the common startup and MCP entry. A real Harbor built-in Codex run installed CLI 0.164.0; a filtered-environment MCP check listed all nine RoboCasa tools and successfully called describe. Model execution is not yet validated: the configured provider returned HTTP 503, also reproduced by a standalone Responses request. Claude Code has a standard configuration but has not yet been live-validated through this entry. Do not count either configuration as a completed model benchmark on this evidence.
