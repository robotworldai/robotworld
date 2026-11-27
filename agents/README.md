# Harbor agents

[Harbor guide](../docs/HARBOR.md) · [中文指南](../docs/HARBOR.zh-CN.md)

These are standard Harbor custom **installed agents**: Harbor orchestrates each
trial, while the model loop actually runs inside the task's `main` container.
Use `--agent-import-path module:Class`, not a separate RobotWorld evaluation API.

| Import path | Runtime | Purpose |
|---|---|---|
| `agents.robotworld:RobotWorldProbeAgent` | Python, no model | CloseDrawer's native 450-step transport/budget probe |
| `agents.robotworld:RobotWorldCodexAgent` | Verified source-built Codex app-server | Persistent robot episode, explicit Responses image window |
| `agents.robotworld:RobotWorldClaudeCodeAgent` | Claude Code 2.1.292 | Persistent robot episode, optional Messages-to-Chat conversion |

The internal `harnesses/` modules implement the already-tested control loops.
They are not copies of Codex or Claude Code. An installed-agent adapter handles
runtime provisioning, task instruction delivery, process invocation and error
propagation. Native traces are retained; ATIF export is not yet implemented for
these custom adapters. Do not label this as lossless ATIF output.

## Shared setup

Run commands from the checkout after provisioning simulator images and assets:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORLD_ROOT="$PWD"
export WORLD_ASSETS=/absolute/path/to/Assets
export WORLD_EPISODE_TOKEN="$(python -c 'import secrets;print(secrets.token_hex(32))')"
```

Use an absolute `--jobs-dir "$PWD/outputs/harbor"` for reliable sidecar artifact
collection with the pinned Harbor Docker backend. Example job configurations
are in `configs/jobs/`; change the agent, model, task list and `n_attempts` there.
Do not put actual provider keys in job files or `--agent-env` command arguments.

## Codex

Prepare a portable runtime from an existing verified source build:

```bash
export WORLD_CODEX_SOURCE=/absolute/path/to/codex
python scripts/prepare_harbor_codex.py \
  --provider-config /private/path/provider.json \
  --output var/harbor-runtimes/codex-provider
export WORLD_HARBOR_CODEX_RUNTIME="$PWD/var/harbor-runtimes/codex-provider"
# Load the API key variable named in provider.json privately.
harbor run --path tasks/close-drawer \
  --agent-import-path agents.robotworld:RobotWorldCodexAgent --model YOUR_MODEL_ID \
  --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

The provider JSON uses the schema in [provider configuration](../docs/HARBOR_VALIDATION.md):
`manifest`, `provider`, TOML `overrides`, and `observation_window`.
The preparer verifies the source revision, clean worktree and binary SHA, creates
a shallow source checkout, copies the same binary, and relocates build/catalog
paths for the agent container. It does not modify Codex source or rebuild with
an undocumented patch. The container re-verifies the result before launch.
Current packaging supports clean upstream builds, not locally patched builds.

A private `CODEX_HOME` is created in the container. Provider credentials are
inherited by the process; no ChatGPT account is used. Outgoing image counts
are recorded under `agent/robotworld/events/model-image-window.jsonl`.

## Claude Code

Either preinstall Claude Code 2.1.292 in the agent image or reuse a local official
CLI with `WORLD_HARBOR_CLAUDE_BINARY=/absolute/path/to/claude`. The runtime version
and reused binary checksum are recorded in the trial.

For a reachable native Messages provider, privately load `ANTHROPIC_BASE_URL`
and `ANTHROPIC_API_KEY`. For a Chat Completions provider:

```bash
export WORLD_HARBOR_CLAUDE_BINARY=/absolute/path/to/claude
export WORLD_HARBOR_CHAT_BASE_URL=https://YOUR_PROVIDER/v1
# Privately load WORLD_MODEL_API_KEY; never commit its value.
harbor run --path tasks/close-drawer \
  --agent-import-path agents.robotworld:RobotWorldClaudeCodeAgent --model YOUR_MODEL_ID \
  --n-attempts 1 --jobs-dir "$PWD/outputs/harbor"
```

The optional converter is LiteLLM 1.105.1 in its own Python environment inside
`main`, listening on container loopback. It is separate from MCP's pinned
environment and never acts as a model itself. Without a Chat endpoint setting,
the adapter uses the configured native Messages endpoint directly. This
optional first-run dependency installation takes time; prebuild a pinned agent
image to amortize it across trials.

## Add your own agent

Use Harbor's `BaseInstalledAgent` or one of its built-ins that supports task MCP
servers. Standard task images include the RobotWorld HTTP/MCP client; no
RobotWorld-specific model loop is mandatory. Follow task `describe` and native
tool schemas, and submit actions through `execute`. Forward `instruction.md`
as task instructions, not just the native environment's goal text.

Keep continuation, image history, permitted auxiliary tools and timeout handling
explicit. Built-in Harbor agents are separate profiles until these policies
have been checked; merely swapping a class name does not prove matched settings.
Do not give the agent simulator files, assets, Docker access or score-writing
authority. `tests/` belongs to the verifier, not the agent.

Infrastructure probes are named and recorded as probes. They may legitimately
end with reward 0; their job is to verify execution and artifacts, not solve tasks.
