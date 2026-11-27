# Claude Code pilot with GPT-6

The upstream `claude-code` repository contains public documentation/plugins, not
all CLI implementation source. This adapter runs the official CLI without patching
it. Locally inspected version: **2.1.292**. MCP SDK: **1.30.0**.

Claude Code speaks Anthropic Messages at `/v1/messages`. An OpenAI-compatible
`/v1/chat/completions` URL alone is insufficient. Use a gateway that supports the
Messages tools, images and streaming protocol and actually routes the requested
model. No model-name substitution or fallback is implemented here. Anthropic does
not officially support non-Claude models in Claude Code; this is an experimental
compatibility integration, not an upstream-supported configuration.

```bash
python -m pip install mcp==1.30.0
# Load these from your private credential store:
export ANTHROPIC_BASE_URL='https://YOUR-MESSAGES-ENDPOINT'
read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY
export WORLD_EPISODE_URL='http://127.0.0.1:18765'
# WORLD_EPISODE_TOKEN must match the evaluator-created service.
python harnesses/claude_code/run.py --claude /path/to/claude \
  --model gpt-6-astra --output outputs/claude-close-drawer/run-001
```

Run from the repository with `PYTHONPATH=.`. Launch the simulation separately;
the harness cannot reset it. Built-in CLI tools are disabled in this pilot; only
RobotWorld MCP is allowed. Code control is not exposed. This ensures a harness
swap cannot accidentally introduce direct simulator control.

The wrapper records each CLI turn and resumes the same Claude session while the
environment remains active, up to `--max-continuations` (default 20). This is a
continuation count, not a physics-step limit. `--turn-timeout` bounds each CLI
invocation. Model errors stop separately from native task failure. Use a dedicated
output directory for every rollout.

Current evidence: real RoboCasa HTTP and MCP calls pass. On 2026-10-16,
`https://api.ai-infer.com/v1/messages` accepted model
`uniaix-claude-opus-5-5` and returned a structured `tool_use` block. Claude Code
2.1.292 connected to RobotWorld MCP with only the five declared tools enabled.
A native-budget CloseDrawer rollout is recorded separately; these protocol checks
alone do not establish task success or GPT-6 compatibility.

For this provider, set `ANTHROPIC_BASE_URL=https://api.ai-infer.com` (the CLI adds
`/v1/messages`). Load `ANTHROPIC_API_KEY` privately, then run:

```bash
PYTHONPATH=. python scripts/run_harness_pilot.py --harness claude-code \
  --claude /path/to/claude --model uniaix-claude-opus-5-5 \
  --output outputs/protocol/close-drawer-claude-opus-001
```

The model name is the provider's routing alias. Record both the requested alias
and response model identifier when reporting model results; do not infer an
independently verified model identity from the alias alone.

References: [gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol),
[provider support boundary](https://code.claude.com/docs/en/llm-gateway).

Latest model-backed pilot (2026-10-16): Claude Code with requested model
`uniaix-claude-opus-5-5` (response identifier `claude-opus-5-5`) executed three
action chunks and reached control step 50/450. The provider then returned
HTTP 402 `insufficient balance`. Video finalized at 51 frames / 20 FPS.
This establishes the model/tool/environment path, not task completion. The trial
is invalid for performance scoring because it stopped on a provider error.
