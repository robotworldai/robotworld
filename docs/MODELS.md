# Connect a model API

RobotWorld uses the source-built agent runtime to run a model through the benchmark's tools. The provider must support streaming Responses events, image input, function calls, and function-call results. Complete [deployment](DEPLOYMENT.md) first; an API key alone does not install the simulators or assets.

## Configure

Run from the repository root in the Python environment:

```bash
python scripts/configure_api.py \
  --base-url https://YOUR_API_HOST/v1 \
  --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"
```

Set `WORLD_MODEL_API_KEY` using your shell or secret manager. Never place its value in a tracked file or command-line argument. The generated private `config.toml` contains:

```toml
model = "YOUR_MULTIMODAL_MODEL_ID"
model_provider = "api"

[model_providers.api]
name = "Configured model API"
base_url = "https://YOUR_API_HOST/v1"
wire_api = "responses"
env_key = "WORLD_MODEL_API_KEY"
requires_openai_auth = false
supports_websockets = false
```

The setup script refuses to overwrite an existing configuration. Use `--output var/auth/model-b` and `--key-env OTHER_API_KEY` for another provider. Optional `--reasoning-effort high` applies only when supported by the provider. For a local service with no authentication, remove `env_key` from its configuration. An endpoint on the host may need a different address when accessed from a simulator container.

## Check the connection

```bash
python scripts/check_api.py --config-dir "$CODEX_AUTH_HOME"
```

This makes two small billable model requests: identify a generated image through a function call, then acknowledge the returned tool result. A passing check confirms this basic API exchange; it does not verify a full simulator rollout or every runtime feature. The checker does not save credentials, response bodies, or provider error bodies.

Requests use the endpoint and model you configure. No private platform route or implicit model alias is selected.

## Evaluate

```bash
bash scripts/run_robocasa.sh list
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --dry-run
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --batch api-check
```

The batch runner reads the model from `config.toml`. An explicit `--model` or `MODEL` overrides it. Use a fresh batch name when changing model, task budget, control assistance, or simulator settings. Keep the same evaluation conditions when comparing providers.

Credentials are resolved into a temporary private runtime configuration. Batch metadata records the model and provider names, not the credential value. Model conversations and videos can contain task content; review generated `outputs/` separately before sharing them.
