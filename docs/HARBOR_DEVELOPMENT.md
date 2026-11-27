# Maintaining the Harbor integration

[中文](HARBOR_DEVELOPMENT.zh-CN.md) · [Run a task](../README.md#standard-harbor-workflow) · [Robot API](harness-protocol.md)

## Ownership and execution

Harbor owns job/trial scheduling, installed-agent provisioning, timeouts, collection and verifier execution. RobotWorld owns the task, robot transport and simulator adapter. The model provider supplies inference; it does not own task completion.

`harbor run → main agent → MCP/HTTP → simulator → final episode.json → collect → separate verifier → reward`

`main` receives only public robot observations and actions. The simulator owns physical state and success/failure. Do not expose its filesystem, Docker socket or scoring implementation to the agent. Shell access does not grant direct simulator control. `WORLD_CODE_CONTROL=1` only enables the configured feedback interface.

## Change a task

1. Pick the existing task and read its instruction, TOML, compose configuration and `tests/task.json`. Identify the simulator adapter and upstream task. Record the existing seed, horizon, action units/reference frames and success source.
2. Change the authoritative implementation: simulator/adapter for physics and actions; job YAML for agent/model; task instruction for public guidance; TOML for collection and wall-clock limits. The physical success predicate must not be reimplemented independently in the verifier.
3. Update public instructions and tool schemas together when action semantics change. State absolute versus delta commands, units, frames, repetition semantics and examples.
4. Shared generated files need synchronization: `environment/harbor_tasks/export.py` exports clients, grading files and task specs. Inspect `python scripts/export_harbor_tasks.py --help` and export to a temporary directory first. Review the diff before replacing published tasks; regeneration can overwrite manual edits. Keep the catalog and job task selection consistent when adding/removing tasks.
5. Version any changed task, runtime or scoring configuration in the release record. Do not silently compare results across different physical definitions.

## Add an agent or model

For a built-in Harbor agent, select its name and provide its documented credentials/configuration. No simulator edit is needed. For a custom harness, implement the Agent interface of pinned Harbor 0.24.0 and install your module into the same Python environment as Harbor. Select it with `--agent-import-path package.module:AgentClass`.

The adapter must start its runtime inside `main`, consume the instruction, connect to the declared MCP service (or the public Episode protocol), handle images and tool receipts, propagate errors, and preserve its logs. Study `agents/robotworld.py` for lifecycle plumbing, but do not inherit its legacy continuation/image-window policy accidentally. Those behaviours are part of the experiment configuration.

A different model service must support the harness's wire protocol. Test text, image input, streamed tool names/arguments, tool results and final-turn completion independently. Preserve request/call IDs; never silently drop calls, repeat an uncertain action, substitute a model or count protocol errors as physical failure.

## Validate changes

Start with the focused local tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_make_harbor_job.py tests/test_model_output_replay.py \
  tests/test_harbor_runtime_paths.py tests/test_source_archive.py
```

Then run one task with the actual agent and provider using the README. A schema test or no-model smoke does not prove a model can act. Confirm that the trace contains robot actions, the simulator advances, videos are collected, and the independent verifier reads the environment outcome.

For an existing valid model trial, replay its actual action chunks without another model call:

```bash
# Use the same WORLD_ROOT / WORLD_ASSETS / WORLD_EPISODE_TOKEN setup as the Harbor guide.
python scripts/validate_harbor_model_replay.py \
  --model-config configs/jobs/robotworld-builtin-codex.yaml \
  --task close-drawer --source-trial /absolute/path/to/job/task__trial \
  --output var/validation/my-replay
```

Without `--source-trial`, the script first runs the configured model, then replays. Replace the model placeholder before doing so. `--continue-on-error` records a blocked task and proceeds; use it only when failures are not a shared provider/configuration problem. Reusing an existing trial requires matching scene, seed, runtime and budget; the script checks some metadata but **does not prove initial-state identity**. A divergence requires investigation, not a changed reward. Zero-action sources are reported separately. Diagnostic capped runs do not certify full-budget execution.

Inspect `pairs.json`, both trial `result.json` files, `agent/`, `artifacts/simulation/` and `verifier/`. A match certifies only the recorded comparison: it is not task-solving success, universal determinism or correctness of the physical predicate. Check successful and unsuccessful cases and retain infrastructure errors separately.

## Release checklist and open work

- Include all task clients, grader copies, Docker definitions, licenses and pinned-source metadata. Keep upstream Python packages named `assets`; they are code, not downloadable scene data.
- Exclude credentials, local outputs, caches and private reference controllers. Assets follow `docs/ASSETS.md` and their own licences.
- Run tests from the exported directory and check README links. Keep English and Chinese instructions aligned.
- Current tasks still mount repository code through WORLD_ROOT. Fully immutable standalone task images and public `-d robotworld` registration remain pending.
- `RELEASE_NOTES.md` records limitations. Do not describe all 84 tasks or every agent/model pairing as certified until actual evidence supports it.
