# Harbor job configuration

A **task** defines one environment and its goal. An **agent** runs a model and
interacts with that environment. A **trial** is one attempt by an agent on a task.
A **job** schedules a collection of trials. These names follow Harbor.

| Change | Edit |
|---|---|
| Which tasks to evaluate | `jobs/*.yaml`: `tasks[].path` |
| Agent implementation / harness | `jobs/*.yaml`: `agents[].import_path` (or `name` for a built-in agent) |
| Model | `jobs/*.yaml`: `agents[].model_name` |
| Attempts per task and agent | `jobs/*.yaml`: `n_attempts` |
| Concurrent trials | `jobs/*.yaml`: `n_concurrent_trials` |
| Task instructions | `../tasks/<task>/instruction.md` |
| Agent/verifier wall-clock limits | `../tasks/<task>/task.toml` |
| Simulator, assets, native control-step budget | `../tasks/<task>/environment/` |
| Physical success validation | `../tasks/<task>/tests/` |

Wall-clock limits and physical control-step budgets are different. Changing a
task's budget or checker defines a new evaluation profile; changing a model must
not silently change either. `n_attempts` repeats the configured task; it does not
automatically change the environment seed.

From the repository root, after following [the Harbor setup guide](../docs/HARBOR.md):

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
harbor run -c configs/jobs/close-drawer-probe.yaml \
  --jobs-dir "$PWD/outputs/harbor"
```

The probe does not use a model or solve the task. For model evaluation select the
Codex or Claude Code job after provisioning its [agent runtime](../agents/README.md).
Use an absolute output path with the pinned Harbor Docker backend. API keys are
runtime environment variables, never YAML values. `providers/` contains provider
configuration examples, not credentials.

中文：任务、执行器、单次尝试、批量评测分别对应 Harbor 的 **task、agent、trial、job**。
换模型或 harness 修改 `jobs/*.yaml`；造题修改 `tasks/<task>/`。`n_attempts` 是每个
task/agent 组合的尝试次数，不会自动更换场景种子；`task.toml` 的时间限制不是仿真步数。
完整中文操作见 [Harbor 指南](../docs/HARBOR.zh-CN.md)。
