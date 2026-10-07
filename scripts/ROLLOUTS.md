# Batch evaluations

Each `run_<benchmark>.sh` configures one benchmark. `run_all.sh` invokes all 20 sequentially. Without an action, an entry point lists tasks; `run` explicitly starts evaluation. See the [root README](../README.md) for installation and examples.

## Settings

The shell wrappers define per-task `TASK_STEPS=("TASK=native" ...)` and `TASK_CODE_CONTROL=("TASK=default" ...)` arrays.

- `native` selects the original budget, `profile` selects the scenario-profile budget, and a positive integer explicitly overrides it. A request cannot exceed a known scenario limit or disable native early termination.
- BEHAVIOR is capped at 2,000 steps, while the original budget remains recorded. `CountertopCleanup` uses an explicit 600-step World budget because its official horizon is unverified.
- Adapted World objectives use the registered profile horizon. `SCORING_PROFILE=native` restores the original goal, scoring, and horizon. A shortened diagnostic that cannot cover the complete evaluation interval has indeterminate success.
- Code control defaults to `off`. Per-task `on/off` overrides `CODE_CONTROL`; `default` inherits it. A later `--task-code-control TASK=on` overrides the corresponding array entry.
- Unsupported code control remains unavailable and is recorded as effectively false. This option enables a bounded program inside the environment; offline workspace code still operates only on permitted observations.
- `ROLLOUTS` / `--rollouts` is the number of trials per task, default 3. All trials count; this is not best-of-N selection.
- `--tasks TASK1,TASK2` selects tasks. `--batch` identifies the experiment. `MODEL` and `CODEX_AUTH_HOME` select the model and provider credentials. Outputs default to `outputs/`.
- `--seed-stride` defaults to zero: trials use the same initial condition with new agent sessions. Fixed-instance RoboDojo, BEHAVIOR, and Bench2Dex tasks reject unsupported seed changes.
- `--timeout` defaults to 12 hours of wall time, separate from simulation steps. Simulation pauses during model reasoning.

Later CLI arguments take precedence over wrapper defaults. Effective settings are recorded in `run.json`. Use a new batch ID after changing the model, tools, or budgets.

## Outputs and resume

`outputs/<benchmark>/<task>/run-<batch>-0001/` contains the original artifacts, relative video/event/program links, and the agent workspace where supported. Preserve `artifacts/` when copying a run. Full and image-free event streams are retained.

With identical settings, `run --batch SAME --resume` skips completed successes and valid failures. Interrupted or infrastructure-error slots restart from the task's initial state in a new retry directory, preserving earlier artifacts. It does not restore intermediate simulator state. Summaries count logical rollout slots rather than double-counting retries.

## Statistics

Per-task summaries include individual outcomes and raw scores/rewards, means, population variance (`ddof=0`), and sample variance (`ddof=1`). Sample variance is `null` for a single observation. `<batch>-runs.csv` contains per-rollout results; `<batch>.csv` contains per-task metric summaries.

In the benchmark JSON, `overall.task_equal_success_rate` summarises per-task success rates with equal task weights. `overall.pooled_rollouts.success_rate` summarises all valid Boolean rollout outcomes. Their denominators differ; means agree when all tasks have equal, complete trial counts. `full_benchmark_success_rate` is populated only when all planned Boolean outcomes are available.

Raw scores with different task-specific units are not pooled. Infrastructure errors and indeterminate outcomes are unscored, with coverage reported explicitly. A per-benchmark summary is not evidence that the complete 84-task suite was evaluated.

`run_all.sh` continues to later benchmarks after a benchmark error and returns nonzero overall. Ctrl-C interrupts the campaign. Set `BENCHMARKS='robocasa robolab'` to run a subset.
