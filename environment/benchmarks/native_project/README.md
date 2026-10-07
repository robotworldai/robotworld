# Shared native-project transport

This package provides the source-built agent tool loop, native action validation, actor observations, and trace recording. Task construction, assets, dependencies, action semantics, and scoring remain in each project's adapter and third-party integration directory.

It does not supply a complete task policy, assume unbounded actions belong in [-1, 1], or substitute survival for native scoring.

`project.json` specifies pinned source, Docker configuration, tasks, and step limits. The default engine uses IsaacLab. Projects with custom stepping implement `make_env`; legacy TorchRL projects use `custom/create_sim`. Set `runtime_python_paths` before importing AppLauncher to avoid mixing IsaacLab forks.

Only allowlisted actor observations are provided to the policy. Projects can implement `policy_groups` or `policy_observation(env, obs)`. Critic observations, rewards, evaluator state, and review cameras stay outside this interface.

`apply_action` accepts the full native action vector. Optional `coding_control` invokes a bounded generated program at each native step. `observe` does not advance physics. The shared history uses the current observation and up to four earlier tool observations at interval two; video is recorded every control step.

Simulation pauses during model reasoning. This setting does not test real-robot inference latency; native actuator delays and disturbances still apply during execution.
