# TTRL adapter

Original environment resources are in `third_party/benchmarks/ttrl/`; this adapter maintains the custom `Simulator` and `ServeMetrics`.

The integration uses the author's `TTEnv`, not the generic `ManagerBasedRLEnv`. `SingleEpisode` disables automatic robot-episode resets without intercepting `reset_ball`. Original actor history is allowlisted separately; critic data in `extras` is excluded from policy inputs.

The source-built agent controls the environment through shared dynamic tools. See the [project README](../../../third_party/benchmarks/ttrl/README.md) for interface boundaries and scoring.
