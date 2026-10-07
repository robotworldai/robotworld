# Configuration layers

Planned layers are `defaults/` for shared runtime/budgets, `benchmarks/` for integration defaults, `profiles/` for model/container/tool combinations, and ignored `local/` files for machine paths and private settings.

The intended precedence is defaults < benchmark < profile < local < CLI. Record the sanitised effective configuration in the run manifest.

The runtime binary must resolve to a verified build from this repository's `codex/` source. Use the runtime's supported provider configuration, with credentials supplied through environment variables or secret mounts. Verify provider compatibility explicitly.
