# volleybots adapter

T05: drone 1v1 volleyball using native flight actions and a fixed independent opponent. A 3v3 policy or serving-only checkpoint is not an equivalent opponent.

- `project.py` loads the pinned environment, defines action documentation and permitted model observations, and reads native scores.
- [Source, assets, licences, and Docker recipes](../../../third_party/benchmarks/volleybots/README.md) are maintained separately from the adapter. Upstream source files are left unchanged.
- Trajectories, videos, and compatibility outputs are written under `var/runs/`, outside the source checkout.

Use `scripts/eval/volleybots.sh` for lower-level diagnostics, or consult the [integration overview](../../../docs/native17/README.md). The shared `../native_project/` loop provides `observe`, `apply_action`, and optional `coding_control`. Action dimensions, scaling, and observation semantics remain specific to the original environment.

Model runs use the locally built `codex/` runtime and Docker simulator. An implemented adapter does not establish environment readiness: consult the [validation record](../../../docs/native17/VALIDATION.md) for asset and hardware limitations and recorded episodes.
