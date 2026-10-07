# aerial_balance adapter

T04: cable-suspended beam and ball positioning, with the original scalar vertical-velocity increment, 15-step delay, and 11-dimensional state.

- `project.py` loads the pinned environment, defines action documentation and permitted model observations, and reads native scores.
- [Source, assets, licences, and Docker recipes](../../../third_party/benchmarks/aerial_balance/README.md) are maintained separately from the adapter. Upstream source files are left unchanged.
- Trajectories, videos, and compatibility outputs are written under `var/runs/`, outside the source checkout.

Use `scripts/eval/aerial_balance.sh` for lower-level diagnostics, or consult the [integration overview](../../../docs/native17/README.md). The shared `../native_project/` loop provides `observe`, `apply_action`, and optional `coding_control`. Action dimensions, scaling, and observation semantics remain specific to the original environment.

Model runs use the locally built `codex/` runtime and Docker simulator. An implemented adapter does not establish environment readiness: consult the [validation record](../../../docs/native17/VALIDATION.md) for asset and hardware limitations and recorded episodes.
