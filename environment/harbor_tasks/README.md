# Harbor task export

`python scripts/export_harbor_tasks.py --list` prints all selected benchmark tasks,
including tasks whose original budget is unresolved. `--output tasks`
exports only implemented Episode adapters. Existing output directories are never
overwritten; use a new destination for each versioned dataset release.

Useful selections:

```bash
python scripts/export_harbor_tasks.py --bench robocasa --tasks CloseDrawer,CoffeeSetupMug --output tasks/casa-pilot
python scripts/export_harbor_tasks.py --bench digit --bench reflexbench --output tasks/native-pilot
python scripts/export_harbor_tasks.py --output tasks
```

Every generated directory is a standard Harbor task with `instruction.md`,
`task.toml`, `environment/`, and `tests/`. A task includes a small MCP client in
its environment image, so an MCP-capable Harbor installed agent need not use
RobotWorld's custom Codex or Claude Code wrapper. The wrappers are optional agent
implementations, not alternate task formats.

`catalog.json` records all 84 task identities; `robotworld-task.json` records each
exported task's source budget, selected horizon, scoring profile, seed, and
validation status. `adapter_ready` means a service implementation exists; it does
**not** mean asset installation, simulator startup, task feasibility, or model
performance has passed. Runtime validation and asset validation are separate fields.

Budgets preserve the selected evaluation protocol. BEHAVIOR is capped at 2,000
native control steps, while retaining its original budget in metadata. Soccer uses
the already selected fixed 20-second / 1,000-step protocol, explicitly distinct
from the upstream CLI's six-second / 300-step default. Approved World success
profiles retain their configured horizons and checker; other tasks retain their
native success checker. CountertopCleanup has no official dataset-registry horizon
in the pinned source; it retains the existing World handoff budget of 600 from
`scripts/run_robocasa.sh`, explicitly recorded as a World fallback, not official.

Native-project adapters reuse the existing simulator construction, actor
observation allowlist, action validation, runtime compatibility, and scoring hooks.
They do not import or launch Codex. A state or sensor query never advances physics;
`execute` validates the whole action chunk before applying actions, records actual
native steps, and uses the shared Episode idempotency/termination rules. Feedback
code control is deliberately disabled in this initial standard profile.

Install assets with the repository download/restore workflow before starting
Harbor. `restore_assets.py` now makes relative directory links so relocating the
checkout into the simulator container preserves them. Do not replace mixed source
and asset directories wholesale with an asset-only directory: for example, doing
so can hide the Python modules beside AI-CPS USD files.
