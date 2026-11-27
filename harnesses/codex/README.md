# Source-built Codex harness

RobotWorld retains both harnesses:

- `harnesses/codex/`: public namespace and documentation for the existing Codex integration.
- `harnesses/claude_code/`: Claude Code integration with an explicitly selected model.
- `codex/`: the local upstream Codex source checkout; retained in the repository.

The current implementation is in `environment/runtime/codex_session.py` and
`environment/runtime/isolated_codex.py`. Keeping those paths preserves existing
benchmark launchers and source-build verification. `CodexSession` is also exported
from this package; it is the same implementation, not another model client.

Build and source identity checks remain unchanged. This change does not move,
duplicate or modify upstream Codex sources. The existing runtime supports build
manifests including its pre-existing patch metadata; adding this namespace does
not certify a particular historical binary as an unmodified upstream build.

Migration status: existing Codex benchmark runners remain operational through
their original launchers. A neutral-protocol pilot runner is now available in `run.py`; its Qwen
Responses integration has executed real CloseDrawer actions. Full task success
and broader adapter coverage remain separate validation requirements. Do not claim arbitrary harness interchangeability across all tasks
until the corresponding adapter and runner pass the same contract tests.
