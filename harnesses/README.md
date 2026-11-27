# Agent harnesses

The harness and model are independent choices. RobotWorld's public episode API
connects a harness to the simulator; provider configuration connects it to a model.

| Harness | RobotWorld adapter | Actual runtime in the latest local pilot | Status |
|---|---|---|---|
| Codex | `harnesses/codex/run.py` | Verified source build from the explicitly configured `WORLD_CODEX_SOURCE` | DeepSeek CloseDrawer completed 450 steps; task unsuccessful |
| Claude Code | `harnesses/claude_code/run.py` | Official Claude Code CLI | DeepSeek CloseDrawer completed 450 steps; task unsuccessful |

These were standalone pilots with host-side harness processes and Docker-side
simulation, not model-backed Harbor trials. The Codex pilot reused the verified
external `World/codex` build, rather than the bundled checkout. See each run's
`launch.json` and build manifest for exact provenance.

Provider credentials belong in private environment configuration. Claude Code
requires its native Messages protocol; the current third-party pilot uses a
separate LiteLLM Messages-to-Chat adapter. Codex uses Responses and a request-only
observation-image window. Neither converts a ChatGPT account into an API.

The next integration target is a Harbor installed agent running the existing
RobotWorld wrapper inside the task's `main` container, alongside a separate
simulator and verifier. Harbor's built-in agents remain a distinct evaluation
profile until their continuation, tools and image handling are checked against
the RobotWorld wrappers. Do not assume changing `--agent` preserves those policies.

- [Episode protocol and Harbor migration design](../docs/harness-protocol.md)
- [Third-party provider pilot and image window](../docs/qwen-harness-pilot.md)
- [Existing Harbor task](../tasks/robocasa-close-drawer/README.md)
