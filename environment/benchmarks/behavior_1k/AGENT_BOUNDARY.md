# BEHAVIOR agent boundary

The boundary restricts benchmark data and environment access, not computation.
Codex and BEHAVIOR source checkouts remain unchanged.

## Runtime layout

The Docker launcher starts a host-owned, source-verified Codex app-server inside
Linux bubblewrap. Its filesystem, PID, user, IPC and UTS namespaces are separate
from the simulator. All Linux capabilities are dropped. Startup fails if this
isolation cannot be created; there is no unrestricted fallback.

The evaluator container talks to app-server through a Unix socket managed by
World. The socket itself is **not** mounted inside the agent namespace. The
agent receives dynamic robot tools through Codex's normal RPC mechanism.

Agent-visible paths:

| Path | Purpose / access |
| --- | --- |
| `/workspace` | Read/write code, plans, derived maps and persistent notes |
| `/observations/rollout-000/<n>` | Read-only RGB PNG, depth NPY, proprioception and observation history exported by the allowlist |
| `/runtime/codex-app-server` | Read-only verified local source build |
| `/runtime/models.json` | Read-only model catalog |
| `/agent-home` | Temporary Codex runtime home; only login and model selection are copied, no old sessions, MCP servers, plugins or memories |
| `/usr/bin`, `/usr/lib`, `/usr/share`, library aliases | Read-only host system tools and libraries; Python is `python3` |

No World checkout, benchmark source, scene assets, evaluator outputs, Docker
socket, simulator process namespace, or host home directory is mounted. A
symlink in `/workspace` cannot grant access outside this namespace. System
Python includes pip; optional observation-processing packages can be installed
under `/workspace` rather than exposing the simulator's Python environment.

The current implementation shares host networking so the source Codex runtime
can contact its model service. This is **not a network allowlist or a general
untrusted-code security sandbox**. Do not expose simulator control / debug
services to that network; any future network service needs an explicit access
review. The temporary login is accessible to the agent runtime, as with a normal
Codex process, and is deleted when the relay closes. It is not copied to run
artifacts. No user config containing external tool registrations is imported.

## Policy contract

- Shell and code mode are enabled. `environments: []` is removed because an empty
  selection disables Codex's execution environment, including shell tools.
- Code, files, observation processing and memory are permitted. Web, apps,
  plugins and multi-agent remain unconfigured/disabled in this benchmark
  profile; this is not an official ban on those capability categories.
- Actions and fresh sensor observations pass through benchmark tools and the
  official evaluator. Tool validation, action budgets and logging still apply.
- The agent is expected to keep trying while the episode is active. `give_up`
  is not registered; a stray legacy call is rejected without stopping. A normal
  completed assistant turn starts a continuation in the same thread with current
  observations; this neither resets the simulator nor inserts a hold action.
  Grasp/IK failures call for a revised approach or subgoal, not task termination.
  Explicit interruption, runtime failure, wall timeout and action-budget limits
  still stop execution. They are recorded separately from evaluator completion.
- Task instructions and controller semantics are supplied. Maps/localization
  derived from permitted observations are allowed. Hidden scene truth, global
  robot pose, evaluator state and score are not exported to the policy.
- The app-server uses `sandbox: danger-full-access` **inside the outer Linux
  namespace**, allowing computation without approval prompts. This does not
  grant access to the host or simulator files omitted from that namespace.
- Probe-only rendering runs do not launch a model or require the relay.

## Outputs and verification

The run contains `agent-workspace/`, `agent-observations/`,
`agent-boundary.json` and `agent-runtime.stderr.log`. Full Codex tool and shell
events plus automatically generated no-image events remain in
`rollout-000/events/`. Evaluator results remain outside the agent namespace.

Host prerequisites: Linux user namespaces, `bubblewrap`, Python 3.11+, and the
existing locally built Codex artifact and login. The simulator Docker image and
its upstream scene are unchanged. Namespace isolation is host-side, so no Docker
socket, privileged mode or relaxed container seccomp is required.

Validation on 2026-09-26:

- Kernel-backed isolation test: hidden files and escape symlinks inaccessible,
  observation writes rejected, host environment cleared, only private PIDs visible.
- Source-Codex smoke test: actual shell execution wrote persistent JSON and then
  called a synthetic dynamic tool, recorded at
  `World/var/diagnostics/agent-boundary-smoke-02/`.
- The existing BEHAVIOR Docker image connected to the isolated source app-server
  and completed the initialization RPC; record:
  `World/var/diagnostics/agent-boundary-docker-transport-01/`.
- `features.code_mode` is enabled; the selected model used native shell calls
  in this smoke test. This is not evidence of a separate code-mode execution.
- No new 5000-step simulation or task-success claim is part of this boundary test.

Repeat the real model smoke test from World (uses the existing model account):

```bash
python -m environment.scripts.test_agent_boundary \
  --codex-home var/auth/robodojo-codex \
  --output var/diagnostics/agent-boundary-new
```
