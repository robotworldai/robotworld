# Harness-independent episodes (v1 pilot)

RobotWorld owns physical execution and scoring; a harness owns model inference,
conversation management and action planning. Harbor owns trial orchestration.
These are independent choices. Swapping a harness must not swap task physics,
initial-state sampling, controller semantics or the success checker.

## Three boundaries

| Boundary | Input | Output | Responsibility |
|---|---|---|---|
| Harness | task, public observations, receipts | ordered action commands | model/provider, context, planning, continuation |
| Episode service | authenticated commands and queries | observations, execution receipts, status | serialization, idempotency, native-step budget, logs |
| Benchmark adapter | validated native actions | native observations and termination | simulator, cameras, controllers, original scoring |

`environment/service/episode.py` has no model dependency. The RoboCasa adapter
reuses `environment/benchmarks/robocasa/control.py` and `env.step(convert_action(...))`.
It does not modify RoboCasa or its success predicate. `environment/clients/episode.py`
is a standard-library HTTP client. `environment/bridges/mcp.py` converts this same API
to MCP tools and image blocks, without adding control authority.

## Public API

All calls require `Authorization: Bearer $WORLD_EPISODE_TOKEN`. Do not publish the
service on a public network. The pilot uses a local Docker network or loopback.

| Method | Path | Meaning |
|---|---|---|
| GET | `/v1/describe` | instruction, tool argument schemas, units, frames, frequency, budget |
| GET | `/v1/observe` | public images and proprioception; no simulation advance |
| GET | `/v1/status` | episode ID, observation ID, executed steps, success and termination |
| POST | `/v1/execute` | execute an ordered command chunk |
| GET | `/v1/receipts/{request_id}` | recover the original execution receipt |

Example: after querying the current episode and observation IDs:

```json
{
  "request_id": "action-0001",
  "episode_id": "<ID from status>",
  "observation_id": "<ID from observe>",
  "commands": [
    {"tool": "move_eef", "arguments": {
      "note": "Raise the hand slightly", "steps": 2,
      "targets": {"eef_delta": [0, 0, 0.1, 0, 0, 0]}
    }},
    {"tool": "set_gripper", "arguments": {
      "note": "Open the gripper", "steps": 1,
      "targets": {"gripper_close": 0}
    }}
  ]
}
```

In RoboCasa, 0.1 in this z component requests a 0.005 m base-frame controller
increment **per control step**. Two repeats request two increments, not one fixed
pose held for two steps. Contact and controller dynamics mean actual displacement
is not guaranteed. Read the adapter's description for each benchmark; do not
transfer RoboCasa's units to another robot.

The harness chooses its chunk length. A transport safety cap is 256 commands;
each native command also retains its original limits (RoboCasa: 1–30 repeats).
Chunks execute sequentially. Simultaneous base/arm/torso motion belongs in one
`move_robot` command, not two concurrent writes. Native termination or the episode
budget interrupts a chunk, and the receipt lists actual steps per command.
A control step is not a physics substep, model call, or wall-clock second.

## Lifecycle and failure semantics

Only the evaluator starts or closes a simulator. There is no agent reset, state
setter, hidden-state endpoint, or success override. One service process owns one
episode. Queries do not consume steps or create artificial history entries.

All commands in a chunk are validated before execution. A stale observation ID
is rejected. Repeating the exact same request ID and payload returns the cached
receipt without executing again; reusing an ID with different content is rejected.
After a timeout, query the receipt before doing anything else. This guarantee is
in-process: a crashed service invalidates the episode, rather than restoring and
silently replaying an uncertain action.

`success`, native termination/truncation, budget exhaustion, agent stopping and
infrastructure error are separate outcomes. A simulator exception marks execution
uncertain and stops the episode; it must not be counted as an ordinary model
failure. The agent's final answer never sets success. The Claude pilot explicitly
uses a persistent-episode wrapper (at most 20 continuation requests); Harbor's
built-in agents may use different turn-ending semantics and must be labelled
accordingly in comparisons.

## Observation, recording and isolation

RoboCasa returns its native RGB and proprioception. History is current plus up to
four earlier frames spaced two **control steps** apart. Repeated reads have the
same IDs and history. Video contains every control step, independent of history
sampling. MCP returns text labels plus actual image blocks, not filenames pointing
to the simulator. Harness-side context/image pruning is a separate responsibility;
Claude Code's long-horizon image behaviour still requires model-backed validation.

Simulator artifacts: `video.mp4`, `evaluation-config.json`, `native-steps.jsonl`,
`events.jsonl`, `episode.json`. Harness artifacts are separate and include raw CLI
stream JSON, stderr, selected configuration, and native session files. Secrets
are passed through the environment and must not be bundled in published runs.

The Harbor agent sees only the clients/bridges modules. It has no simulator source
mount, assets mount, Docker socket or output volume. The simulator is a sidecar;
a separate verifier grades simulator artifacts collected by Harbor. The private
orchestrator sends SIGUSR1 to finalize and flush recordings before collection;
this management operation is not an agent tool.

## Migration status

This is a pilot, not a claim that all 84 tasks already use the new protocol.
RoboCasa CloseDrawer is the first native adapter. Existing evaluation entrypoints
remain available and unchanged by this additive protocol. Migrate each benchmark
by implementing `describe`, `observe`, `prepare`, `step`, `close`, preserving its
native action contract and evaluator. Require fake-adapter contract tests, a
native no-model run, image/step/video checks, then a model-backed trial before
switching its default launcher.

Harbor packages can be added independently after adapter acceptance. Do not create
84 nominal task directories and label them validated without testing their adapters.

## Quick local pilot

```bash
# Reuse the existing RoboCasa image and downloaded Assets.
PYTHONPATH=. python scripts/run_harness_pilot.py \
  --output outputs/protocol/close-drawer-probe-001
# Model-driven run, after loading the Messages-compatible credentials:
PYTHONPATH=. python scripts/run_harness_pilot.py --harness claude-code \
  --claude /path/to/claude --model gpt-6-astra \
  --output outputs/protocol/close-drawer-claude-001
```

The first command is explicitly a non-solving native-budget probe. It does not
use a model or establish task reachability. The second retains native physics
and budget while replacing only the harness. The launcher creates a fresh
simulator, waits for readiness, records outputs, and stops it to finalize video.

For Harbor see [the pilot task](../tasks/close-drawer/README.md). For a
new MCP-capable harness, configure `python -m environment.bridges.mcp`, inherit
`WORLD_EPISODE_URL` and `WORLD_EPISODE_TOKEN`, then call `describe` and `observe`.
HTTP-only harnesses can use the same operations through `EpisodeClient`.

## Pilot validation record

On 2026-10-16, 19 protocol/native-control tests passed. A real CloseDrawer
request containing 600 requested steps stopped at the original 450-step budget;
its repeated request returned the identical receipt with no extra steps. The
native journal contains steps 1 through 450 and the video has 451 frames at
20 FPS (22.55 seconds including the initial frame). MCP initialization, tool
listing, state queries and image blocks were checked against the real service.
Harbor completed one no-op trial with zero exceptions and reward 0 through a
separate verifier. These are infrastructure results, not model success results.

## Harbor migration design — reviewed 2026-10-17

This is the implementation target, not a claim that the full task set has migrated.

### Current evidence and gaps

- One Harbor 0.24.0 task exists: `tasks/close-drawer`. Its recorded
  no-op validation covers startup, simulator artifact collection and separate
  verification, not model control.
- Standalone DeepSeek runs completed CloseDrawer's 450-step budget through
  both Claude Code and source-built Codex. Both were unsuccessful; these
  establish completed integration runs rather than task reachability.
- Standalone harness processes still run on the host. The Harbor task already
  separates `main`, `simulator`, and the verifier, but model-backed execution
  of our wrappers inside `main` remains to be implemented and validated.
- Harbor's pilot wall-clock allowance is currently 7200 seconds, versus the
  standalone default of 43200 seconds. Align these before making comparisons;
  neither number is the physical control-step budget.
- Pilot Compose files depend on host source mounts, and the Codex pilot reuses
  an external absolute build path. These must become versioned image contents
  or explicitly provisioned dependencies for a portable release.
- Repetition currently keeps seed 7. A seed schedule and per-trial metadata
  must be implemented before claiming independent environment sampling.

### Target deployment

```text
Host: Harbor scheduler (trial lifecycle, GPU allocation, collection, aggregation)
  |
  +-- main: selected harness + RobotWorld wrapper + HTTP/MCP client
  |      +-- provider client / protocol converter / image-window relay
  |      +-- writable agent workspace and native trace directory
  |      +-- outbound model API access
  |
  +-- simulator: benchmark image + episode server + benchmark adapter
  |      +-- assets, GPU, native physics, control-step budget, checker
  |      +-- simulator-owned videos, actions, outcome artifacts
  |
  +-- separate verifier: consumes Harbor-collected simulator artifacts
         +-- emits reward and detailed outcome
```

Harbor's reserved `main` service is the agent container. It can reach
`http://simulator:8765` on the trial network. The harness needs no GPU when
using a remote model API. The simulator does not receive model credentials.
The agent must not receive the Docker socket, simulator sources, assets or a
writable simulator-results mount. Its shell may process public observations
and write local programs; physical execution still goes through episode tools.
Feedback code, when enabled, runs under the benchmark's restricted executor,
not an unrestricted simulator shell.

Use one shared agent base with pinned harness runtimes or separate small Codex
and Claude Code variants. Reuse each benchmark's existing simulation image.
Do not create one heavyweight image for every task/harness pair. GPU driver
requirements remain properties of the host and simulator stack.

### Integration choice

Implement Harbor custom installed agents for the existing RobotWorld wrappers,
rather than silently replacing their execution semantics with built-in agents.
A thin installed-agent class handles setup, process launch, artifact collection
and outcome metadata. RobotWorld keeps its physics-facing episode protocol.

The wrappers must retain the same instruction, tool permissions, image policy,
normal-turn continuation, interruption handling and budgets in host and container
runs. Pin runtime versions and record their build hashes. Built-in Harbor Codex
and Claude Code can be offered separately as explicitly labelled profiles after
equivalence checks. Native traces remain authoritative; ATIF conversion is an
additional export and must not discard action receipts or fabricate reasoning.

A provider profile declares model ID, endpoint protocol and credential variable.
Messages/Chat conversion is provider infrastructure, not benchmark logic.
An agent-container image relay must connect to an address accessible inside that
container; host `127.0.0.1` is not the agent's loopback.

### Task packages and results

Each accepted task gets standard `instruction.md`, `task.toml`,
`environment/Dockerfile`, `environment/docker-compose.yaml`, and
`tests/test.sh` plus the trusted grader. Include `solution/solve.sh` only when
a genuine verified reference exists. Keep model names and keys out of tasks.

Generate packages from a single task registry containing benchmark, native ID,
seed policy, step budget, checker version and asset manifest. Preserve original
task semantics and any disclosed World-authored criteria. Generated directory
presence is not an integration acceptance signal.

The verifier reads simulator-owned outcomes after finalization, never an
agent-authored success file. For valid trials write success as reward 0 or 1
and retain native scores separately. Infrastructure failures remain errors with
explicit reasons; report their rate and denominator rather than silently
discarding them or presenting them as ordinary physical failures.

Use Harbor's job/trial directories as the authoritative results:
agent native trace and generated files, simulation video and step journal,
verifier reward/details, and job aggregates. Optionally expose benchmark/task
indexes for users; avoid maintaining two independently mutable result trees.
A restarted trial creates a fresh episode and preserves the interrupted attempt.

### Acceptance sequence

1. Pin Harbor and the agent runtime versions; containerize one wrapper without
   changing task physics. Remove host-specific code paths from the release image.
2. Run CloseDrawer's no-model protocol checks inside Harbor: initialization,
   image transport, idempotent actions, budget termination, video finalization,
   trusted verification, and isolation from simulator files.
3. Run DeepSeek through containerized Codex and Claude Code on the same task
   configuration. Check complete trajectories and actual outgoing images;
   do not infer equivalence from a matching final reward alone.
4. Test model final-answer continuation, provider failure, timeout, simulator
   failure and interrupted collection. Keep agent failure distinct from task
   failure and invalid infrastructure.
5. Migrate each remaining benchmark adapter, then generate its selected tasks.
   Each requires native smoke validation and at least one model-backed trial.
6. Add dataset-wide jobs, per-task budgets, repeats and seeds, error-aware
   success statistics, and README commands verified on a clean checkout.

### Official references

- [Task format](https://docs.harborframework.com/tasks/overview)
- [Multi-container environments](https://docs.harborframework.com/tasks/multi-container)
- [Custom installed agents](https://docs.harborframework.com/agents/custom-agents)
- [Separate verification](https://docs.harborframework.com/tasks/separate-verifier)
- [Built-in agents and MCP support](https://docs.harborframework.com/agents/pre-integrated-agents)

These references describe current upstream capabilities. Our existing pilot is
pinned to Harbor 0.24.0; validate APIs against that installed version before
using features from newer documentation.
