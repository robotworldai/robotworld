# Optional Agent Docker backend

The default `IsolatedCodex` backend remains bubblewrap. On a host where user/PID
namespaces cannot be created by bubblewrap, set `WORLD_AGENT_BACKEND=docker`.
This is an explicit local runtime variation, not a change to task or scoring rules.

Required host settings:

- `WORLD_AGENT_DOCKER_BIN`: Docker CLI path (default `docker`).
- `DOCKER_HOST`: the selected daemon endpoint, if not the default.
- `WORLD_AGENT_DOCKER_IMAGE`: immutable local `sha256:<64 hex>` image ID.
- `WORLD_AGENT_MODEL_SOCKET`: dedicated, existing `relay.sock` served by a trusted
  host-side model proxy. That proxy must enforce the selected endpoint/model and
  request budget and keep authentication outside the Agent container.

The image must supply Python 3 and shared libraries compatible with the verified
source-built Codex binary. The current Linux implementation additionally mounts
host libssl 1.1, libcrypto 1.1 and libbz2 from `/lib64`; it is not portable to every
host without adapting and revalidating those library paths. The host needs
permission to create mounts and assign workspace ownership to UID/GID 65532.

The container is nonroot, has no network, drops all capabilities, uses a read-only
root filesystem, and does not receive credentials, simulator assets, evaluator
files or the Docker socket. Only the per-episode workspace is writable; observation
exports are read-only. The model socket is an intentional, restricted capability,
not arbitrary network access. Do not substitute an unrestricted proxy.

The original source-build validation remains mandatory. Only `model` and
`model_reasoning_effort` from the supplied auth-home config are copied into a
credential-free config. Provider authentication/routing is the trusted proxy's
responsibility. Thread/turn model selections remain controlled by the benchmark.
The JSON-RPC relay selects `externalSandbox` with restricted network for turn and
command execution, without rewriting images, tools or task instructions.

## Validation status

2026-09-30: main `IsolatedCodex` and `CodexSession` were exercised with a real
source-built app-server. Nonroot/process/filesystem/observation-write checks and
`command/exec` passed. Two GPT-6 Astra synthetic tests correctly read the initial
image, called a tool, and read the new tool-returned image; containers were removed.
The configured provider returned token-rate-limit failures before successful retries,
so strict all-requests-success acceptance did not pass. Historical attempts are
retained in the local validation archive and are not included in this repository.

These tests disabled built-in model tools only for the synthetic one-tool test.
Formal task tool profiles were not changed. This backend alone does not provision
the task-specific trusted proxy, change simulator GPU mounts, or start evaluations.
Real robot control and eight-concurrent model episodes remain to be validated.
