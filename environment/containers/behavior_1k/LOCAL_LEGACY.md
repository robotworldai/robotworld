# Historical local deployment (before 2026-09-30 merge)

Preserved for the local legacy launcher; not the default upstream protocol.

# BEHAVIOR-1K: Existing Local Docker

Reuse the already validated RobotWorld local environment; no new image or
Isaac 6.x migration is needed for this adapter.

Image: robotworld-behavior-python-compiler:ubuntu24.04-py311.
The existing launcher mounts its independent installed venv at /opt/behavior:
OmniGibson 3.9.3, Isaac Sim 5.1.0.0, Python 3.11, Torch 2.7.0+cu128.
This is an image plus host-local runtime mounts, not a portable baked image.

Use the existing runtime/behavior1k/docker/run_native.sh unchanged. It owns
the private daemon lifecycle and GPU locks, uses matching 580.105.08 graphics
libraries, GPU0 only, no container network and read-only licensed partial
assets. No model credentials or Docker socket enter the simulator container.

The outer main adapter is environment/benchmarks/behavior_1k/run.py.
Codex stays on the host. The launcher requires the existing private asset
receipt and machine-local installation; it does not silently download assets
or accept licenses.
