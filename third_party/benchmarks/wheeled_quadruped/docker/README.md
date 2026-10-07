# wheeled_quadruped container

Reuse existing `world/wheeledlab:isaac6.0.1-experimental`; build from the project Dockerfile with `bash scripts/eval/wheeled_quadruped.sh build` at World root. Sources/assets mount read-only; outputs go to a separate run directory. No benchmark or Codex source modifications.

This is an experimental Isaac6.0.1/IsaacLab2.2 compatibility runtime, not the upstream verified version. Upstream exact version is recorded per task in project.json. Source-built local World/codex app-server runs in a separate sandbox; Docker communicates through the isolated socket, not a direct model API. `WORLD_MODEL` selects a configured model; supply `--codex-home` for its provider credentials to the host launcher.

`bash scripts/eval/wheeled_quadruped.sh probe T09` must pass before reporting simulator compatibility or starting model evaluation. Images have now been built and inspected; see World/var/runs/docker/native17/builds/wheel-project-images.json. Native physics/model rollout is still pending; no task success is claimed.
