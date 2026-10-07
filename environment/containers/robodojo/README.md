# RoboDojo / Isaac Sim 6.0.1 Docker

Local mirror: `world/robodojo:isaac6.0.1-local`, manifest `sha256:ab5fa941de440148940ea88d68230c720d322955365c63b931b0d24d887f237e`. The mirror contents are about 25.4 GB, Docker and the decoy layer occupy about 51.6 GB. It is not a proven zero-installed formula that has been installed using a snapshot.

The real four-step move_eef test driven by source-built Codex passed; For full task results see [STATUS.md](../../benchmarks/robodojo/STATUS.md). [External Compatibility Layer](../../benchmarks/robodojo/compat/README.md) is explicitly authorized by the user without modifying the upstream source code.

## Run

```bash
export WORLD_ROOT=/opt/robotworld/World
export WORLD_CODEX_HOME="$WORLD_ROOT/var/auth/robodojo-codex"
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" doctor
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" probe
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" smoke
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" task
```

Requires the current shell to have Docker permissions. doctor inspects the internal version of the mirror, probe initializes the real scene and camera, smoke allows the source code Codex to lift the left hand 1 cm, return, keep the right claw open and submit a cross-border target. task uses official conveyor belt instructions; The default 32 sub-tool calls / model round 600 seconds, plus `--max-actions`, `--timeout`. When policy ceases early, the uncompleted tasks are liquidated by the environmental disclosure according to the original RoboProbe logic; smoke retains unterminated result null.

The default packaging RAM cap 32 GiB (`--memory`) with a total time limit of 900 seconds (`--wall-timeout`, including simulation native calls). Output path printed on startup. `progress.json` recording phase, `episode.json` recording tool call and task results, `frames/` save RGB, `exit.json` record container exit code.

## Content and Mount

The mirror consists of Python 3.12, Isaac Sim 6.0.1.0, IsaacLab package 6.1.11, CuRobo, unchanged RoboDojo export source code, ffmpeg, control protocol dependency and necessary sharing library. The source list is in the mirror `/opt/world-runtime-manifest.json`.

Runs with only World (source code build Codex to external bridge), single scene Assets, cache, result and independent Codex configuration directory. Host Python / RoboDojo / IsaacLab is not mounted. Model requests are still processed by the construction product of World/codex, do not call the installed Codex, and do not write the model HTTP client. The dedicated Codex configuration directory contains authentication information and does not include mirrors or GitHub.

Single scenario asset from local copy: `var/datasets/robodojo/match_and_pick_from_conveyor_0/Assets`, 216 files, approximately 1 GiB, fixed eval seed 1/ layout index 0. They do not enter the mirror; The original asset reference audit showed partial external material citations and could not be declared completely offline.

## Build and Open

See [SNAPSHOT_BUILD.md](SNAPSHOT_BUILD.md). Build with independent BuildKit, hard memory limit 8 GiB; Client address space limit 32 GiB. Builds by disk directory, does not compress local export, and avoids a problem with the already large stdin tar/ PAX error recognition.

Open source submission of Dockerfile, build/run script, compatible code and source description; Do not submit the entire mirror, Python installation directory, authentication or data asset. If you need to move a mirror, you can move it in a separate `docker save` or push it to the mirror warehouse after being authorized to do so.

`run_isaac601_local.py`, when not `--image`, is a mounted diagnosis, which is primarily used for checking dependence; Regularly run using `run-conveyor.sh` above. Default always uses an independent mirror.

The old 5.1 Dockerfile, build.sh, run.sh, and source-build instructions remain in [README.isaac51.md](README.isaac51.md); they are no longer the default path.
