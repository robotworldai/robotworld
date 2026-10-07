# RoboLab integration validation (2026-09-27)

## Implemented

- Unmodified NVlabs/RoboLab `ad45d4f974725d020f82c2b0d77d78533aeba2b3` is under `checkout/`.
- Official evaluator entry point is `environment/integrations/robolab_eval.py`; official environment stepping, time limits, success predicates, HDF5 and videos remain upstream.
- Source-built World/codex app-server relay, isolated agent workspace, native DROID joint-position / absolute-IK / relative-IK profiles, simultaneous arm+gripper tool, current + 4 historical tool-feedback observations at interval 2, complete events and automatic no-images events.
- Docker images built: `world/robolab:0.3.1-isaac5.0`, `world/robolab:0.3.1-isaac5.1` and `world/robolab:0.3.1-isaac6.0.1-experimental`.
- Selected official tools-container assets: manifest contains 42 files / 257,729,621 bytes. Remote NVIDIA MDL/texture references are preserved and listed; this is not a fully offline bundle.
- Main World regression suite: 85 tests passed, including 11 new RoboLab control/policy/lifecycle tests. Policy unit tests use a mocked Codex transport, not a live model.

## Runtime evidence and limits

Evidence is in `World/var/runs/docker/robolab/` (excluded from public release).

- `startup-probe01`: official Isaac Sim 5.0 / IsaacLab 2.2 image failed during native RTX startup on this RTX 5090 / driver 595.80 host. Image build/import success is not GPU runtime success.
- `isaac51-retry-01`: official NVIDIA IsaacLab 2.3.2 / Sim 5.1 Docker stack built successfully, but both standalone SimulationApp and camera-enabled AppLauncher probes crash in `librtx.scenedb.plugin.so` before any RoboLab scene or action is created. A standalone 5.0 control reproduces the same class of crash. All three probes failed; no 5.1 camera/physics/evaluation success is claimed. Host driver remains 595.80; upstream source remains clean; no 6.0.1 compatibility patches are loaded. See that run's README and logs, and NVIDIA's driver guidance at https://github.com/isaac-sim/IsaacSim/discussions/648 . This does not establish the cause of the earlier 6.0.1 penetration artifact.
- `startup-probe-isaac601-01` through `06`: preliminary IsaacLab 3 path is rejected. Beyond API changes, its XYZW configuration convention differs from the original WXYZ task/controller contract. These runs are not valid benchmark evidence.
- `startup-probe-isaac601-lab22-*`: current experimental path retains original IsaacLab 2.2 source/configuration and adapts APIs externally. The final path uses the Core 6.0.1 experience, original 2.2 render presets and ordinary single-env Camera with unchanged camera parameters. `startup-probe-isaac601-lab22-19` passed: 3 native joint-position steps, original predicates, two real 1280x720 RGB images and all six proprioception fields. Live source-Codex smoke validation passed as detailed below; no official task success has been established.

`probe.json` together with `exit.json` indicating `probe_passed: true` is required for a successful environment probe. Exit code 0 alone is insufficient: SimulationApp shutdown can mask exceptions. `error.txt` preserves Python failures.

## Task mapping

The nine screenshot entries are task families / RobotWorld scenarios, not nine uniquely identified upstream classes. See `robotworld-tasks.json`. Related official tasks are recorded without inventing full-container rules, injected drops, false completion messages or sensor dropout. Custom recovery protocols need separate definitions and must not be reported as the original official benchmark.

## Live source-Codex validation

Source commit: `8ae55c863db26d417e83390c5854f1144114276b`; binary SHA-256: `6ca8992ea4cda34050a14ca409239e4231a13196bdd2357a632ba5b1606a2892`. This is the locally built app-server, not the installed CLI or the assistant executing robot actions itself.

- `codex-smoke01`: official ToolOrganizationBothTask, absolute IK, model called move_robot / set_gripper / move_eef; 70 requested and 70 confirmed control steps. Both videos contain 70 frames at 15 FPS. Stopped at the configured 3-call limit. Its initial HDF5 is truncated and must not be used; external abort cleanup was added after detecting this.
- `codex-smoke02-relative`: same official task, relative IK, model called move_robot; 10 requested and 10 confirmed control steps. Stopped at the configured 1-call limit. Partial HDF5 reopens correctly, actions shape `(10, 7)`, with no cleanup errors. Full and no-images event streams exist.
- Both stops are infrastructure budget aborts, not complete official episodes or task-success results. The upstream recorder's `success=false` in partial data is not an official failure score. `episode-aborted.json` and HDF5 `world_episode_aborted` / `world_official_episode_complete` attributes identify partial data. No benchmark result aggregation runs for aborted episodes.
- The abort wrapper calls the original recorder's `flush_buffer`, closes files and the environment, and does not call `export_episodes` to invent final task status. Normal official evaluation is delegated unchanged.

The experimental 6.0.1 engine is not claimed numerically equivalent to the original 5.0 baseline. Only this selected rigid-object scene has been GPU tested; the full 120-task benchmark and nine requested scenario families are not validated.
