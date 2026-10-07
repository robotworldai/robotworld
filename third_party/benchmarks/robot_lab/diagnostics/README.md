# Physical compatibility of T11 A1

These procedures are used to distinguish between the impact of action playback, asset conversion, running time version and CPU/GPU path. ** is not a new benchmark and does not output success. ** does not call the model and does not modify the robot_lab upstream or Codex source code.

See `var/runs/docker/robot_lab/physics-friction-fix-report.md` for following up the problem of locating and repairing A1 zero joint frictions that did not take effect at 6.0.1. Default `WORLD_A1_FRICTION_COMPAT=0` for diagnostic scripts to preserve old baselines; `--friction-compat` must be passed to verify the production restoration. `--zero-joint-friction` is a single variable digesting zero after reset, and `--friction-compat` is a production repair performed during the initialization phase, not mixed. `--steps-override 50` can shorten the positioning experiment.

`physics-stage-attributes.json` records scene physics properties. `effective-parameters.json` reads both old and new joint-friction APIs so that neither representation is missed.

## Documentation

- `run.py`: Host starter, using an independent Docker process in each case, to save commands, program snapshots, logs and return codes.
- `compare.py`: Carry out a fixed action in the container to record the joints, force rectangles, attitude, contact and original terminations by control step.
- `summarize.py`: Align the tracks by joint name, check the actions for step index and calculate the differences.
- `isaaclab22/`: frame files extracted from the current 6.0.1 mirror `/opt/isaaclab22` for comparison with 4.5 CPU; Not complete Git checkout. `isaaclab22-source.json` records the source mirror and SHA256 of each file.

## Experiment definition

| Control | Steps | Meaning |
|---|---:|---|
| hold | 500 | action complete zero, i.e. maintain the default joint target; Not shut down the power, not a reliable balance controller. |
| pulse | 200 | Zero-based action ranges 40–59, 90–109, and 140–159 add 0.2 to FR hip/thigh/calf respectively, corresponding to a 0.05 rad target offset |
| replay | 101 | Replay original requested_action for `a1-feet-gpt01/events/environment.jsonl` |

The control period is 0.02 s, with four 0.005 s physics substeps. The exposure record is the peak of the most recent 3 physical samples retained by the end-of-control sensor at each control step, following the original termination method; ** is not a complete contact trajectory of each physical substep **.

`native-*` keeps the original mission randomized, rewarded, terminated and observed noise and omits to review the camera. Two precise replays of the order in which the assets are converted and the environment constructed, and validates the initial actor observation, not just setting the same seed.

`fixed-*`, `isaac*-cpu-*` are controlled diagnostics: turn off events/rewards/terminations/curriculum and observe noise, reset the default position and zero speed, and remove the high-level scanner that actor/critic does not use. Continue to operate after exposure for comparative response purposes and cannot be counted according to the original task.

Three asset variants are compared: the original merged-foot USD (13 bodies), the 6.0.1-compatible USD (17 bodies), and a 4.5 USD exported from the same foot-preserving URDF (17 bodies). 13 calf contact with the body asset includes normal foot bottom contact and cannot be directly called illegal contact.

## Revert

Executes from `World`'s parent directory. Two mirrors and reference rounds are required. The output directory must be a new directory:

```bash
python World/third_party/benchmarks/robot_lab/diagnostics/run.py \
  --output "$PWD/World/var/runs/docker/robot_lab/my-native-controls" \
  --cases native-replay-a native-replay-b native-hold

python World/third_party/benchmarks/robot_lab/diagnostics/run.py \
  --output "$PWD/World/var/runs/docker/robot_lab/my-45-controls" \
  --cases isaac45-lab22-cpu-hold isaac45-lab22-cpu-pulse isaac45-lab22-cpu-replay

python World/third_party/benchmarks/robot_lab/diagnostics/run.py \
  --output "$PWD/World/var/runs/docker/robot_lab/my-6-shared-usd-controls" \
  --shared-usd "$PWD/World/var/runs/docker/robot_lab/my-45-controls/isaac45-lab22-cpu-hold/old-import/a1.usd" \
  --cases isaac6-oldusd-cpu-hold isaac6-oldusd-cpu-pulse isaac6-oldusd-cpu-replay

python World/third_party/benchmarks/robot_lab/diagnostics/summarize.py \
  World/var/runs/docker/robot_lab/my-native-controls \
  World/var/runs/docker/robot_lab/my-45-controls \
  World/var/runs/docker/robot_lab/my-6-shared-usd-controls \
  --output World/var/runs/docker/robot_lab/my-comparison.json
```

Replace reference rounds with `--reference`; Current playback verification is limited to this 101 trail. Execute the entire matrix when `--cases` is not provided, and the final shared USD group by default relies on this `physics-controls-07` export; Suggests a visible grouping and a specified path as above.

Each case: `configuration.json` for actual configuration, `effective-parameters.json` for operational parameters, `physics.jsonl` for no image track, `result.json` for trajectory conclusion, `launcher.log` for parent directory `index.json` for process state. Even with complete result, it is necessary to check whether the process collapsed during the clean-up phase.

This result is in `World/var/runs/docker/robot_lab/physics-controls-report.md`. The cross-version CPU test cannot claim an equal value to the official GPU, nor can it position all differences to the PhysX kernel: Python, Torch, extension and operational integration remain different.
