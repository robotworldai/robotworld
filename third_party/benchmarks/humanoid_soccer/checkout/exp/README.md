# MuJoCo Sim2Sim Rollout

This folder contains standalone MuJoCo sim2sim rollout code for the stage-2 soccer policy.

The rollout samples robot and ball initial states, selects a kicking motion from the released motion set, runs the exported policy in MuJoCo, and writes per-trial outputs under `exp/results/`. Generated outputs are ignored by git.

Install the ONNX runtime before using `--policy` or `--policy-dir`:

```bash
python -m pip install onnxruntime
```

## Run

Run directly with the released ONNX policy and a MuJoCo viewer:

```bash
python exp/mujoco_soccer_experiment.py \
  --policy ckp/policy_30000.onnx \
  --motion-path motions/soccer-standard \
  --num-trials 1 \
  --visualize
```

Remove `--visualize` for batch evaluation:

```bash
python exp/mujoco_soccer_experiment.py \
  --policy ckp/policy_30000.onnx \
  --motion-path motions/soccer-standard \
  --num-trials 100
```

Useful options:

```bash
python exp/mujoco_soccer_experiment.py \
  --experiment-name test \
  --policy ckp/policy_30000.onnx \
  --motion-path motions/soccer-standard \
  --num-trials 1000 \
  --birth-radius 0.25 \
  --ball-forward-range 0.5 3.0 \
  --ball-lateral-range -1.5 1.5 \
  --goal 5.0 0.0 \
  --goal-width 2.0 \
  --enable-ball-vel \
  --ball-speed-range 0.1 0.3 \
  --ball-spawn-max-attempts 128
```

To export a custom policy from an Isaac Lab checkpoint, use the existing play script:

```bash
python scripts/rsl_rl/play_multi.py \
  --task Tracking-Flat-G1-SoccerMoving-RNN-v0 \
  --motion_path motions/soccer-standard \
  --load_run <run-name> \
  --checkpoint <checkpoint-file> \
  --export_motion_name all \
  --num_envs 1 \
  --headless
```

This creates ONNX files under the checkpoint directory's `exported/` folder. Use `--policy-dir <exported-dir>` when you exported one policy per motion, or `--policy <file.onnx>` when evaluating a single shared ONNX file.

The ball spawn ranges and goal coordinates are interpreted in the robot initial local frame by default. Thus the defaults sample the ball in front of the robot with local `x` in `[0.5, 5.0]` and local `y` in `[-2.5, 2.5]`, and put the goal center at `(6, 0)`.

Each run writes:

- `config.json`: resolved experiment configuration and generated MJCF path
- `episodes.csv`: one row per randomized trial
- `summary.json`: aggregate success, tracking, smoothness, contact, and ball metrics

## Evaluation

The MuJoCo experiment already computes per-trial tracking metrics and writes the aggregate results to `summary.json`. For offline trajectory evaluation on `.pkl` rollout folders, use `eval_baseline.py`:

```bash
python exp/mujoco_soccer/eval_baseline.py \
  input.rollout_traj=<rollout-pkl-dir> \
  input.reference_traj=<reference-pkl-dir> \
  input.save_path=.tmp/tracker_eval.json
```

The evaluator expects each input directory to contain matching `.pkl` motion files. With `input.reference_traj` set, it reports imitation accuracy metrics such as `E_gmpjpe`, `E_mpjpe`, `E_dof_mpjpe`, `E_vel`, and `E_acc`; without a reference directory, it reports smoothness metrics only. Results are saved to the requested JSON file plus a `_raw.json` companion file.

Useful overrides:

```bash
python exp/mujoco_soccer/eval_baseline.py \
  input.rollout_traj=exp/results/<rollout-pkl-dir> \
  input.reference_traj=motions/<reference-pkl-dir> \
  input.save_path=exp/results/tracker_eval.json \
  skeleton=evaluator/config/skeleton/g1_29dof.yaml \
  delta_per_frame=true
```

## Code Layout

- `mujoco_soccer_experiment.py`: thin CLI entry point
- `mujoco_soccer/cli.py`: argument parsing and top-level orchestration
- `mujoco_soccer/runner.py`: rollout execution and spawn validation
- `mujoco_soccer/scene.py`: MuJoCo XML generation and simulator wrapper
- `mujoco_soccer/motion.py`: motion loading and selection
- `mujoco_soccer/policy.py`: ONNX policy adapter
- `mujoco_soccer/eval_baseline.py`: offline trajectory accuracy and smoothness evaluator
