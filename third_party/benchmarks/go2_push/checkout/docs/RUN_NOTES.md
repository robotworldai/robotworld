# Baseline Run Notes — Session 1

## Run metadata

| Field             | Value                                                       |
|-------------------|-------------------------------------------------------------|
| Date              | 2026-06-01                                                  |
| Start time        | 20:11:01 local (UTC+2)                                      |
| End time          | 20:18:03 local                                              |
| Wall clock time   | 7 min 2 sec total (init ~27 s + training 6 min 35 sec)      |
| Task              | `Isaac-Velocity-Flat-Unitree-Go2-v0`                        |
| Seed              | 42                                                          |
| num_envs          | 2048                                                        |
| max_iterations    | 500                                                         |

## Training command

Run from `/home/msi/isaaclab-go2-locomotion/` (CWD determines log root path):

```bash
conda run -n isaaclab_env bash -c \
  "python /home/msi/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
   --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
   --num_envs 2048 \
   --max_iterations 500 \
   --seed 42 \
   --headless \
   --experiment_name go2_flat_baseline"
```

**Note:** `--experiment_name` is silently ignored by `update_rsl_rl_cfg` in Isaac Lab 2.1.1
(the flag is parsed but never applied to `agent_cfg`). The actual log directory uses the
task-default experiment name `unitree_go2_flat`.

Log directory: `logs/rsl_rl/unitree_go2_flat/2026-06-01_20-11-14/`

## Play / video command

```bash
conda run -n isaaclab_env bash -c \
  "python /home/msi/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
   --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
   --num_envs 32 \
   --video \
   --video_length 400 \
   --headless \
   --checkpoint logs/rsl_rl/unitree_go2_flat/2026-06-01_20-11-14/model_499.pt"
```

Video saved to: `logs/videos/go2_flat_baseline_iter499.mp4` (2.2 MB, 400 steps)

## Hardware metrics

### VRAM (nvidia-smi, training)

| Time     | VRAM (MB) | % of 6141 MB |
|----------|-----------|---------------|
| 20:12:50 | 3482      | 56.7%         |
| 20:13:50 | 3480      | 56.7%         |
| 20:14:50 | 3466      | 56.4%         |
| 20:15:50 | 3468      | 56.5%         |
| 20:16:50 | 3463      | 56.4%         |
| 20:17:50 | 3463      | 56.4%         |

**Peak VRAM (training): 3482 MB (56.7%)**  
Threshold was 5500 MB. Margin: ~2018 MB (~33 pp). Training is well within budget.

**Peak VRAM (play.py, cameras enabled, 32 envs): ~4781 MB (77.9%)**  
Rendering adds ~1.3 GB vs headless training. Still under threshold.

### RAM (from `free -h` during peak training)

| Metric      | Value |
|-------------|-------|
| Total       | 15 GB |
| Used (peak) | ~6.5 GB |
| Available   | ~7.3 GB |
| Swap used   | ~2.7 GB (post-session, likely OS paging, not OOM) |

## Training performance

| Metric                       | Value             |
|------------------------------|-------------------|
| Throughput                   | ~65,330 steps/s   |
| Iteration time               | 0.75 s/iteration  |
| Total environment steps      | 24,576,000        |
| rsl_rl training time         | 00:06:35          |

## Final results (iteration 499/500)

| Metric                     | Value    |
|----------------------------|----------|
| Mean reward                | 34.66    |
| Mean episode length        | 1000.00  |
| Mean action noise std      | 0.35     |
| track_lin_vel_xy_exp       | 1.4248   |
| track_ang_vel_z_exp        | 0.6379   |
| error_vel_xy (m/s)         | 0.1902   |
| error_vel_yaw (rad/s)      | 0.3498   |
| base_contact terminations  | 0.0000   |
| timeout terminations       | 1.9167   |

`base_contact = 0.0` means the robot never fell on its body during evaluation.
`episode_length = 1000` (max) was reached from iteration ~150 onwards.

## Learning curve

| Iteration | Mean Reward | Mean Ep. Length |
|-----------|-------------|-----------------|
| 1         | -0.57       | 11.8            |
| 50        | -3.87       | 991.4           |
| 100       | 8.18        | 996.1           |
| 150       | 12.57       | 1000.0          |
| 200       | 20.38       | 992.8           |
| 250       | 27.31       | 1000.0          |
| 300       | 31.05       | 1000.0          |
| 350       | 32.90       | 1000.0          |
| 400       | 33.38       | 1000.0          |
| 450       | 33.90       | 1000.0          |
| 499       | 34.66       | 1000.0          |

The negative reward at iteration 50 despite long episode length is expected: the robot
was surviving but accumulating penalties (action rate, joint torques, etc.). Positive
rewards begin around iteration 80–100 as velocity tracking improves.

Reward appears to be approaching a plateau (~34–35) — 500 iterations is not enough for
full convergence, but the policy clearly demonstrates learned locomotion.

## Log directory contents

```
logs/rsl_rl/unitree_go2_flat/2026-06-01_20-11-14/
├── events.out.tfevents.*        (TensorBoard log)
├── git/IsaacLab.diff            (IsaacLab repo state at run time)
├── model_0.pt
├── model_50.pt
├── model_100.pt
├── model_150.pt
├── model_200.pt
├── model_250.pt
├── model_300.pt
├── model_350.pt
├── model_400.pt
├── model_450.pt
├── model_499.pt                 (final checkpoint)
└── params/
    ├── agent.yaml / agent.pkl
    └── env.yaml   / env.pkl
```

## Anomalies and warnings

1. **`--experiment_name` CLI flag silently ignored (Isaac Lab 2.1.1 bug).**
   `update_rsl_rl_cfg` in `cli_args.py` does not apply `args_cli.experiment_name` to
   `agent_cfg`. The actual directory name always comes from the task's default agent
   config. For this task it is `unitree_go2_flat`. When running `play.py` without
   `--checkpoint`, use `--experiment_name unitree_go2_flat` (not `go2_flat_baseline`).

2. **`omni.usd` MDL source-asset warnings** — emitted at startup for `UsdPreviewSurface.mdl`,
   `UsdUVTexture.mdl`, etc. These are standard Isaac Sim 4.5 warnings about missing MDL
   shader discovery paths in headless mode. No functional impact.

3. **No `--log_dir` flag exists.** Log root is always `logs/rsl_rl/<experiment_name>`
   relative to CWD. Running the script from the project root is the correct way to redirect
   logs into this repo.

4. **2.7 GB swap in use post-session.** Likely from OS page-out of Isaac Sim's large virtual
   memory during training. No swap thrashing was observed during the run itself.
