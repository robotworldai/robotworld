# Run Notes — Session 2

## Run metadata

| Field           | Value                                                                          |
|-----------------|--------------------------------------------------------------------------------|
| Date            | 2026-06-02                                                                     |
| Start time      | 11:58:18 local (UTC+2)                                                         |
| End time        | 12:29:xx local (model_1999.pt timestamp 12:29)                                 |
| Wall clock time | ~31 min total (startup ~1.5 min + RL training ~29.5 min)                       |
| Task            | `Isaac-Velocity-Flat-Unitree-Go2-v0`                                           |
| Seed            | 42                                                                             |
| num_envs        | 3072                                                                           |
| max_iterations  | 2000                                                                           |

## Training command

Training was started manually by the author. Reconstructed command:

```bash
# Run from: /home/msi/isaaclab-go2-locomotion/  (CWD)
python /home/msi/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 3072 \
  --max_iterations 2000 \
  --seed 42 \
  --headless
```

Log directory: `logs/rsl_rl/unitree_go2_flat/2026-06-02_11-58-18/`

(`--experiment_name` not specified; experiment name always defaults to task's agent config
name `unitree_go2_flat` due to Isaac Lab 2.1.1 bug — see Session 1 notes.)

## Play / video command

```bash
python /home/msi/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 32 \
  --video \
  --video_length 800 \
  --headless \
  --checkpoint logs/rsl_rl/unitree_go2_flat/2026-06-02_11-58-18/model_1999.pt
```

Video saved to: `logs/videos/go2_flat_run2_final.mp4`

## Hardware metrics

### VRAM during training (nvidia-smi, every 2 min)

| Time     | VRAM (MB) | Temp (°C) |
|----------|-----------|-----------|
| 12:03:25 | 3917      | 74        |
| 12:05:25 | 3909      | 75        |
| 12:07:25 | 3916      | 76        |
| 12:09:25 | 3872      | 76        |
| 12:11:26 | 3882      | 78        |
| 12:13:26 | 3878      | 76        |
| 12:15:26 | 3868      | 76        |
| 12:17:26 | 3845      | 73        |
| 12:19:26 | 3850      | 73        |
| 12:21:26 | 3850      | 73        |
| 12:23:26 | 3893      | 74        |
| 12:25:26 | 3865      | 73        |
| 12:27:26 | 3880      | 74        |

**Peak VRAM (training): 3917 MB (63.8% of 6141 MB)**  
Threshold was 5800 MB. Margin: ~1883 MB. Headroom available to scale further.

**Peak VRAM (play.py, cameras enabled, 32 envs): ~4684 MB (76.3%)**

**Peak temperature: 78°C** (12:11, early in sustained training). Settled at 73–76°C.

### RAM during training

| Stage           | Used     |
|-----------------|----------|
| At first check  | ~7.0 GB  |
| Peak observed   | ~7.5 GB  |
| After cleanup   | ~1.6 GB  |
| Swap used (post)| ~2.7 GB (normal Isaac Sim paging) |

## Training performance

| Metric                    | Value              |
|---------------------------|--------------------|
| Throughput (last iter)    | 84,212 steps/s     |
| Throughput (mean)         | ~81,172 steps/s    |
| Steps per iteration       | 73,728 (24 × 3072) |
| Total environment steps   | 147,456,000        |

## Final results (iteration 1999/2000)

| Metric                     | Value    |
|----------------------------|----------|
| Mean reward                | 37.99    |
| Mean episode length        | 1000.00  |
| track_lin_vel_xy_exp       | 1.4554   |
| track_ang_vel_z_exp        | 0.6753   |
| error_vel_xy (m/s)         | 0.1365   |
| error_vel_yaw (rad/s)      | 0.2722   |
| base_contact terminations  | 0.0000   |
| timeout terminations       | ~1.9     |

`base_contact = 0.0` means no body-ground falls during evaluation.
Episode length at maximum from iteration ~50 onward.

## Learning curve

| Iteration | Mean Reward | Mean Ep. Length |
|-----------|-------------|-----------------|
| 1         | -1.04       | 31.2            |
| 50        | -1.59       | 1000.0          |
| 100       | 10.19       | 972.8           |
| 150       | 21.10       | 991.0           |
| 200       | 28.83       | 986.5           |
| 300       | 34.22       | 1000.0          |
| 400       | 34.85       | 990.5           |
| 500       | 35.73       | 1000.0          |
| 700       | 36.44       | 1000.0          |
| 1000      | 37.30       | 1000.0          |
| 1200      | 37.04       | 995.7           |
| 1500      | 37.16       | 991.5           |
| 1800      | 36.36       | 995.7           |
| 1999      | 37.99       | 1000.0          |

Reward rises steeply to ~35 by iteration 300, then grows slowly. By iteration 1999
the reward is still marginally improving (not fully converged). A run of 3000–5000
iterations would be needed for convergence, but the current policy is mature.

## Log directory contents

```
logs/rsl_rl/unitree_go2_flat/2026-06-02_11-58-18/
├── events.out.tfevents.*          (TensorBoard log)
├── git/IsaacLab.diff
├── model_0.pt
├── model_50.pt  ...  model_1999.pt  (41 checkpoints every 50 iterations)
└── params/
    ├── agent.yaml / agent.pkl
    └── env.yaml   / env.pkl
```

## Plot files

Generated from TFEvents via matplotlib (no browser available in this session;
author can run `tensorboard --logdir logs/rsl_rl/unitree_go2_flat/2026-06-02_11-58-18/`
for interactive TensorBoard, or substitute these with browser screenshots):

- `docs/images/run2_mean_reward.png`
- `docs/images/run2_mean_episode_length.png`
- `docs/images/run2_rew_track_linvel.png`
- `docs/images/run2_rew_track_angvel.png`

## Session 1 vs Session 2 comparison

| Metric                         | Session 1 (iter 499) | Session 2 (iter 1999) | Change      |
|--------------------------------|----------------------|------------------------|-------------|
| num_envs                       | 2048                 | 3072                   | +50%        |
| max_iterations                 | 500                  | 2000                   | +4×         |
| Wall clock time                | 7 min 2 sec          | ~31 min                | +4.4×       |
| VRAM peak (training)           | 3482 MB (56.7%)      | 3917 MB (63.8%)        | +435 MB     |
| Throughput (steps/s)           | ~65,330              | ~84,212                | +29%        |
| Final mean reward              | 34.66                | 37.99                  | +9.6%       |
| Final mean episode length      | 1000.00              | 1000.00                | unchanged   |
| error_vel_xy (m/s)             | 0.1902               | 0.1365                 | −28%        |
| error_vel_yaw (rad/s)          | 0.3498               | 0.2722                 | −22%        |

Higher num_envs improved GPU utilization (56% → 64%) and throughput (+29%).
More iterations brought reward from 34.66 → 37.99 and noticeably improved velocity
tracking precision.

## Anomalies and warnings

1. **Alphabetical-vs-numerical sort confusion during monitoring.** During live monitoring,
   `ls *.pt | sort | tail -5` showed model_950 as the latest checkpoint even though
   model_1000–1150 already existed. String sort puts "1" < "9", so model_1000.pt sorts
   before model_950.pt. Corrected to `sort -V` (version-aware) midway. No impact on
   training; purely a monitoring artifact. Future sessions: always use `ls *.pt | sort -V`.

2. **Transient GPU utilization dips to 20–40%.** Seen mid-training at elapsed ~17–18 min.
   Caused by the gap between the rollout collection phase (high utilization) and the
   policy update phase (lower utilization). Normal PPO behavior. Not a stall — the
   process remained in 'Rl' state throughout.

3. **VRAM peak 3924 MB at elapsed ~24:53** (model_1600 checkpoint). This is the highest
   single reading; all others were 3845–3917 MB. Well under 5800 MB threshold.

4. **Standard omni.usd / MDL shader warnings** — same as session 1, cosmetic only.

5. **2.7 GB swap in use post-session** — same as session 1, expected.

6. **TensorBoard screenshots generated via matplotlib.** No browser was available on the
   path. Plots are saved as PNG in `docs/images/`. To obtain actual TensorBoard browser
   screenshots, run:
   ```
   conda activate isaaclab_env
   tensorboard --logdir logs/rsl_rl/unitree_go2_flat/2026-06-02_11-58-18/
   # then open http://localhost:6006 in a browser
   ```
