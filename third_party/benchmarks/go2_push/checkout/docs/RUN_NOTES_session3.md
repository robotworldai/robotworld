# Run Notes — Session 3

## Run metadata

| Field           | Value                                                                              |
|-----------------|------------------------------------------------------------------------------------|
| Date            | 2026-06-02                                                                         |
| Start time      | 13:56:47 local (UTC+2)                                                             |
| End time        | ~14:55 local (58 min 30 sec total)                                                 |
| Wall clock time | 58 min 30 sec total (startup ~1.5 min + RL training ~57 min)                       |
| Task            | `Isaac-Velocity-Flat-Unitree-Go2-v0`                                               |
| Seed            | 42                                                                                 |
| num_envs        | 3072                                                                               |
| max_iterations  | 4000                                                                               |

## Training command

```bash
# Run from: /home/msi/isaaclab-go2-locomotion/  (CWD)
/home/msi/miniforge3/envs/isaaclab_env/bin/python \
  /home/msi/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 3072 \
  --max_iterations 4000 \
  --seed 42 \
  --headless
```

Note: Full Python path required because `python` is not on the system PATH
(must use conda env Python binary directly, or `conda run -n isaaclab_env python ...`).

Log directory: `logs/rsl_rl/unitree_go2_flat/2026-06-02_13-56-52/`

## Play / video command

Best checkpoint: `model_2050.pt` (reward 37.8524 — best available checkpoint; see
"Best checkpoint identification" section below).

```bash
/home/msi/miniforge3/envs/isaaclab_env/bin/python \
  /home/msi/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 32 \
  --video \
  --video_length 800 \
  --headless \
  --checkpoint /home/msi/isaaclab-go2-locomotion/logs/rsl_rl/unitree_go2_flat/2026-06-02_13-56-52/model_2050.pt
```

Video saved to: `logs/videos/go2_flat_run3_best_iter2050.mp4` (3.8 MB, 800 steps)

## Hardware metrics

### VRAM during training

VRAM monitoring was not captured during session 3 (the training completed in ~58 min
while monitoring setup was in progress). Based on session 2 data using identical
parameters (same task, num_envs=3072, seed=42), estimated VRAM: ~3900 MB ±100 MB
(~63–64% of 6141 MB). No stop conditions were triggered.

**Estimated peak VRAM (training): ~3900 MB (~63.5%) — within safe range**

### Iteration time

Consistent throughout: 0.88–0.90 s/iteration. No signs of thermal throttle
(no increasing trend over the run). Peak temperature during session 2 (identical
config) was 78°C; session 3 expected similar or slightly higher due to longer duration.

## Training performance

| Metric                    | Value               |
|---------------------------|---------------------|
| Throughput (final iter)   | 82,351 steps/s      |
| Throughput (mean)         | ~84,067 steps/s     |
| Throughput (min/max)      | 36,661 / 87,529     |
| Steps per iteration       | 73,728 (24 × 3072)  |
| Total environment steps   | 294,912,000         |
| Iteration time            | 0.88–0.90 s/iter    |

## Best checkpoint identification

TFEvents logs every iteration. The maximum logged `Train/mean_reward` occurred at
**iteration 1999** (reward 37.9890). However, rsl_rl saves checkpoints every 50
iterations, so `model_1999.pt` does not exist in session 3 (it was only the final
checkpoint in session 2's 2000-iteration run).

Checking rewards at available checkpoints near the peak:

| Checkpoint       | Reward   |
|------------------|----------|
| model_1950.pt    | 37.4939  |
| model_2000.pt    | 37.7739  |
| **model_2050.pt**| **37.8524** ← best available |
| model_2100.pt    | 37.2849  |

**Best available checkpoint: `model_2050.pt` (reward 37.8524)**

This is the checkpoint used for the demo video and designated as the official
session 3 baseline.

## Final results (iteration 3999/4000)

| Metric                     | Value    |
|----------------------------|----------|
| Mean reward (final)        | 37.26    |
| Mean reward (best ckpt)    | 37.85    |
| Mean episode length        | 1000.00  |
| track_lin_vel_xy_exp       | 1.4485   |
| track_ang_vel_z_exp        | 0.6748   |
| error_vel_xy (m/s)         | 0.1483   |
| error_vel_yaw (rad/s)      | 0.2732   |
| base_contact terminations  | 0.0000   |
| timeout terminations       | ~3.1     |

`base_contact = 0.0` means no body-ground falls during evaluation.

## Learning curve

| Iteration | Mean Reward | Mean Ep. Length |
|-----------|-------------|-----------------|
| 0         | -0.58       | 13.7            |
| 50        | -1.59       | 1000.0          |
| 100       | 10.19       | 972.8           |
| 150       | 21.10       | 991.0           |
| 200       | 28.83       | 986.5           |
| 300       | 34.22       | 1000.0          |
| 400       | 34.85       | 990.5           |
| 500       | 35.73       | 1000.0          |
| 700       | 36.44       | 1000.0          |
| 1000      | 37.30       | 1000.0          |
| 1500      | 37.16       | 991.5           |
| 2000      | 37.77       | 1000.0          |
| 2500      | 37.02       | 995.4           |
| 3000      | 36.38       | 986.5           |
| 3500      | 37.07       | 995.7           |
| 3999      | 37.26       | 1000.0          |

The reward curve rises steeply to ~35 by iter 300, then plateaus around 37–38 from
iter ~1000 onward. From iter 2000 to 4000 the reward oscillates between 36.4 and 37.9
with no clear upward trend, confirming convergence. The policy is fully trained.

## Convergence verdict

**The policy was converged by ~iter 1000–2000.** Extending from 2000 to 4000 iterations
produced no measurable improvement (reward range 2000–4000: 36.4–37.9, peak at iter 2050
= 37.85). This confirms that the session 2 checkpoint (`model_1999.pt` from
`2026-06-02_11-58-18`) was already near-optimal. Session 3's value is the confirmation of
convergence, not additional improvement.

## Log directory contents

```
logs/rsl_rl/unitree_go2_flat/2026-06-02_13-56-52/
├── events.out.tfevents.*          (TensorBoard log)
├── git/IsaacLab.diff
├── model_0.pt
├── model_50.pt  ...  model_3999.pt  (81 checkpoints every 50 iterations)
├── videos/play/rl-video-step-0.mp4  (raw play output)
└── params/
    ├── agent.yaml / agent.pkl
    └── env.yaml   / env.pkl
```

## Plot files

Generated from TFEvents via matplotlib (same style/colors as session 2):

- `docs/images/run3_mean_reward.png`
- `docs/images/run3_mean_episode_length.png`
- `docs/images/run3_rew_track_linvel.png`
- `docs/images/run3_rew_track_angvel.png`
- `docs/images/run2_vs_run3_mean_reward.png` (overlay: run2 vs run3)
- `docs/images/run2_vs_run3_rew_track_linvel.png` (overlay: run2 vs run3)

To view interactively:
```
conda activate isaaclab_env
tensorboard --logdir logs/rsl_rl/unitree_go2_flat/2026-06-02_13-56-52/
# then open http://localhost:6006
```

## Session comparison: sessions 1–3

| Metric                         | Session 1 (500 iter) | Session 2 (2000 iter) | Session 3 (4000 iter) |
|--------------------------------|----------------------|------------------------|------------------------|
| num_envs                       | 2048                 | 3072                   | 3072                   |
| Wall clock time                | 7 min 2 sec          | ~31 min                | ~58 min 30 sec         |
| VRAM peak (training)           | 3482 MB (56.7%)      | 3917 MB (63.8%)        | ~3900 MB (~63.5%) est. |
| Throughput (steps/s, mean)     | ~65,330              | ~84,212                | ~84,067                |
| Total env steps                | 24,576,000           | 147,456,000            | 294,912,000            |
| Final mean reward              | 34.66                | 37.99                  | 37.26                  |
| Best ckpt reward               | 34.66                | 37.99                  | 37.85 (model_2050)     |
| error_vel_xy (m/s)             | 0.1902               | 0.1365                 | 0.1483                 |
| error_vel_yaw (rad/s)          | 0.3498               | 0.2722                 | 0.2732                 |
| Policy converged?              | No (still growing)   | Near-converged         | Yes (confirmed)        |

Session 2 `model_1999.pt` (reward 37.99) remains the highest-reward snapshot.
Session 3 confirms convergence and demonstrates the plateau is stable, not a local dip.

## Anomalies and warnings

1. **Training completed before VRAM monitoring was established.** The background
   monitoring setup (waiting for "Simulation App Starting" in the log) did not match
   the actual log string, so the monitoring check returned immediately as failed. The
   training ran unmonitored. No stop conditions were triggered (inferred from successful
   completion and consistent throughput). Future sessions: monitor VRAM with a separate
   `watch nvidia-smi` or a polling loop with a different trigger phrase.

2. **`python` not on system PATH.** `conda activate isaaclab_env` in the author's
   terminal does not affect Claude Code's subprocess environment. Always use the full
   path `/home/msi/miniforge3/envs/isaaclab_env/bin/python` or
   `conda run -n isaaclab_env python ...` in scripts launched by Claude Code.
   (Rule added to pipeline feedback memory.)

3. **play.py requires absolute checkpoint path.** Using a relative path
   (`logs/rsl_rl/.../model_X.pt`) causes `FileNotFoundError` because play.py resolves
   relative to its own directory, not CWD. Always pass the full absolute path to
   `--checkpoint`.

4. **No `model_1999.pt` in session 3.** rsl_rl saves the final checkpoint under its
   actual iteration number (so `model_1999.pt` for a 2000-iter run, `model_3999.pt` for
   a 4000-iter run). Intermediate checkpoints are every 50 iterations. The TFEvents
   peak at iter 1999 is between saved checkpoints; `model_2050.pt` is the best available.

5. **Standard omni.usd / MDL shader warnings** — same as all previous sessions, cosmetic.
