# Session 4b — Push-Recovery Full Training Run

Date: 2026-06-02 (training), 2026-06-03 (post-processing)  
Status: **PASS — full 2500-iteration run completed; best checkpoint identified.**

---

## 1. Command

```bash
cd ~/isaaclab-go2-locomotion
conda activate isaaclab_env
python scripts/train_pushrecovery.py \
    --num_envs 3072 \
    --max_iterations 2500 \
    --seed 42 \
    --headless
```

Log directory: `logs/rsl_rl/unitree_go2_flat_pushrecovery/2026-06-02_23-49-30/`

---

## 2. Hardware peaks

| Metric | Value |
|--------|-------|
| VRAM peak | 3831 MiB / 6141 MiB (62%) |
| GPU utilization (sustained) | ~68–88% |
| RAM | No OOM; within safe range for 16 GB + 16 GB swap |
| Wall time (training only) | **35.9 min** (2152 s) |
| Mean throughput | **86 468 steps/s** |
| Peak throughput | 89 926 steps/s |

VRAM was below the expected 4000–4200 MB estimate and well within the 5800 MB stop limit.
Throughput was above the 75–80k expectation (and above baseline session 2's 81k mean,
likely due to cuDNN cache warming during the longer run).

---

## 3. Training results

### Final metrics (iter 2499)

| Metric | Value |
|--------|-------|
| Final mean reward | 35.296 |
| Mean episode length | 984.9 steps |
| Lin vel tracking error (last iter) | 0.1853 m/s |
| Ang vel tracking error (last iter) | 0.3191 rad/s |
| Mean lin vel error (last 50 iter) | **0.1654 m/s** |
| Mean ang vel error (last 50 iter) | **0.3130 rad/s** |
| `base_contact` termination rate | 0.125 (12.5%) |

### Best checkpoint

| Field | Value |
|-------|-------|
| Best reward (TFEvents) | **36.757** at iteration 2374 |
| Best saved checkpoint | `model_2350.pt` (nearest saved; reward 36.437 at iter 2350) |
| Official checkpoint | `logs/rsl_rl/unitree_go2_flat_pushrecovery/2026-06-02_23-49-30/model_2350.pt` |

Checkpoints saved every 50 iterations. Iteration 2374 (TFEvents peak) falls between
model_2350 and model_2400; model_2350 has marginally higher saved reward (36.437 vs 36.427).

---

## 4. Learning curve summary

The reward trajectory matched the expected behavior from the brief:

- **Iter 0–200**: Rapid climb from −4 to ~30; policy learns basic locomotion under
  minimal disturbance (impulse at 30 N, sustained not yet active).
- **Iter 200–800**: Continued growth to ~34, with slightly slower rise as the impulse
  curriculum ramps from 30 N to 120 N and the sustained curriculum begins (10→40 N).
  No visible stagnation dip — the policy adapted without reward collapse.
- **Iter 800–1000**: Impulse curriculum saturated at 120 N; sustained ramp completes
  at ~iter 1000. Reward plateaus near 34–35.
- **Iter 1000–2500**: Slow continued growth and consolidation, reaching peak 36.76 at
  iter 2374. Slight oscillation in the final ~500 iter is normal PPO noise.

No reward collapse, no NaN, no OOM. `base_contact` rate at 12.5% at convergence reflects
the robot occasionally falling under full 120 N impulse loads, which is expected behavior
at this force magnitude.

---

## 5. Curriculum status at end of training

| Disturbance | Start force | End force | Ramp iterations | Status |
|-------------|-------------|-----------|-----------------|--------|
| Impulse (`push_robot_impulse`) | 30 N | 120 N | 0–800 | **Saturated at iter 800** |
| Sustained (`push_robot_sustained`) | 10 N | 40 N | 200–1000 | **Saturated at iter 1000** |

Both curricula reached their maximum force levels and ran at full difficulty for the
remaining 1500+ iterations.

---

## 6. Total push events triggered across training

| Event type | Cumulative triggers | Notes |
|------------|---------------------|-------|
| Impulse | **~371 251** total | Averaged over 3072 envs, ~120.8 per env total |
| Sustained | **~1 171** total | Lower rate due to 25–40 s interval and episode resets |

Impulse frequency is lower-than-naive expectation because episode resets re-randomize
the trigger delay, so short early-training episodes see fewer triggers per env.
At convergence (episode length ~985 steps ≈ ~49 s sim time) the effective impulse
rate is ~4–8 pushes per episode per env (interval 6–10 s), consistent with design.

---

## 7. Comparison: baseline vs push-recovery

| Metric | Baseline (session 2) | Push-Recovery (session 4b) | Delta |
|--------|---------------------|---------------------------|-------|
| Iterations | 2000 | 2500 | +500 |
| num_envs | 3072 | 3072 | — |
| Best reward | 37.989 @ iter 1999 | 36.757 @ iter 2374 | −1.23 |
| Final reward | 37.989 | 35.296 | −2.69 |
| Mean ep length (final) | 1000.0 steps | 984.9 steps | −15.1 |
| Lin vel error (mean last 50) | 0.1430 m/s | 0.1654 m/s | +0.023 |
| Ang vel error (mean last 50) | 0.2766 rad/s | 0.3130 rad/s | +0.036 |
| Mean throughput | 81 172 steps/s | 86 468 steps/s | +5296 |
| VRAM peak | 3917 MiB | 3831 MiB | −86 MiB |
| Wall time | ~40 min | 35.9 min | −4 min |

**Interpretation**: The push-recovery policy pays a ~3% reward penalty and ~16% higher
tracking error compared to the undisturbed baseline, while absorbing repeated 120 N lateral
impulses and sustained 40 N loads. This is the expected cost of robustness; no unexpected
degradation. The lower VRAM and higher throughput vs baseline are within run-to-run variance.

---

## 8. Replay video

```
logs/videos/go2_pushrecovery_best_iter2350.mp4
```

- num_envs: 32, video_length: 1200 steps (~24 s sim)
- Checkpoint: model_2350.pt
- Source: `logs/rsl_rl/unitree_go2_flat_pushrecovery/2026-06-02_23-49-30/videos/play/rl-video-step-0.mp4`

At 32 envs × 24 s sim time × ~0.125 push/s average, the video contains ~96 impulse push
events in total. Visual recovery behavior should be clearly visible across multiple envs.

---

## 9. Play script fix

`scripts/play_pushrecovery.py` was updated: `runner.alg.actor_critic` → `runner.alg.policy`
(rsl_rl 2.3.3 renamed the attribute). This is a scaffolding fix unrelated to the
push-recovery implementation.

---

## 10. Plots generated

**Push-recovery specific** (saved in `docs/images/`):
- `pushrecovery_mean_reward.png`
- `pushrecovery_mean_episode_length.png`
- `pushrecovery_rew_track_linvel.png`
- `pushrecovery_rew_track_angvel.png`
- `pushrecovery_curriculum.png`

**Comparison** (saved in `docs/images/`):
- `baseline_vs_pushrecovery_mean_reward.png`
- `baseline_vs_pushrecovery_rew_track_linvel.png`
- `baseline_vs_pushrecovery_episode_length.png`

Plotting script: `scripts/plot_session4b.py`

---

## 11. Anomalies and notes

- **TLAS warning during play**: Isaac Sim RTX renderer printed "TLAS limit: within: false"
  during play initialization. This is a benign warning; rendering proceeded without issue.
  Seen on this hardware with 32 envs + RTX rendering.

- **rsl_rl 2.3.3 API mismatch**: `PPO.actor_critic` was renamed to `PPO.policy` in
  rsl_rl 2.x. Play script fixed accordingly. Training was unaffected (train.py does not
  call export functions). Worth noting if upgrading rsl_rl further.

- **`base_contact` rate 12.5%**: At full 120 N impulse force (~8× body weight laterally),
  occasional falls are unavoidable. The policy recovers in the majority of cases; the
  termination rate is not catastrophic and is consistent with the difficulty level.

- **No reward collapse, no NaN, no OOM across 2500 iterations.** Training was fully clean.
