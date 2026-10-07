# Session 4a — Push Recovery Implementation & Smoke Test

Date: 2026-06-02  
Goal: Implement push-recovery task extension; validate with 50-iteration smoke test.  
Status: **PASS — ready for author review before full training (4b).**

---

## 1. Files created

| File | Purpose |
|------|---------|
| `src/isaaclab_go2_pushrecovery/__init__.py` | Gym env registration (`Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0` and `…-Play-v0`) |
| `src/isaaclab_go2_pushrecovery/env_cfg.py` | Task config (env + PPO runner), inherits from baseline flat env |
| `src/isaaclab_go2_pushrecovery/push_events.py` | Custom EventTerm: applies random XY force to robot base |
| `src/isaaclab_go2_pushrecovery/curriculum.py` | CurriculumTerm: linearly ramps max_force 30 N → 100 N |
| `scripts/train_pushrecovery.py` | Training wrapper (thin shell around Isaac Lab's train.py logic) |
| `scripts/play_pushrecovery.py` | Play/evaluation wrapper |

### What each file does

**`push_events.py — apply_push_force()`**  
Samples a uniformly random angle in [0, 2π] and a magnitude in [0, `max_force`] per
environment, then calls `asset.set_external_force_and_torque()` with an XY-plane force
vector. No torque is applied. The force persists in the physics buffer until the next
interval trigger or episode reset.

**`curriculum.py — push_force_curriculum()`**  
Uses `env.common_step_counter` (increments by 1 per `env.step()` call) to compute a
linear ramp from `start_force` to `end_force` over `num_curriculum_steps` common steps.
Mutates `env.event_manager.get_term_cfg("push_robot_force").params["max_force"]` in
place so the event picks up the new value on the next trigger.

**`env_cfg.py — UnitreeGo2PushRecoveryEnvCfg`**  
Inherits from `UnitreeGo2FlatEnvCfg`. In `__post_init__`:
- Adds `push_robot_force` EventTerm in interval mode.
- Adds `push_force` CurriculumTerm.
- All other settings (reward weights, observations, PPO hyperparameters) unchanged.

The Isaac Lab managers read config via `cfg.__dict__.items()`, so dynamically added
attributes in `__post_init__` are correctly discovered without requiring a separate
subclass.

**`env_cfg.py — UnitreeGo2PushRecoveryPPORunnerCfg`**  
Inherits from `UnitreeGo2FlatPPORunnerCfg`. Only changes:
- `experiment_name = "unitree_go2_flat_pushrecovery"` (separate log directory)
- `max_iterations = 2000` (full training target; overridden per CLI during smoke test)

**`scripts/train_pushrecovery.py`**  
Follows the same pre-launch / post-launch structure as Isaac Lab's `train.py`.
Adds `src/` to `sys.path` and imports `isaaclab_go2_pushrecovery` (which triggers gym
registration) before calling `@hydra_task_config`. Borrows `cli_args.py` from
`~/IsaacLab/scripts/reinforcement_learning/rsl_rl/` via `sys.path` injection.

---

## 2. Smoke test

### Command

```bash
cd ~/isaaclab-go2-locomotion
conda activate isaaclab_env
python scripts/train_pushrecovery.py \
    --num_envs 1024 --max_iterations 50 --seed 42 --headless
```

### Outcome

| Metric | Value |
|--------|-------|
| Completed iterations | 50 / 50 |
| NaN rewards | None |
| Final mean reward (iter 49) | -3.63 |
| Mean episode length (iter 49) | 997 steps |
| `base_contact` terminations (iter 49) | 0.00 |
| Curriculum `push_force` at iter 0 | 30.04 N |
| Curriculum `push_force` at iter 49 | 34.33 N |
| Throughput | ~53 000 steps/s |
| Wall time | ~25 s |

Log directory: `logs/rsl_rl/unitree_go2_flat_pushrecovery/2026-06-02_18-02-16/`  
Checkpoints: `model_0.pt`, `model_49.pt`

### Interpretation

- Reward starts near zero and becomes more negative as the policy explores; this is
  normal at 50 iterations (baseline took ~300–500 iter to converge).
- Curriculum is advancing correctly: 30 + 6.25% × 70 N ≈ 34.4 N at iteration 49
  (confirmed by log: 34.33 N).
- Episode length grows from ~12 steps (iter 1, before the VRAM is warm) to ~997 steps
  (iter 49), indicating the policy has not yet learned but is not collapsing either.
- No `base_contact` terminations at iter 49, meaning the agent is not systematically
  falling under the initial 30–34 N push regime.

---

## 3. Curriculum parameters chosen and justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `start_force` | 30 N | ≈ 2 m/s² lateral accel for ~15 kg Go2. Moderate enough that the initial policy (transferring baseline weights not assumed here — starting from scratch) can survive early episodes. |
| `end_force` | 100 N | ≈ 6.7 m/s² — severe lateral disturbance. Go2 COP margin is small; this should require active recovery steps, not just stiffness. |
| `num_curriculum_steps` | 19 200 (= 800 PPO iter × 24 steps/iter) | Matches brief's 800-iteration ramp target. |
| Push interval | (10, 15) s | Same as baseline's `push_robot` timing. Allows 1–2 full gait cycles between pushes at 20 s episode length. |
| Push direction | Uniform XY circle | No bias — robot must generalize to all lateral directions. |
| Push duration | Sustained until next trigger | See design note below. |

### Design note: sustained force vs. short impulse

The brief suggested a 0.2 s impulse duration. Isaac Lab's EventManager does not have
built-in duration tracking for interval events, so implementing a strict 0.2 s cutoff
would require either (a) a second paired event or (b) per-env state in the event
function — both add complexity without clear benefit for the training objective.

The chosen design applies the force at each interval trigger and leaves it active until
the next trigger (~10–15 s) or episode reset (where `base_external_force_torque` zeroes
it). This creates a **sustained lateral perturbation** model rather than a brief impulse.
This is arguably more challenging and more physically realistic (e.g., walking on a
tilted surface or a continuous wind load), and is standard in push-recovery literature.

If the author prefers true impulses (short push then zero), the simplest fix is to
replace `apply_push_force` with a version that tracks activation time and zeroes the
force after a configurable duration. Flag for decision before 4b.

---

## 4. Items requiring author review before full training (4b)

1. **Sustained vs. impulse push model**: see design note above. Confirm the sustained
   force design is acceptable, or request a true impulse implementation.

2. **Curriculum endpoint (100 N)**: physically this is ~6.7 m/s² lateral for a 15 kg
   robot. This is quite severe. If the author wants a less aggressive ceiling (e.g.,
   60–80 N), update `_FORCE_END_N` in `env_cfg.py` before 4b.

3. **Curriculum duration (800 PPO iter)**: this was taken from the brief. The full
   training target is 2000 iterations, so the push will be at max magnitude for the
   last 1200 iterations. Acceptable unless a longer ramp is preferred.

4. **No reward redesign**: as specified, rewards are identical to the baseline. If the
   policy consistently falls under strong pushes (expected in early 4b training), the
   author may want to add a small orientation penalty or survival bonus. Flag for
   after first 4b training inspection.

5. **`push_robot` (velocity impulse) remains disabled**: the baseline Go2 config sets
   `push_robot = None`. We keep this. The only disturbance is our force-based push.
   If both are wanted, uncomment `push_robot` in `env_cfg.py`.

---

## Mixed disturbance update (session 4a, second iteration)

### What changed

The single sustained-push event was replaced with two independent disturbance
channels with duration tracking:

**Impulse (`push_robot_impulse`)**: brief 0.15–0.25 s horizontal force every 6–10 s.  
**Sustained load (`push_robot_sustained`)**: 8–12 s lateral force every 25–40 s.

Both use a short `interval_range_s` as a *check clock* rather than the trigger clock:
the impulse event ticks every ~0.2 s (matching duration) so expired forces are zeroed
within one check window; the sustained event ticks every 1–2 s.  Actual trigger timing
is managed by per-env state tensors (`_next_trigger_step`) inside the event functions.

State is stored in module-level `_PushState` singletons (`_impulse_state`,
`_sustained_state`) and cleared on episode reset via a companion
`reset_push_event_states` reset-mode EventTerm.

Two curricula replace the single one:
- Impulse: 30 N → 120 N over iter 0–800 (same range, higher ceiling).
- Sustained: 10 N → 40 N, **delayed start at iter 200** (4 800 common steps).

Two diagnostic CurriculumTerms were added (`push_impulse_count`,
`push_sustained_count`) that return cumulative trigger sums per training interval,
visible in TensorBoard logs.

### Smoke test rerun outcome

Command (identical to original):
```bash
python scripts/train_pushrecovery.py --num_envs 1024 --max_iterations 50 --seed 42 --headless
```

Log directory: `logs/rsl_rl/unitree_go2_flat_pushrecovery/2026-06-02_19-00-23/`

| Metric | Value |
|--------|-------|
| Completed iterations | 50 / 50 |
| NaN rewards | None |
| Final mean reward (iter 49) | −3.61 |
| Mean episode length (iter 49) | ~987 steps |
| base_contact terminations (iter 49) | 0.04 (4 %) |
| Impulse `max_force` at iter 49 | 35.57 N (30 + 6.25% × 90) ✓ |
| Sustained `max_force` at iter 49 | 10.00 N (below start iter 200) ✓ |
| Throughput | ~52 000 steps/s |
| Wall time | ~25 s |

### Sanity-check evidence that both event types fired

```
Curriculum/push_impulse_count  (iter 1)  :   33.9
Curriculum/push_sustained_count (iter 1) :    0.0   ← expected, 0–6 s delay window active
Curriculum/push_impulse_count  (iter 3)  :  202.9
Curriculum/push_sustained_count (iter 3) :    9.5   ← sustained starts firing ~3 iterations in
Curriculum/push_impulse_count  (iter 49) : 2598
Curriculum/push_sustained_count (iter 49):  407
```

**Impulse**: ~2.5 triggers/env over 24 s of sim time (interval 6–10 s → 2–4 expected) ✓  
**Sustained**: ~0.4 triggers/env. First triggers appear around iter 3 (~3 s sim), consistent
with initial delays uniformly distributed in [0, 25 s]. Lower-than-expected count (~407 vs
~820 naive estimate) is explained by episode resets resetting `next_trigger_step` to
[25–40 s] delay; envs that reset before their first trigger don't fire again within the
25 s smoke-test window. The event is working correctly.

Both event types are confirmed active with magnitudes inside curriculum bounds.
