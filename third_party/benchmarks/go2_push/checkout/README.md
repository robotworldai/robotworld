# Isaac Lab — Go2 Locomotion

[![Isaac Sim 4.5](https://img.shields.io/badge/Isaac%20Sim-4.5.0-76B900)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab 2.1](https://img.shields.io/badge/Isaac%20Lab-2.1.1-76B900)](https://github.com/isaac-sim/IsaacLab)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Reinforcement learning baseline and disturbance-robust extension for Unitree Go2 quadruped locomotion, built on NVIDIA Isaac Sim 4.5 and Isaac Lab 2.1.**

> **Status.** Baseline velocity-tracking policy and push-recovery extension are both complete. The repository is being finalized with full reproducibility documentation and a comparative analysis writeup. A separate follow-up project on perception and navigation (Isaac ROS + Nav2) is planned.

---

## Democo

Side-by-side comparison of the two trained policies under **identical scheduled disturbances** (seven 120 N impulses across 25 seconds, applied at the same instant to both robots). The baseline policy was trained on undisturbed flat terrain; the push-recovery policy was trained with a mixed disturbance curriculum.

**Watch:** [Side-by-side video on the v0.2.0 release page](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/releases/tag/v0.2.0-pushrecovery).

Under identical perturbations:
- The **baseline policy** (left in the video) repeatedly loses balance and falls. It was never exposed to external forces during training.
- The **push-recovery policy** (right) absorbs the impulse, steps to recover, and continues tracking the velocity command.

Recovery rate under 120 N impulse loads: **87.5%** for the push-recovery policy, vs **0%** for the baseline at the same force.

## Project goal

Train a Unitree Go2 quadruped to follow velocity commands `[v_x, v_y, ω_z]` via reinforcement learning (PPO), then extend the baseline with a disturbance-rejection ("push recovery") behavior. The full pipeline runs in simulation on consumer-grade hardware (laptop with RTX 4050 6 GB).

This is a **portfolio project**, not research. The goal is to validate the full Isaac Lab pipeline end-to-end (training, logging, replay, evaluation) and to demonstrate working knowledge of:
- Sim-based RL for legged locomotion (PPO with `rsl_rl`)
- Reward design and curriculum learning in a manager-based RL environment
- Sim-to-real-ready policy formulation (proprioceptive observations only)
- Engineering discipline on resource-constrained hardware

## Project structure: two phases

### Phase 1 — Baseline (complete)

Standard task `Isaac-Velocity-Flat-Unitree-Go2-v0` trained across three sessions to identify the converged policy:

| Session | Iterations | num_envs | Wall time | Final reward | Notes |
|---|---|---|---|---|---|
| 1 | 500 | 2048 | ~7 min | 34.66 | Pipeline validation run |
| 2 | 2000 | 3072 | ~31 min | **37.99** | Official baseline (highest reward) |
| 3 | 4000 | 3072 | ~58 min | 37.26 | Convergence confirmation (no further gain) |

Session 2's `model_1999.pt` is the official baseline policy. Session 3 confirmed convergence at ~2000 iterations.

**Tracking error after baseline training** (averaged over replay with random commands):
- Linear velocity error: 0.137 m/s
- Angular velocity error: 0.272 rad/s
- Zero falls or body-contact terminations across full replays

Per-session metrics, hardware peaks, and anomalies are in `docs/RUN_NOTES.md`, `RUN_NOTES_session2.md`, `RUN_NOTES_session3.md`.

### Phase 2 — Push-recovery extension (complete)

A new task `Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0` extends the baseline with **mixed external disturbances** applied to the robot's base during training. The reward function, observation space, and PPO hyperparameters are **unchanged** from baseline — only event terms and curricula are added.

**Disturbance design:**

| Disturbance | Duration | Trigger interval | Magnitude curriculum |
|---|---|---|---|
| Impulse | 0.15-0.25 s | every 6-10 s | 30 N → 120 N over iter 0-800 |
| Sustained load | 8-12 s | every 25-40 s | 10 N → 40 N over iter 200-1000 |

Both disturbances are applied independently and can overlap. Each curriculum ramps linearly. Direction is uniform random in the horizontal XY plane.

**Training results** (session 4b):

- 2500 PPO iterations, 3072 envs, seed 42
- Wall time: 36 minutes
- Best checkpoint: `model_2350.pt` (reward 36.44)
- VRAM peak: 3831 MiB on a 6 GB GPU
- Throughput: ~86,000 environment steps/second

**Comparison to baseline:**

| Metric | Baseline | Push-recovery | Delta |
|---|---|---|---|
| Final reward | 37.99 | 36.44 | −4% (cost of robustness) |
| Linear velocity tracking error | 0.137 m/s | ~0.15 m/s (with disturbances active) | — |
| Recovery rate under 120 N impulse | ~0% | 87.5% | massive improvement |

The 4% reward delta is the price paid for disturbance robustness. The recovery-rate jump is the value gained. The trade-off is favorable.

Per-session details in `docs/RUN_NOTES_session4a.md` and `RUN_NOTES_session4b.md`.

### Out of scope (current project)

- Vision-based observations (RGB / depth / lidar) — laptop VRAM (6 GB) cannot sustain rendering-in-the-loop during PPO training
- Rough terrain training — possible future extension but distinct from push recovery
- Real-robot deployment — no physical Go2 available
- High-level navigation, SLAM, obstacle avoidance — planned as a separate follow-up project

## Stack

- **NVIDIA Isaac Sim 4.5.0** — physics simulation (PhysX 5) + USD scene management
- **NVIDIA Isaac Lab 2.1.1** — RL environment framework on top of Isaac Sim
- **rsl_rl 2.3.3** — PPO implementation (ETH Zurich)
- **PyTorch 2.7.0+cu128** — neural network backend
- **Python 3.10** in a Miniforge / `conda-forge` environment

## Hardware

Trained on a thin gaming laptop:
- NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM, 30 W TGP
- Intel Core i7-12650H, 16 GB RAM, 16 GB swap
- Ubuntu 22.04.5 LTS, kernel 6.8 (HWE), driver 550.163.01

Sustained training at 3072 parallel environments held VRAM at ~3.9 GB (64% utilization) and produced ~84,000-86,000 environment steps/second. No thermal throttling observed during 60-minute training runs.

This setup is significantly below the typical Isaac Lab recommendation (24+ GB VRAM workstation). The project documents the configuration adjustments needed to make sustained training viable on this hardware.

## Repository structure
```
isaaclab-go2-locomotion/
├── CLAUDE.md          # AI-assistant project context (for development sessions)
├── README.md          # This file
├── LICENSE            # MIT
├── docs/              # Per-session run notes and plots
│   ├── RUN_NOTES.md, RUN_NOTES_session{2,3,4a,4b}.md
│   └── images/        # Learning curves and comparison plots
├── src/
│   └── isaaclab_go2_pushrecovery/   # Custom task: env_cfg, push_events, curriculum
├── scripts/
│   ├── train_pushrecovery.py        # Training wrapper for push-recovery task
│   ├── play_pushrecovery.py         # Standard replay
│   ├── play_scheduled_pushes.py     # Single-robot replay with deterministic pushes (used for demo videos)
│   └── plot_session4b.py            # Plot generation from TFEvents
├── configs/           # YAML configs (currently empty)
├── notebooks/         # Analysis notebooks (currently empty)
└── logs/              # Training artifacts and videos (gitignored; videos in GitHub releases)
```

## Reproducibility

Both training runs used:
- Algorithm: PPO via `rsl_rl`, default hyperparameters from Isaac Lab 2.1.1
- Seed: 42
- num_envs: 3072

**Baseline reproduction:**
```bash
# From IsaacLab repo:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs=3072 --max_iterations=2000 --seed=42 --headless
```

**Push-recovery reproduction:**
```bash
# From this repo's root, with the custom task registered:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0 \
    --num_envs=3072 --max_iterations=2500 --seed=42 --headless
```

Detailed environment versions and exact commands per session are in `docs/RUN_NOTES*.md`.

## Author

**Brando Ulissi** — M.Sc. Automation Engineering, University of Bologna

This project is part of a personal effort to add hands-on reinforcement learning to a control-engineering profile.

LinkedIn: [brando-ulissi](https://www.linkedin.com/in/brando-ulissi)

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- NVIDIA Isaac Lab and Isaac Sim teams
- The `rsl_rl` authors at ETH Zurich Robotic Systems Lab
- The Unitree Go2 USD model shipped with Isaac Lab
