# Project Context — isaaclab-go2-locomotion

## Project goal

Build a reinforcement learning portfolio project: train a Unitree Go2 quadruped
to perform velocity tracking locomotion using PPO on NVIDIA Isaac Lab.
Final deliverable is a public GitHub repo with code, plots, video demo,
and professional README, targeted at robotics/RL roles.

## Author technical background

M.Sc. Automation Engineering, University of Bologna. Strong foundations in
classical control (MPC, LQR, sliding mode, geometric SO(3) control), ROS2,
Python, MATLAB/Simulink. Theoretical RL exposure but no prior hands-on
experience with Isaac Lab, Isaac Sim, MuJoCo, or RL libraries.
C++ reading proficiency, not production-level.

## Hardware constraints (important!)

- NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM (small)
- 16 GB RAM, 16 GB swap
- Single GPU, no distributed training
- VRAM-constrained: keep num_envs ≤ 2048 for quadruped training
- Always train headless (no real-time rendering during training)
- Vision-based RL (rendering during training) is not viable on this hardware

## Environment

- Ubuntu 22.04.5 LTS, kernel 6.8 HWE
- Conda environment: `isaaclab_env` (Miniforge), Python 3.10.20
- Isaac Sim 4.5.0, Isaac Lab 2.1.1 (at ~/IsaacLab/, installed in editable mode)
- PyTorch 2.7.0+cu128 (NVIDIA forward-compatibility with driver 550 works)
- rsl_rl 2.3.3 as RL library

To activate: `conda activate isaaclab_env`

## Project conventions

- Code and comments in English (this is a public portfolio repo)
- Chat with the author can be in Italian, but written deliverables are English
- Honest engineering tone in README: no overclaim, no LinkedIn-style hype
- Iterative approach: small validated steps, not big leaps
- Reward design: start from the standard Isaac Lab velocity locomotion reward,
  modify only with clear motivation
- Hyperparameters: stay close to rsl_rl defaults until baseline works,
  then tune with intent

## Out of scope

- UAV/drone applications (author has prior DTU NDA on this topic)
- Real robot deployment (no physical Go2 available)
- Multi-GPU or distributed training
- Vision-based observations (camera RGB/depth)
- Manipulation tasks (this is a locomotion-focused portfolio)

## Working pattern

The author works on the "thinking and review" side via Anthropic chat (claude.ai),
and uses Claude Code in VS Code for hands-on coding execution. When in doubt
about a non-trivial architectural decision (task formulation, reward shaping
strategy, training hyperparameter tuning beyond defaults), prefer to flag it
back to the author for input rather than make silent choices.

## File organization

- `src/` — Python source files for custom tasks, wrappers, utilities
- `scripts/` — executable training, play, evaluation scripts
- `configs/` — YAML configuration files
- `logs/` — training output (gitignored)
- `docs/images/` — screenshots, plots
- `notebooks/` — analysis notebooks
