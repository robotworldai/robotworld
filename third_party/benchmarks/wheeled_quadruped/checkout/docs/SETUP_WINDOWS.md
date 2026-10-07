# Windows bring-up runbook

This is the concrete, machine-specific procedure to get **Isaac Sim 5.1 + Isaac Lab 2.3.2** running on this Windows workstation and train the wheeled-quadruped tasks. It is a runbook, not a tutorial — follow the steps in order.

> Companion docs: [README.md](../README.md) for the project overview, [TRAINING.md](TRAINING.md) for the RL design.

## Current state of this machine

| Item | Status |
|---|---|
| Isaac Sim | **Not installed** |
| Isaac Lab | **Not installed** |
| Windows Python | **Not installed** (no system `python`/`py` on PATH yet) |
| GPU | **RTX 4090 Laptop, 16 GB VRAM** |
| GPU driver | **591.44** — new enough for Isaac Sim 5.1 |
| Free disk on `C:` | **~18 GB** — **not enough**; a full install needs ~60–80 GB |

Two facts drive the whole procedure:

1. **Disk is the blocker.** ~18 GB free vs ~60–80 GB needed. Freeing space is **step 0** and is non-negotiable.
2. **WSL2 does not help.** Isaac Sim's GPU stack is unsupported on WSL2, *and* the WSL virtual disk (`ext4.vhdx`) lives on `C:` anyway — so moving into WSL neither runs Isaac Sim nor escapes the `C:` disk pressure. Install **Windows-native**.

## Step 0 — Free disk space (mandatory)

You need roughly **60+ GB free on `C:`** before installing. Safe, high-yield cleanups (in rough order of payoff):

- **Compact the WSL virtual disk** if you use WSL — this has reclaimed ~90 GB here before. Shut WSL down first (`wsl --shutdown`), then compact `ext4.vhdx` via `diskpart` / `Optimize-VHD`, or `wsl --manage <distro> --set-sparse true`.
- **`conda clean --all`** — removes unused conda packages, tarballs, and caches.
- **`pip cache purge`** — clears the pip wheel cache.
- Empty the Recycle Bin; run **Disk Cleanup** / Storage Sense on Windows temp files.

> [!CAUTION]
> **Do not delete anything under `…\Temp\claude\**`.** That directory is off-limits on this machine. Skip it in every cleanup pass.

Confirm you have the headroom before continuing:

```powershell
Get-PSDrive C | Select-Object Used, Free
```

## Step 1 — Install Python 3.11

Isaac Sim 5.1 / Isaac Lab 2.3.2 target **Python 3.11**. Install the official python.org 3.11 build (**per-user**, no admin needed):

- Download the "Windows installer (64-bit)" for the latest 3.11.x from [python.org](https://www.python.org/downloads/windows/).
- In the installer, check **"Add python.exe to PATH"** and choose **"Install for me only"**.

Verify:

```powershell
python --version   # should print Python 3.11.x
```

## Step 2 — Enable Windows long paths

Isaac Sim's dependency tree contains paths longer than the legacy 260-character limit. Enable long paths (admin PowerShell, one-time):

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
```

A reboot (or at least a new shell) is recommended so the setting takes effect for the install.

## Step 3 — Create a virtual environment

From the repo root (`wheeled_quadruped_robot`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `Activate.ps1` is blocked, allow scripts for this session:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

## Step 4 — Install Isaac Sim, PyTorch, Isaac Lab, and this package

Run these **in order**, inside the activated venv:

```powershell
# Isaac Sim 5.1
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# PyTorch 2.7.0, CUDA 12.8 build (must match Isaac Sim's CUDA)
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab 2.3.2 (pulls in rsl-rl >= 3.0, skrl, sb3, rl-games)
pip install "isaaclab[all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# This project's task package (editable)
pip install -e source\wheeled_quadruped
```

> [!NOTE]
> **First run is slow.** The very first time you import Isaac Sim / launch any script, Omniverse pulls and compiles its extension registry and shader cache. Expect the process to sit for **10+ minutes** with little output before the first window or log line appears. This is normal; do not kill it. Subsequent launches are fast.

## Step 5 — Acceptance gate: verify the environment

Before training anything, confirm the environment loads and steps cleanly:

```powershell
python scripts\verify_env.py --num_envs 8
```

This boots the simulator headless, builds `Wheeled-Quadruped-Balance-v0`, prints the robot's joints/bodies and per-observation-group shapes, runs a short random rollout, checks every observation and reward tensor for NaN/Inf, and prints **PASS/FAIL**. Do not proceed to training until this prints PASS.

Sanity checks to eyeball in its output:

- `num_joints = 4` (2 rear wheels + 2 front thighs), `robot` present in the scene.
- Policy observation width **16**, critic observation width **20**.
- No NaN/Inf reported; resets are observed; mean reward is finite.

## Step 6 — Train

On the **16 GB** laptop GPU, **start at `--num_envs 2048`.** 4096 is the config default and the number the rl_games/sb3 YAMLs assume, but it may not fit in 16 GB alongside rendering; move up to 4096 only after confirming VRAM headroom with `nvidia-smi`.

```powershell
# Balance policy, headless
python scripts\rsl_rl\train.py --task Wheeled-Quadruped-Balance-v0 --headless --num_envs 2048

# Once VRAM is confirmed to have room:
python scripts\rsl_rl\train.py --task Wheeled-Quadruped-Balance-v0 --headless --num_envs 4096

# Velocity policy
python scripts\rsl_rl\train.py --task Wheeled-Quadruped-Velocity-v0 --headless --num_envs 2048
```

Watch a trained checkpoint (loads the latest run by default):

```powershell
python scripts\rsl_rl\play.py --task Wheeled-Quadruped-Balance-Play-v0 --num_envs 32
```

## Checkpoints, logs, and TensorBoard

- Runs are written to **`logs\rsl_rl\<experiment_name>\<timestamp>\`**, where `<experiment_name>` is `wheeled_quadruped_balance` or `wheeled_quadruped_velocity`.
- Checkpoints are `model_<iteration>.pt`; `save_interval = 100`.
- Launch TensorBoard against the experiment folder:

  ```powershell
  tensorboard --logdir logs\rsl_rl
  ```

To resume, pass `--resume` (optionally with `--load_run <timestamp>` and `--checkpoint <n>`) to `train.py`.

## rl_games / SB3 batch-size coupling

The rl_games and Stable-Baselines3 YAML configs are written for **`num_envs = 4096`**:

- **rl_games** (`rl_games_ppo_cfg.yaml`): `horizon_length = 16`, `minibatch_size = 8192`. Rollout = `num_actors × horizon_length`. With 4096 actors that is 65536, and 8192 divides it (8 minibatches).
- **SB3** (`sb3_ppo_cfg.yaml`): `n_steps = 16`, `batch_size = 4096`. Rollout = `num_envs × n_steps = 65536`, which 4096 divides.

If you train those two frameworks with a **different** `--num_envs`, adjust `minibatch_size` / `batch_size` so it still evenly divides `num_envs × (horizon_length | n_steps)`, or training will error out or silently mis-batch. The **rsl_rl** config is not affected (it uses `num_mini_batches = 4`, a divisor, so it scales with any `num_envs`).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Install fails with a path/filename-too-long error | Long paths not enabled — redo **Step 2**, open a fresh shell |
| `No module named 'isaaclab'` when running scripts | venv not activated, or Isaac Lab install (Step 4) did not complete |
| `No module named 'wheeled_quadruped'` | `pip install -e source\wheeled_quadruped` not run in the active venv |
| First launch appears frozen for minutes | Expected — Omniverse extension-registry pull (**Step 4** note) |
| Out-of-memory / VRAM error at 4096 envs | Drop to `--num_envs 2048`; keep `--headless` |
| rsl-rl import/version error in train.py | Isaac Lab 2.3.2 needs rsl-rl-lib ≥ 3.0; upgrade it |
| Disk fills mid-install | Return to **Step 0**; the extension cache alone is many GB |
