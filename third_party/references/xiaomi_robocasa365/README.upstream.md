# RoboCasa365 Evaluation

This guide walks through evaluating **Xiaomi-Robotics-1** on the [RoboCasa365](https://robocasa.ai/) benchmark: setting up the simulation environment, launching the inference server, running the evaluation client, and reading the results.

> **Architecture.** Evaluation uses a client–server split: the **server** loads the model and serves actions over a socket; the **client** runs the RoboCasa365 simulator, builds inputs with the Hugging Face `AutoProcessor`, and decodes returned actions. They run in **two separate conda environments**.

---

## Prerequisites

### 1. Deploy environment (`mibot`)

The server runs here. Install it once by following **[docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md)**.

### 2. RoboCasa365 simulation environment (`robocasa_365`)

For the installation of the RoboCasa365 environment, please refer to the [official RoboCasa installation guide](https://robocasa.ai/docs/build/html/introduction/installation.html). Then install the XR-1 evaluation client dependencies. **transformers must be exactly `4.57.1`**; other versions are unverified:

```bash
pip install transformers==4.57.1 imageio[ffmpeg] tqdm scipy
```

For headless rendering, install the EGL/OpenGL runtime libraries provided by your operating system, and set `MUJOCO_GL=egl`.

### 3. Checkpoint

Download the fine-tuned checkpoint from the [RoboCasa365 collection](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365). The same model directory must be readable by both the server and the client:

```bash
export MODEL_PATH="$PWD/checkpoints/Xiaomi-Robotics-1-RoboCasa365"
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 --local-dir "$MODEL_PATH"
```

---

## Evaluation

### Step 1 — Start the inference server

In a terminal with the **deploy** environment, from the repository root:

```bash
conda activate mibot
bash scripts/deploy.sh "$MODEL_PATH" 8 8   # <model_path> <num_ports> <num_gpus>
```

This starts 8 servers on ports `10086`–`10093` inside the `model_servers` tmux session. Inspect with `tmux attach -t model_servers`. **Start the server before launching the client.**

### Step 2 — Run the evaluation client

In another terminal. The launcher activates the **RoboCasa365** environment itself from the `CONDA_ENV` variable, so you do not need to `conda activate` beforehand:

```bash
CONDA_ENV=robocasa_365 \
    bash scripts/launch_robocasa365.sh \
    8 eval_results/robocasa365 "$MODEL_PATH"
```

| Arg | Meaning |
| :--- | :--- |
| `8` | Number of client workers (must match the number of servers) |
| `eval_results/robocasa365` | Output directory for results |
| `"$MODEL_PATH"` | Checkpoint path (used by the client to load the processor) |

Workers connect to ports `10086`–`10093` by default. Each worker pulls from a shared rollout queue and atomically claims the next rollout after finishing its current one — rollouts are **not** statically assigned to GPUs. Workers preserve the launcher's CUDA visibility and leave `MUJOCO_EGL_DEVICE_ID` unset, matching the reference runtime.

**Smoke test** (single task, one trial):

```bash
CONDA_ENV=robocasa_365 NUM_TRIALS=1 \
    bash scripts/launch_robocasa365.sh \
    1 eval_results/robocasa365-smoke "$MODEL_PATH" \
    --task-name CloseBlenderLid --horizon 20
```

**Configuration knobs.** Override machine-specific settings with `BASE_PORT`, `SERVER_ADDR`, `PYTHON`, or `CONDA_ENV`, and the evaluator with `SPLIT`, `TASK_SET`, `NUM_TRIALS`, `REPLAN_STEPS`, `OBS_HISTORY`, `OBS_INTERVAL`, `SEED`, and `CROP_RATIO`.

---

## Results

For a run ID `<run-id>`, outputs are written as:

```text
eval_results/robocasa365/<run-id>/
  summary.json
  <TaskName>/
    stats.json
    episode_<episode>_seed_<seed>_<success|failure>.mp4

eval_results/robocasa365/scheduler/<run-id>/
  manifest.json  pending/  running/  results/  errors/  logs/
```

The final merge validates global episode indices (missing / duplicate / inconsistent) before writing the summary. A failed worker leaves an error record and causes the launcher to exit non-zero.

**Reference configuration.** split `pretrain`, task set `target50`, 50 trials per task, observation history 4, observation interval 2, 16 actions per query, environment seed 7, and crop ratio 0.95. It produced 1432 successes over 2500 episodes, for a 57.28% episode success rate.
