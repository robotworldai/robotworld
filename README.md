<div align="center">

# 🤖 RobotWorld

**English** | [简体中文](README.zh-CN.md)

### Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments

[![Project Page](https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge)](https://robotworldai.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2610.10409)
[![Code](https://img.shields.io/badge/Code-GitHub-2ea44f?style=for-the-badge)](https://github.com/robotworldai/robotworld)
[![Hugging Face Papers](https://img.shields.io/badge/Hugging_Face-Papers-ffd21e?style=for-the-badge)](https://huggingface.co/papers/2610.10409)
[![WeChat](https://img.shields.io/badge/WeChat-Community-07C160?style=for-the-badge&logo=wechat&logoColor=white)](#community)

<a href="https://robotworldai.github.io/#showreel">
  <img src="https://robotworldai.github.io/assets/images/showreel-poster.png" alt="Watch RobotWorld in 45 seconds" width="800">
</a>

🎬 **[▶ Watch RobotWorld in 45 seconds](https://robotworldai.github.io/#showreel)** · [中文版](https://robotworldai.github.io/zh/#showreel)

🦾 [Tasks & Recordings](https://robotworldai.github.io/#gallery) · 📊 [Results](https://robotworldai.github.io/#leaderboard) · 🧪 [Protocol](https://robotworldai.github.io/#protocol)

📝 **Analysis:** [English](https://robotworldai.github.io/blog/) · [中文](https://robotworldai.github.io/blog/zh/)

**84 tasks · 20 benchmark integrations · Manipulation, locomotion, driving & flight**

</div>

---

<a id="community"></a>

## 💬 Join the RobotWorld WeChat Group

交流 Robot Use、具身智能体、评测复现与任务扩展，欢迎研究者和开发者加入。也欢迎通过反馈、新任务或代码贡献，成为 RobotWorld 下一版本的贡献者。

<p align="center">
  <b>👇 微信扫码加入</b><br>
  <a href="docs/images/wechat-community.png"><img src="docs/images/wechat-community.png" alt="RobotWorld 微信群二维码" width="320"></a><br>
  二维码有效期至 2026 年 10 月 15 日。无法扫码？点击图片查看原图。
</p>

---

## ✨ What Is RobotWorld?

RobotWorld connects a source-built multimodal agent runtime to robot simulators through explicit observation and action interfaces. Agents can analyse observations, issue bounded robot commands, and use execution feedback to revise their actions. Task evaluators assess the resulting trajectories independently of the agent's completion claims.

This source snapshot registers **84 tasks across 20 benchmark integrations**. Registration does not imply that every task has a verified successful physical trajectory. The default batch protocol runs three rollouts per task with environment-side code control disabled. Simulation pauses during agent reasoning and workspace analysis.

## 🧭 Start Here

- **[Deployment guide](docs/DEPLOYMENT.md)** — host requirements, source restoration, assets, Docker images, runtime build, authentication, verification, and evaluation.
- **[Assets](docs/ASSETS.md)** — pinned downloads, upstream asset sources, and restricted datasets.
- **[Model configuration](docs/MODELS.md)** — API credentials and model configuration.
- **[Evaluation guide](scripts/ROLLOUTS.md)** — per-task budgets, control modes, repeated trials, resume, and summaries.
- **[Source and deployment limitations](docs/HANDOFF.md)** — what this snapshot can reproduce and which dependencies still need provisioning.

> **Prebuilt containers:** download the 20-image bundle from [RobotWorld Images](https://huggingface.co/datasets/visity/RobotWorld-Images). Access is currently restricted to authorised accounts. Follow the [download and import guide](docs/PREBUILT_IMAGES.md) before evaluation; some legacy task profiles need additional images.

## 🛠️ Installation

Run the simulator stack on Linux with an NVIDIA GPU, a compatible driver, Docker with Compose/Buildx and NVIDIA Container Toolkit, Git/Git LFS, Python 3.11 or later, Rust/rustup, and bubblewrap with user namespaces enabled. Simulator-specific hardware and software requirements also apply.

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' huggingface_hub
export WORLD_PYTHON="$PWD/.venv/bin/python"

# Restore independent source repositories at the locked revisions.
# Do this before restoring assets.
bash scripts/setup_handoff.sh sources

# Authenticate if the asset repository requires access.
hf auth login

# Download the pinned asset release, verify hashes, and restore paths.
bash scripts/setup_handoff.sh assets --bench robocasa

# Inspect the build plan, then build this benchmark's images.
bash scripts/setup_handoff.sh docker --bench robocasa --dry-run
bash scripts/setup_handoff.sh docker --bench robocasa

# Build the local app-server, login CLI, and code-mode helper.
bash scripts/setup_handoff.sh codex
```

For all integrations, omit `--bench` from the asset and image commands after provisioning the required base images. BEHAVIOR assets are downloaded separately after the user accepts the upstream licence. The [full deployment guide](docs/DEPLOYMENT.md) covers these dependencies and verification steps.

Use the [prebuilt image bundle](docs/PREBUILT_IMAGES.md) instead of the Docker build commands if the required tags are already available. Existing installations can be reused after checking the pinned source revisions, asset hashes, and image tags. Do not replace them with newer versions silently.

### Direct-tool runtime option

If `codex-code-mode-host` cannot be built (for example, an unavailable upstream V8 archive), use direct tool calling for the isolated agent runtime:

```bash
export WORLD_CODEX_DISABLE_CODE_MODE=1
bash scripts/setup_handoff.sh codex
```

Keep this variable set when running the evaluation. The relay selects direct tool exposure in a private model catalogue without changing the model ID or Codex source, and records `runtime-tool-mode.json`. This was used for the RoboCasa API smoke test. It does **not** enable environment-side `code_control`; action tools, observations, budgets and scoring remain unchanged. Record this interaction setting when comparing results. Other runtime entry points require separate verification.

## 🔌 Connect Your Model API

After deploying the simulator and building the agent runtime, configure your model API. The endpoint must accept streaming **Responses API** requests with images and function calls; a Chat Completions-only endpoint is not sufficient. The examples assume the endpoint is reachable from the runtime.

```bash
python scripts/configure_api.py \
  --base-url https://YOUR_API_HOST/v1 \
  --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"
# Enter the key without echoing it or putting it in shell history.
read -rsp "Model API key: " WORLD_MODEL_API_KEY; printf '\n'
export WORLD_MODEL_API_KEY
python scripts/check_api.py
```

The configuration stores the name of the credential environment variable, never its value. The API check makes two small model requests to verify image input and a tool-result round trip; normal provider charges apply. It does not start a simulator.

Evaluation inherits the configured model. Set `MODEL` or pass `--model` only when intentionally overriding it. Requests use the configured provider without model-name-based substitutions. See [API configuration](docs/MODELS.md).

## 🚀 Run an Evaluation

Run commands from the repository root with `WORLD_PYTHON` and `CODEX_AUTH_HOME` set as above.

```bash
# Inspect tasks, effective budgets, scoring profiles, and control settings.
bash scripts/run_robocasa.sh list

# Inspect the complete campaign without calling a model or simulator.
ROLLOUTS=1 bash scripts/run_all.sh run --dry-run

# Start with one task and one rollout.
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --batch drawer-smoke

# Run all selected RoboCasa tasks, three times each.
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01

# Override a task budget explicitly.
bash scripts/run_robocasa.sh run --tasks CloseDrawer \
  --task-steps CloseDrawer=450 --rollouts 3 --batch drawer-01

# Enable environment-side code control for a supported BEHAVIOR task.
bash scripts/run_behavior_1k.sh run --tasks clean_up_your_desk \
  --task-code-control clean_up_your_desk=on --rollouts 2 --batch desk-code-01

# Run selected benchmarks sequentially.
BENCHMARKS='robocasa robolab' ROLLOUTS=2 bash scripts/run_all.sh run --batch subset-01

# Run the full registered suite after provisioning all dependencies.
ROLLOUTS=3 bash scripts/run_all.sh run --batch model-a-full

# Retry unfinished task/rollout slots with the same settings and batch ID.
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01 --resume
```

`--resume` restarts unfinished rollouts from their initial state and preserves previous attempts. It does not restore an interrupted physics state. Use a new batch ID when changing the model, budgets, assets, or tool conditions.

### Evaluation settings

| Setting | Meaning |
| --- | --- |
| `TASK_STEPS` / `--task-steps` | Per-task `native`, `profile`, or a positive step limit within the known scenario limit. |
| `CODE_CONTROL` | Benchmark default for environment-side generated control programs; initially `off`. |
| `TASK_CODE_CONTROL` / `--task-code-control` | Per-task `on`, `off`, or `default`; explicit values override the benchmark default. |
| `ROLLOUTS` / `--rollouts` | Trials **per task**; default: 3. |
| `MODEL` / `CODEX_AUTH_HOME` | Model ID and authentication/provider directory. |
| `SCORING_PROFILE` | `auto` uses the registered World rules for adapted scenarios and native rules elsewhere; `native` selects original objectives, budgets, and scoring. |
| `--batch` | Experiment identifier. |
| `--output-root` | Output location; defaults to `outputs/`. |
| `--timeout` | Wall-clock limit per run; default: 43,200 seconds, separate from simulation steps. |
| `--seed` / `--seed-stride` | Seed configuration where the integration supports randomisation. |

Later CLI arguments override the corresponding script settings. RoboDojo, RoboCasa, and RoboLab do not expose environment-side code control in this snapshot; requesting it does not add that capability, and the effective setting is recorded as false. Disabling code control still allows offline computation on permitted observations.

BEHAVIOR uses `min(native_budget, 2000)`. `CountertopCleanup` has an explicit 600-step World budget because its official horizon is unverified in the pinned source. Other task budgets are recorded in the catalogue. Native termination conditions remain active.

## 📊 Outputs and Metrics

```text
outputs/<benchmark>/<task>/run-<batch>-0001/
  run.json                 # Settings, effective budgets, status, success, and raw score
  videos/                  # Relative links to MP4 files
  events/                  # Agent, tool, and environment events; image-free copies
  programs/                # Index of generated control programs
  agent-workspace/         # Agent files, where supported by the integration
  artifacts/               # Original benchmark output; retain with the links above
outputs/<benchmark>/summaries/<batch>.json
outputs/<benchmark>/summaries/<batch>.csv
outputs/<benchmark>/summaries/<batch>-runs.csv
```

Per-task success rate is the mean of valid Boolean outcomes. Benchmark summaries average task success rates with equal task weights and also provide pooled rollout statistics. Population and sample variance are reported separately; sample variance is undefined for a single observation. Raw scores with different units are retained per task.

A valid failure is `false`. Infrastructure errors and incomplete or indeterminate outcomes are unscored (`null`), with coverage reported explicitly. Incomplete coverage does not constitute a complete benchmark score. Copy the whole run directory when archiving results so that relative artifact links remain usable.

## 🗂️ Repository Layout

| Path | Contents |
| --- | --- |
| `environment/` | Adapters, observation/action interfaces, runtime, scoring, and validation. |
| `scripts/` | Setup, evaluation, asset preparation, and reporting commands. |
| `docs/` | Deployment and protocol documentation, task specifications, and historical audits. |
| `codex/` | Bundled agent-runtime source; the independent checkout is restored from `sources.lock.json`. |
| `third_party/` | Source snapshots, integration wrappers, Docker recipes, and upstream licences. |
| `sources.lock.json` | Pinned source repositories and revisions. |
| `Assets/` | Downloaded assets; created during installation. |
| `var/`, `outputs/` | Local builds, authentication, caches, and run artifacts; excluded from source export. |

## 📋 Release Checks

See [release review](docs/RELEASE_REVIEW.md) for the source/privacy scan and checks performed on this snapshot. The current full test suite is not entirely passing; API-backed simulator validation is pending.

## 🔎 Validation and Provenance

The source for this release is `visitworld123/RoboWorld_Scaffold`, branch `main`, commit `132972eec7ea4e9a5b59f8be7398f19d7681fc52`. Deployment instructions describe that snapshot. Historical reports and task notes remain in the repository; their recorded results are not new release-validation results.

Run the CPU tests with `python -m pytest`. Task-list and Docker-plan checks do not verify physical task feasibility. GPU simulation, model connectivity, and successful trajectories require separate checks. Experimental Isaac compatibility layers are identified in run metadata and should not be treated as proof of equivalence to the original simulator version.

Third-party code and assets retain their respective licences and notices. BEHAVIOR assets and decryption keys are not redistributed. This snapshot has no top-level licence for RobotWorld-owned code; the maintainers need to select one before presenting it as a fully licensed open-source release.

## 📚 Citation

If you use RobotWorld in your research, please cite:

```bibtex
@misc{yang2026robotworld,
  title  = {{RobotWorld}: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments},
  author = {Yang, Zhiqin and Li, Chenxin and Hu, Xiaomeng and Liu, Yibin and Huang, Weidong and Sun, Jiankai and Li, Haitao and Wu, Zijian and Huang, Yuzhi and Huang, Fanding and Sun, Hanwen and Liu, Jiashun and Tong, Jingqi and Huang, Mingxin and Hu, Shaoli and Huang, Shijue and Bai, Tianyi and Wang, Xinyuan and Lin, Yunlong and Tang, Zhengyang and Zhang, Zhexin and Chen, Zhuo and Song, Xierui and Dai, Juntao and Chen, Boyuan and Ji, Jiaming and Zhan, Fangneng and Hu, Mengkang and Xue, Wei and Zhang, Yonggang and Hu, Han and Ho, Tsung-Yi and Guo, Yike},
  year   = {2026},
  url    = {https://github.com/robotworldai/robotworld}
}
```
