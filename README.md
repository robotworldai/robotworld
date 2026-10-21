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
  <a href="docs/images/wechat-community.png"><img src="docs/images/wechat-community.png" alt="RobotWorld 微信群二维码" width="200"></a><br>
  二维码有效期至 2026 年 10 月 24 日。无法扫码？点击图片查看原图。
</p>

---

## ✨ What Is RobotWorld?

RobotWorld connects a source-built multimodal agent runtime to robot simulators through explicit observation and action interfaces. Agents can analyse observations, issue bounded robot commands, and use execution feedback to revise their actions. Task evaluators assess the resulting trajectories independently of the agent's completion claims.

This source snapshot registers **84 tasks across 20 benchmark integrations**. Registration does not imply that every task has a verified successful physical trajectory. The default batch protocol runs three rollouts per task with environment-side code control disabled. Simulation pauses during agent reasoning and workspace analysis.

## Quick Start

Start with one RoboCasa task before provisioning the full suite. You need a Linux/NVIDIA host, Docker with NVIDIA Container Toolkit, Python 3.11+, and the pinned build dependencies. **L40S 48GB is the current host reference, not an all-task compatibility guarantee.** See the [deployment guide](docs/DEPLOYMENT.md) for prerequisites and troubleshooting.

<details>
<summary>GPU compatibility and host dependencies</summary>

### Hardware and dependencies

- **Current host reference:** NVIDIA L40S 48GB (driver `580.105.08`); this describes the deployment host, not certification of all 84 tasks.
- **Other GPUs:** RTX 5090 has historical partial-task validation with experimental Isaac Sim 6.0.1 profiles; older Isaac/CUDA stacks can be incompatible. RTX 4090, RTX A6000, RTX 6000 Ada and L40 are RT-capable candidates, not a verified support list.
- **Isaac rendering:** use a GPU with RT Cores. A100/H100 are [not supported by Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html), although they can host a separate model inference service. With a remote model API, local GPUs primarily run simulation and rendering.
- **Compatibility:** match the task's pinned image, driver and CUDA/Isaac versions, then run a smoke episode before scaling. No suite-wide minimum VRAM, RAM or disk requirement has been established.

Host dependencies: Linux, a compatible NVIDIA driver, Docker with Compose/Buildx and NVIDIA Container Toolkit, Git/Git LFS, Python 3.11+, the pinned Rust toolchain, C/C++ build tools (CMake, Clang and OpenSSL headers), bubblewrap with user namespaces, FFmpeg/ffprobe and zstd. Provision the pinned simulator images and assets, including any required access or licences; see the [deployment guide](docs/DEPLOYMENT.md).

</details>

### 1. Prepare one environment

Run these commands from the repository root after cloning. Restore pinned sources **before** assets.

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' huggingface_hub
export WORLD_PYTHON="$PWD/.venv/bin/python"

bash scripts/setup_handoff.sh sources
# Authenticate with your own account if asset access requires it.
hf auth login
bash scripts/setup_handoff.sh assets --bench robocasa
bash scripts/setup_handoff.sh docker --bench robocasa --dry-run
bash scripts/setup_handoff.sh docker --bench robocasa
bash scripts/setup_handoff.sh codex
```

You can [import prebuilt images](docs/PREBUILT_IMAGES.md) instead of building them; access currently requires authorisation, and some legacy profiles need additional images. For runtime helper build failures, see the [direct-tool option](docs/DEPLOYMENT.md#direct-tool-runtime-option). Other environments may require additional assets, base images and licences.

### 2. Connect a model and run one task

Use a reachable streaming **Responses API** endpoint with image and function-call support. A Chat Completions-only endpoint is not sufficient.

```bash
python scripts/configure_api.py \
  --base-url https://YOUR_API_HOST/v1 \
  --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"
# Read the key without echoing it or saving it in shell history.
read -rsp "Model API key: " WORLD_MODEL_API_KEY; printf '\n'
export WORLD_MODEL_API_KEY
python scripts/check_api.py

bash scripts/run_robocasa.sh list
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --dry-run
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --batch drawer-smoke
```

Replace the endpoint and model placeholders with your provider's values. The API check makes two small paid model requests, but does not start a simulator. Configuration stores the credential variable's name, not its value. In a new shell, reactivate the virtual environment and restore the environment variables.

### 3. Inspect results, then scale up

Inspect `run.json`, videos and events under `outputs/robocasa/CloseDrawer/`. A valid task failure is `false`; infrastructure or indeterminate outcomes are unscored (`null`). Preserve the whole run directory, including `artifacts/`, when archiving.

```bash
# Run all RoboCasa tasks, three trials per task.
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01
# Retry unfinished slots with the same settings.
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01 --resume
```

**Current defaults:** 3 rollouts per task, environment-side `code_control=off`, per-task control-step budgets, and `--timeout=43200` (12 hours). Simulation pauses during reasoning; offline analysis is still allowed. Native termination conditions remain active. `--resume` starts unfinished rollouts from the initial state, not a saved physics state. Use a new batch ID when changing evaluation conditions.

## Documentation

| Guide | Contents |
| --- | --- |
| [Deployment](docs/DEPLOYMENT.md) | Host setup, sources, runtime builds and troubleshooting |
| [Assets](docs/ASSETS.md) / [Prebuilt images](docs/PREBUILT_IMAGES.md) | Pinned downloads, access and licences |
| [Model API](docs/MODELS.md) | Providers, credentials and model configuration |
| [Evaluation](scripts/ROLLOUTS.md) | Task budgets, control modes, full-suite runs, resume and metrics |
| [Repository layout](DIRECTORY_GUIDE.md) | Source and output directory map |
| [Release boundaries](docs/HANDOFF.md) / [Checks](docs/RELEASE_REVIEW.md) | Reproducibility limits and recorded validation |

## Release Status

The release documentation records outstanding full-suite test failures and pending API-backed simulator validation. Historical trajectories are not a fresh validation of this source release; validate your chosen environment before scaling.

Third-party code and assets retain their own licences. BEHAVIOR assets and decryption keys are not redistributed. **A top-level licence for RobotWorld-owned code has not yet been selected.**

<details>
<summary>Source provenance and validation scope</summary>

The source for this release is `visitworld123/RoboWorld_Scaffold`, branch `main`, commit `132972eec7ea4e9a5b59f8be7398f19d7681fc52`. Run CPU tests with `python -m pytest`. Task lists, dry runs and API checks do not replace GPU rollout validation. Experimental Isaac compatibility layers are recorded in run metadata and do not establish equivalence to the original simulator.

</details>

## 📚 Citation

If you use RobotWorld in your research, please cite:

```bibtex
@misc{yang2026robotworld,
  title  = {{RobotWorld}: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments},
  author = {Yang, Zhiqin and Li, Chenxin and Hu, Xiaomeng and Liu, Yibin and Huang, Weidong and Sun, Jiankai and Li, Haitao and Wu, Zijian and Huang, Yuzhi and Huang, Fanding and Sun, Hanwen and Liu, Jiashun and Tong, Jingqi and Huang, Mingxin and Hu, Shaoli and Huang, Shijue and Bai, Tianyi and Wang, Xinyuan and Lin, Yunlong and Tang, Zhengyang and Zhang, Zhexin and Chen, Zhuo and Song, Xierui and Dai, Juntao and Chen, Boyuan and Ji, Jiaming and Zhan, Fangneng and Hu, Mengkang and Xue, Wei and Zhang, Yonggang and Hu, Han and Ho, Tsung-Yi and Guo, Yike},
  year   = {2026},
  eprint = {2610.10409},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2610.10409}
}
```
