<div align="center">

# 🤖 RobotWorld

[English](README.md) | **简体中文**

### 面向多种任务与机器人形态的通用多模态智能体评测

[![项目主页](https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge)](https://robotworldai.github.io/zh/)
[![论文](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2610.10409)
[![代码](https://img.shields.io/badge/Code-GitHub-2ea44f?style=for-the-badge)](https://github.com/robotworldai/robotworld)
[![Hugging Face Papers](https://img.shields.io/badge/Hugging_Face-Papers-ffd21e?style=for-the-badge)](https://huggingface.co/papers/2610.10409)
[![WeChat](https://img.shields.io/badge/WeChat-Community-07C160?style=for-the-badge&logo=wechat&logoColor=white)](#community)

<a href="https://robotworldai.github.io/zh/#showreel">
  <img src="https://robotworldai.github.io/assets/images/showreel-poster.png" alt="点击观看：45 秒了解 RobotWorld" width="800">
</a>

🎬 **[▶ 45 秒了解 RobotWorld](https://robotworldai.github.io/zh/#showreel)** · [English](https://robotworldai.github.io/#showreel)

🦾 [任务与录像](https://robotworldai.github.io/zh/#gallery) · 📊 [评测结果](https://robotworldai.github.io/zh/#leaderboard) · 🧪 [评测协议](https://robotworldai.github.io/zh/#protocol)

📝 **分析文章：**[中文](https://robotworldai.github.io/blog/zh/) · [英文](https://robotworldai.github.io/blog/)

**84 项任务 · 20 个 benchmark 集成 · 操作、运动、驾驶与飞行**

</div>

---

<a id="community"></a>

## 💬 加入 RobotWorld 微信群

交流 Robot Use、具身智能体、评测复现与任务扩展，欢迎研究者和开发者加入。也欢迎通过反馈、新任务或代码贡献，成为 RobotWorld 下一版本的贡献者。

<p align="center">
  <b>👇 微信扫码加入</b><br>
  <a href="docs/images/wechat-community.png"><img src="docs/images/wechat-community.png" alt="RobotWorld 微信群二维码" width="200"></a><br>
  二维码有效期至 2026 年 10 月 24 日。无法扫码？点击图片查看原图。
</p>

---

## ✨ RobotWorld 是什么？

RobotWorld 将从本仓库源码构建的智能体运行时接入机器人模拟器。模型通过明确的观测与动作工具控制机器人，并根据执行反馈修正动作；任务是否完成由环境判据决定，而不是由模型的文字声明决定。

当前发布版注册了 **20 个 benchmark、84 道题**。注册不代表所有任务都已有经过验证的成功轨迹。默认每道题执行 3 次 rollout，关闭环境内 `code_control`；模型推理和离线分析期间仿真暂停。

## 快速开始

建议先跑通一道 RoboCasa 任务，再准备完整任务集。需要 Linux/NVIDIA 主机、Docker 与 NVIDIA Container Toolkit、Python 3.11+ 及固定版本的构建依赖。**L40S 48GB 是当前主机参考，不代表全部任务的兼容性保证。** 完整准备流程见[部署指南](docs/DEPLOYMENT.md)。

<details>
<summary>显卡兼容性与主机依赖</summary>

### 硬件与依赖

- **当前主机参考：** NVIDIA L40S 48GB（驱动 `580.105.08`）；这是部署主机配置，不代表全部 84 题均已在该配置上验证。
- **其他显卡：** RTX 5090 在 Isaac Sim 6.0.1 实验兼容配置下有部分任务历史验证，旧版 Isaac/CUDA 可能不兼容。RTX 4090、RTX A6000、RTX 6000 Ada 和 L40 具备 RT Core，可作为候选，但不是已验证支持名单。
- **Isaac 渲染：** 需要具备 RT Core 的显卡；A100/H100 [不受 Isaac Sim 支持](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)，但可用于独立的模型推理服务。使用远程模型 API 时，本机 GPU 主要承担仿真和渲染。
- **兼容性：** 按任务匹配固定镜像、驱动及 CUDA/Isaac 版本，先跑短回合再批量评测。目前未建立整套环境统一的最低显存、内存和磁盘要求。

主机依赖：Linux、兼容的 NVIDIA 驱动、Docker（Compose/Buildx）和 NVIDIA Container Toolkit、Git/Git LFS、Python 3.11+、固定版本的 Rust 工具链、C/C++ 构建工具（CMake、Clang、OpenSSL 开发头文件）、支持用户命名空间的 bubblewrap、FFmpeg/ffprobe 和 zstd。另需准备固定版本的模拟器镜像和资产，获取必要访问权限并接受相应许可，详见[部署指南](docs/DEPLOYMENT.md)。

</details>

### 1. 准备一个环境

克隆后在仓库根目录执行。先恢复固定版本源码，**再恢复资产**。

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' huggingface_hub
export WORLD_PYTHON="$PWD/.venv/bin/python"

bash scripts/setup_handoff.sh sources
# 资产库需要权限时，使用自己的账户登录。
hf auth login
bash scripts/setup_handoff.sh assets --bench robocasa
bash scripts/setup_handoff.sh docker --bench robocasa --dry-run
bash scripts/setup_handoff.sh docker --bench robocasa
bash scripts/setup_handoff.sh codex
```

也可[导入预构建镜像](docs/PREBUILT_IMAGES.md)，代替本地构建；当前需要授权，部分旧配置还需额外镜像。运行时 helper 构建失败时，参见[直接工具调用模式](docs/DEPLOYMENT.md#direct-tool-runtime-option)。其他环境可能需要额外资产、基础镜像及许可。

### 2. 接入模型，运行一道题

接口须能从运行环境访问，并支持流式 **Responses API**、图像和函数调用；仅支持 Chat Completions 的接口不足以运行此发布版。

```bash
python scripts/configure_api.py \
  --base-url https://YOUR_API_HOST/v1 \
  --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"
# 隐藏输入，不将 key 写入命令历史。
read -rsp "Model API key: " WORLD_MODEL_API_KEY; printf '\n'
export WORLD_MODEL_API_KEY
python scripts/check_api.py

bash scripts/run_robocasa.sh list
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --dry-run
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --batch drawer-smoke
```

将接口地址和模型名替换为服务商提供的值。API 检查会发起两次小型付费请求，但不启动模拟器；配置只保存凭据变量名，不保存 key。新开终端时需重新激活虚拟环境并设置环境变量。

### 3. 查看结果，再扩大评测

查看 `outputs/robocasa/CloseDrawer/` 下的 `run.json`、录像和事件。有效任务失败记为 `false`，基础设施错误或无法确定的结果不计分（`null`）。归档时保留整个 run 目录，包括 `artifacts/`。

```bash
# 全部 RoboCasa 任务，每题三次。
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01
# 相同配置下重试未完成的 rollout。
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01 --resume
```

**当前默认值：** 每题 3 次 rollout、环境内 `code_control=off`、每题独立的控制步数预算，以及 `--timeout=43200`（12 小时）。模型推理时仿真暂停，仍允许离线分析；原生终止条件继续生效。`--resume` 从初始状态重跑未完成回合，不恢复中途物理状态。改变评测条件时请使用新 batch 名称。

## 文档

| 文档 | 内容 |
| --- | --- |
| [部署指南](docs/DEPLOYMENT.md) | 主机准备、源码、运行时构建与排错 |
| [资产](docs/ASSETS.md) / [预构建镜像](docs/PREBUILT_IMAGES.md) | 固定版本下载、访问权限和许可 |
| [模型 API](docs/MODELS.md) | 服务接入、凭据和模型配置 |
| [评测指南](scripts/ROLLOUTS.md) | 每题预算、控制模式、全套运行、续跑及统计 |
| [目录说明](DIRECTORY_GUIDE.md) | 源码与输出目录 |
| [发布范围](docs/HANDOFF.md) / [核查记录](docs/RELEASE_REVIEW.md) | 可复现边界与已记录的验证 |

详细文档目前以英文为主。

## 发布状态

发布文档仍记录有全量测试未通过项，真实 API 与模拟器联调验证待完成。历史轨迹不能替代本发布版的重新验证，批量运行前请先验证选定环境。

第三方源码与资产保留各自许可；BEHAVIOR 资产和解密密钥不分发。**RobotWorld 自有代码的顶层许可证尚未选定。**

<details>
<summary>源码来源与验证范围</summary>

本发布版源码来自 `visitworld123/RoboWorld_Scaffold` 的 `main` 分支，提交 `132972eec7ea4e9a5b59f8be7398f19d7681fc52`。CPU 测试可执行 `python -m pytest`。任务列表、启动计划和 API 检查不能替代 GPU rollout 验证；实验性 Isaac 兼容层在运行元数据中记录，不代表与原生模拟器等价。

</details>

## 📚 引用

如果你的研究使用了 RobotWorld，请引用：

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
