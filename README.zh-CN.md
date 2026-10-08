<div align="center">

# 🤖 RobotWorld

[English](README.md) | **简体中文**

### 面向多种任务与机器人形态的通用多模态智能体评测

[![项目主页](https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge)](https://robotworldai.github.io/zh/)
[![论文](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2610.10409)
[![代码](https://img.shields.io/badge/Code-GitHub-2ea44f?style=for-the-badge)](https://github.com/robotworldai/robotworld)
[![Hugging Face Papers](https://img.shields.io/badge/Hugging_Face-Papers-ffd21e?style=for-the-badge)](https://huggingface.co/papers/2610.10409)
[![WeChat](https://img.shields.io/badge/WeChat-Community-07C160?style=for-the-badge)](#community)

<a href="https://robotworldai.github.io/zh/#showreel">
  <img src="https://robotworldai.github.io/assets/images/showreel-poster.png" alt="点击观看：45 秒了解 RobotWorld" width="800">
</a>

🎬 **[▶ 45 秒了解 RobotWorld](https://robotworldai.github.io/zh/#showreel)** · [English](https://robotworldai.github.io/#showreel)

🦾 [任务与录像](https://robotworldai.github.io/zh/#gallery) · 📊 [评测结果](https://robotworldai.github.io/zh/#leaderboard) · 🧪 [评测协议](https://robotworldai.github.io/zh/#protocol)

📝 **分析文章：**[中文](https://robotworldai.github.io/blog/zh/) · [英文](https://robotworldai.github.io/blog/)

**84 项任务 · 20 个 benchmark 集成 · 操作、运动、驾驶与飞行**

</div>

---

## ✨ RobotWorld 是什么？

RobotWorld 将从本仓库源码构建的智能体运行时接入机器人模拟器。模型通过明确的观测与动作工具控制机器人，并根据执行反馈修正动作；任务是否完成由环境判据决定，而不是由模型的文字声明决定。

当前发布版注册了 **20 个 benchmark、84 道题**。注册不代表所有任务都已有经过验证的成功轨迹。默认每道题执行 3 次 rollout，关闭环境内 `code_control`；模型推理和离线分析期间仿真暂停。

## 🧭 文档导航

- [完整部署流程](docs/DEPLOYMENT.md)：主机、源码、资产、Docker、运行时构建与验证。
- [资产下载](docs/ASSETS.md)：固定版本、官方来源及需单独授权的数据。
- [预构建镜像](docs/PREBUILT_IMAGES.md)：下载与导入，避免本地逐个构建。
- [模型配置](docs/MODELS.md)：API 地址、模型名和凭据。
- [评测参数](scripts/ROLLOUTS.md)：步数、重复次数、续跑和统计。
- [发布限制](docs/HANDOFF.md)：依赖和可复现范围。

上述详细文档目前以英文为主。本页提供完整的中文启动说明。

## 🛠️ 1. 安装主机依赖和 Python 包

使用 Linux、NVIDIA GPU、兼容的主机驱动，以及配置好 GPU 支持的 Docker、Compose/Buildx 和 NVIDIA Container Toolkit。还需要 Git/Git LFS、Python 3.11+、Rust/rustup、C/C++ 构建工具、bubblewrap（支持用户命名空间）、FFmpeg 和 zstd。不同模拟器的硬件要求以对应集成为准。

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' huggingface_hub
export WORLD_PYTHON="$PWD/.venv/bin/python"
```

后续命令均在仓库根目录执行。新开终端时，重新激活虚拟环境并设置 `WORLD_PYTHON`。主机 Python 负责调度，模拟器依赖放在各自的 Docker 镜像中。

## 📦 2. 准备源码、资产和镜像

先恢复固定版本源码，再恢复资产，避免源码替换导致资产链接丢失：

```bash
python scripts/fetch_sources.py --all --dry-run
bash scripts/setup_handoff.sh sources

# 资产库需要权限时，先用自己的 Hugging Face 账户登录。
hf auth login

# 先准备一个 benchmark；去掉 --bench robocasa 则准备全部分发资产。
bash scripts/setup_handoff.sh assets --bench robocasa
```

资产默认从 [visity/RobotWorld-Assets](https://huggingface.co/datasets/visity/RobotWorld-Assets) 下载，固定版本和清单哈希见 `environment/datasets/asset-source.json`。下载器核对 SHA-256 并恢复运行所需路径，资产放在 `Assets/<benchmark>/`。

已有匹配的资产导出目录时，可以离线复用：

```bash
bash scripts/setup_handoff.sh assets --local-source /path/to/upload-hf
```

**BEHAVIOR 资产不在公共资产包内。** 阅读并接受上游许可后，单独执行：

```bash
python scripts/prepare_behavior_assets.py --accept-license --dry-run
python scripts/prepare_behavior_assets.py --accept-license
```

镜像可直接从 [visity/RobotWorld-Images](https://huggingface.co/datasets/visity/RobotWorld-Images) 下载并导入；当前镜像库需要授权账户，具体命令见[镜像说明](docs/PREBUILT_IMAGES.md)。导入恢复预构建镜像，不需要逐个重新编译。清单包含 20 个镜像；某些旧运行配置需要额外镜像，不能假定镜像包覆盖所有配置。

也可以自行构建，例如 RoboCasa：

```bash
bash scripts/setup_handoff.sh docker --bench robocasa --dry-run
bash scripts/setup_handoff.sh docker --bench robocasa
```

大多数 Isaac 集成需要先准备 `world/robodojo:isaac6.0.1-local` 基础镜像，见[部署说明](docs/DEPLOYMENT.md)。本地已有的镜像、资产和安装可以复用，但需核对源码版本、资产哈希及镜像标签。实验性 Isaac 兼容版本不代表与上游物理行为完全等价。

## ⚙️ 3. 构建本地智能体运行时

```bash
bash scripts/setup_handoff.sh codex
```

使用仓库固定版本的 Codex 源码构建，不依赖系统全局安装的 Codex CLI。构建产物在 `var/build/codex/`。安装细节和编译依赖见[部署说明](docs/DEPLOYMENT.md)。

### 直接工具调用模式

如果 `codex-code-mode-host` 因上游 V8 下载不可用等原因无法构建，可为隔离运行时使用直接工具调用：

```bash
export WORLD_CODEX_DISABLE_CODE_MODE=1
bash scripts/setup_handoff.sh codex
```

评测时也保持该变量。转发层在私有模型目录配置中指定直接工具模式，不改变模型名或 Codex 源码，并记录 `runtime-tool-mode.json`。本次 RoboCasa API 冒烟测试使用此配置。它与环境内 `code_control` 是两个独立设置，不改变机器人动作工具、观测、预算或评分。对比成绩时应记录该配置；其他运行入口需要另行验证。

## 🔌 4. 接入自己的模型 API

默认 API 可从运行环境直接访问。接口需要支持流式 **Responses API**、图像输入、函数调用和工具结果回传；仅有 Chat Completions 接口不足以运行此发布版。

```bash
python scripts/configure_api.py \
  --base-url https://YOUR_API_HOST/v1 \
  --model YOUR_MULTIMODAL_MODEL_ID
export CODEX_AUTH_HOME="$PWD/var/auth/api"

# 隐藏输入，不将 key 写入命令历史或源码。
read -rsp "Model API key: " WORLD_MODEL_API_KEY; printf '\n'
export WORLD_MODEL_API_KEY
python scripts/check_api.py
```

将示例地址和模型名替换成服务提供方给出的值。配置只保存凭据环境变量名，不保存 key。检查会进行两次小型模型请求，验证图像输入和工具调用往返，会产生正常 API 费用，但不会启动模拟器。

模型默认读取配置文件。只在需要覆盖时设置 `MODEL` 或 `--model`。切换服务可用 `--output var/auth/model-b` 新建配置，并将 `CODEX_AUTH_HOME` 指向该目录；原配置不会被自动覆盖。可用 `--key-env OTHER_API_KEY` 指定其他凭据环境变量。推理强度等参数见[模型配置](docs/MODELS.md)。

## 🚀 5. 从一道题开始运行

```bash
# 查看选题和生效的步数、工具、评分配置。
bash scripts/run_robocasa.sh list

# 只检查启动计划，不调用模型或模拟器。
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --dry-run

# 一道题，一次 rollout。
bash scripts/run_robocasa.sh run --tasks CloseDrawer --rollouts 1 --batch drawer-smoke

# 一个 benchmark 的全部选题，每题三次。
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01

# 覆盖某道题的步数。
bash scripts/run_robocasa.sh run --tasks CloseDrawer \
  --task-steps CloseDrawer=450 --rollouts 3 --batch drawer-01

# 为支持代码控制的题单独开启该模式。
bash scripts/run_behavior_1k.sh run --tasks clean_up_your_desk \
  --task-code-control clean_up_your_desk=on --rollouts 2 --batch desk-code-01

# 运行指定的多个 benchmark。
BENCHMARKS='robocasa robolab' ROLLOUTS=2 bash scripts/run_all.sh run --batch subset-01

# 全部依赖准备好后，运行全部注册选题。
ROLLOUTS=3 bash scripts/run_all.sh run --batch model-a-full

# 相同配置、相同 batch 下重试未完成的 rollout。
ROLLOUTS=3 bash scripts/run_robocasa.sh run --batch model-a-01 --resume
```

`--resume` 会从初始状态重跑未完成的 rollout，并保留之前的尝试，不恢复中途物理状态。改变模型、预算、资产或工具条件时，请换一个 batch 名称。

### 可以修改哪些参数

每个 benchmark 对应 `scripts/run_<benchmark>.sh`。脚本顶部可以修改默认值，命令行后面的同名参数优先。

| 参数 | 含义 |
| --- | --- |
| `TASK_STEPS` / `--task-steps` | 每题预算：`native` 为原生预算，`profile` 为场景预算，也可填写不超过已知场景限制的正整数。脚本中是数组。 |
| `CODE_CONTROL` | 整个 benchmark 的环境内反馈代码控制开关，默认 `off`。 |
| `TASK_CODE_CONTROL` / `--task-code-control` | 每题 `on`、`off` 或 `default`；`default` 继承 benchmark 设置。脚本中是数组。 |
| `ROLLOUTS` / `--rollouts` | 每道题的重复次数，默认 3。 |
| `MODEL` / `CODEX_AUTH_HOME` | 模型名与模型配置目录。 |
| `SCORING_PROFILE` | `auto` 为改编题使用已注册的 World 判据，其余使用原生判据；`native` 使用原始目标、预算和评分。 |
| `--tasks` | 仅运行指定任务。 |
| `--batch` | 实验标识，同一实验续跑时保持一致。 |
| `--output-root` | 输出根目录，默认 `outputs/`。 |
| `--timeout` | 每轮墙钟时间上限，默认 43,200 秒，与仿真步数分开。 |
| `--seed` / `--seed-stride` | 对支持随机种子的集成设置初始种子与递增量。 |

BEHAVIOR 默认预算为 `min(原生预算, 2000)`。`CountertopCleanup` 的原生时限尚未核实，采用明确标注的 600 步 World 预算；其他题见任务清单。原生失败终止条件仍生效。

本发布版 RoboDojo、RoboCasa、RoboLab 不提供环境内代码控制，开启参数不会凭空增加该能力，结果中会记录实际为关闭。**关闭 `code_control` 不等于关闭离线计算工具**：模型仍可分析获准读取的观测，但不能绕过动作接口控制仿真。

## 📊 6. 结果、视频与统计

```text
outputs/<benchmark>/<task>/run-<batch>-0001/
  run.json                 # 配置、有效预算、状态、成功与原始分数
  videos/                  # MP4 的相对链接
  events/                  # 模型、工具、环境轨迹及不含图像副本
  programs/                # 生成的控制程序索引
  agent-workspace/         # 模型产物（取决于集成是否支持）
  artifacts/               # 原始输出，归档时必须一并保留
outputs/<benchmark>/summaries/<batch>.json
outputs/<benchmark>/summaries/<batch>.csv
outputs/<benchmark>/summaries/<batch>-runs.csv
```

每题成功率为有效布尔结果的均值；benchmark 汇总对各题成功率等权平均，同时提供汇总 rollout 统计。总体方差与样本方差分别记录，只有一次观测时样本方差无定义。不同单位的原始 score 按题保留，不直接混加。

有效任务失败记为 `false`。基础设施错误、中断或无法确定的结果记为 `null`，不计为模型失败，并单独报告覆盖率。覆盖不完整不能视作完成整个 benchmark。复制结果时保留完整 run 文件夹，以免视频和轨迹相对链接失效。

## 🗂️ 7. 目录与验证范围

| 目录 | 内容 |
| --- | --- |
| `environment/` | 集成、观测和动作接口、运行时、评分与验证代码。 |
| `scripts/` | 安装、资产处理、评测与报告脚本。 |
| `docs/` | 协议、部署、任务说明及历史审计。 |
| `codex/` | 固定版本智能体运行时源码。 |
| `third_party/` | 上游环境源码、外部包装、Docker 构建及许可证。 |
| `sources.lock.json` | 上游仓库与固定提交。 |
| `Assets/` | 安装时下载的资产，不进入代码发布。 |
| `var/`、`outputs/` | 本地构建、认证、缓存和评测产物，不进入代码发布。 |

本发布目录的检查记录见[发布核查](docs/RELEASE_REVIEW.md)。目前全量测试并非全绿，真实 API 与模拟器联调待验证。

CPU 测试可执行 `python -m pytest`。列出任务、检查启动计划、API 连接通过，分别只验证对应环节，不能替代真实 GPU rollout 或成功轨迹验证。历史任务报告不等于本发布版的新增实测结果。

第三方源码与资产保留各自许可证和限制，BEHAVIOR 资产及解密密钥不分发。本快照尚无适用于 RobotWorld 自有代码的顶层许可证，维护者应在正式开源发布前选定。

## 📚 引用

如果你的研究使用了 RobotWorld，请引用：

```bibtex
@misc{yang2026robotworld,
  title  = {{RobotWorld}: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments},
  author = {Yang, Zhiqin and Li, Chenxin and Hu, Xiaomeng and Liu, Yibin and Huang, Weidong and Sun, Jiankai and Li, Haitao and Wu, Zijian and Huang, Yuzhi and Huang, Fanding and Sun, Hanwen and Liu, Jiashun and Tong, Jingqi and Huang, Mingxin and Hu, Shaoli and Huang, Shijue and Bai, Tianyi and Wang, Xinyuan and Lin, Yunlong and Tang, Zhengyang and Zhang, Zhexin and Chen, Zhuo and Song, Xierui and Dai, Juntao and Chen, Boyuan and Ji, Jiaming and Zhan, Fangneng and Hu, Mengkang and Xue, Wei and Zhang, Yonggang and Hu, Han and Ho, Tsung-Yi and Guo, Yike},
  year   = {2026},
  url    = {https://github.com/robotworldai/robotworld}
}
```

<a id="community"></a>

## 欢迎加入 RobotWorld 社区

<a href="docs/images/wechat-community.png"><img src="docs/images/wechat-community.png" alt="RobotWorld 微信群二维码" width="300"></a>
