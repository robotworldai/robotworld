# RobotWorld

[项目主页](https://robotworldai.github.io/) · [任务录像](https://robotworldai.github.io/zh/#gallery) · [论文](https://arxiv.org/abs/2610.10409) · [English](README.md)

RobotWorld 将 20 个仿真环境中的 84 个机器人控制任务封装为 Harbor 任务。智能体通过 MCP 获取观测并发出动作，模拟器记录轨迹和终止原因，Harbor 在独立容器中评分。

## 安装

需要 Linux amd64、Docker、NVIDIA Container Toolkit、兼容的 GPU 驱动及 `zstd`。以下命令均在仓库根目录执行。

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install 'harbor==0.24.0' PyYAML huggingface_hub
```

## 准备资产和镜像

关抽屉任务使用 RoboCasa 资产。运行其他环境时，用 `--bench` 选择相应资产。20 个模拟器镜像共用一份 12.42 GiB 的压缩归档，每台 Docker 主机导入一次。访问受限仓库时，使用已有读取权限的 Hugging Face 账号登录。

部分任务会挂载上游源码。先按 `sources.lock.json` 恢复相应 checkout，再下载资产。下方双任务示例还需要依次运行 `python scripts/fetch_sources.py humanoid_soccer` 和 `python scripts/download_assets.py --bench humanoid_soccer`。需要全部上游源码时使用 `python scripts/fetch_sources.py --all`。

```bash
python scripts/download_assets.py --bench robocasa
hf download visity/RobotWorld-Images --repo-type dataset \
  --revision 2f8c0e5f33d481c2dfa699dd523041c917a39273 --local-dir var/image-bundle
bash var/image-bundle/import_images.sh var/image-bundle
```

下载脚本校验资产清单和文件哈希，镜像导入脚本校验分卷和镜像 ID。Harbor 从任务目录构建智能体和评分容器。BEHAVIOR-1K 需要单独接受上游许可并获取数据。[资产说明](docs/ASSETS.md) · [镜像说明](docs/PREBUILT_IMAGES.md)

OmniIsaacGymEnvs 需要原版 Isaac Sim 4.0 Gym 扩展及许可归档。使用[扩展提取脚本](third_party/benchmarks/omniisaacgymenvs/compat/extract_gym.py)从固定版本的镜像准备该依赖。

[资产版本](https://huggingface.co/datasets/visity/RobotWorld-Assets/tree/c7fcb40e7c503cc137a24ce53e5198b7222d0798) 包含 OmniDrones 和 RoboLab 的补充资源，其中 5 个文件由下载器从 NVIDIA 获取并校验 SHA-256。[镜像仓库](https://huggingface.co/datasets/visity/RobotWorld-Images) 提供任务包引用的模拟器镜像。

## 配置凭据

在启动 Harbor 的终端设置环境变量，也可以通过密钥管理器注入。使用 Claude Code 时，可在 Bash 中隐藏输入：

```bash
read -rp '模型 ID: ' MODEL_ID
read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY
```

将 `MODEL_ID` 设为账号可用的模型 ID。Claude Code 读取 `ANTHROPIC_API_KEY`，自定义 Messages 服务可另设 `ANTHROPIC_BASE_URL`。使用 Codex 时，设置 `OPENAI_API_KEY`，将运行参数改为 `--agent codex --model "openai/$MODEL_ID"`。Codex 也支持订阅账号登录。Hugging Face CLI 单独管理下载凭据，模拟器使用的回合令牌由启动脚本生成。[智能体配置](docs/HARBOR_BUILTIN_AGENTS.zh-CN.md)

## 运行单个任务

```bash
export WORLD_ASSETS="$PWD/Assets"
bash scripts/harbor_run.sh --path tasks/close-drawer --agent claude-code \
  --model "anthropic/${MODEL_ID:?Set MODEL_ID to your provider model ID}" \
  --n-attempts 1 --n-concurrent 1 --jobs-dir "$PWD/outputs/harbor"
```

## 批量运行

```bash
python scripts/make_harbor_job.py --agent claude-code --model "anthropic/$MODEL_ID" \
  --task close-drawer --task play-soccer --attempts 1 --concurrency 1 \
  --output var/jobs/two-tasks.yaml
bash scripts/harbor_run.sh --config var/jobs/two-tasks.yaml
```

用 `--task all` 选择全部任务。在作业目录下，每次试验的 `agent/` 保存智能体轨迹，`artifacts/simulation/` 保存仿真状态与录像，`verifier/reward.txt` 保存分数。作业根目录的 `result.json` 汇总评分与异常。[任务列表](docs/TASK_NAMES.md) · [任务开发](docs/HARBOR_DEVELOPMENT.zh-CN.md)

## 任务与结果

Harbor 使用 [tasks/](tasks/) 下的目录名作为任务 ID，可用 `ls tasks` 列出任务。每个目录的 `instruction.md` 说明机器人接口和原生步数上限。智能体思考时仿真暂停，任务成功后可提前终止。检查试验目录中的 `result.json` 可确认是否发生异常，`artifacts/simulation/episode.json` 记录实际步数和终止原因。

使用已生成的任务包时，安装该任务的运行依赖即可。重新生成任务包前需恢复对应的上游源码。[开发指南](docs/HARBOR_DEVELOPMENT.zh-CN.md) 介绍任务生成与验证。模拟器代码和数据各自遵循上游许可。

## 社区交流

扫码加入 RobotWorld 社区微信群。此群码在 2026 年 12 月 4 日前有效。

<a href="docs/images/robotworld-community.png"><img src="docs/images/robotworld-community.png" width="300" alt="RobotWorld 社区微信群二维码"></a>

## 引用

```bibtex
@misc{yang2026robotworld,
  title  = {{RobotWorld}: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments},
  author = {Yang, Zhiqin and Li, Chenxin and Hu, Xiaomeng and Liu, Yibin and Huang, Weidong and Sun, Jiankai and Li, Haitao and Wu, Zijian and Huang, Yuzhi and Huang, Fanding and Sun, Hanwen and Liu, Jiashun and Tong, Jingqi and Huang, Mingxin and Hu, Shaoli and Huang, Shijue and Bai, Tianyi and Wang, Xinyuan and Lin, Yunlong and Tang, Zhengyang and Zhang, Zhexin and Chen, Zhuo and Song, Xierui and Dai, Juntao and Chen, Boyuan and Ji, Jiaming and Zhan, Fangneng and Hu, Mengkang and Xue, Wei and Zhang, Yonggang and Hu, Han and Ho, Tsung-Yi and Guo, Yike},
  year   = {2026},
  url    = {https://github.com/robotworldai/robotworld}
}
```
